"""
WebSocket Transport — holds WebSocket connections, normalizes to CoreMsg.

This transport is a pure connection handler. It:
  1. Accepts WebSocket connections from FastAPI
  2. Parses the client wire protocol (JSON messages)
  3. Normalizes every message into a CoreMsg
  4. Passes CoreMsg to the manager (which routes to the core)
  5. Receives CoreMsg responses and serializes them back to the client protocol

It does NOT contain any pipeline logic (STT, LLM, TTS, memory, etc.).
That's the core's job.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from aether.core.config import config
from aether.transport.base import Transport
from aether.transport.core_msg import (
    AudioContent,
    ConnectionState,
    CoreMsg,
    EventContent,
    TextContent,
)

logger = logging.getLogger(__name__)


class WebSocketTransport(Transport):
    """
    WebSocket transport — the facade between WebSocket clients and the core.

    Supports:
    - Text messages (dashboard chat)
    - Streaming audio chunks (iOS voice)
    - Batch audio (fallback)
    - Vision (image upload)
    - Session configuration (voice/text mode switching)
    - Notifications (outbound push to clients)

    The transport only speaks the client wire protocol.
    Everything else is CoreMsg in, CoreMsg out.
    """

    name = "websocket"

    def __init__(self):
        super().__init__()
        # session_id → WebSocket
        self._connections: dict[str, WebSocket] = {}
        # session_id → session metadata
        self._sessions: dict[str, dict[str, Any]] = {}
        # user_id → session_id
        self._user_sessions: dict[str, str] = {}

    # ─── Transport Interface ─────────────────────────────────────

    async def start(self) -> None:
        """WebSocket transport doesn't start a server — FastAPI handles that."""
        self._running = True
        logger.info("WebSocket transport ready (accepts connections from FastAPI)")

    async def stop(self) -> None:
        """Stop — close all connections."""
        self._running = False
        for session_id, ws in list(self._connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        self._sessions.clear()
        self._user_sessions.clear()
        logger.info("WebSocket transport stopped")

    async def send(self, user_id: str, msg: CoreMsg) -> None:
        """Send a CoreMsg to a specific user's WebSocket."""
        session_id = self._user_sessions.get(user_id)
        if not session_id or session_id not in self._connections:
            logger.debug(f"No active WS for user {user_id}")
            return
        ws = self._connections[session_id]
        await self._send_ws(ws, msg)

    async def broadcast(self, msg: CoreMsg) -> None:
        """Broadcast a CoreMsg to all connected clients."""
        for session_id, ws in list(self._connections.items()):
            try:
                await self._send_ws(ws, msg)
            except Exception as e:
                logger.debug(f"Broadcast error to {session_id}: {e}")

    async def get_connected_users(self) -> list[str]:
        return list(self._user_sessions.keys())

    async def is_connected(self, user_id: str) -> bool:
        return user_id in self._user_sessions

    async def get_status(self) -> dict:
        return {
            "transport": self.name,
            "connections": len(self._connections),
            "users": len(self._user_sessions),
        }

    # ─── FastAPI Integration ─────────────────────────────────────

    async def handle_connection(self, websocket: WebSocket, user_id: str = "") -> None:
        """
        Handle a new WebSocket connection from FastAPI's /ws endpoint.

        This is the entry point. It:
        1. Accepts the connection
        2. Registers with the manager
        3. Runs the message loop (normalizing → CoreMsg → manager)
        4. Cleans up on disconnect
        """
        await websocket.accept()

        session_id = str(uuid.uuid4())[:8]
        if not user_id:
            user_id = f"anon-{session_id}"

        # Register
        self._connections[session_id] = websocket
        self._user_sessions[user_id] = session_id
        self._sessions[session_id] = {
            "user_id": user_id,
            "mode": "voice",
        }

        # Notify manager of connection
        await self._notify_connection(user_id, ConnectionState.CONNECTED)
        logger.info(f"WS connected: user={user_id}, session={session_id}")

        try:
            await self._message_loop(websocket, session_id, user_id)
        except WebSocketDisconnect:
            logger.info(f"WS disconnected: user={user_id}")
        except Exception as e:
            logger.error(f"WS error: {e}", exc_info=True)
        finally:
            # Notify core of disconnect so it can summarize session, etc.
            disconnect_msg = CoreMsg.event(
                event_type="disconnect",
                user_id=user_id,
                session_id=session_id,
                transport=self.name,
                session_mode=self._sessions.get(session_id, {}).get("mode", "voice"),
            )
            await self._notify_message(disconnect_msg)

            # Clean up local state
            self._connections.pop(session_id, None)
            self._sessions.pop(session_id, None)
            self._user_sessions.pop(user_id, None)
            await self._notify_connection(user_id, ConnectionState.DISCONNECTED)
            logger.info(f"WS cleaned up: session={session_id}")

    # ─── Message Loop (Inbound: Wire Protocol → CoreMsg) ─────────

    async def _message_loop(self, ws: WebSocket, session_id: str, user_id: str) -> None:
        """Read messages from the WebSocket and normalize to CoreMsg."""
        while True:
            raw = await ws.receive_text()
            logger.debug(f"← WS IN ({session_id}): {raw[:200]}")

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                continue

            msg_type = msg.get("type")
            mode = self._sessions[session_id]["mode"]

            # ── Normalize each client message type into a CoreMsg ──

            if msg_type == "session_config":
                new_mode = msg.get("mode", "voice")
                if new_mode in ("text", "voice"):
                    self._sessions[session_id]["mode"] = new_mode
                core_msg = CoreMsg.event(
                    event_type="session_config",
                    user_id=user_id,
                    session_id=session_id,
                    payload={"mode": new_mode},
                    transport=self.name,
                    session_mode=new_mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "stream_start":
                self._sessions[session_id]["mode"] = "voice"
                core_msg = CoreMsg.event(
                    event_type="stream_start",
                    user_id=user_id,
                    session_id=session_id,
                    payload={"reconnect": msg.get("reconnect", False)},
                    transport=self.name,
                    session_mode="voice",
                )
                await self._notify_message(core_msg)

            elif msg_type == "stream_stop":
                core_msg = CoreMsg.event(
                    event_type="stream_stop",
                    user_id=user_id,
                    session_id=session_id,
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "mute":
                core_msg = CoreMsg.event(
                    event_type="mute",
                    user_id=user_id,
                    session_id=session_id,
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "unmute":
                core_msg = CoreMsg.event(
                    event_type="unmute",
                    user_id=user_id,
                    session_id=session_id,
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "text":
                core_msg = CoreMsg.text(
                    text=msg.get("data", ""),
                    user_id=user_id,
                    session_id=session_id,
                    role="user",
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "audio":
                # Batch audio blob
                try:
                    audio_data = base64.b64decode(msg["data"])
                    core_msg = CoreMsg.audio(
                        audio_data=audio_data,
                        user_id=user_id,
                        session_id=session_id,
                        transport=self.name,
                        session_mode=mode,
                    )
                    await self._notify_message(core_msg)
                except Exception as e:
                    logger.error(f"Audio decode error: {e}")

            elif msg_type == "audio_chunk":
                # Streaming audio chunk → forward as event with raw b64
                core_msg = CoreMsg.event(
                    event_type="audio_chunk",
                    user_id=user_id,
                    session_id=session_id,
                    payload={"data": msg.get("data", "")},
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "image":
                core_msg = CoreMsg.event(
                    event_type="image",
                    user_id=user_id,
                    session_id=session_id,
                    payload={
                        "data": msg.get("data", ""),
                        "mime": msg.get("mime", "image/jpeg"),
                    },
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "notification_feedback":
                core_msg = CoreMsg.event(
                    event_type="notification_feedback",
                    user_id=user_id,
                    session_id=session_id,
                    payload=msg.get("data", {}),
                    transport=self.name,
                    session_mode=mode,
                )
                await self._notify_message(core_msg)

            elif msg_type == "pong":
                pass  # Keep-alive, no action

            else:
                logger.warning(f"Unknown WS message type: {msg_type}")

    # ─── Outbound (CoreMsg → Wire Protocol) ──────────────────────

    async def _send_ws(self, ws: WebSocket, msg: CoreMsg) -> None:
        """
        Serialize a CoreMsg to the client wire protocol and send.

        Uses timeout protection to avoid blocking on dead connections.
        Matches the exact JSON format the client expects.
        """
        try:
            if ws.client_state.name != "CONNECTED":
                return

            payload = self._serialize(msg)
            if payload is None:
                return

            json_str = json.dumps(payload)
            logger.debug(f"→ WS OUT: {json_str[:200]}")

            await asyncio.wait_for(
                ws.send_text(json_str),
                timeout=config.server.ws_send_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timeout")
        except Exception as e:
            logger.debug(f"WebSocket send skipped: {e}")

    def _serialize(self, msg: CoreMsg) -> dict | None:
        """
        Convert a CoreMsg to the client wire protocol dict.

        This is the ONLY place that knows the client JSON format.
        """
        content = msg.content
        meta = msg.metadata
        transport = meta.transport

        if isinstance(content, TextContent):
            if transport == "text_chunk" or content.role == "assistant":
                return {
                    "type": "text_chunk",
                    "data": content.text,
                    "index": meta.sentence_index,
                }
            elif transport == "transcript":
                return {"type": "transcript", "data": content.text, "interim": False}
            elif transport == "transcript_interim":
                return {"type": "transcript", "data": content.text, "interim": True}
            else:
                # status, connected, listening, muted, etc.
                return {"type": "status", "data": content.text}

        elif isinstance(content, AudioContent):
            b64 = base64.b64encode(content.audio_data).decode("utf-8")
            extra: dict[str, Any] = {
                "type": "audio_chunk",
                "data": b64,
                "index": meta.sentence_index,
            }
            if transport == "status_audio":
                extra["index"] = -1
                extra["status_audio"] = True
            return extra

        elif isinstance(content, EventContent):
            if content.event_type == "stream_end":
                return {"type": "stream_end"}
            elif content.event_type == "tool_result":
                return {
                    "type": "tool_result",
                    "data": json.dumps(content.payload),
                }
            elif content.event_type == "notification":
                return {
                    "type": "notification",
                    "data": json.dumps(content.payload)
                    if isinstance(content.payload, dict)
                    else content.payload,
                }
            elif content.event_type == "ready":
                return {"type": "status", "data": "listening..."}
            else:
                logger.debug(f"Unhandled event type for WS: {content.event_type}")
                return None

        return None

    def __repr__(self) -> str:
        return f"<WebSocketTransport(connections={len(self._connections)})>"
