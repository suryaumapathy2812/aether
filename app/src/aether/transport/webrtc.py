"""
SmallWebRTC Transport — self-hosted WebRTC via aiortc.

This transport handles WebRTC peer connections for real-time voice.
It:
  1. Accepts SDP offers via HTTP signaling endpoints
  2. Creates aiortc peer connections with audio transceivers
  3. Reads incoming raw PCM audio → normalizes to CoreMsg
  4. Receives CoreMsg responses → pushes audio to RawAudioTrack → WebRTC
  5. Supports a data channel for text messages, events, status updates
     (same JSON protocol as WebSocket so clients can reuse message handling)

It does NOT contain any pipeline logic (STT, LLM, TTS, memory, etc.).
That's the core's job via CoreHandler.

Requires: pip install aiortc av numpy
If aiortc is not installed, the transport gracefully skips registration.
"""

from __future__ import annotations

import asyncio
import base64
import fractions
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from aether.transport.base import Transport
from aether.transport.core_msg import (
    AudioContent,
    ConnectionState,
    CoreMsg,
    EventContent,
    MsgMetadata,
    TextContent,
)

logger = logging.getLogger(__name__)

# ── Optional aiortc imports (graceful degradation) ────────────

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from aiortc.sdp import candidate_from_sdp
    from av import AudioFrame

    AIORTC_AVAILABLE = True
except ModuleNotFoundError:
    AIORTC_AVAILABLE = False
    AudioStreamTrack = object  # type: ignore[misc,assignment]
    logger.info(
        "aiortc not installed — WebRTC transport unavailable. "
        "Install with: pip install aiortc av"
    )


# ── RawAudioTrack (outbound audio to client) ─────────────────


class RawAudioTrack(AudioStreamTrack):  # type: ignore[misc]
    """
    Custom AudioStreamTrack that queues PCM audio in 10ms chunks.

    The CoreHandler yields CoreMsg with AudioContent (TTS output).
    SmallWebRTCTransport converts that to raw bytes and pushes them
    here. aiortc calls recv() at its own pace to feed the WebRTC
    connection.

    Adapted from Pipecat's RawAudioTrack pattern.

    Only usable when aiortc is installed.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 24000):
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc is required for RawAudioTrack")
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_10ms = sample_rate * 10 // 1000
        self._bytes_per_10ms = self._samples_per_10ms * 2  # 16-bit PCM
        self._timestamp = 0
        self._start = time.time()
        self._chunk_queue: deque[bytes] = deque()

    def add_audio_bytes(self, audio_bytes: bytes) -> None:
        """
        Queue raw PCM bytes for transmission.

        Breaks input into 10ms chunks. If the input is not a multiple
        of 10ms, the last chunk is zero-padded.
        """
        for i in range(0, len(audio_bytes), self._bytes_per_10ms):
            chunk = audio_bytes[i : i + self._bytes_per_10ms]
            # Pad last chunk if needed
            if len(chunk) < self._bytes_per_10ms:
                chunk = chunk + bytes(self._bytes_per_10ms - len(chunk))
            self._chunk_queue.append(chunk)

    async def recv(self):
        """Return the next 10ms audio frame (or silence if queue empty)."""
        # Timing synchronization
        if self._timestamp > 0:
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        if self._chunk_queue:
            chunk = self._chunk_queue.popleft()
        else:
            chunk = bytes(self._bytes_per_10ms)  # silence

        samples = np.frombuffer(chunk, dtype=np.int16)
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += self._samples_per_10ms
        return frame


# ── WebRTC Session (per peer connection) ─────────────────────


@dataclass
class WebRTCSession:
    """Tracks state for a single WebRTC peer connection."""

    pc_id: str
    pc: Any  # RTCPeerConnection
    user_id: str
    audio_track: Optional[RawAudioTrack] = None
    data_channel: Any = None  # RTCDataChannel
    mode: str = "voice"
    connected: bool = False
    audio_read_task: Optional[asyncio.Task] = None
    last_ping_time: float = field(default_factory=time.time)


# ── SmallWebRTCTransport ─────────────────────────────────────


class SmallWebRTCTransport(Transport):
    """
    WebRTC transport using aiortc (self-hosted, no external service).

    Signaling flow:
      1. Client POSTs SDP offer to /webrtc/offer → gets SDP answer + pc_id
      2. Client PATCHes ICE candidates to /webrtc/ice
      3. WebRTC peer connection established
      4. Raw audio flows bidirectionally
      5. Data channel carries text/events (same JSON as WebSocket)

    Audio flow:
      Inbound:  WebRTC audio track → read PCM → CoreMsg.event("audio_chunk") → CoreHandler
      Outbound: CoreHandler yields CoreMsg(AudioContent) → RawAudioTrack → WebRTC

    Data channel flow (same JSON protocol as WebSocket):
      Inbound:  {"type": "text", "data": "hello"} → CoreMsg.text()
      Outbound: {"type": "text_chunk", "data": "...", "index": 0}
    """

    name = "webrtc"

    def __init__(
        self,
        ice_servers: Optional[list[dict[str, Any]]] = None,
        sample_rate_in: int = 16000,
        sample_rate_out: int = 24000,
    ):
        """
        Args:
            ice_servers: List of ICE server configs, e.g.
                [{"urls": "stun:stun.l.google.com:19302"}]
                None = no STUN/TURN (works on local network).
            sample_rate_in: Expected input audio sample rate from client.
            sample_rate_out: Output audio sample rate for TTS playback.
        """
        super().__init__()

        if not AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc is required for WebRTC transport. "
                "Install with: pip install aiortc av"
            )

        self._ice_servers: list[RTCIceServer] = []
        if ice_servers:
            for srv in ice_servers:
                self._ice_servers.append(RTCIceServer(**srv))

        self._sample_rate_in = sample_rate_in
        self._sample_rate_out = sample_rate_out

        # pc_id → WebRTCSession
        self._sessions: dict[str, WebRTCSession] = {}
        # user_id → pc_id (for routing outbound messages)
        self._user_sessions: dict[str, str] = {}

        self._lock = asyncio.Lock()

    # ─── Transport Interface ─────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info("WebRTC transport ready (signaling via /webrtc/offer, /webrtc/ice)")

    async def stop(self) -> None:
        self._running = False
        async with self._lock:
            for session in list(self._sessions.values()):
                await self._close_session(session)
            self._sessions.clear()
            self._user_sessions.clear()
        logger.info("WebRTC transport stopped")

    async def send(self, user_id: str, msg: CoreMsg) -> None:
        """Send a CoreMsg to a user's WebRTC connection."""
        pc_id = self._user_sessions.get(user_id)
        if not pc_id or pc_id not in self._sessions:
            logger.debug(f"No active WebRTC session for user {user_id}")
            return

        session = self._sessions[pc_id]
        await self._send_to_session(session, msg)

    async def broadcast(self, msg: CoreMsg) -> None:
        for session in list(self._sessions.values()):
            try:
                await self._send_to_session(session, msg)
            except Exception as e:
                logger.debug(f"Broadcast error to {session.pc_id}: {e}")

    async def get_connected_users(self) -> list[str]:
        return list(self._user_sessions.keys())

    async def is_connected(self, user_id: str) -> bool:
        return user_id in self._user_sessions

    async def get_status(self) -> dict:
        return {
            "transport": self.name,
            "connections": len(self._sessions),
            "users": len(self._user_sessions),
            "aiortc_available": AIORTC_AVAILABLE,
        }

    # ─── Signaling: SDP Offer/Answer ─────────────────────────────

    async def handle_offer(
        self,
        sdp: str,
        sdp_type: str,
        user_id: str = "",
        pc_id: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Handle an SDP offer from a client.

        Creates (or reuses) a peer connection, sets up audio transceivers
        and data channel, returns the SDP answer.

        Args:
            sdp: The SDP offer string.
            sdp_type: The SDP type (usually "offer").
            user_id: The user ID (from auth/query param).
            pc_id: Optional existing peer connection ID for renegotiation.

        Returns:
            {"sdp": str, "type": str, "pc_id": str}
        """
        async with self._lock:
            # Renegotiation: reuse existing connection
            if pc_id and pc_id in self._sessions:
                session = self._sessions[pc_id]
                logger.info(f"Renegotiating WebRTC for pc_id={pc_id}")
                await self._renegotiate(session, sdp, sdp_type)
                answer = session.pc.localDescription
                return {
                    "sdp": answer.sdp,
                    "type": answer.type,
                    "pc_id": session.pc_id,
                }

            # New connection
            new_pc_id = f"pc-{uuid.uuid4().hex[:12]}"
            if not user_id:
                user_id = f"anon-{new_pc_id[:8]}"

            # If user already has a session, close the old one
            old_pc_id = self._user_sessions.get(user_id)
            if old_pc_id and old_pc_id in self._sessions:
                logger.info(f"Replacing old WebRTC session for user {user_id}")
                await self._close_session(self._sessions[old_pc_id])

            rtc_config = RTCConfiguration(iceServers=self._ice_servers)
            pc = RTCPeerConnection(rtc_config)

            # Check if the client's offer includes audio
            # (dashboard = data-channel only, iOS = audio + data channel)
            offer_has_audio = "m=audio" in sdp

            # Create outbound audio track only if client offered audio
            audio_track: Optional[RawAudioTrack] = None
            if offer_has_audio:
                audio_track = RawAudioTrack(sample_rate=self._sample_rate_out)

            session = WebRTCSession(
                pc_id=new_pc_id,
                pc=pc,
                user_id=user_id,
                audio_track=audio_track,
                mode="voice" if offer_has_audio else "text",
            )

            # Set up event handlers
            self._setup_pc_handlers(session)

            # Process the SDP offer
            offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
            await pc.setRemoteDescription(offer)

            # Add outbound audio track only if client offered audio
            # (adding a track to a data-channel-only connection causes
            # aiortc to fail with "None is not in list" direction error)
            if audio_track is not None:
                pc.addTrack(audio_track)

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Store session
            self._sessions[new_pc_id] = session
            self._user_sessions[user_id] = new_pc_id

            local_desc = pc.localDescription
            logger.info(f"WebRTC offer handled: user={user_id}, pc_id={new_pc_id}")

            return {
                "sdp": local_desc.sdp,
                "type": local_desc.type,
                "pc_id": new_pc_id,
            }

    async def handle_ice_candidate(
        self,
        pc_id: str,
        candidate: str,
        sdp_mid: Optional[str] = None,
        sdp_mline_index: Optional[int] = None,
    ) -> None:
        """
        Add a remote ICE candidate to a peer connection.

        Args:
            pc_id: The peer connection ID.
            candidate: The ICE candidate SDP string.
            sdp_mid: The SDP mid for the candidate.
            sdp_mline_index: The SDP mline index.
        """
        session = self._sessions.get(pc_id)
        if not session:
            raise ValueError(f"Unknown peer connection: {pc_id}")

        ice_candidate = candidate_from_sdp(candidate)
        if sdp_mid is not None:
            ice_candidate.sdpMid = sdp_mid
        if sdp_mline_index is not None:
            ice_candidate.sdpMLineIndex = sdp_mline_index

        await session.pc.addIceCandidate(ice_candidate)
        logger.debug(f"Added ICE candidate for pc_id={pc_id}")

    # ─── Peer Connection Event Handlers ──────────────────────────

    def _setup_pc_handlers(self, session: WebRTCSession) -> None:
        """Wire up aiortc event handlers for a peer connection."""
        pc = session.pc

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = pc.connectionState
            logger.info(f"WebRTC connection state: {state} (pc_id={session.pc_id})")

            if state == "connected":
                session.connected = True
                await self._notify_connection(
                    session.user_id, ConnectionState.CONNECTED
                )
                # Send a stream_start event so CoreHandler initializes
                # the voice session (STT, greeting, etc.)
                start_msg = CoreMsg.event(
                    event_type="stream_start",
                    user_id=session.user_id,
                    session_id=session.pc_id,
                    payload={"reconnect": False},
                    transport=self.name,
                    session_mode="voice",
                )
                await self._notify_message(start_msg)

            elif state in ("disconnected", "failed", "closed"):
                if session.connected:
                    session.connected = False
                    await self._handle_disconnect(session)

        @pc.on("track")
        async def on_track(track):
            logger.info(
                f"WebRTC track received: kind={track.kind} (pc_id={session.pc_id})"
            )
            if track.kind == "audio":
                session.audio_read_task = asyncio.create_task(
                    self._read_audio_loop(session, track)
                )

                @track.on("ended")
                async def on_track_ended():
                    logger.info(f"Audio track ended (pc_id={session.pc_id})")
                    if session.audio_read_task:
                        session.audio_read_task.cancel()

        @pc.on("datachannel")
        async def on_datachannel(channel):
            logger.info(f"Data channel opened: {channel.label} (pc_id={session.pc_id})")
            session.data_channel = channel

            @channel.on("message")
            async def on_message(message):
                await self._handle_data_channel_message(session, message)

    # ─── Inbound Audio (WebRTC → CoreMsg) ────────────────────────

    async def _read_audio_loop(self, session: WebRTCSession, track) -> None:
        """
        Background task: read PCM audio from the WebRTC audio track
        and forward as CoreMsg to the core.

        Each recv() returns ~20ms of audio. We convert to base64 and
        send as audio_chunk events (same format as WebSocket streaming).
        """
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    if session.connected:
                        logger.debug(f"Audio recv timeout (pc_id={session.pc_id})")
                    continue
                except MediaStreamError:
                    logger.info(f"Audio stream ended (pc_id={session.pc_id})")
                    break

                # Convert AudioFrame → raw PCM bytes
                pcm_array = frame.to_ndarray().astype(np.int16)
                pcm_bytes = pcm_array.tobytes()

                # Forward as audio_chunk event (matches WebSocket protocol)
                b64_data = base64.b64encode(pcm_bytes).decode("utf-8")
                core_msg = CoreMsg.event(
                    event_type="audio_chunk",
                    user_id=session.user_id,
                    session_id=session.pc_id,
                    payload={"data": b64_data},
                    transport=self.name,
                    session_mode="voice",
                )
                await self._notify_message(core_msg)

        except asyncio.CancelledError:
            logger.debug(f"Audio read loop cancelled (pc_id={session.pc_id})")
        except Exception as e:
            logger.error(
                f"Audio read loop error (pc_id={session.pc_id}): {e}",
                exc_info=True,
            )

    # ─── Data Channel Messages (same JSON protocol as WebSocket) ─

    async def _handle_data_channel_message(
        self, session: WebRTCSession, message: str
    ) -> None:
        """
        Handle a message from the WebRTC data channel.

        Supports the same JSON protocol as the WebSocket transport
        so clients can reuse their message handling code.
        """
        # Ping/pong keep-alive
        if isinstance(message, str) and message.startswith("ping"):
            session.last_ping_time = time.time()
            if session.data_channel and session.data_channel.readyState == "open":
                session.data_channel.send("pong")
            return

        try:
            msg = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid data channel message: {message[:100]}")
            return

        msg_type = msg.get("type")

        if msg_type == "text":
            core_msg = CoreMsg.text(
                text=msg.get("data", ""),
                user_id=session.user_id,
                session_id=session.pc_id,
                role="user",
                transport=self.name,
                session_mode=session.mode,
            )
            await self._notify_message(core_msg)

        elif msg_type == "session_config":
            new_mode = msg.get("mode", "voice")
            if new_mode in ("text", "voice"):
                session.mode = new_mode
            core_msg = CoreMsg.event(
                event_type="session_config",
                user_id=session.user_id,
                session_id=session.pc_id,
                payload={"mode": new_mode},
                transport=self.name,
                session_mode=new_mode,
            )
            await self._notify_message(core_msg)

        elif msg_type == "mute":
            core_msg = CoreMsg.event(
                event_type="mute",
                user_id=session.user_id,
                session_id=session.pc_id,
                transport=self.name,
                session_mode=session.mode,
            )
            await self._notify_message(core_msg)

        elif msg_type == "unmute":
            core_msg = CoreMsg.event(
                event_type="unmute",
                user_id=session.user_id,
                session_id=session.pc_id,
                transport=self.name,
                session_mode=session.mode,
            )
            await self._notify_message(core_msg)

        elif msg_type == "image":
            core_msg = CoreMsg.event(
                event_type="image",
                user_id=session.user_id,
                session_id=session.pc_id,
                payload={
                    "data": msg.get("data", ""),
                    "mime": msg.get("mime", "image/jpeg"),
                },
                transport=self.name,
                session_mode=session.mode,
            )
            await self._notify_message(core_msg)

        elif msg_type == "notification_feedback":
            core_msg = CoreMsg.event(
                event_type="notification_feedback",
                user_id=session.user_id,
                session_id=session.pc_id,
                payload=msg.get("data", {}),
                transport=self.name,
                session_mode=session.mode,
            )
            await self._notify_message(core_msg)

        elif msg_type == "pong":
            pass  # Keep-alive ack

        else:
            logger.debug(f"Unknown data channel message type: {msg_type}")

    # ─── Outbound (CoreMsg → WebRTC) ─────────────────────────────

    async def _send_to_session(self, session: WebRTCSession, msg: CoreMsg) -> None:
        """
        Send a CoreMsg to a WebRTC session.

        Audio → push to RawAudioTrack (WebRTC audio stream)
        Text/Events → serialize to JSON and send via data channel
        """
        content = msg.content

        if isinstance(content, AudioContent):
            # Push raw PCM to the outbound audio track
            if session.audio_track:
                session.audio_track.add_audio_bytes(content.audio_data)

        elif isinstance(content, (TextContent, EventContent)):
            # Serialize to JSON (same format as WebSocket) and send via data channel
            payload = self._serialize_for_data_channel(msg)
            if payload and session.data_channel:
                if session.data_channel.readyState == "open":
                    try:
                        session.data_channel.send(json.dumps(payload))
                    except Exception as e:
                        logger.debug(f"Data channel send error: {e}")

    def _serialize_for_data_channel(self, msg: CoreMsg) -> dict | None:
        """
        Convert a CoreMsg to the client JSON protocol.

        Same format as WebSocketTransport._serialize() so clients
        can reuse their message handling code.
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
                return {"type": "status", "data": content.text}

        elif isinstance(content, AudioContent):
            # Audio goes via WebRTC track, not data channel
            # But if someone explicitly sends audio as event, base64 it
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
                logger.debug(
                    f"Unhandled event type for data channel: {content.event_type}"
                )
                return None

        return None

    # ─── Session Lifecycle ───────────────────────────────────────

    async def _handle_disconnect(self, session: WebRTCSession) -> None:
        """Handle a WebRTC peer disconnection."""
        # Notify core of disconnect (session summary, cleanup, etc.)
        disconnect_msg = CoreMsg.event(
            event_type="disconnect",
            user_id=session.user_id,
            session_id=session.pc_id,
            transport=self.name,
            session_mode=session.mode,
        )
        await self._notify_message(disconnect_msg)

        # Clean up
        async with self._lock:
            self._sessions.pop(session.pc_id, None)
            if self._user_sessions.get(session.user_id) == session.pc_id:
                self._user_sessions.pop(session.user_id, None)

        await self._notify_connection(session.user_id, ConnectionState.DISCONNECTED)
        logger.info(
            f"WebRTC disconnected: user={session.user_id}, pc_id={session.pc_id}"
        )

    async def _close_session(self, session: WebRTCSession) -> None:
        """Close a peer connection and clean up resources."""
        if session.audio_read_task:
            session.audio_read_task.cancel()
            try:
                await session.audio_read_task
            except asyncio.CancelledError:
                pass

        try:
            await session.pc.close()
        except Exception as e:
            logger.debug(f"Error closing peer connection: {e}")

        self._sessions.pop(session.pc_id, None)
        if self._user_sessions.get(session.user_id) == session.pc_id:
            self._user_sessions.pop(session.user_id, None)

        logger.debug(f"Closed WebRTC session: pc_id={session.pc_id}")

    async def _renegotiate(
        self, session: WebRTCSession, sdp: str, sdp_type: str
    ) -> None:
        """Renegotiate an existing peer connection with a new SDP offer."""
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await session.pc.setRemoteDescription(offer)
        answer = await session.pc.createAnswer()
        await session.pc.setLocalDescription(answer)
        logger.info(f"Renegotiated WebRTC: pc_id={session.pc_id}")

    # ─── Helpers ─────────────────────────────────────────────────

    def get_session_by_pc_id(self, pc_id: str) -> Optional[WebRTCSession]:
        """Get a session by peer connection ID (for signaling endpoints)."""
        return self._sessions.get(pc_id)

    def __repr__(self) -> str:
        return f"<SmallWebRTCTransport(connections={len(self._sessions)})>"
