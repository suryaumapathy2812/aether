"""
WS Notification Sidecar — push-only WebSocket for dashboard.

Receives: notifications, status updates, transcript display.
No voice, no text chat — those go through HTTP and WebRTC respectively.

Client messages:
- notification_feedback: User engaged/dismissed/muted a notification
  → stored as memory fact for future notification filtering

Server pushes:
- notification: Plugin event surfaced by NotificationService
- status: Agent status updates
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from aether.agent import AgentCore

logger = logging.getLogger(__name__)


class WSSidecar:
    """
    Push-only WebSocket for dashboard notifications.

    Each connected WebSocket subscribes to AgentCore notifications.
    On disconnect, the subscription is automatically cleaned up.
    """

    def __init__(self, agent: "AgentCore") -> None:
        self.agent = agent
        self._connections: list[WebSocket] = []

    async def handle_connection(self, ws: WebSocket) -> None:
        """Handle a new sidecar WebSocket connection."""
        await ws.accept()
        self._connections.append(ws)

        # Subscribe to agent notifications
        async def on_notification(notif: dict) -> None:
            await self._send(ws, "notification", notif)

        self.agent.subscribe_notifications(on_notification)
        logger.info("WS sidecar connected (%d total)", len(self._connections))

        try:
            while True:
                # Listen for client messages
                data = await ws.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "notification_feedback":
                    feedback = data.get("data", {})
                    await self._handle_feedback(feedback)

                elif msg_type == "ping":
                    await self._send(ws, "pong", {})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("WS sidecar error: %s", e)
        finally:
            if ws in self._connections:
                self._connections.remove(ws)
            self.agent.unsubscribe_notifications(on_notification)
            logger.info(
                "WS sidecar disconnected (%d remaining)", len(self._connections)
            )

    async def push(self, event_type: str, data: dict) -> None:
        """Push event to all connected sidecar clients."""
        for ws in list(self._connections):
            try:
                await self._send(ws, event_type, data)
            except Exception:
                if ws in self._connections:
                    self._connections.remove(ws)

    async def _send(self, ws: WebSocket, event_type: str, data: dict) -> None:
        """Send a typed JSON message to a WebSocket."""
        await ws.send_json({"type": event_type, "data": data})

    async def _handle_feedback(self, feedback: dict) -> None:
        """Store notification feedback as a memory fact.

        This teaches the notification filter the user's preferences:
        - "engaged" → user reads these notifications
        - "dismissed" → user ignores these
        - "muted" → user wants to mute this sender/plugin
        """
        action = feedback.get("action", "")
        plugin = feedback.get("plugin", "unknown")
        sender = feedback.get("sender", "")

        fact = ""
        if action == "engaged":
            fact = f"User immediately reads {plugin} notifications from {sender}"
        elif action == "dismissed":
            fact = f"User dismisses {plugin} notifications from {sender}"
        elif action == "muted":
            fact = f"User wants to mute all {plugin} notifications from {sender}"

        if fact:
            try:
                await self.agent._memory_store._store_fact(fact, 0)
                logger.info("Notification feedback stored: %s", fact)
            except Exception as e:
                logger.warning("Failed to store notification feedback: %s", e)

    @property
    def connection_count(self) -> int:
        return len(self._connections)
