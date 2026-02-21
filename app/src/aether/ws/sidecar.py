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
- task_completed: Sub-agent task finished (from EventBus)

Session-level streaming (text chunks, tool calls) is SSE-only via
/v1/sessions/{id}/events. The WS sidecar handles system-level
notifications, not per-session event streams.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.kernel.event_bus import EventBus

logger = logging.getLogger(__name__)


class WSSidecar:
    """
    Push-only WebSocket for dashboard notifications.

    Each connected WebSocket subscribes to AgentCore notifications
    and EventBus task.completed events. On disconnect, subscriptions
    are automatically cleaned up.
    """

    def __init__(
        self,
        agent: "AgentCore",
        event_bus: "EventBus | None" = None,
    ) -> None:
        self.agent = agent
        self._event_bus = event_bus
        self._connections: list[WebSocket] = []

    async def handle_connection(self, ws: WebSocket) -> None:
        """Handle a new sidecar WebSocket connection."""
        await ws.accept()
        self._connections.append(ws)

        # Subscribe to agent notifications
        async def on_notification(notif: dict) -> None:
            await self._send(ws, "notification", notif)

        self.agent.subscribe_notifications(on_notification)

        # Subscribe to EventBus task.completed events
        event_queue = None
        event_listener_task = None
        if self._event_bus is not None:
            event_queue = self._event_bus.subscribe("task.completed")
            event_listener_task = asyncio.create_task(
                self._forward_events(ws, event_queue)
            )

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

            # Clean up EventBus subscription
            if event_listener_task is not None:
                event_listener_task.cancel()
                try:
                    await event_listener_task
                except asyncio.CancelledError:
                    pass
            if self._event_bus is not None and event_queue is not None:
                self._event_bus.unsubscribe("task.completed", event_queue)

            logger.info(
                "WS sidecar disconnected (%d remaining)", len(self._connections)
            )

    async def _forward_events(self, ws: WebSocket, queue: asyncio.Queue) -> None:
        """Forward EventBus events to a WebSocket connection."""
        try:
            assert self._event_bus is not None
            async for event in self._event_bus.listen(queue):
                try:
                    await self._send(ws, "task_completed", event)
                except Exception:
                    break  # WebSocket closed
        except asyncio.CancelledError:
            pass

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
