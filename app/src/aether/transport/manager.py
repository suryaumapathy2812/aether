"""
TransportManager — the central facade that orchestrates all transports.

The TransportManager:
  - Registers and manages transports (WebSocket, WebRTC, Push, etc.)
  - Routes incoming messages from transports → core
  - Routes outgoing messages from core → correct transport(s)
  - Manages per-user connection state
  - Handles notifications and broadcasts
  - Wires the status_audio_callback so background core events reach clients
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from aether.transport.base import Transport
from aether.transport.core_msg import ConnectionState, CoreMsg, MsgDirection
from aether.transport.interface import CoreInterface

logger = logging.getLogger(__name__)


class TransportManager:
    """
    Central manager for all transports.

    This is the facade that orchestrates communication between:
    - Multiple transports (WebSocket, WebRTC, Push, etc.)
    - The Core (LLM, Memory, Tools, Plugins via CoreHandler)
    - Connected clients

    Flow:
      Client → Transport → TransportManager.handle_incoming → Core.process_message
      Core yields CoreMsg → TransportManager.send_to_user → Transport.send → Client
    """

    def __init__(self, core: CoreInterface):
        self.core = core
        self.transports: dict[str, Transport] = {}

        # Track connections: user_id → {transport_name → session_info}
        self._connections: dict[str, dict[str, Any]] = {}

        self._lock = asyncio.Lock()

        # Wire the status_audio_callback so background events
        # (STT interim transcripts, status updates from debounce/trigger)
        # can reach clients without going through process_message's yield.
        if hasattr(core, "set_status_audio_callback"):
            core.set_status_audio_callback(self._handle_background_msg)

    # ─── Transport Registration ──────────────────────────────────

    async def register_transport(self, transport: Transport) -> None:
        """Register a transport with the manager."""
        async with self._lock:
            if transport.name in self.transports:
                logger.warning(
                    f"Transport {transport.name} already registered, replacing"
                )

            # Wire callbacks
            transport.on_message(self.handle_incoming)
            transport.on_connection_change(self._handle_connection_change)

            self.transports[transport.name] = transport
            logger.info(f"Registered transport: {transport.name}")

    async def unregister_transport(self, name: str) -> None:
        """Unregister and stop a transport."""
        async with self._lock:
            if name in self.transports:
                await self.transports[name].stop()
                del self.transports[name]
                logger.info(f"Unregistered transport: {name}")

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start the core and all registered transports."""
        logger.info("Starting transport layer...")

        # Core first
        await self.core.start()

        # Then transports
        for name, transport in self.transports.items():
            try:
                await transport.start()
                logger.info(f"Started transport: {name}")
            except Exception as e:
                logger.error(f"Failed to start transport {name}: {e}", exc_info=True)

        logger.info(f"Transport layer ready: {list(self.transports.keys())}")

    async def stop_all(self) -> None:
        """Stop all transports and the core."""
        logger.info("Stopping transport layer...")

        for name, transport in self.transports.items():
            try:
                await transport.stop()
                logger.info(f"Stopped transport: {name}")
            except Exception as e:
                logger.error(f"Error stopping transport {name}: {e}")

        await self.core.stop()
        self._connections.clear()
        logger.info("Transport layer stopped")

    # ─── Message Routing (Inbound) ───────────────────────────────

    async def handle_incoming(self, msg: CoreMsg) -> None:
        """
        Handle an incoming message from a transport.

        Called by transports when they receive a message from a client.
        Routes through the core and sends responses back to the user.
        """
        msg.direction = MsgDirection.INBOUND

        try:
            async for response in self.core.process_message(msg):
                await self.send_to_user(msg.user_id, response)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error_msg = CoreMsg.text(
                text="Sorry, something went wrong. Please try again.",
                user_id=msg.user_id,
                session_id=msg.session_id,
                role="assistant",
                transport="error",
            )
            await self.send_to_user(msg.user_id, error_msg)

    # ─── Message Routing (Outbound) ──────────────────────────────

    async def send_to_user(self, user_id: str, msg: CoreMsg) -> None:
        """
        Send a message to a specific user.

        Routes to all transports the user is connected through.
        """
        msg.direction = MsgDirection.OUTBOUND
        msg.user_id = user_id

        user_connections = self._connections.get(user_id, {})
        if not user_connections:
            logger.debug(f"No connections for user {user_id}")
            return

        # Send to all transports this user is connected through
        for t_name in user_connections:
            if t_name in self.transports:
                try:
                    await self.transports[t_name].send(user_id, msg)
                except Exception as e:
                    logger.error(f"Error sending to {user_id} via {t_name}: {e}")

    async def broadcast(self, msg: CoreMsg) -> None:
        """Broadcast a message to all connected users."""
        msg.direction = MsgDirection.OUTBOUND
        for user_id in list(self._connections.keys()):
            await self.send_to_user(user_id, msg)

    async def send_notification(self, user_id: str, notification: CoreMsg) -> None:
        """Send a notification to a specific user."""
        await self.send_to_user(user_id, notification)

    # ─── Background Message Callback ─────────────────────────────

    async def _handle_background_msg(self, msg: CoreMsg) -> None:
        """
        Callback for messages generated outside the request/response cycle.

        This handles:
        - STT interim transcripts (from the background STT event loop)
        - Status updates from debounce/trigger (thinking..., listening...)
        - Status audio (fire-and-forget TTS for tool status)
        - Voice pipeline responses triggered by STT streaming

        These messages originate in CoreHandler's background tasks and need
        to be routed to the correct client.
        """
        await self.send_to_user(msg.user_id, msg)

    # ─── Connection Management ───────────────────────────────────

    async def _handle_connection_change(
        self, user_id: str, state: ConnectionState, transport_name: str
    ) -> None:
        """
        Handle connection state changes from transports.

        This is the callback transports call when clients connect/disconnect.
        The transport passes its own name so we know exactly which transport
        triggered the change — no need to poll all transports.
        """
        async with self._lock:
            if state == ConnectionState.CONNECTED:
                if user_id not in self._connections:
                    self._connections[user_id] = {}
                self._connections[user_id][transport_name] = {"connected": True}
                logger.info(f"User {user_id} connected via {transport_name}")

            elif state == ConnectionState.DISCONNECTED:
                if user_id in self._connections:
                    self._connections[user_id].pop(transport_name, None)
                    # Remove user entirely if no transports left
                    if not self._connections[user_id]:
                        del self._connections[user_id]
                logger.info(f"User {user_id} disconnected from {transport_name}")

            elif state == ConnectionState.RECONNECTED:
                if user_id not in self._connections:
                    self._connections[user_id] = {}
                self._connections[user_id][transport_name] = {"connected": True}
                logger.info(f"User {user_id} reconnected via {transport_name}")

    # ─── Status & Info ───────────────────────────────────────────

    async def get_status(self) -> dict:
        """Get status of all transports and the core."""
        transport_status = {}
        for name, transport in self.transports.items():
            try:
                transport_status[name] = await transport.get_status()
            except Exception as e:
                transport_status[name] = {"error": str(e)}

        try:
            core_health = await self.core.health_check()
        except Exception as e:
            core_health = {"error": str(e)}

        return {
            "core": core_health,
            "transports": transport_status,
            "connections": {
                uid: list(conns.keys()) for uid, conns in self._connections.items()
            },
        }

    async def get_connected_users(self) -> list[str]:
        """Get list of all connected user IDs."""
        return list(self._connections.keys())

    async def is_user_connected(self, user_id: str) -> bool:
        """Check if a user has any active connections."""
        return user_id in self._connections and len(self._connections[user_id]) > 0

    def get_transport(self, name: str) -> Optional[Transport]:
        """Get a transport by name."""
        return self.transports.get(name)

    def __repr__(self) -> str:
        return (
            f"<TransportManager(transports={list(self.transports.keys())}, "
            f"users={len(self._connections)})>"
        )
