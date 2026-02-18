"""
Base Transport Interface - Abstract base class for all transports.

All transports (WebSocket, WebRTC, Push, etc.) implement this interface.
The TransportManager communicates with transports through this interface.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

from aether.transport.core_msg import ConnectionState, CoreMsg

logger = logging.getLogger(__name__)


class Transport(ABC):
    """
    Base class for all transports.

    A transport handles the connection between clients and the transport layer.
    Each transport type (WebSocket, WebRTC, Push, etc.) implements this interface.
    """

    # Transport name - must be unique
    name: str = "base"

    def __init__(self):
        self._message_callback: Optional[Callable[[CoreMsg], Awaitable[None]]] = None
        self._connection_callback: Optional[
            Callable[[str, ConnectionState, str], Awaitable[None]]
        ] = None
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """
        Start the transport (begin listening for connections).

        This should set up any servers, webhooks, or connections needed
        to receive client messages.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the transport (graceful shutdown).

        This should cleanly close any connections or stop servers.
        """
        pass

    @abstractmethod
    async def send(self, user_id: str, msg: CoreMsg) -> None:
        """
        Send a message to a specific user.

        Args:
            user_id: The user to send to
            msg: The message to send
        """
        pass

    @abstractmethod
    async def broadcast(self, msg: CoreMsg) -> None:
        """
        Broadcast a message to all connected users.

        Args:
            msg: The message to broadcast
        """
        pass

    @abstractmethod
    async def get_connected_users(self) -> list[str]:
        """
        Get list of connected user IDs.

        Returns:
            List of user IDs currently connected via this transport
        """
        pass

    @abstractmethod
    async def is_connected(self, user_id: str) -> bool:
        """
        Check if a user is connected via this transport.

        Args:
            user_id: The user to check

        Returns:
            True if the user has an active connection
        """
        pass

    @abstractmethod
    async def get_status(self) -> dict:
        """
        Get transport status.

        Returns:
            Dict with status information (connections, errors, etc.)
        """
        pass

    # ─── Callback Registration ────────────────────────────────────

    def on_message(self, callback: Callable[[CoreMsg], Awaitable[None]]) -> None:
        """
        Set callback for incoming messages.

        The transport calls this when it receives a message from a client.
        The callback should process the message and potentially send responses.

        Args:
            callback: Async function that receives CoreMsg
        """
        self._message_callback = callback

    def on_connection_change(
        self, callback: Callable[[str, ConnectionState, str], Awaitable[None]]
    ) -> None:
        """
        Set callback for connection state changes.

        The transport calls this when a client connects or disconnects.

        Args:
            callback: Async function that receives (user_id, state, transport_name)
        """
        self._connection_callback = callback

    # ─── Helper Methods ───────────────────────────────────────────

    async def _notify_message(self, msg: CoreMsg) -> None:
        """Internal: notify message callback."""
        if self._message_callback:
            try:
                await self._message_callback(msg)
            except Exception as e:
                logger.error(
                    f"Error in message callback for {self.name}: {e}", exc_info=True
                )

    async def _notify_connection(self, user_id: str, state: ConnectionState) -> None:
        """Internal: notify connection callback, passing this transport's name."""
        if self._connection_callback:
            try:
                await self._connection_callback(user_id, state, self.name)
            except Exception as e:
                logger.error(
                    f"Error in connection callback for {self.name}: {e}", exc_info=True
                )

    @property
    def is_running(self) -> bool:
        """Check if transport is running."""
        return self._running

    def __repr__(self) -> str:
        return f"<{self.name} Transport>"
