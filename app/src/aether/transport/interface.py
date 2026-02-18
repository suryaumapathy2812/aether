"""
Core Interface — abstract interface between transport and Aether Core.

This defines how the TransportManager communicates with the Core.
The CoreHandler implements this using existing Aether components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from aether.transport.core_msg import CoreMsg


class CoreInterface(ABC):
    """
    Abstract interface that the transport layer uses to communicate with the Core.

    The Core (LLMProcessor, Memory, Tools, Plugins) doesn't know about transports.
    It only understands this interface.
    """

    @abstractmethod
    def process_message(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
        """
        Process a message from the transport layer and yield responses.

        This is an async generator — implementations use ``async def`` with ``yield``.

        Args:
            msg: The incoming message from the transport layer

        Yields:
            Response messages from the core (text, audio, status, events, etc.)
        """
        ...  # pragma: no cover
        # Make the type checker happy — this is overridden as an async generator.
        yield  # type: ignore[misc]

    @abstractmethod
    async def send_notification(self, user_id: str, notification: CoreMsg) -> None:
        """
        Send a notification to a specific user.

        Used for plugin events, background updates, etc.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict:
        """Check core health."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the core (initialize providers, etc.)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the core (cleanup providers, etc.)."""
        ...
