"""
Modality Adapter — abstract base for all modality adapters.

A modality adapter sits between the transport layer and the kernel/LLM layer.
It converts transport-level I/O (CoreMsg) into kernel-level jobs (KernelRequest)
and converts kernel events (KernelEvent) back into transport-level responses (CoreMsg).

Responsibilities:
- Input conversion: CoreMsg → KernelRequest (or LLMRequestEnvelope)
- Output conversion: KernelEvent/LLMEventEnvelope → CoreMsg
- Modality-specific orchestration (STT/TTS for voice, stream framing for text)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator

if TYPE_CHECKING:
    from aether.transport.core_msg import CoreMsg


class ModalityAdapter(ABC):
    """
    Abstract base for modality adapters.

    Each adapter handles a specific modality (text, voice, system) and
    converts between transport-level CoreMsg and kernel-level contracts.
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the modality name: 'text', 'voice', or 'system'."""
        ...

    @abstractmethod
    async def handle_input(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """
        Process an incoming message from the transport layer.

        Converts the transport message into the appropriate processing
        pipeline for this modality, yielding response CoreMsg objects.

        Args:
            msg: Incoming message from transport
            session_state: Mutable session state dict for this session

        Yields:
            Response CoreMsg objects to send back through transport
        """
        ...
        yield  # type: ignore[misc]

    @abstractmethod
    async def handle_output(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """
        Convert a kernel/LLM event into transport-level output.

        Used for push-style events (notifications, background results)
        that originate from the kernel rather than a transport request.

        Args:
            event_type: Type of event (text_chunk, audio_chunk, status, etc.)
            payload: Event payload
            session_state: Mutable session state dict

        Yields:
            CoreMsg objects to send through transport
        """
        ...
        yield  # type: ignore[misc]

    async def on_session_start(self, session_state: dict[str, Any]) -> None:
        """Called when a session starts. Override for setup logic."""

    async def on_session_end(self, session_state: dict[str, Any]) -> None:
        """Called when a session ends. Override for cleanup logic."""
