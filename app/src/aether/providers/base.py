"""
Provider base classes — the three clean boundaries.

These abstract classes define what it means to be an STT, LLM, or TTS provider.
Implementations must honor these contracts. That's the whole deal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class STTProvider(ABC):
    """Speech-to-text provider interface."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize the provider (API clients, connections, etc.)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Clean up resources."""
        ...

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str | None:
        """Transcribe a complete audio buffer. Returns text or None."""
        ...

    @abstractmethod
    async def connect_stream(self) -> None:
        """Open a live streaming connection for real-time transcription."""
        ...

    @abstractmethod
    async def disconnect_stream(self) -> None:
        """Close the live streaming connection."""
        ...

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Send an audio chunk for live transcription."""
        ...

    @abstractmethod
    async def stream_events(self) -> AsyncGenerator:
        """Yield transcription events from the live stream."""
        yield  # type: ignore

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the streaming connection is alive."""
        ...

    async def health_check(self) -> dict:
        """Return provider health status."""
        return {"provider": self.__class__.__name__, "status": "unknown"}


class LLMProvider(ABC):
    """Language model provider interface."""

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens. Yields strings as they arrive."""
        yield  # type: ignore

    async def health_check(self) -> dict:
        return {"provider": self.__class__.__name__, "status": "unknown"}


class TTSProvider(ABC):
    """Text-to-speech provider interface."""

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3)."""
        ...

    async def health_check(self) -> dict:
        return {"provider": self.__class__.__name__, "status": "unknown"}
