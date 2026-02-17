"""
Provider base classes — the three clean boundaries.

These abstract classes define what it means to be an STT, LLM, or TTS provider.
Implementations must honor these contracts. That's the whole deal.

v0.05: LLMProvider upgraded to support tool calling via generate_stream_with_tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator


class STTProvider(ABC):
    """Speech-to-text provider interface."""

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str | None:
        ...

    @abstractmethod
    async def connect_stream(self) -> None:
        ...

    @abstractmethod
    async def disconnect_stream(self) -> None:
        ...

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        ...

    @abstractmethod
    async def stream_events(self) -> AsyncGenerator:
        yield  # type: ignore

    @abstractmethod
    async def is_connected(self) -> bool:
        ...

    async def health_check(self) -> dict:
        return {"provider": self.__class__.__name__, "status": "unknown"}


@dataclass
class LLMToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMStreamEvent:
    """
    An event from the LLM stream.

    type:
      - "token": a text token (content is the token string)
      - "tool_calls": the LLM wants to call tools (tool_calls is populated)
      - "done": stream is finished
    """
    type: str  # "token", "tool_calls", "done"
    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)


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

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """
        Stream response with tool calling support.

        Yields LLMStreamEvents. Default wraps generate_stream (no tools).
        Override in providers that support function calling.
        """
        async for token in self.generate_stream(messages, max_tokens, temperature):
            yield LLMStreamEvent(type="token", content=token)
        yield LLMStreamEvent(type="done")

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
