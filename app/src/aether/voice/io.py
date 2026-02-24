"""
Audio I/O abstractions for voice transports.

Boundary contract:
- Audio input: 16kHz PCM16 mono LE
- Audio output: 24kHz PCM16 mono LE
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator


class AudioInputEvent(Enum):
    """Speech-boundary events emitted by AudioInput."""

    START_OF_SPEECH = "start_of_speech"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class AudioFrame:
    """A chunk of PCM audio with optional speech metadata."""

    data: bytes
    sample_rate: int
    samples_per_channel: int
    event: AudioInputEvent | None = None


class AudioInput(ABC):
    """Audio source implemented by transports."""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[AudioFrame]: ...

    @abstractmethod
    async def close(self) -> None: ...


class AudioOutput(ABC):
    """Audio sink implemented by transports."""

    @abstractmethod
    async def push_frame(self, frame: AudioFrame) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class TextOutput(ABC):
    """Text sink for transcripts and state updates."""

    @abstractmethod
    async def push_text(self, text: str, *, final: bool = False) -> None: ...

    @abstractmethod
    async def push_state(self, state: str) -> None: ...


class NullTextOutput(TextOutput):
    """No-op TextOutput for transports without a text channel."""

    async def push_text(self, text: str, *, final: bool = False) -> None:
        return None

    async def push_state(self, state: str) -> None:
        return None
