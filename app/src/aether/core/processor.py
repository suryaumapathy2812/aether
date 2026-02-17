"""
Aether Processor — the building block of the pipeline.

A Processor takes frames in and yields frames out. That's the entire interface.
Every pipeline stage (STT, LLM, TTS, Memory) implements this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from aether.core.frames import Frame


class Processor(ABC):
    """Base class for all pipeline processors."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    async def start(self) -> None:
        """Called once when the pipeline starts. Override for setup."""
        pass

    @abstractmethod
    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """
        Process a frame and yield zero or more output frames.

        This is the only method you must implement.
        """
        # Must be an async generator (use yield)
        yield  # type: ignore  # pragma: no cover

    async def stop(self) -> None:
        """Called once when the pipeline stops. Override for cleanup."""
        pass

    def __repr__(self) -> str:
        return f"<{self.name}>"
