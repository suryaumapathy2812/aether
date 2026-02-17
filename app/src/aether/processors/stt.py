"""
STT Processor — speech-to-text using provider interface.

v0.03: Uses provider abstraction. The STTProcessor handles batch mode.
StreamingSTTProcessor is replaced by the provider's streaming capabilities
directly in main.py (it was never a proper Processor subclass anyway).

The batch STTProcessor remains as a Processor for pipeline compatibility.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from aether.core.frames import Frame, FrameType, text_frame
from aether.core.processor import Processor
from aether.providers.base import STTProvider

logger = logging.getLogger(__name__)


class STTProcessor(Processor):
    """Batch STT — transcribes a complete audio buffer via provider."""

    def __init__(self, provider: STTProvider):
        super().__init__("STT")
        self.provider = provider

    async def start(self) -> None:
        await self.provider.start()

    async def stop(self) -> None:
        await self.provider.stop()

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if frame.type != FrameType.AUDIO:
            yield frame
            return

        try:
            transcript = await self.provider.transcribe(frame.data)
            if transcript:
                logger.info(f"STT: '{transcript[:80]}'")
                yield text_frame(transcript, role="user")
            else:
                logger.debug("STT: empty transcription, skipping")
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
