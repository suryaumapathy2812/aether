"""
TTS Processor — text-to-speech via provider interface.

v0.03: Uses TTSProvider abstraction. Timeout protection stays here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from aether.core.config import config
from aether.core.frames import Frame, FrameType, audio_frame
from aether.core.processor import Processor
from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)


class TTSProcessor(Processor):
    def __init__(self, provider: TTSProvider):
        super().__init__("TTS")
        self.provider = provider

    async def start(self) -> None:
        await self.provider.start()

    async def stop(self) -> None:
        await self.provider.stop()

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Convert assistant text to audio."""
        if frame.type != FrameType.TEXT or frame.metadata.get("role") != "assistant":
            yield frame
            return

        text = frame.data
        if not text or not text.strip():
            return

        timeout = config.tts.timeout

        try:
            audio_bytes = await asyncio.wait_for(
                self.provider.synthesize(text),
                timeout=timeout,
            )

            logger.debug(
                f"TTS chunk: {len(audio_bytes) / 1024:.0f}KB for '{text[:40]}'"
            )
            yield audio_frame(audio_bytes, sample_rate=24000)
            yield frame  # Also yield text so client can display it

        except asyncio.TimeoutError:
            logger.warning(f"TTS timeout after {timeout}s for: '{text[:40]}'")
            yield frame
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            yield frame
