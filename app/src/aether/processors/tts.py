"""
TTS Processor — OpenAI text-to-speech.

Takes text frames (assistant responses), yields audio frames.
Uses OpenAI TTS for simplicity and cost-effectiveness in v0.01.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from aether.core.frames import Frame, FrameType, audio_frame
from aether.core.processor import Processor

logger = logging.getLogger(__name__)

# Consistent voice identity — this is a product decision, not a technical one.
VOICE = "nova"  # Warm, natural, conversational
MODEL = "tts-1"  # Faster, slightly lower quality. tts-1-hd for production.


class TTSProcessor(Processor):
    def __init__(self):
        super().__init__("TTS")
        self.client = AsyncOpenAI()

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Convert assistant text to audio."""
        # Only process assistant text frames
        if frame.type != FrameType.TEXT or frame.metadata.get("role") != "assistant":
            yield frame
            return

        text = frame.data
        if not text or not text.strip():
            return

        try:
            response = await self.client.audio.speech.create(
                model=MODEL,
                voice=VOICE,
                input=text,
                response_format="mp3",
            )

            audio_bytes = response.content
            logger.info(f"TTS: generated {len(audio_bytes)} bytes of audio")

            yield audio_frame(audio_bytes, sample_rate=24000)

            # Also yield the text frame so the client can display it
            yield frame

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            # If TTS fails, still send the text so user gets the response
            yield frame
