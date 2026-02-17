"""
TTS Processor — OpenAI text-to-speech.

Takes text frames (assistant responses), yields audio frames.
Uses OpenAI TTS for simplicity and cost-effectiveness in v0.01.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from aether.core.frames import Frame, FrameType, audio_frame
from aether.core.processor import Processor

logger = logging.getLogger(__name__)

# Consistent voice identity — this is a product decision, not a technical one.
VOICE = "nova"  # Warm, natural, conversational
MODEL = "tts-1"  # Faster, slightly lower quality. tts-1-hd for production.
TTS_TIMEOUT = 15.0  # seconds — prevent hanging on slow/failed API calls


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
            audio_bytes = await asyncio.wait_for(
                self._synthesize(text),
                timeout=TTS_TIMEOUT,
            )

            logger.info(f"TTS: generated {len(audio_bytes)} bytes of audio")
            yield audio_frame(audio_bytes, sample_rate=24000)

            # Also yield the text frame so the client can display it
            yield frame

        except asyncio.TimeoutError:
            logger.warning(f"TTS timeout after {TTS_TIMEOUT}s for: '{text[:40]}'")
            yield frame
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            # If TTS fails, still send the text so user gets the response
            yield frame

    async def _synthesize(self, text: str) -> bytes:
        """Call OpenAI TTS and read the full response body.

        Wrapped in a separate method so asyncio.wait_for covers both
        the HTTP request AND the response body read (which can hang
        independently of the initial 200 OK).
        """
        response = await self.client.audio.speech.create(
            model=MODEL,
            voice=VOICE,
            input=text,
            response_format="mp3",
        )
        # .content reads the response body — this is where hangs happen
        # when the connection pool is under pressure
        return response.content
