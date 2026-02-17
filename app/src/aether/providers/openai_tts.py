"""
OpenAI TTS Provider — text-to-speech synthesis.

Returns MP3 audio bytes. Timeout protection is handled at the processor level.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from aether.core.config import config
from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)


class OpenAITTSProvider(TTSProvider):
    def __init__(self):
        self.client: AsyncOpenAI | None = None

    async def start(self) -> None:
        self.client = AsyncOpenAI()
        logger.info(f"OpenAI TTS provider ready (voice: {config.tts.voice})")

    async def stop(self) -> None:
        self.client = None

    async def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio bytes."""
        if not self.client:
            raise RuntimeError("OpenAI TTS not started")

        response = await self.client.audio.speech.create(
            model=config.tts.model,
            voice=config.tts.voice,
            input=text,
            response_format="mp3",
        )
        return response.content

    async def health_check(self) -> dict:
        status = "ready" if self.client else "not_started"
        return {
            "provider": "openai",
            "model": config.tts.model,
            "voice": config.tts.voice,
            "status": status,
        }
