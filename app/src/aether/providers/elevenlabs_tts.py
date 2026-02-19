"""
ElevenLabs TTS Provider — high-quality text-to-speech.

REST API: POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
Returns raw audio bytes directly (no base64).
Uses httpx (already a transitive dep via openai).
"""

from __future__ import annotations

import logging
import time

import httpx

from aether.core.config import config
from aether.core.metrics import metrics
from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsTTSProvider(TTSProvider):
    def __init__(self):
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self.client:
            return  # Already started
        if not config.tts.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.tts.timeout),
            headers={
                "xi-api-key": config.tts.elevenlabs_api_key,
                "Content-Type": "application/json",
            },
        )
        logger.info(
            f"ElevenLabs TTS ready (model={config.tts.elevenlabs_model}, voice={config.tts.elevenlabs_voice_id})"
        )

    async def stop(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio bytes."""
        if not self.client:
            raise RuntimeError("ElevenLabs TTS not started")

        started = time.time()
        metrics.inc("provider.tts.requests", labels={"provider": "elevenlabs"})

        try:
            url = f"{ELEVENLABS_TTS_URL}/{config.tts.elevenlabs_voice_id}"
            response = await self.client.post(
                url,
                json={
                    "text": text,
                    "model_id": config.tts.elevenlabs_model,
                },
            )
            response.raise_for_status()
            audio = response.content

            elapsed_ms = (time.time() - started) * 1000
            metrics.observe(
                "provider.tts.latency_ms",
                elapsed_ms,
                labels={"provider": "elevenlabs"},
            )
            return audio
        except Exception:
            metrics.inc("provider.tts.errors", labels={"provider": "elevenlabs"})
            raise

    async def health_check(self) -> dict:
        status = "ready" if self.client else "not_started"
        return {
            "provider": "elevenlabs",
            "model": config.tts.elevenlabs_model,
            "voice_id": config.tts.elevenlabs_voice_id,
            "status": status,
        }
