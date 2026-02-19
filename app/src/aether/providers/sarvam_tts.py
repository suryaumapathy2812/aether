"""
Sarvam TTS Provider — Indian language text-to-speech via Bulbul v3.

REST API: POST https://api.sarvam.ai/text-to-speech
Returns base64-encoded audio. We decode it to raw bytes.
Uses httpx (already a transitive dep via openai).
"""

from __future__ import annotations

import base64
import logging
import time

import httpx

from aether.core.config import config
from aether.core.metrics import metrics
from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


class SarvamTTSProvider(TTSProvider):
    def __init__(self):
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self.client:
            return  # Already started
        if not config.tts.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY not set")

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.tts.timeout),
            headers={
                "api-subscription-key": config.tts.sarvam_api_key,
                "Content-Type": "application/json",
            },
        )
        logger.info(
            f"Sarvam TTS ready (model={config.tts.sarvam_model}, speaker={config.tts.sarvam_speaker})"
        )

    async def stop(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV)."""
        if not self.client:
            raise RuntimeError("Sarvam TTS not started")

        started = time.time()
        metrics.inc("provider.tts.requests", labels={"provider": "sarvam"})

        try:
            response = await self.client.post(
                SARVAM_TTS_URL,
                json={
                    "text": text,
                    "target_language_code": config.tts.sarvam_language,
                    "model": config.tts.sarvam_model,
                    "speaker": config.tts.sarvam_speaker,
                    "speech_sample_rate": config.tts.sarvam_sample_rate,
                },
            )
            response.raise_for_status()

            data = response.json()
            # API returns {"audios": ["base64..."]}
            audio_b64 = "".join(data["audios"])
            audio = base64.b64decode(audio_b64)

            elapsed_ms = (time.time() - started) * 1000
            metrics.observe(
                "provider.tts.latency_ms", elapsed_ms, labels={"provider": "sarvam"}
            )
            # Estimate audio duration: WAV 16kHz 16-bit mono = 32000 bytes/sec
            # Subtract 44-byte WAV header
            audio_duration_ms = max(0, len(audio) - 44) / 32.0
            metrics.observe("provider.tts.audio_duration_ms", audio_duration_ms)
            return audio
        except Exception:
            metrics.inc("provider.tts.errors", labels={"provider": "sarvam"})
            raise

    async def health_check(self) -> dict:
        status = "ready" if self.client else "not_started"
        return {
            "provider": "sarvam",
            "model": config.tts.sarvam_model,
            "speaker": config.tts.sarvam_speaker,
            "status": status,
        }
