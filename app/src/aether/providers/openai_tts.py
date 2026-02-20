"""
OpenAI TTS Provider — text-to-speech synthesis.

Returns raw PCM audio (24kHz, 16-bit mono, little-endian).
Supports both batch and streaming synthesis.
Timeout protection is handled at the processor level.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from openai import AsyncOpenAI

from aether.core.config import config
from aether.core.metrics import metrics
from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)

# Streaming chunk size: 4800 bytes = 100ms at 24kHz mono 16-bit PCM
_STREAM_CHUNK_SIZE = 4800


class OpenAITTSProvider(TTSProvider):
    def __init__(self):
        self.client: AsyncOpenAI | None = None

    async def start(self) -> None:
        if self.client:
            return  # Already started
        self.client = AsyncOpenAI()
        logger.info(f"OpenAI TTS ready (voice={config.tts.voice})")

    async def stop(self) -> None:
        self.client = None

    async def synthesize(self, text: str) -> bytes:
        """Convert text to raw PCM audio (24kHz, 16-bit mono, little-endian)."""
        if not self.client:
            raise RuntimeError("OpenAI TTS not started")

        started = time.time()
        metrics.inc("provider.tts.requests", labels={"provider": "openai"})

        try:
            response = await self.client.audio.speech.create(
                model=config.tts.model,
                voice=config.tts.voice,
                input=text,
                response_format="pcm",
            )
            audio = response.content
            elapsed_ms = (time.time() - started) * 1000
            metrics.observe(
                "provider.tts.latency_ms", elapsed_ms, labels={"provider": "openai"}
            )
            # Estimate audio duration: PCM 24kHz 16-bit mono = 48000 bytes/sec
            audio_duration_ms = len(audio) / 48.0
            metrics.observe("provider.tts.audio_duration_ms", audio_duration_ms)
            return audio
        except Exception:
            metrics.inc("provider.tts.errors", labels={"provider": "openai"})
            raise

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream PCM audio chunks as they're synthesized by OpenAI.

        Yields ~100ms chunks (4800 bytes at 24kHz mono 16-bit).
        This enables first-audio-byte latency of ~200-400ms instead of
        waiting for the full synthesis to complete.
        """
        if not self.client:
            raise RuntimeError("OpenAI TTS not started")

        started = time.time()
        metrics.inc("provider.tts.requests", labels={"provider": "openai", "mode": "stream"})
        total_bytes = 0

        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=config.tts.model,
                voice=config.tts.voice,
                input=text,
                response_format="pcm",
            ) as response:
                buffer = b""
                async for raw_chunk in response.iter_bytes():
                    buffer += raw_chunk
                    # Yield complete chunks of _STREAM_CHUNK_SIZE
                    while len(buffer) >= _STREAM_CHUNK_SIZE:
                        chunk = buffer[:_STREAM_CHUNK_SIZE]
                        buffer = buffer[_STREAM_CHUNK_SIZE:]
                        total_bytes += len(chunk)
                        yield chunk

                # Yield any remaining bytes
                if buffer:
                    total_bytes += len(buffer)
                    yield buffer

            elapsed_ms = (time.time() - started) * 1000
            metrics.observe(
                "provider.tts.latency_ms", elapsed_ms, labels={"provider": "openai"}
            )
            audio_duration_ms = total_bytes / 48.0
            metrics.observe("provider.tts.audio_duration_ms", audio_duration_ms)
            logger.info(
                "[pipeline] TTS stream: %d bytes (%.1fms audio) in %.0fms",
                total_bytes, audio_duration_ms, elapsed_ms,
            )
        except Exception:
            metrics.inc("provider.tts.errors", labels={"provider": "openai"})
            raise

    async def health_check(self) -> dict:
        status = "ready" if self.client else "not_started"
        return {
            "provider": "openai",
            "model": config.tts.model,
            "voice": config.tts.voice,
            "status": status,
        }
