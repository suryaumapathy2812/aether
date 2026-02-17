"""
STT Processor — Deepgram speech-to-text.

Takes audio frames, yields text frames with the transcription.
Uses Deepgram's REST API for simplicity in v0.01.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from deepgram import AsyncDeepgramClient

from aether.core.frames import Frame, FrameType, text_frame
from aether.core.processor import Processor

logger = logging.getLogger(__name__)


class STTProcessor(Processor):
    def __init__(self):
        super().__init__("STT")
        self.client: AsyncDeepgramClient | None = None

    async def start(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        self.client = AsyncDeepgramClient(api_key=api_key)
        logger.info("STT processor ready (Deepgram)")

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Transcribe audio frame to text frame."""
        # Pass through non-audio frames unchanged
        if frame.type != FrameType.AUDIO:
            yield frame
            return

        if not self.client:
            raise RuntimeError("STT processor not started")

        try:
            response = await self.client.listen.v1.media.transcribe_file(
                request=frame.data,
                model="nova-3",
                smart_format=True,
                language="en",
            )

            # Navigate the response to get transcript
            transcript = response.results.channels[0].alternatives[0].transcript

            if transcript and transcript.strip():
                logger.info(f"STT: '{transcript[:80]}'")
                yield text_frame(transcript.strip(), role="user")
            else:
                logger.debug("STT: empty transcription, skipping")

        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
