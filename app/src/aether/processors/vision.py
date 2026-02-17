"""
Vision Processor — passes vision frames through to LLM context.

In v0.01 this is a simple pass-through. The LLMProcessor handles
multimodal messages directly with GPT-4o.

This processor exists as a boundary — future versions will add
preprocessing, frame extraction from video, etc.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from aether.core.frames import Frame, FrameType
from aether.core.processor import Processor

logger = logging.getLogger(__name__)


class VisionProcessor(Processor):
    def __init__(self):
        super().__init__("Vision")

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Pass vision frames through. Log for observability."""
        if frame.type == FrameType.VISION:
            size_kb = len(frame.data) / 1024
            logger.info(f"Vision: received image ({size_kb:.1f} KB, {frame.metadata.get('mime_type', 'unknown')})")

        yield frame
