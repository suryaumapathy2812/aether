"""
Aether Pipeline — chains processors together and runs frames through them.

The pipeline is just a list. Frame goes in one end, flows through each processor,
output comes out the other end. Simple as Hono, not complex as NestJS.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from aether.core.frames import Frame, FrameType
from aether.core.processor import Processor

logger = logging.getLogger(__name__)


class Pipeline:
    """
    A sequential chain of Processors.

    Frames flow through each processor in order. Each processor can yield
    zero or more frames, and each yielded frame is fed to the next processor.
    """

    def __init__(self, processors: list[Processor]):
        self.processors = processors

    async def start(self) -> None:
        """Start all processors."""
        for p in self.processors:
            logger.info(f"Starting processor: {p.name}")
            await p.start()

    async def stop(self) -> None:
        """Stop all processors in reverse order."""
        for p in reversed(self.processors):
            logger.info(f"Stopping processor: {p.name}")
            await p.stop()

    async def run(self, initial_frames: list[Frame]) -> AsyncGenerator[Frame, None]:
        """
        Run frames through the pipeline.

        Each processor receives frames from the previous one and yields
        frames to the next. The final processor's output is yielded.
        """
        # Start with the initial frames
        current_frames = initial_frames

        for processor in self.processors:
            next_frames = []
            for frame in current_frames:
                try:
                    async for output_frame in processor.process(frame):
                        next_frames.append(output_frame)
                except Exception as e:
                    logger.error(f"Error in {processor.name}: {e}", exc_info=True)
                    # Don't kill the pipeline on processor errors — log and continue
                    continue
            current_frames = next_frames

        # Yield final output frames
        for frame in current_frames:
            yield frame

    async def run_single(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Convenience: run a single frame through the pipeline."""
        async for output in self.run([frame]):
            yield output

    def __repr__(self) -> str:
        chain = " → ".join(p.name for p in self.processors)
        return f"Pipeline[{chain}]"
