"""
Memory Retriever Processor — searches memory and injects context.

Takes a text frame (user message), searches for relevant memories,
and yields a memory frame + the original text frame for the LLM.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from aether.core.frames import Frame, FrameType, memory_frame
from aether.core.processor import Processor
from aether.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryRetrieverProcessor(Processor):
    def __init__(self, store: MemoryStore):
        super().__init__("MemoryRetriever")
        self.store = store

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Search memory for relevant context, then pass through the frame."""
        # Only search memory for user text frames
        if frame.type != FrameType.TEXT or frame.metadata.get("role") != "user":
            yield frame
            return

        user_text = frame.data

        try:
            results = await self.store.search(user_text, limit=5)

            if results:
                memories = []
                for r in results:
                    if r["similarity"] > 0.3:  # Only include reasonably relevant memories
                        memories.append(
                            f"[Previous conversation] User said: {r['user_message']} — "
                            f"You replied: {r['assistant_message']}"
                        )

                if memories:
                    logger.info(f"Found {len(memories)} relevant memories")
                    yield memory_frame(memories, query=user_text)
                else:
                    logger.debug("No memories above similarity threshold")
            else:
                logger.debug("No memories found")

        except Exception as e:
            logger.error(f"Memory search error: {e}", exc_info=True)
            # Don't block the pipeline if memory fails

        # Always yield the original text frame
        yield frame
