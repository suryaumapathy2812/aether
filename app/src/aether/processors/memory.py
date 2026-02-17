"""
Memory Retriever Processor — searches memory and injects context.

v0.03: Now includes both conversations and extracted facts.
Facts are presented separately so the LLM can use them naturally.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from aether.core.config import config
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
        if frame.type != FrameType.TEXT or frame.metadata.get("role") != "user":
            yield frame
            return

        user_text = frame.data

        try:
            results = await self.store.search(
                user_text, limit=config.memory.search_limit
            )

            if results:
                memories = []
                for r in results:
                    if r.get("type") == "fact":
                        memories.append(f"[Known fact] {r['fact']}")
                    elif r.get("type") == "conversation":
                        memories.append(
                            f"[Previous conversation] User said: {r['user_message']} — "
                            f"You replied: {r['assistant_message']}"
                        )
                    else:
                        # Backward compat
                        if r.get("user_message"):
                            memories.append(
                                f"[Previous conversation] User said: {r['user_message']} — "
                                f"You replied: {r['assistant_message']}"
                            )

                if memories:
                    logger.info(f"Memory: {len(memories)} matches")
                    yield memory_frame(memories, query=user_text)
                else:
                    logger.debug("Memory: no matches above threshold")
            else:
                logger.debug("Memory: empty")

        except Exception as e:
            logger.error(f"Memory search error: {e}", exc_info=True)

        yield frame
