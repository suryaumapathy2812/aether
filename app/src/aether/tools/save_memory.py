"""Save-to-memory tool — lets the LLM explicitly store facts and preferences.

This is a built-in tool (always available, no plugin required). It gives the
LLM the ability to proactively remember things during multi-tool workflows
or whenever it decides something is worth persisting across sessions.

Stored facts are deduplicated via canonical key matching and embedded for
semantic search, so they surface automatically in future conversations.
"""

from __future__ import annotations

import logging

from aether.memory.store import MemoryStore
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)


class SaveMemoryTool(AetherTool):
    """Save a fact, preference, or instruction to long-term memory."""

    name = "save_memory"
    description = (
        "Save a fact, preference, or instruction to long-term memory so you "
        "can recall it in future conversations. Use this when the user tells "
        "you something worth remembering, or when you discover something "
        "important during a task."
    )
    status_text = "Saving to memory..."
    parameters = [
        ToolParam(
            name="content",
            type="string",
            description=(
                "The fact, preference, or instruction to remember. "
                "Write it as a clear, self-contained statement "
                "(e.g. 'User prefers dark mode', 'Project uses FastAPI with SQLite')."
            ),
            required=True,
        ),
    ]

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    async def execute(self, content: str, **_) -> ToolResult:
        content = content.strip()
        if not content:
            return ToolResult.fail("Cannot save empty memory.")

        try:
            await self._memory_store.store_preference(content)
            logger.info(f"LLM saved memory: {content[:80]}")
            return ToolResult.success(f"Saved to memory: {content}")
        except Exception as e:
            logger.error(f"Error saving memory: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to save memory: {e}")
