"""Search-memory tool — lets the LLM query long-term memory on demand.

This is a built-in tool (always available, no plugin required). It gives the
LLM the ability to proactively look up user preferences, past decisions,
prior context, and previous actions before responding.

Searches across all four memory tiers (facts, conversations, actions, sessions)
using semantic similarity and returns the most relevant results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from aether.memory.store import MemoryStore
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)


class SearchMemoryTool(AetherTool):
    """Search long-term memory for relevant facts, preferences, and context."""

    name = "search_memory"
    description = (
        "Search your long-term memory for user preferences, past decisions, "
        "prior conversations, and previous actions. Use this when you need to "
        "recall something the user told you before, check if you already know "
        "something, or look up context from a previous session."
    )
    status_text = "Searching memory..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description=(
                "What to search for. Be specific and descriptive "
                "(e.g. 'user timezone preference', 'what project framework was chosen', "
                "'previous conversation about deployment')."
            ),
            required=True,
        ),
        ToolParam(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default 5).",
            required=False,
            default=5,
        ),
    ]

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    async def execute(self, query: str, limit: int = 5, **_) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult.fail("Cannot search with empty query.")

        try:
            results = await self._memory_store.search(query, limit=limit)

            if not results:
                return ToolResult.success("No relevant memories found.")

            lines = []
            for i, r in enumerate(results, 1):
                ts = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
                date_str = ts.strftime("%Y-%m-%d")

                if r["type"] == "fact":
                    lines.append(f"{i}. [fact] {r['fact']} ({date_str})")
                elif r["type"] == "conversation":
                    user_msg = r["user_message"][:120]
                    lines.append(f"{i}. [conversation] User: {user_msg} ({date_str})")
                elif r["type"] == "action":
                    lines.append(
                        f"{i}. [action] {r['tool_name']}: {r['output'][:120]} ({date_str})"
                    )
                elif r["type"] == "session":
                    lines.append(f"{i}. [session] {r['summary'][:150]} ({date_str})")

            output = f"Found {len(results)} relevant memories:\n" + "\n".join(lines)
            logger.info(f"LLM searched memory: {query[:80]} → {len(results)} results")
            return ToolResult.success(output)
        except Exception as e:
            logger.error(f"Error searching memory: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to search memory: {e}")
