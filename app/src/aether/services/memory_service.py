"""
Memory Service — background memory operations through LLMCore.

Handles memory_* job kinds:
- memory_fact_extract: Extract facts from conversation turns
- memory_session_summary: Summarize a session for cross-session continuity
- memory_action_compact: Compact old tool actions into summaries

All LLM calls go through LLMCore, ensuring consistent contracts.
These are background (system modality) jobs — no user-facing output.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aether.core.metrics import metrics
from aether.llm.contracts import LLMEventType, LLMRequestEnvelope

if TYPE_CHECKING:
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Prompts for memory operations
SESSION_SUMMARY_PROMPT = """Summarize this conversation session in 2-3 sentences.
Focus on what was accomplished (files created, questions answered, tasks completed), not what was said.
If tools were used, mention what they did.

Conversation:
{conversation}

Summary:"""

FACT_EXTRACTION_PROMPT = """Extract key facts from this conversation turn. Focus on:
- Personal details (name, location, job, preferences)
- Preferences and opinions ("I like...", "I prefer...", "I hate...")
- Plans and goals ("I'm working on...", "I want to...")
- Relationships ("my wife...", "my friend...")
- Workflow patterns ("User always asks for Python projects")
- Any specific factual information the user shared

Return a JSON array of fact strings. Each fact should be a short, standalone statement.
If no significant facts are present, return an empty array [].

Example output: ["User's name is Alex", "User works at Google", "User prefers Python"]

Conversation:
User: {user_message}
Assistant: {assistant_message}

Facts (JSON array):"""


class MemoryService:
    """
    Background memory operations through LLMCore.

    All LLM calls use the shared LLMCore interface with system modality.
    Results are stored directly to MemoryStore.
    """

    def __init__(
        self,
        llm_core: "LLMCore",
        memory_store: "MemoryStore",
    ) -> None:
        self._llm_core = llm_core
        self._memory_store = memory_store

    async def extract_facts(
        self,
        user_message: str,
        assistant_message: str,
        conversation_id: int | None = None,
    ) -> list[str]:
        """
        Extract facts from a conversation turn using LLMCore.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            conversation_id: Optional conversation ID for linking

        Returns:
            List of extracted fact strings
        """
        started = time.time()
        metrics.inc("service.memory.extraction.started")

        prompt = FACT_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        envelope = LLMRequestEnvelope(
            kind="memory_fact_extract",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 200, "temperature": 0.0},
        )

        # Collect full response
        response_text = await self._collect_response(envelope)

        # Parse JSON array
        facts = self._parse_facts(response_text)

        # Store each fact
        for fact in facts:
            try:
                await self._memory_store._store_fact(fact, conversation_id or 0)
            except Exception as e:
                logger.debug(f"Fact store error: {e}")

        elapsed_ms = round((time.time() - started) * 1000)
        metrics.observe("service.memory.extraction_ms", elapsed_ms)
        metrics.inc("service.memory.facts_extracted", value=len(facts))

        if facts:
            logger.info(f"Extracted {len(facts)} facts in {elapsed_ms}ms: {facts}")

        return facts

    async def summarize_session(
        self,
        session_id: str,
        conversation_history: list[dict[str, Any]],
        started_at: float,
        turn_count: int,
        tools_used: list[str] | None = None,
    ) -> str:
        """
        Summarize a session for cross-session continuity.

        Args:
            session_id: The session ID
            conversation_history: List of conversation messages
            started_at: Session start timestamp
            turn_count: Number of turns in the session
            tools_used: List of tools used during the session

        Returns:
            Summary text
        """
        # Build conversation text from history
        conv_lines = []
        tool_names: set[str] = set(tools_used or [])

        for msg in conversation_history[-20:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conv_lines.append(f"User: {content}")
            elif role == "assistant" and content:
                conv_lines.append(f"Aether: {content}")
            elif role == "tool":
                tool_names.add("tool")

        if not conv_lines:
            return ""

        conv_text = "\n".join(conv_lines)
        prompt = SESSION_SUMMARY_PROMPT.format(conversation=conv_text)

        envelope = LLMRequestEnvelope(
            kind="memory_session_summary",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 100, "temperature": 0.3},
        )

        summary = await self._collect_response(envelope)
        summary = summary.strip()

        if summary:
            try:
                await self._memory_store.add_session_summary(
                    session_id=session_id,
                    summary=summary,
                    started_at=started_at,
                    ended_at=time.time(),
                    turns=turn_count,
                    tools_used=list(tool_names),
                )
                logger.info(f"Session {session_id} summarized: {summary[:80]}...")
            except Exception as e:
                logger.error(f"Session summary store failed: {e}")

        return summary

    async def compact_actions(self) -> None:
        """
        Compact old tool actions into summaries.

        Delegates to MemoryStore's existing compaction logic.
        In the future, this could use LLMCore for smarter compaction.
        """
        try:
            await self._memory_store._compact_old_actions()
            logger.info("Action compaction completed")
        except Exception as e:
            logger.error(f"Action compaction failed: {e}")

    # ─── Helpers ─────────────────────────────────────────────────

    async def _collect_response(self, envelope: LLMRequestEnvelope) -> str:
        """Collect full text response from LLMCore."""
        chunks: list[str] = []

        async for event in self._llm_core.generate_with_tools(envelope):
            if event.event_type == LLMEventType.TEXT_CHUNK.value:
                chunks.append(event.payload.get("text", ""))

        return " ".join(chunks)

    def _parse_facts(self, response_text: str) -> list[str]:
        """Parse a JSON array of facts from LLM response."""
        import json
        import re

        response_text = response_text.strip()

        try:
            if response_text.startswith("["):
                facts = json.loads(response_text)
            else:
                match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if match:
                    facts = json.loads(match.group())
                else:
                    facts = []
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"Could not parse facts from: {response_text[:100]}")
            facts = []

        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
