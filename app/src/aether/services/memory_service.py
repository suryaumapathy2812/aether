"""
Memory Service — background memory operations through LLMCore.

Handles memory_* job kinds:
- memory_fact_extract: Extract facts, memories, and decisions from conversation turns
  (three-bucket model per Requirements.md §7.1)
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
SESSION_SUMMARY_PROMPT = """You are writing Aether's session handoff note for a "Jarvis/second-brain" assistant.

Write a concise summary that helps future sessions resume effectively.
Focus on continuity, not transcript replay.

Include:
1) Outcome: what was accomplished or decided
2) Key preferences/constraints discovered or confirmed
3) Open loops: unresolved questions, pending tasks, follow-ups
4) Next-best action when the user returns

Style rules:
- 3-5 short sentences total
- Concrete and specific; avoid fluff
- Mention important tools/actions only when they materially changed progress
- Do not fabricate missing details

Conversation:
{conversation}

Summary:"""

MEMORY_EXTRACTION_PROMPT = """You are Aether's memory extractor. After each conversation turn, extract three types of information:

## Facts
Objective, stable information about the user. Things that are true and unlikely to change.
Examples: "User's name is Surya", "User works at Acme Corp", "User's timezone is Asia/Kolkata"

## Memories
Contextual, episodic information — what happened, what was discussed, behavioral patterns.
Examples: "User was stressed about Q3 deadline", "User dismissed 3 notifications in a row"

## Decisions
Rules about how the agent should behave for this user. Learned from patterns or explicit feedback.
Examples: "Don't notify about calendar after 9pm", "User prefers bullet points over paragraphs"

Rules:
- Write concise strings, third-person ("User ...")
- Only extract genuinely useful information — skip small talk
- Facts: stable, durable info. Memories: episodic, time-bound. Decisions: behavioral rules.
- If nothing valuable, return empty arrays

Return ONLY a JSON object:
{{"facts": ["..."], "memories": ["..."], "decisions": ["..."]}}

Conversation:
User: {user_message}
Assistant: {assistant_message}"""


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
        Extract facts, memories, and decisions from a conversation turn using LLMCore.

        Despite the name (kept for backward compatibility), this now extracts all three
        bucket types: facts, memories, and decisions. Returns only the facts list for
        backward compatibility, but stores all three types.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            conversation_id: Optional conversation ID for linking

        Returns:
            List of extracted fact strings (for backward compatibility)
        """
        started = time.time()
        metrics.inc("service.memory.extraction.started")

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        envelope = LLMRequestEnvelope(
            kind="memory_fact_extract",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 400, "temperature": 0.0},
        )

        # Collect full response
        response_text = await self._collect_response(envelope)

        # Parse JSON object with three arrays
        extracted = self._parse_extraction_response(response_text)
        facts = extracted.get("facts", [])
        memories = extracted.get("memories", [])
        decisions = extracted.get("decisions", [])

        conv_id = conversation_id or 0

        # Store facts (existing path)
        for fact in facts:
            try:
                await self._memory_store._store_fact(fact, conv_id)
            except Exception as e:
                logger.debug(f"Fact store error: {e}")

        # Store memories (new v0.08 path)
        for memory in memories:
            try:
                await self._memory_store.store_memory(
                    memory, category="episodic", conv_id=conv_id
                )
            except Exception as e:
                logger.debug(f"Memory store error: {e}")

        # Store decisions (new v0.08 path)
        for decision in decisions:
            try:
                await self._memory_store.store_decision(
                    decision, category="preference", source="extracted", conv_id=conv_id
                )
            except Exception as e:
                logger.debug(f"Decision store error: {e}")

        elapsed_ms = round((time.time() - started) * 1000)
        metrics.observe("service.memory.extraction_ms", elapsed_ms)
        metrics.inc("service.memory.facts_extracted", value=len(facts))
        metrics.inc("service.memory.memories_extracted", value=len(memories))
        metrics.inc("service.memory.decisions_extracted", value=len(decisions))

        total = len(facts) + len(memories) + len(decisions)
        if total:
            logger.info(
                f"Extracted {len(facts)} facts, {len(memories)} memories, "
                f"{len(decisions)} decisions in {elapsed_ms}ms"
            )

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
        """Parse a JSON array of facts from LLM response (legacy helper)."""
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

    def _parse_extraction_response(self, response_text: str) -> dict[str, list[str]]:
        """Parse a JSON object with facts, memories, and decisions arrays.

        Expected format: {"facts": [...], "memories": [...], "decisions": [...]}
        Falls back to treating the response as a facts-only JSON array for
        backward compatibility with older model responses.
        """
        import json
        import re

        response_text = response_text.strip()
        empty: dict[str, list[str]] = {"facts": [], "memories": [], "decisions": []}

        try:
            # Try parsing as JSON object first
            if response_text.startswith("{"):
                parsed = json.loads(response_text)
            else:
                # Model may wrap in markdown code block
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                else:
                    # Fallback: try as JSON array (old format → treat as facts only)
                    facts = self._parse_facts(response_text)
                    return {"facts": facts, "memories": [], "decisions": []}

            if not isinstance(parsed, dict):
                logger.debug(f"Extraction response is not a dict: {type(parsed)}")
                return empty

            # Validate and clean each bucket
            result: dict[str, list[str]] = {}
            for key in ("facts", "memories", "decisions"):
                raw = parsed.get(key, [])
                if not isinstance(raw, list):
                    result[key] = []
                    continue
                result[key] = [
                    item.strip()
                    for item in raw
                    if isinstance(item, str) and item.strip()
                ]

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(
                f"Could not parse extraction response: {e} — {response_text[:100]}"
            )
            # Last resort: try legacy array parse
            facts = self._parse_facts(response_text)
            return {"facts": facts, "memories": [], "decisions": []}
