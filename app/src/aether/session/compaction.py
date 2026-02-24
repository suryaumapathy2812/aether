"""
Session Compaction — summarize old messages when context is full.

When a session's messages exceed the context budget, old messages
are summarized into a compact summary message and the originals
are deleted. Recent messages and pending tool results are preserved.

This enables truly long-running agent sessions that would otherwise
exceed the model's context window.

Invariants:
- Pending tool calls (assistant messages with tool_calls whose results
  haven't arrived yet) are NEVER compacted — they stay in context.
- The summary message is marked with part_type "compacted_summary"
  so it can be identified and re-compacted in future passes.
- Compaction is idempotent — running it twice on the same session
  produces the same result (the existing summary is included in the
  next compaction pass if the context is still over budget).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aether.llm.core import LLMCore
    from aether.session.models import Message
    from aether.session.store import SessionStore

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────

# Default context budget (tokens). 80% of 128k = ~102k tokens.
DEFAULT_MAX_CONTEXT_TOKENS = 100_000

# Rough chars-per-token estimate for budget checking.
# Conservative (slightly over-counts) to trigger compaction early
# rather than hitting the hard limit.
CHARS_PER_TOKEN = 4

# Number of recent turns to preserve (not compacted).
# Per implementation_plan.md Phase 6: default 10.
PRESERVE_RECENT_TURNS = 10

# Maximum chars of conversation text sent to the summarization LLM.
# Keeps the summarization call itself within budget.
MAX_SUMMARIZATION_INPUT_CHARS = 12_000

# Marker prefix for compacted summary messages so they can be identified.
COMPACTION_MARKER = "[Compacted conversation summary]"

# ─── Prompts ──────────────────────────────────────────────────────

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer. Given a conversation history, produce a concise summary that preserves:
1. Key facts and decisions made
2. Important tool results and their outcomes
3. The current state of any ongoing tasks
4. Any user preferences or constraints mentioned

Be concise but complete. Use bullet points. Focus on information the assistant will need to continue the conversation effectively."""

COMPACTION_USER_PROMPT = """\
Summarize the following conversation history concisely, preserving all facts, decisions, and context needed to continue the work.

Conversation:
{conversation}

Summary:"""


# ═══════════════════════════════════════════════════════════════════
# SessionCompactor
# ═══════════════════════════════════════════════════════════════════


class SessionCompactor:
    """
    Manages context window compaction for long-running sessions.

    When the total message content exceeds the context budget,
    old messages are summarized into a compact summary message
    and the originals are deleted.

    Usage (called automatically by SessionLoop):
        compactor = SessionCompactor(session_store, llm_core)
        if await compactor.needs_compaction(session_id):
            await compactor.compact(session_id)
    """

    def __init__(
        self,
        session_store: "SessionStore",
        llm_core: "LLMCore | None" = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        preserve_recent: int = PRESERVE_RECENT_TURNS,
    ) -> None:
        self._session_store = session_store
        self._llm_core = llm_core
        self._max_context_tokens = max_context_tokens
        self._preserve_recent = preserve_recent

    # ─── Public API ───────────────────────────────────────────────

    async def needs_compaction(self, session_id: str) -> bool:
        """Check if a session's messages exceed the context budget.

        Returns True when the estimated token count of ALL messages
        (including content, tool_calls, and tool_call_ids) exceeds
        the configured max_context_tokens threshold.
        """
        messages = await self._session_store.get_messages(session_id)
        if len(messages) <= self._preserve_recent:
            return False  # Too few messages to compact

        estimated_tokens = self._estimate_tokens(messages)
        over_budget = estimated_tokens > self._max_context_tokens

        if over_budget:
            logger.debug(
                "Session %s: ~%d tokens (budget %d) — compaction needed",
                session_id,
                estimated_tokens,
                self._max_context_tokens,
            )

        return over_budget

    async def compact(self, session_id: str) -> bool:
        """
        Compact a session's messages by summarizing old ones.

        Returns True if compaction was performed, False if skipped.

        Algorithm:
        1. Load all messages
        2. Find the compaction boundary (preserving recent turns AND
           any messages with pending/unresolved tool calls)
        3. Summarize old messages using LLM (or simple truncation fallback)
        4. Delete old messages from store
        5. Insert summary as a system message BEFORE the preserved messages
        6. Mark the summary with a compacted_summary MessagePart
        """
        messages = await self._session_store.get_messages(session_id)

        if len(messages) <= self._preserve_recent:
            logger.info("Session %s: too few messages to compact", session_id)
            return False

        # Find the compaction boundary — everything before this index
        # gets summarized; everything from this index onward is preserved.
        boundary_idx = self._find_compaction_boundary(messages)

        if boundary_idx <= 0:
            logger.info("Session %s: no messages eligible for compaction", session_id)
            return False

        old_messages = messages[:boundary_idx]
        preserved_messages = messages[boundary_idx:]

        logger.info(
            "Session %s: compacting %d messages, preserving %d",
            session_id,
            len(old_messages),
            len(preserved_messages),
        )

        # Build conversation text from old messages
        conversation_text = self._build_conversation_text(old_messages)

        # Generate summary
        summary = await self._generate_summary(conversation_text)

        if not summary:
            logger.warning("Session %s: compaction summary empty, skipping", session_id)
            return False

        # Atomically replace old messages with summary.
        # 1. Delete old messages (sequence < cutoff)
        # 2. Insert summary at a sequence just before the preserved messages
        cutoff_sequence = old_messages[-1].sequence + 1
        deleted = await self._session_store.delete_messages_before(
            session_id, cutoff_sequence
        )

        # Insert summary as a system message.
        # Use insert_message_at_sequence to place it BEFORE preserved messages.
        # The sequence is (cutoff_sequence - 1) so it sorts before the
        # preserved messages which start at cutoff_sequence.
        summary_content = f"{COMPACTION_MARKER}\n{summary}"
        summary_msg = await self._session_store.insert_message_at_sequence(
            session_id,
            role="system",
            content=summary_content,
            sequence=cutoff_sequence - 1,
        )

        # Mark with a compacted_summary part for identification
        from aether.session.models import PartType

        await self._session_store.add_part(
            message_id=summary_msg.message_id,
            part_type=PartType.COMPACTION.value,
            content={"compacted_count": len(old_messages), "deleted": deleted},
        )

        logger.info(
            "Session %s: compacted %d messages into summary (%d chars)",
            session_id,
            deleted,
            len(summary),
        )
        return True

    # ─── Boundary Detection ───────────────────────────────────────

    def _find_compaction_boundary(self, messages: list["Message"]) -> int:
        """
        Find the index that splits messages into [old | preserved].

        Preserved region includes:
        - The last N turns (self._preserve_recent)
        - Any messages with pending/unresolved tool calls

        A tool call is "pending" if there's an assistant message with
        tool_calls but no corresponding tool-role message with a matching
        tool_call_id later in the conversation.

        Returns the index of the first preserved message.
        """
        n = len(messages)

        # Start with the basic split: preserve the last N messages
        basic_boundary = max(0, n - self._preserve_recent)

        # Now walk backwards from the basic boundary to find any
        # pending tool calls that must also be preserved.
        # Collect all tool_call_ids that have results (tool-role messages)
        resolved_tool_ids: set[str] = set()
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id:
                resolved_tool_ids.add(msg.tool_call_id)

        # Walk backwards from the boundary to find unresolved tool calls
        adjusted_boundary = basic_boundary
        for i in range(basic_boundary - 1, -1, -1):
            msg = messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                # Check if ANY tool call in this message is unresolved
                has_pending = False
                for tc in msg.tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in resolved_tool_ids:
                        has_pending = True
                        break

                if has_pending:
                    # This message and everything after it must be preserved.
                    # Also preserve any tool results that belong to this
                    # message's tool calls (they might be between i and
                    # the current boundary).
                    adjusted_boundary = i
                    logger.debug(
                        "Compaction boundary moved from %d to %d "
                        "(pending tool calls in message seq=%d)",
                        basic_boundary,
                        i,
                        msg.sequence,
                    )

        # Also check: if the message right before the boundary is a tool
        # result, we need to include its parent assistant message too.
        # Walk backwards to ensure we don't split a tool_call/tool_result pair.
        while adjusted_boundary > 0:
            msg = messages[adjusted_boundary]
            if msg.role == "tool":
                # This is a tool result — its parent assistant message
                # (with the tool_calls) must also be preserved.
                adjusted_boundary -= 1
            else:
                break

        return adjusted_boundary

    # ─── Token Estimation ─────────────────────────────────────────

    def _estimate_tokens(self, messages: list["Message"]) -> int:
        """
        Estimate the total token count for a list of messages.

        Uses a rough chars/4 heuristic. Counts content, tool_calls
        (serialized), and tool_call_ids to avoid under-counting.
        """
        total_chars = 0
        for msg in messages:
            # Message content
            content = msg.content
            if content is not None:
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    # Multimodal content — count text parts
                    for part in content:
                        if isinstance(part, dict):
                            total_chars += len(part.get("text", ""))
                else:
                    total_chars += len(str(content))

            # Tool calls (serialized JSON can be large)
            if msg.tool_calls:
                total_chars += len(json.dumps(msg.tool_calls))

            # Tool call ID
            if msg.tool_call_id:
                total_chars += len(msg.tool_call_id)

            # Role overhead (~4 tokens per message for role/formatting)
            total_chars += 16

        return total_chars // CHARS_PER_TOKEN

    # ─── Conversation Text Builder ────────────────────────────────

    def _build_conversation_text(self, messages: list["Message"]) -> str:
        """
        Build a human-readable conversation transcript from messages.

        Used as input to the summarization LLM. Each message is formatted
        as "ROLE: content" with tool call/result annotations.
        """
        lines: list[str] = []
        for msg in messages:
            role = msg.role.upper()
            content = str(msg.content or "")

            # Annotate tool calls
            if msg.tool_calls:
                tool_names = [
                    tc.get("function", {}).get("name", "?") for tc in msg.tool_calls
                ]
                content += f" [called tools: {', '.join(tool_names)}]"

            # Annotate tool results
            if msg.tool_call_id:
                content = f"[tool result for {msg.tool_call_id}] {content}"

            # Truncate very long individual messages to keep the
            # summarization input manageable
            lines.append(f"{role}: {content[:500]}")

        return "\n".join(lines)

    # ─── Summary Generation ───────────────────────────────────────

    async def _generate_summary(self, conversation_text: str) -> str | None:
        """Generate a summary of the conversation using LLM or fallback."""
        if self._llm_core is not None:
            return await self._llm_summarize(conversation_text)
        return self._simple_summarize(conversation_text)

    async def _llm_summarize(self, conversation_text: str) -> str | None:
        """Use LLM to generate a high-quality summary."""
        assert self._llm_core is not None

        from aether.llm.contracts import LLMEventType, LLMRequestEnvelope

        # Truncate conversation text to keep the summarization call
        # within a reasonable input size
        truncated = conversation_text[:MAX_SUMMARIZATION_INPUT_CHARS]

        envelope = LLMRequestEnvelope(
            kind="compaction",
            modality="system",
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMPACTION_USER_PROMPT.format(conversation=truncated),
                },
            ],
            tools=[],  # No tools for summarization
            policy={"max_tokens": 1500, "temperature": 0.3},
        )

        collected: list[str] = []
        try:
            async for event in self._llm_core.generate_with_tools(envelope):
                if event.event_type == LLMEventType.TEXT_CHUNK.value:
                    collected.append(event.payload.get("text", ""))

            summary = "".join(collected).strip()
            return summary if summary else None

        except Exception as e:
            logger.error("LLM compaction failed: %s", e, exc_info=True)
            # Fall back to simple summarization
            return self._simple_summarize(conversation_text)

    def _simple_summarize(self, conversation_text: str) -> str:
        """Simple truncation-based summary when LLM is not available.

        Keeps the first few and last few lines of the conversation
        to preserve the beginning context and most recent state.
        """
        lines = conversation_text.split("\n")
        if len(lines) <= 20:
            return conversation_text

        kept = lines[:5] + ["... (earlier messages omitted) ..."] + lines[-10:]
        return "\n".join(kept)
