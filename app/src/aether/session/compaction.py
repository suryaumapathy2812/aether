"""
Session Compaction — summarize old messages when context is full.

When a session's messages exceed the context budget, old messages
are summarized into a compact summary message and the originals
are deleted. Recent messages and pending tool results are preserved.

This enables truly long-running agent sessions that would otherwise
exceed the model's context window.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.llm.core import LLMCore
    from aether.session.store import SessionStore

logger = logging.getLogger(__name__)

# Default context budget (tokens). Models vary, but 80% of 128k is safe.
DEFAULT_MAX_CONTEXT_TOKENS = 100_000

# Rough chars-per-token estimate for budget checking
CHARS_PER_TOKEN = 4

# Number of recent turns to preserve (not compacted)
PRESERVE_RECENT_TURNS = 6

# Summarization prompt
COMPACTION_SYSTEM_PROMPT = """You are a conversation summarizer. Given a conversation history, produce a concise summary that preserves:
1. Key facts and decisions made
2. Important tool results and their outcomes
3. The current state of any ongoing tasks
4. Any user preferences or constraints mentioned

Be concise but complete. Use bullet points. Focus on information the assistant will need to continue the conversation effectively."""

COMPACTION_USER_PROMPT = """Summarize the following conversation history into a compact summary. Preserve all important context needed to continue the conversation.

Conversation:
{conversation}

Summary:"""


class SessionCompactor:
    """
    Manages context window compaction for long-running sessions.

    When the total message content exceeds the context budget,
    old messages are summarized into a compact summary message
    and the originals are deleted.
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

    async def needs_compaction(self, session_id: str) -> bool:
        """Check if a session's messages exceed the context budget."""
        messages = await self._session_store.get_messages(session_id)
        if len(messages) <= self._preserve_recent:
            return False  # Too few messages to compact

        total_chars = sum(len(str(m.content or "")) for m in messages)
        estimated_tokens = total_chars // CHARS_PER_TOKEN
        return estimated_tokens > self._max_context_tokens

    async def compact(self, session_id: str) -> bool:
        """
        Compact a session's messages by summarizing old ones.

        Returns True if compaction was performed, False if skipped.

        Process:
        1. Load all messages
        2. Split into old (to compact) and recent (to preserve)
        3. Summarize old messages using LLM (or simple truncation if no LLM)
        4. Delete old messages from store
        5. Insert summary as a system message
        """
        messages = await self._session_store.get_messages(session_id)

        if len(messages) <= self._preserve_recent:
            logger.info("Session %s: too few messages to compact", session_id)
            return False

        # Split: old messages to summarize, recent to preserve
        split_point = len(messages) - self._preserve_recent
        old_messages = messages[:split_point]
        # recent_messages = messages[split_point:]  # These stay as-is

        if not old_messages:
            return False

        # Build conversation text from old messages
        conversation_lines = []
        for msg in old_messages:
            role = msg.role.upper()
            content = str(msg.content or "")
            if msg.tool_calls:
                tool_names = [
                    tc.get("function", {}).get("name", "?")
                    for tc in (msg.tool_calls or [])
                ]
                content += f" [called tools: {', '.join(tool_names)}]"
            if msg.tool_call_id:
                content = f"[tool result for {msg.tool_call_id}] {content}"
            conversation_lines.append(f"{role}: {content[:500]}")

        conversation_text = "\n".join(conversation_lines)

        # Generate summary
        summary = await self._generate_summary(conversation_text)

        if not summary:
            logger.warning("Session %s: compaction summary empty, skipping", session_id)
            return False

        # Delete old messages (by sequence number)
        cutoff_sequence = old_messages[-1].sequence + 1
        deleted = await self._session_store.delete_messages_before(
            session_id, cutoff_sequence
        )

        # Insert summary as a system message at the beginning
        await self._session_store.add_message(
            session_id,
            role="system",
            content=f"[Compacted conversation summary]\n{summary}",
        )

        logger.info(
            "Session %s: compacted %d messages into summary (%d chars)",
            session_id,
            deleted,
            len(summary),
        )
        return True

    async def _generate_summary(self, conversation_text: str) -> str | None:
        """Generate a summary of the conversation using LLM or fallback."""
        if self._llm_core is not None:
            return await self._llm_summarize(conversation_text)
        return self._simple_summarize(conversation_text)

    async def _llm_summarize(self, conversation_text: str) -> str | None:
        """Use LLM to generate a high-quality summary."""
        assert self._llm_core is not None

        from aether.llm.contracts import LLMEventType, LLMRequestEnvelope

        envelope = LLMRequestEnvelope(
            kind="compaction",
            modality="system",
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMPACTION_USER_PROMPT.format(
                        conversation=conversation_text[:8000]
                    ),
                },
            ],
            tools=[],  # No tools for summarization
            policy={"max_tokens": 1000, "temperature": 0.3},
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
        """Simple truncation-based summary when LLM is not available."""
        lines = conversation_text.split("\n")
        # Keep first few and last few lines
        if len(lines) <= 20:
            return conversation_text

        kept = lines[:5] + ["...(earlier messages omitted)..."] + lines[-10:]
        return "\n".join(kept)
