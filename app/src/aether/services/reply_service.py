"""
Reply Service — user-facing response generation.

Handles reply_text and reply_voice job kinds. This is the primary
user-facing service that:
1. Builds context via ContextBuilder (skills, plugins, memory)
2. Generates response via LLMCore (with automatic tool calling)
3. Stores conversation to memory
4. Yields LLMEventEnvelope events for the caller to consume

The caller (modality adapter or kernel) handles TTS synthesis and
transport-level framing.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from aether.core.metrics import metrics
from aether.llm.contracts import LLMEventEnvelope, LLMEventType

if TYPE_CHECKING:
    from aether.llm.context_builder import ContextBuilder, SessionState
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class ReplyService:
    """
    Generates user-facing responses through LLMCore.

    This service owns the reply_text and reply_voice job kinds.
    It builds context, generates the response, and stores the
    conversation to memory.

    The service does NOT handle:
    - TTS synthesis (that's the VoiceAdapter's job)
    - Transport framing (that's the modality adapter's job)
    - Session management (that's the kernel's job)
    """

    def __init__(
        self,
        llm_core: "LLMCore",
        context_builder: "ContextBuilder",
        memory_store: "MemoryStore",
    ) -> None:
        self._llm_core = llm_core
        self._context_builder = context_builder
        self._memory_store = memory_store

    async def generate_reply(
        self,
        user_message: str,
        session: "SessionState",
        enabled_plugins: list[str] | None = None,
        pending_memory: str | None = None,
        pending_vision: dict[str, Any] | None = None,
    ) -> AsyncGenerator[LLMEventEnvelope, None]:
        """
        Generate a reply to the user's message.

        Yields LLMEventEnvelope events (text_chunk, tool_result, stream_end).
        The caller is responsible for converting these to transport messages.

        Args:
            user_message: The user's message text
            session: Session state with history
            enabled_plugins: List of enabled plugin names
            pending_memory: Pre-retrieved memory context string
            pending_vision: Vision context (image data)

        Yields:
            LLMEventEnvelope events
        """
        enabled_plugins = enabled_plugins or []
        kind = session.mode  # "text" or "voice"
        started = time.time()

        metrics.inc("service.reply.started", labels={"kind": kind})

        # Build LLM request envelope
        envelope = await self._context_builder.build(
            user_message=user_message,
            session=session,
            enabled_plugins=enabled_plugins,
            pending_memory=pending_memory,
            pending_vision=pending_vision,
        )

        # Stream LLM response with automatic tool execution
        collected_text: list[str] = []
        token_count = 0

        async for event in self._llm_core.generate_with_tools(envelope):
            if event.event_type == LLMEventType.TEXT_CHUNK.value:
                chunk = event.payload.get("text", "")
                collected_text.append(chunk)
                token_count += len(chunk.split())

            yield event

        # Store conversation to memory (fire-and-forget style)
        assistant_text = " ".join(collected_text).strip()
        if assistant_text:
            try:
                await self._memory_store.add(user_message, assistant_text)
            except Exception as e:
                logger.error(f"Memory store failed: {e}")

        elapsed_ms = round((time.time() - started) * 1000)
        metrics.observe("service.reply.duration_ms", elapsed_ms, labels={"kind": kind})
        metrics.observe("service.reply.tokens", token_count, labels={"kind": kind})
        metrics.inc("service.reply.completed", labels={"kind": kind})
        logger.info(
            f"Reply generated in {elapsed_ms}ms (kind={kind}, ~{token_count} tokens)"
        )
