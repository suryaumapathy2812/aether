"""
Text Adapter — handles text-only I/O (HTTP /chat, WebSocket text mode).

NO STT/TTS. Converts user text into LLM requests and streams text chunks
back to the transport. This is the simplest adapter.

Pipeline: user text → ContextBuilder → LLMCore → text_chunk stream → transport
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from aether.kernel.contracts import JobModality
from aether.modality.base import ModalityAdapter

if TYPE_CHECKING:
    from aether.llm.context_builder import ContextBuilder, SessionState
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore
    from aether.transport.core_msg import CoreMsg


logger = logging.getLogger(__name__)


class TextAdapter(ModalityAdapter):
    """
    Text modality adapter — no STT/TTS.

    Handles:
    - User text messages → LLM pipeline → text_chunk responses
    - Stream framing (sentence-level chunks with indices)
    - Memory retrieval before LLM call
    - Tool result and status event forwarding
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

    @property
    def modality(self) -> str:
        return JobModality.TEXT.value

    async def handle_input(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """
        Process text input through the LLM pipeline.

        Flow: text → memory retrieval → context build → LLM generate → text chunks
        """
        from aether.llm.context_builder import SessionState
        from aether.llm.contracts import LLMEventType
        from aether.transport.core_msg import CoreMsg, EventContent, TextContent

        content = msg.content
        if not isinstance(content, TextContent):
            return

        user_text = content.text
        user_id = msg.user_id
        session_id = msg.session_id

        # Track turn count
        session_state.setdefault("turn_count", 0)
        session_state["turn_count"] += 1

        # Build session state for context builder
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            mode="text",
            history=session_state.get("history", []),
        )

        # Memory retrieval
        pending_memory = await self._retrieve_memory(user_text)

        # Build LLM request envelope
        envelope = await self._context_builder.build(
            user_message=user_text,
            session=session,
            enabled_plugins=session_state.get("enabled_plugins", []),
            pending_memory=pending_memory,
        )

        # Stream LLM response
        sentence_index = 0
        started = time.time()

        async for event in self._llm_core.generate_with_tools(envelope):
            if event.event_type == LLMEventType.TEXT_CHUNK.value:
                text = event.payload.get("text", "")
                yield CoreMsg.text(
                    text=text,
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    transport="text_chunk",
                    sentence_index=sentence_index,
                )
                sentence_index += 1

            elif event.event_type == LLMEventType.TOOL_RESULT.value:
                yield CoreMsg.event(
                    event_type="tool_result",
                    user_id=user_id,
                    session_id=session_id,
                    payload={
                        "name": event.payload.get("tool_name", "unknown"),
                        "output": event.payload.get("output", "")[:500],
                        "error": event.payload.get("error", False),
                    },
                    transport="tools",
                )

            elif event.event_type == LLMEventType.STREAM_END.value:
                yield CoreMsg.event(
                    event_type="stream_end",
                    user_id=user_id,
                    session_id=session_id,
                    transport="control",
                )

            elif event.event_type == LLMEventType.ERROR.value:
                yield CoreMsg.text(
                    text=event.payload.get("message", "Something went wrong."),
                    user_id=user_id,
                    session_id=session_id,
                    role="system",
                    transport="status",
                )

        elapsed_ms = round((time.time() - started) * 1000)
        logger.info(f"[text] {sentence_index} sentences | {elapsed_ms}ms")

    async def handle_output(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Convert kernel events to text transport messages."""
        from aether.transport.core_msg import CoreMsg

        user_id = session_state.get("user_id", "")
        session_id = session_state.get("session_id", "")

        if event_type == "text_chunk":
            yield CoreMsg.text(
                text=payload.get("text", ""),
                user_id=user_id,
                session_id=session_id,
                role=payload.get("role", "assistant"),
                transport="text_chunk",
            )
        elif event_type == "status":
            yield CoreMsg.text(
                text=payload.get("text", ""),
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )
        elif event_type == "stream_end":
            yield CoreMsg.event(
                event_type="stream_end",
                user_id=user_id,
                session_id=session_id,
                transport="control",
            )

    async def _retrieve_memory(self, user_text: str) -> str | None:
        """Retrieve relevant memories for context injection."""
        try:
            from aether.core.config import config

            results = await self._memory_store.search(
                user_text, limit=config.memory.search_limit
            )
            if not results:
                return None

            lines = []
            for r in results:
                if r.get("type") == "fact":
                    lines.append(f"[Known fact] {r['fact']}")
                elif r.get("type") == "action":
                    output_preview = r.get("output", "")[:100]
                    status = "failed" if r.get("error") else "succeeded"
                    lines.append(
                        f"[Past action] Used {r['tool_name']}({r.get('arguments', '{}')}) — "
                        f"{status}: {output_preview}"
                    )
                elif r.get("type") == "session":
                    lines.append(f"[Previous session] {r['summary']}")
                elif r.get("type") == "conversation":
                    lines.append(
                        f"[Previous conversation] User said: {r['user_message']} — "
                        f"You replied: {r['assistant_message']}"
                    )

            return "\n".join(lines) if lines else None

        except Exception as e:
            logger.error(f"Memory retrieval error: {e}", exc_info=True)
            return None
