"""
Session Loop — the outer agent loop for autonomous execution.

This is the core capability that makes Aether work like Claude Code / OpenCode.
It wraps LLMCore.generate_with_tools() (the inner tool loop) in a persistent
outer loop that:

1. Loads messages from SessionStore (durable state)
2. Checks exit conditions (agent decided it's done)
3. Checks context overflow → triggers compaction
4. Builds context and calls LLM with tools
5. Persists results to SessionStore
6. Publishes events to EventBus (for SSE streaming, notifications)
7. Loops back until the agent stops calling tools

The inner loop (LLMCore) handles: LLM → tool_call → execute → feed_result → LLM
The outer loop (SessionLoop) handles: session state, persistence, compaction, events

Usage:
    loop = SessionLoop(session_store, llm_core, context_builder, event_bus, ...)
    await loop.run(session_id)  # Runs until agent is done or canceled

    # Or with abort control:
    abort = asyncio.Event()
    task = asyncio.create_task(loop.run(session_id, abort=abort))
    # ... later ...
    abort.set()  # Cancel the loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from aether.agents.agent_types import (
    AgentTypeDefinition,
    get_agent_type,
    get_filtered_tools,
)
from aether.kernel.contracts import KernelEvent
from aether.llm.contracts import LLMEventEnvelope, LLMEventType
from aether.session.models import SessionStatus

if TYPE_CHECKING:
    from aether.kernel.event_bus import EventBus
    from aether.llm.context_builder import ContextBuilder, SessionState
    from aether.llm.core import LLMCore
    from aether.session.compaction import SessionCompactor
    from aether.session.store import SessionStore
    from aether.tools.orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)

# Maximum outer loop iterations to prevent runaway agents
MAX_OUTER_ITERATIONS = 50

# Maximum time a session loop can run (seconds)
MAX_SESSION_DURATION = 600  # 10 minutes


class SessionLoop:
    """
    The outer agent loop — manages session-level concerns while
    LLMCore handles the inner tool-calling loop.

    This is what makes the agent autonomous: it keeps running until
    the LLM decides it's done (no more tool calls, final text response).

    The loop is designed to be run as an asyncio.Task so it doesn't
    block the main thread. Sub-agents each get their own SessionLoop
    instance with their own session_id.
    """

    def __init__(
        self,
        session_store: "SessionStore",
        llm_core: "LLMCore",
        context_builder: "ContextBuilder",
        event_bus: "EventBus | None" = None,
        compactor: "SessionCompactor | None" = None,
        tool_orchestrator: "ToolOrchestrator | None" = None,
        max_iterations: int = MAX_OUTER_ITERATIONS,
        max_duration: float = MAX_SESSION_DURATION,
        agent_type_override: str | None = None,
    ) -> None:
        self._session_store = session_store
        self._llm_core = llm_core
        self._context_builder = context_builder
        self._event_bus = event_bus
        self._compactor = compactor
        self._tool_orchestrator = tool_orchestrator
        self._max_iterations = max_iterations
        self._max_duration = max_duration
        self._agent_type_override = agent_type_override

    async def run(
        self,
        session_id: str,
        abort: asyncio.Event | None = None,
        enabled_plugins: list[str] | None = None,
    ) -> str | None:
        """
        Run the agent loop for a session until completion.

        Returns the final assistant text, or None if canceled/errored.

        Args:
            session_id: The session to run
            abort: Optional event to signal cancellation
            enabled_plugins: List of enabled plugin names for context building
        """
        abort = abort or asyncio.Event()
        enabled_plugins = enabled_plugins or []
        started_at = time.time()
        iteration = 0
        final_text: str | None = None
        status = SessionStatus.DONE.value  # Default; overridden on error/cancel

        logger.info("SessionLoop started for %s", session_id)
        await self._session_store.update_session_status(
            session_id, SessionStatus.BUSY.value
        )
        await self._publish_status(session_id, "busy")

        try:
            while not abort.is_set() and iteration < self._max_iterations:
                # Check duration limit
                elapsed = time.time() - started_at
                if elapsed > self._max_duration:
                    logger.warning(
                        "SessionLoop %s hit duration limit (%.0fs)",
                        session_id,
                        elapsed,
                    )
                    break

                iteration += 1

                # 1. Load messages from persistent store
                messages = await self._session_store.get_messages_as_openai(session_id)

                if not messages:
                    logger.warning("SessionLoop %s: no messages, exiting", session_id)
                    break

                # 2. Check exit condition: last message is assistant with no tool calls
                if self._should_exit(messages):
                    # Extract final text from last assistant message
                    last = messages[-1]
                    if last.get("role") == "assistant":
                        content = last.get("content")
                        if isinstance(content, str):
                            final_text = content
                    break

                # 3. Check context overflow → compact
                if self._compactor and await self._compactor.needs_compaction(
                    session_id
                ):
                    logger.info("SessionLoop %s: compacting context", session_id)
                    await self._compactor.compact(session_id)
                    continue

                # 4. Build context and call LLM with tools
                session = await self._session_store.get_session(session_id)
                agent_type_name = self._agent_type_override or (
                    session.agent_type if session else "default"
                )
                agent_def = get_agent_type(agent_type_name)

                from aether.llm.context_builder import SessionState

                session_state = SessionState(
                    session_id=session_id,
                    user_id="",
                    mode="text",
                    history=messages,
                )

                envelope = await self._context_builder.build(
                    user_message=self._extract_last_user_text(messages),
                    session=session_state,
                    enabled_plugins=enabled_plugins,
                )

                # Apply agent-type system prompt and tool filtering.
                # For non-default agent types, replace the system prompt
                # and filter the tool schemas to only allowed tools.
                if agent_type_name not in ("default", "general"):
                    self._apply_agent_type(envelope, agent_def)

                # 5. Run inner loop (LLMCore.generate_with_tools)
                collected_text: list[str] = []
                has_tool_calls = False

                async for event in self._llm_core.generate_with_tools(envelope):
                    # Publish to event bus for real-time streaming
                    await self._publish_event(session_id, event)

                    if event.event_type == LLMEventType.TEXT_CHUNK.value:
                        collected_text.append(event.payload.get("text", ""))

                    elif event.event_type == LLMEventType.TOOL_CALL.value:
                        has_tool_calls = True

                # 6. Persist the assistant's response
                assistant_text = "".join(collected_text).strip()

                # The inner loop (LLMCore) already appended tool calls and results
                # to the envelope's messages. We need to persist the final state.
                # The inner loop handles tool_call → tool_result message pairs
                # internally. We persist the final assistant text.
                if assistant_text and not has_tool_calls:
                    # Final response — no more tool calls
                    await self._session_store.append_assistant_message(
                        session_id, assistant_text
                    )
                    final_text = assistant_text
                    # Agent is done — will exit on next iteration's _should_exit check
                    break

                elif has_tool_calls:
                    # Tool calls were made — the inner loop handled them.
                    # We need to persist the full conversation state from the
                    # envelope (which LLMCore mutated with tool results).
                    # The messages in the envelope now include:
                    # - Original messages
                    # - Assistant message with tool_calls
                    # - Tool result messages
                    # We persist only the NEW messages (after the original ones).
                    original_count = len(messages)
                    new_messages = envelope.messages[original_count:]
                    for msg in new_messages:
                        if msg.get("role") == "assistant":
                            await self._session_store.add_message(
                                session_id,
                                role="assistant",
                                content=msg.get("content"),
                                tool_calls=msg.get("tool_calls"),
                            )
                        elif msg.get("role") == "tool":
                            await self._session_store.add_message(
                                session_id,
                                role="tool",
                                content=msg.get("content", ""),
                                tool_call_id=msg.get("tool_call_id", ""),
                            )

                    # If there was also text after the last tool call, persist it
                    if assistant_text:
                        await self._session_store.append_assistant_message(
                            session_id, assistant_text
                        )
                        final_text = assistant_text

                    # Continue the outer loop — LLM may want to do more
                    logger.info(
                        "SessionLoop %s iteration %d: %d tool calls, continuing",
                        session_id,
                        iteration,
                        len(new_messages),
                    )
                    continue

                else:
                    # No text and no tool calls — unusual, break
                    logger.warning(
                        "SessionLoop %s: no text and no tool calls, exiting",
                        session_id,
                    )
                    break

            # Loop finished
            if iteration >= self._max_iterations:
                logger.warning(
                    "SessionLoop %s hit max iterations (%d)",
                    session_id,
                    self._max_iterations,
                )

            status = SessionStatus.DONE.value
            logger.info(
                "SessionLoop %s completed in %d iterations (%.1fs)",
                session_id,
                iteration,
                time.time() - started_at,
            )

        except asyncio.CancelledError:
            status = SessionStatus.CANCELED.value
            logger.info("SessionLoop %s canceled", session_id)

        except Exception as e:
            status = SessionStatus.ERROR.value
            logger.error("SessionLoop %s failed: %s", session_id, e, exc_info=True)
            await self._publish_error(session_id, str(e))

        finally:
            await self._session_store.update_session_status(session_id, status)
            await self._publish_status(session_id, status)
            if self._event_bus:
                await self._event_bus.publish_end(f"session.{session_id}.event")
                await self._event_bus.publish_end(f"session.{session_id}.status")

        return final_text

    # ─── Exit Condition ───────────────────────────────────────────

    def _should_exit(self, messages: list[dict[str, Any]]) -> bool:
        """
        Check if the agent should stop.

        Exit when the last message is an assistant message with text
        content and NO tool calls. This means the agent has given its
        final response.

        Do NOT exit when:
        - Last message is a user message (need to respond)
        - Last message is a tool result (need to process it)
        - Last message is an assistant message WITH tool calls (tools pending)
        """
        if not messages:
            return False

        last = messages[-1]
        role = last.get("role", "")

        if role == "assistant":
            has_tool_calls = bool(last.get("tool_calls"))
            has_content = bool(last.get("content"))
            # Exit if assistant responded with text and no tool calls
            return has_content and not has_tool_calls

        # Don't exit for user messages, tool results, or system messages
        return False

    # ─── Agent Type Application ─────────────────────────────────

    def _apply_agent_type(
        self, envelope: "LLMRequestEnvelope", agent_def: AgentTypeDefinition
    ) -> None:
        """
        Apply agent-type-specific system prompt and tool filtering to an envelope.

        For specialized agents (explore, planner, etc.), this:
        1. Replaces the system prompt with the agent type's prompt
        2. Filters tools to only those allowed by the agent type
        """
        # Replace system prompt
        if envelope.messages and envelope.messages[0].get("role") == "system":
            envelope.messages[0]["content"] = agent_def.system_prompt
        else:
            envelope.messages.insert(
                0, {"role": "system", "content": agent_def.system_prompt}
            )

        # Filter tools
        if envelope.tools:
            all_tool_names = [
                t.get("function", {}).get("name", "") for t in envelope.tools
            ]
            allowed = set(get_filtered_tools(agent_def.name, all_tool_names))
            envelope.tools = [
                t
                for t in envelope.tools
                if t.get("function", {}).get("name", "") in allowed
            ]

        logger.debug(
            "Applied agent type '%s': %d tools allowed",
            agent_def.name,
            len(envelope.tools),
        )

    # ─── Helpers ──────────────────────────────────────────────────

    def _extract_last_user_text(self, messages: list[dict[str, Any]]) -> str:
        """Extract the text from the last user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [
                        p.get("text", "") for p in content if p.get("type") == "text"
                    ]
                    return " ".join(texts)
        return ""

    async def _publish_event(self, session_id: str, event: LLMEventEnvelope) -> None:
        """Publish an LLM event to the event bus."""
        if self._event_bus:
            await self._event_bus.publish(
                f"session.{session_id}.event",
                {
                    "type": event.event_type,
                    "payload": event.payload,
                    "sequence": event.sequence,
                },
            )

    async def _publish_status(self, session_id: str, status: str) -> None:
        """Publish a status change to the event bus."""
        if self._event_bus:
            await self._event_bus.publish(
                f"session.{session_id}.status",
                {"status": status, "timestamp": time.time()},
            )

    async def _publish_error(self, session_id: str, error: str) -> None:
        """Publish an error event."""
        if self._event_bus:
            await self._event_bus.publish(
                f"session.{session_id}.event",
                {"type": "error", "payload": {"message": error}},
            )
