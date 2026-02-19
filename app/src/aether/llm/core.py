"""
LLM Core — single entry point for ALL LLM calls.

This is the heart of the "LLM-kernel OS" architecture. All LLM calls
go through this single interface, ensuring:
- Consistent input/output contracts
- Unified tool calling loop
- Centralized policy resolution
- Observability and tracing

Usage:
    core = LLMCore(provider, tool_orchestrator)

    # Simple streaming (caller handles tools)
    async for event in core.generate(envelope):
        if isinstance(event, LLMEventEnvelope):
            # text_chunk, stream_end, error
            ...
        elif isinstance(event, ToolCallRequest):
            # Execute tool and feed back
            ...

    # Full agentic loop (automatic tool execution)
    async for event in core.generate_with_tools(envelope):
        # All events including tool_result
        ...
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from aether.core.metrics import metrics
from aether.llm.contracts import (
    LLMEventEnvelope,
    LLMRequestEnvelope,
    ToolCallRequest,
    ToolResult,
)
from aether.providers.base import LLMProvider, LLMToolCall

if TYPE_CHECKING:
    from aether.tools.orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)

# Maximum tool calling iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 10

# Sentence boundary pattern for streaming
SENTENCE_BOUNDARY = re.compile(
    r"(?<!\d)"
    r"(?<![A-Z])"
    r"(?<=[.!?])"
    r"\s+"
    r'(?=[A-Z"\'\d(])'
)


class LLMCore:
    """
    Single entry point for ALL LLM calls.

    Handles the agentic tool-calling loop internally. All consumers
    (ReplyService, MemoryService, NotificationService) use this interface.

    The LLM is treated as shared compute, not a session-bound pipeline.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tool_orchestrator: "ToolOrchestrator | None" = None,
    ):
        """
        Initialize LLM Core.

        Args:
            provider: The LLM provider (OpenAI, etc.)
            tool_orchestrator: Optional tool orchestrator for automatic tool execution
        """
        self.provider = provider
        self.tool_orchestrator = tool_orchestrator

    async def generate(
        self,
        envelope: LLMRequestEnvelope,
    ) -> AsyncGenerator[LLMEventEnvelope | ToolCallRequest, None]:
        """
        Generate LLM response with tool calling support.

        Yields:
        - LLMEventEnvelope: text_chunk, stream_end, error events
        - ToolCallRequest: When LLM wants to call a tool

        The caller can either:
        1. Execute tools themselves and call generate() again with results
        2. Use generate_with_tools() which handles the loop automatically

        This method does NOT automatically execute tools - it yields
        ToolCallRequest and expects the caller to handle execution.
        """
        sequence = 0
        iteration = 0

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            buffer = ""
            pending_tool_calls: list[LLMToolCall] = []
            finish_reason = "stop"

            # Get policy settings
            policy = envelope.policy or {}
            max_tokens = policy.get("max_tokens", 500)
            temperature = policy.get("temperature", 0.7)

            try:
                # Stream from provider
                async for event in self.provider.generate_stream_with_tools(
                    envelope.messages,
                    tools=envelope.tools if envelope.tools else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if event.type == "token":
                        buffer += event.content

                        # Stream sentences as they complete
                        parts = SENTENCE_BOUNDARY.split(buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                sentence = sentence.strip()
                                if sentence:
                                    sequence += 1
                                    yield LLMEventEnvelope.text_chunk(
                                        request_id=envelope.request_id,
                                        job_id=envelope.job_id,
                                        text=sentence,
                                        sequence=sequence,
                                    )

                            buffer = parts[-1]

                    elif event.type == "tool_calls":
                        pending_tool_calls = event.tool_calls

                    elif event.type == "done":
                        finish_reason = getattr(event, "finish_reason", "stop")

            except Exception as e:
                logger.error(f"LLM error: {e}", exc_info=True)
                sequence += 1
                yield LLMEventEnvelope.error(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    message=str(e),
                    sequence=sequence,
                    code="llm_error",
                )
                return

            # Flush remaining buffer
            if buffer.strip():
                sequence += 1
                yield LLMEventEnvelope.text_chunk(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    text=buffer.strip(),
                    sequence=sequence,
                )

            # No tool calls? We're done.
            if not pending_tool_calls:
                sequence += 1
                yield LLMEventEnvelope.stream_end(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    sequence=sequence,
                    finish_reason=finish_reason,
                )
                return

            # Yield tool call requests
            for tc in pending_tool_calls:
                yield ToolCallRequest(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    call_id=tc.id,
                )

            # If caller is handling tools, they'll call us again
            # For now, signal that tools are pending and break
            # The caller should execute tools and call generate() again
            # with updated messages
            if finish_reason == "tool_calls":
                # Append assistant message with tool calls to history
                # (caller should do this before calling generate again)
                logger.debug(
                    f"Tool calls pending: {[tc.name for tc in pending_tool_calls]}"
                )
                # We break here - caller handles tool execution
                return

    async def generate_with_tools(
        self,
        envelope: LLMRequestEnvelope,
    ) -> AsyncGenerator[LLMEventEnvelope, None]:
        """
        Generate with automatic tool execution.

        Handles the full agentic loop internally:
        1. Stream LLM response
        2. If tool calls, execute them
        3. Feed results back to LLM
        4. Repeat until LLM stops calling tools

        Yields:
        - LLMEventEnvelope: All events including tool_result
        """
        iteration = 0
        sequence = 0
        call_started = time.time()
        ttft_recorded = False

        metrics.inc("llm.calls", labels={"kind": envelope.kind})

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            tool_results: list[ToolResult] = []
            pending_tool_calls: list[LLMToolCall] = []
            buffer = ""
            finish_reason = "stop"

            # Get policy settings
            policy = envelope.policy or {}
            max_tokens = policy.get("max_tokens", 500)
            temperature = policy.get("temperature", 0.7)

            try:
                # Stream from provider
                async for event in self.provider.generate_stream_with_tools(
                    envelope.messages,
                    tools=envelope.tools if envelope.tools else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if event.type == "token":
                        buffer += event.content

                        # Record TTFT on first token
                        if not ttft_recorded:
                            ttft_ms = (time.time() - call_started) * 1000
                            metrics.observe(
                                "llm.ttft_ms", ttft_ms, labels={"kind": envelope.kind}
                            )
                            ttft_recorded = True

                        # Stream sentences as they complete
                        parts = SENTENCE_BOUNDARY.split(buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                sentence = sentence.strip()
                                if sentence:
                                    sequence += 1
                                    yield LLMEventEnvelope.text_chunk(
                                        request_id=envelope.request_id,
                                        job_id=envelope.job_id,
                                        text=sentence,
                                        sequence=sequence,
                                    )
                            buffer = parts[-1]

                    elif event.type == "tool_calls":
                        pending_tool_calls = event.tool_calls

                    elif event.type == "done":
                        finish_reason = getattr(event, "finish_reason", "stop")

            except Exception as e:
                logger.error(f"LLM error: {e}", exc_info=True)
                sequence += 1
                yield LLMEventEnvelope.error(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    message=str(e),
                    sequence=sequence,
                    code="llm_error",
                )
                return

            # Flush remaining buffer
            if buffer.strip():
                sequence += 1
                yield LLMEventEnvelope.text_chunk(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    text=buffer.strip(),
                    sequence=sequence,
                )

            # No tool calls? We're done.
            if not pending_tool_calls:
                total_ms = (time.time() - call_started) * 1000
                metrics.observe(
                    "llm.duration_ms", total_ms, labels={"kind": envelope.kind}
                )
                sequence += 1
                yield LLMEventEnvelope.stream_end(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    sequence=sequence,
                    finish_reason=finish_reason,
                )
                return

            # Record tool iteration
            metrics.inc("llm.tool_iterations", labels={"kind": envelope.kind})

            # Execute tool calls
            if not self.tool_orchestrator:
                logger.warning("LLM requested tools but no orchestrator configured")
                sequence += 1
                yield LLMEventEnvelope.error(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    message="Tool execution not configured",
                    sequence=sequence,
                    code="no_tool_orchestrator",
                )
                return

            # Add assistant message with tool calls to conversation
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": buffer.strip() or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in pending_tool_calls
                ],
            }
            envelope = LLMRequestEnvelope(
                schema_version=envelope.schema_version,
                request_id=envelope.request_id,
                job_id=envelope.job_id,
                kind=envelope.kind,
                modality=envelope.modality,
                user_id=envelope.user_id,
                session_id=envelope.session_id,
                messages=envelope.messages + [assistant_msg],
                tools=envelope.tools,
                tool_choice=envelope.tool_choice,
                plugin_context=envelope.plugin_context,
                policy=envelope.policy,
                trace=envelope.trace,
            )

            # Execute each tool
            for tc in pending_tool_calls:
                logger.info(f"Tool: {tc.name}({', '.join(tc.arguments.keys())})")

                # Emit status event before execution — voice mode speaks this
                # as an acknowledge phrase while the tool runs
                status_text = getattr(tc, "status_text", None) or f"Using {tc.name}..."
                sequence += 1
                yield LLMEventEnvelope.status(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    message=status_text,
                    sequence=sequence,
                    tool_name=tc.name,
                )

                result = await self.tool_orchestrator.execute(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    call_id=tc.id,
                    plugin_context=envelope.plugin_context,
                )

                # Emit tool result event
                sequence += 1
                yield LLMEventEnvelope.tool_result(
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    tool_name=result.tool_name,
                    output=result.output[:500]
                    if len(result.output) > 500
                    else result.output,
                    call_id=result.call_id,
                    error=result.error,
                    sequence=sequence,
                )

                tool_results.append(result)

            # Add tool results to messages
            for result in tool_results:
                envelope = LLMRequestEnvelope(
                    schema_version=envelope.schema_version,
                    request_id=envelope.request_id,
                    job_id=envelope.job_id,
                    kind=envelope.kind,
                    modality=envelope.modality,
                    user_id=envelope.user_id,
                    session_id=envelope.session_id,
                    messages=envelope.messages
                    + [
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": result.output,
                        }
                    ],
                    tools=envelope.tools,
                    tool_choice=envelope.tool_choice,
                    plugin_context=envelope.plugin_context,
                    policy=envelope.policy,
                    trace=envelope.trace,
                )

            logger.debug(
                f"Tool loop iteration {iteration}: {len(tool_results)} tools executed"
            )

        # Hit max iterations
        if iteration >= MAX_TOOL_ITERATIONS:
            logger.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
            sequence += 1
            yield LLMEventEnvelope.text_chunk(
                request_id=envelope.request_id,
                job_id=envelope.job_id,
                text="I've done a lot of work on that. Let me know if you need more.",
                sequence=sequence,
            )
            sequence += 1
            yield LLMEventEnvelope.stream_end(
                request_id=envelope.request_id,
                job_id=envelope.job_id,
                sequence=sequence,
                finish_reason="max_iterations",
            )
