"""
LLM Contracts — fixed input/output structures for ALL LLM calls.

Every LLM call in the system uses these contracts:
- LLMRequestEnvelope: Input (messages, tools, context, policy)
- LLMEventEnvelope: Streamed output (text_chunk, tool_call, stream_end)
- LLMResultEnvelope: Terminal result (status, output, usage)

This ensures consistency across all consumers:
- ReplyService (user responses)
- MemoryService (fact extraction, summaries)
- NotificationService (classification, compose)
- TaskRunner (sub-agent tasks)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMEventType(str, Enum):
    """Event types emitted by LLM Core."""

    TEXT_CHUNK = "text_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS = "status"  # Tool acknowledge / thinking indicator for voice
    STREAM_END = "stream_end"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ENVELOPES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMRequestEnvelope:
    """
    Fixed input structure for ALL LLM calls.

    Every field is validated. This is the single contract that all
    LLM consumers must use. No direct provider calls.

    Usage:
        envelope = LLMRequestEnvelope(
            kind="reply_text",
            modality="text",
            user_id="user-123",
            session_id="session-456",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[...],
            plugin_context={"gmail": {"access_token": "..."}},
            policy={"model": "gpt-4o", "temperature": 0.7},
        )
    """

    # Identity & routing
    schema_version: str = "1.0"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    kind: str = ""  # reply_text, memory_fact_extract, etc.
    modality: str = ""  # text, voice, system
    user_id: str = ""
    session_id: str = ""

    # Core LLM input
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{role: "system"|"user"|"assistant"|"tool", content: str, tool_call_id?: str}]

    # Tool context
    tools: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{name, description, parameters: {type, properties, required}}]
    tool_choice: str = "auto"  # auto, none, required, or specific tool name

    # Plugin context (injected into tool execution, NOT LLM prompt)
    plugin_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Format: {plugin_name: {access_token, refresh_token, config: {...}}}

    # Policy (provider/model selection)
    policy: dict[str, Any] = field(default_factory=dict)
    # Format: {provider: "openai", model: "gpt-4o", temperature: 0.7, max_tokens: 4096}

    # Tracing
    trace: dict[str, str] = field(default_factory=dict)
    # Format: {trace_id, span_id, parent_span_id}

    def with_job_id(self, job_id: str) -> "LLMRequestEnvelope":
        """Return a copy with job_id set."""
        return LLMRequestEnvelope(
            schema_version=self.schema_version,
            request_id=self.request_id,
            job_id=job_id,
            kind=self.kind,
            modality=self.modality,
            user_id=self.user_id,
            session_id=self.session_id,
            messages=self.messages,
            tools=self.tools,
            tool_choice=self.tool_choice,
            plugin_context=self.plugin_context,
            policy=self.policy,
            trace=self.trace,
        )

    def with_tool_results(self, results: list["ToolResult"]) -> "LLMRequestEnvelope":
        """Return a copy with tool results appended to messages."""
        new_messages = list(self.messages)
        for result in results:
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.output,
                }
            )
        return LLMRequestEnvelope(
            schema_version=self.schema_version,
            request_id=self.request_id,
            job_id=self.job_id,
            kind=self.kind,
            modality=self.modality,
            user_id=self.user_id,
            session_id=self.session_id,
            messages=new_messages,
            tools=self.tools,
            tool_choice=self.tool_choice,
            plugin_context=self.plugin_context,
            policy=self.policy,
            trace=self.trace,
        )


@dataclass
class LLMEventEnvelope:
    """
    Streamed event from LLM.

    Every event is independently processable and includes:
    - sequence: Monotonic per request for ordering
    - idempotency_key: For deduplication

    Event types:
    - text_chunk: Partial text output
    - tool_call: LLM wants to call a tool
    - tool_result: Tool execution result
    - stream_end: LLM finished
    - error: Error occurred
    """

    schema_version: str = "1.0"
    request_id: str = ""
    job_id: str = ""

    event_type: str = ""  # LLMEventType value
    sequence: int = 0  # Monotonic per request
    idempotency_key: str = ""  # request_id:sequence for dedup

    payload: dict[str, Any] = field(default_factory=dict)
    # text_chunk: {text: str, role: "assistant"}
    # tool_call: {tool_name, arguments, call_id}
    # tool_result: {tool_name, output, call_id, error: bool}
    # stream_end: {finish_reason: "stop"|"tool_calls"|"length"}
    # error: {code, message, recoverable: bool}

    metrics: dict[str, Any] = field(default_factory=dict)
    # {latency_ms, tokens_generated, ...}

    def __post_init__(self):
        if not self.idempotency_key:
            self.idempotency_key = f"{self.request_id}:{self.sequence}"

    @classmethod
    def text_chunk(
        cls,
        request_id: str,
        job_id: str,
        text: str,
        sequence: int,
        role: str = "assistant",
    ) -> "LLMEventEnvelope":
        """Create a text_chunk event."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            event_type=LLMEventType.TEXT_CHUNK.value,
            sequence=sequence,
            payload={"text": text, "role": role},
        )

    @classmethod
    def tool_call(
        cls,
        request_id: str,
        job_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str,
        sequence: int,
    ) -> "LLMEventEnvelope":
        """Create a tool_call event."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            event_type=LLMEventType.TOOL_CALL.value,
            sequence=sequence,
            payload={
                "tool_name": tool_name,
                "arguments": arguments,
                "call_id": call_id,
            },
        )

    @classmethod
    def tool_result(
        cls,
        request_id: str,
        job_id: str,
        tool_name: str,
        output: str,
        call_id: str,
        error: bool,
        sequence: int,
    ) -> "LLMEventEnvelope":
        """Create a tool_result event."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            event_type=LLMEventType.TOOL_RESULT.value,
            sequence=sequence,
            payload={
                "tool_name": tool_name,
                "output": output,
                "call_id": call_id,
                "error": error,
            },
        )

    @classmethod
    def status(
        cls,
        request_id: str,
        job_id: str,
        message: str,
        sequence: int,
        tool_name: str | None = None,
    ) -> "LLMEventEnvelope":
        """Create a status event — tool acknowledge or thinking indicator.

        Emitted before each tool execution so voice mode can speak
        an acknowledge phrase (e.g. "Let me check that...") while
        the tool runs in the background.
        """
        return cls(
            request_id=request_id,
            job_id=job_id,
            event_type=LLMEventType.STATUS.value,
            sequence=sequence,
            payload={"message": message, "tool_name": tool_name},
        )

    @classmethod
    def stream_end(
        cls,
        request_id: str,
        job_id: str,
        sequence: int,
        finish_reason: str = "stop",
    ) -> "LLMEventEnvelope":
        """Create a stream_end event."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            event_type=LLMEventType.STREAM_END.value,
            sequence=sequence,
            payload={"finish_reason": finish_reason},
        )

    @classmethod
    def error(
        cls,
        request_id: str,
        job_id: str,
        message: str,
        sequence: int,
        code: str = "unknown",
        recoverable: bool = False,
    ) -> "LLMEventEnvelope":
        """Create an error event."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            event_type=LLMEventType.ERROR.value,
            sequence=sequence,
            payload={
                "code": code,
                "message": message,
                "recoverable": recoverable,
            },
        )


@dataclass
class LLMResultEnvelope:
    """
    Terminal result for non-streaming or summary of streaming call.

    Returned when the LLM call completes (success or failure).
    """

    schema_version: str = "1.0"
    request_id: str = ""
    job_id: str = ""

    status: str = ""  # success, failed, canceled, timeout
    output: dict[str, Any] = field(default_factory=dict)
    # {content: str, tool_calls: [...], finish_reason: str}

    usage: dict[str, int] = field(default_factory=dict)
    # {prompt_tokens, completion_tokens, total_tokens}

    error: dict[str, Any] | None = None
    # {code, message, recoverable: bool}

    @classmethod
    def success(
        cls,
        request_id: str,
        job_id: str,
        content: str,
        finish_reason: str = "stop",
        **usage,
    ) -> "LLMResultEnvelope":
        """Create a successful result."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            status="success",
            output={"content": content, "finish_reason": finish_reason},
            usage=usage,
        )

    @classmethod
    def failed(
        cls,
        request_id: str,
        job_id: str,
        error: dict[str, Any],
    ) -> "LLMResultEnvelope":
        """Create a failed result."""
        return cls(
            request_id=request_id,
            job_id=job_id,
            status="failed",
            error=error,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL CALLING CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolCallRequest:
    """
    Emitted by LLM Core when LLM wants to call a tool.

    The caller (usually LLMCore.generate_with_tools) executes the tool
    and feeds the result back to the LLM.
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str  # OpenAI-style tool call ID


@dataclass
class ToolResult:
    """
    Result of tool execution.

    Fed back to LLM as a tool role message.
    """

    tool_name: str
    output: str
    call_id: str
    error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls, tool_name: str, output: str, call_id: str, **metadata
    ) -> "ToolResult":
        """Create a successful result."""
        return cls(
            tool_name=tool_name,
            output=output,
            call_id=call_id,
            metadata=metadata,
        )

    @classmethod
    def failed(cls, tool_name: str, error_msg: str, call_id: str) -> "ToolResult":
        """Create a failed result."""
        return cls(
            tool_name=tool_name,
            output=error_msg,
            call_id=call_id,
            error=True,
        )
