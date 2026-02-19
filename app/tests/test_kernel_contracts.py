"""
Tests for kernel and LLM contracts.

These tests verify that:
1. All contracts are properly serializable
2. Factory methods work correctly
3. Type validation works
"""

import pytest

from aether.kernel.contracts import (
    JobKind,
    JobModality,
    JobPriority,
    JobStatus,
    KernelEvent,
    KernelRequest,
    KernelResult,
)
from aether.llm.contracts import (
    LLMEventEnvelope,
    LLMEventType,
    LLMRequestEnvelope,
    LLMResultEnvelope,
    ToolCallRequest,
    ToolResult,
)


class TestKernelContracts:
    """Tests for kernel-level contracts."""

    def test_kernel_request_creation(self):
        """Test creating a KernelRequest."""
        request = KernelRequest(
            kind=JobKind.REPLY_TEXT.value,
            modality=JobModality.TEXT.value,
            user_id="user-123",
            session_id="session-456",
            payload={"text": "Hello"},
        )
        assert request.kind == "reply_text"
        assert request.modality == "text"
        assert request.user_id == "user-123"
        assert request.request_id  # Auto-generated
        assert request.priority == JobPriority.INTERACTIVE.value

    def test_kernel_request_with_job_id(self):
        """Test adding job_id to a request."""
        request = KernelRequest(
            kind=JobKind.REPLY_TEXT.value,
            modality=JobModality.TEXT.value,
            user_id="user-123",
            session_id="session-456",
            payload={"text": "Hello"},
        )
        updated = request.with_job_id("job-789")
        assert updated.job_id == "job-789"
        assert request.job_id == ""  # Original unchanged (frozen)

    def test_kernel_event_factory_methods(self):
        """Test KernelEvent factory methods."""
        # text_chunk
        event = KernelEvent.text_chunk(job_id="job-1", text="Hello", sequence=1)
        assert event.stream_type == "text_chunk"
        assert event.payload["text"] == "Hello"
        assert event.sequence == 1

        # tool_result
        event = KernelEvent.tool_result(
            job_id="job-1",
            tool_name="read_file",
            output="file contents",
            call_id="call-1",
            error=False,
            sequence=2,
        )
        assert event.stream_type == "tool_result"
        assert event.payload["tool_name"] == "read_file"

        # stream_end
        event = KernelEvent.stream_end(job_id="job-1", sequence=3, finish_reason="stop")
        assert event.stream_type == "stream_end"
        assert event.payload["finish_reason"] == "stop"

    def test_kernel_result_factory_methods(self):
        """Test KernelResult factory methods."""
        # success
        result = KernelResult.success(
            job_id="job-1", payload={"content": "Hello"}, latency_ms=100
        )
        assert result.status == JobStatus.COMPLETED.value
        assert result.payload["content"] == "Hello"
        assert result.metrics["latency_ms"] == 100

        # failed
        result = KernelResult.failed(
            job_id="job-1", error={"code": "llm_error", "message": "Timeout"}
        )
        assert result.status == JobStatus.FAILED.value
        assert result.error["code"] == "llm_error"

        # canceled
        result = KernelResult.canceled(job_id="job-1")
        assert result.status == JobStatus.CANCELED.value


class TestLLMContracts:
    """Tests for LLM-level contracts."""

    def test_llm_request_envelope_creation(self):
        """Test creating an LLMRequestEnvelope."""
        envelope = LLMRequestEnvelope(
            kind="reply_text",
            modality="text",
            user_id="user-123",
            session_id="session-456",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "test_tool", "description": "A test tool"}],
            policy={"model": "gpt-4o", "temperature": 0.7},
        )
        assert envelope.schema_version == "1.0"
        assert envelope.request_id  # Auto-generated
        assert len(envelope.messages) == 1
        assert envelope.tool_choice == "auto"

    def test_llm_request_envelope_with_tool_results(self):
        """Test adding tool results to envelope."""
        envelope = LLMRequestEnvelope(
            kind="reply_text",
            modality="text",
            user_id="user-123",
            session_id="session-456",
            messages=[{"role": "user", "content": "Hello"}],
        )
        results = [
            ToolResult.success(
                tool_name="read_file",
                output="file contents",
                call_id="call-1",
            )
        ]
        updated = envelope.with_tool_results(results)
        assert len(updated.messages) == 2
        assert updated.messages[1]["role"] == "tool"
        assert updated.messages[1]["tool_call_id"] == "call-1"

    def test_llm_event_envelope_factory_methods(self):
        """Test LLMEventEnvelope factory methods."""
        # text_chunk
        event = LLMEventEnvelope.text_chunk(
            request_id="req-1",
            job_id="job-1",
            text="Hello",
            sequence=1,
        )
        assert event.event_type == LLMEventType.TEXT_CHUNK.value
        assert event.payload["text"] == "Hello"
        assert event.idempotency_key == "req-1:1"

        # tool_call
        event = LLMEventEnvelope.tool_call(
            request_id="req-1",
            job_id="job-1",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            call_id="call-1",
            sequence=2,
        )
        assert event.event_type == LLMEventType.TOOL_CALL.value
        assert event.payload["tool_name"] == "read_file"

        # tool_result
        event = LLMEventEnvelope.tool_result(
            request_id="req-1",
            job_id="job-1",
            tool_name="read_file",
            output="file contents",
            call_id="call-1",
            error=False,
            sequence=3,
        )
        assert event.event_type == LLMEventType.TOOL_RESULT.value

        # stream_end
        event = LLMEventEnvelope.stream_end(
            request_id="req-1",
            job_id="job-1",
            sequence=4,
            finish_reason="stop",
        )
        assert event.event_type == LLMEventType.STREAM_END.value

        # error
        event = LLMEventEnvelope.error(
            request_id="req-1",
            job_id="job-1",
            message="Something went wrong",
            sequence=5,
            code="llm_error",
        )
        assert event.event_type == LLMEventType.ERROR.value
        assert event.payload["code"] == "llm_error"

    def test_llm_result_envelope_factory_methods(self):
        """Test LLMResultEnvelope factory methods."""
        # success
        result = LLMResultEnvelope.success(
            request_id="req-1",
            job_id="job-1",
            content="Hello, world!",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert result.status == "success"
        assert result.output["content"] == "Hello, world!"
        assert result.usage["prompt_tokens"] == 10

        # failed
        result = LLMResultEnvelope.failed(
            request_id="req-1",
            job_id="job-1",
            error={"code": "timeout", "message": "Request timed out"},
        )
        assert result.status == "failed"
        assert result.error["code"] == "timeout"


class TestToolContracts:
    """Tests for tool-related contracts."""

    def test_tool_call_request(self):
        """Test ToolCallRequest creation."""
        request = ToolCallRequest(
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            call_id="call-123",
        )
        assert request.tool_name == "read_file"
        assert request.arguments["path"] == "/tmp/test.txt"
        assert request.call_id == "call-123"

    def test_tool_result_factory_methods(self):
        """Test ToolResult factory methods."""
        # success
        result = ToolResult.success(
            tool_name="read_file",
            output="file contents",
            call_id="call-1",
            path="/tmp/test.txt",
        )
        assert result.tool_name == "read_file"
        assert result.output == "file contents"
        assert result.error is False
        assert result.metadata["path"] == "/tmp/test.txt"

        # failed
        result = ToolResult.failed(
            tool_name="read_file",
            error_msg="File not found",
            call_id="call-1",
        )
        assert result.error is True
        assert result.output == "File not found"


class TestJobKinds:
    """Tests for job kind enumeration."""

    def test_all_kinds_defined(self):
        """Test that all expected job kinds are defined."""
        expected_kinds = [
            "reply_text",
            "reply_voice",
            "memory_fact_extract",
            "memory_session_summary",
            "memory_action_compact",
            "notification_decide",
            "notification_compose",
            "tool_execute",
            "subagent_task",
        ]
        actual_kinds = [k.value for k in JobKind]
        assert set(expected_kinds) == set(actual_kinds)

    def test_modalities_defined(self):
        """Test that all modalities are defined."""
        expected = ["text", "voice", "system"]
        actual = [m.value for m in JobModality]
        assert set(expected) == set(actual)
