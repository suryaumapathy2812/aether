"""
Kernel Contracts — data structures for job orchestration.

These contracts define the interface between:
- Transports/Adapters (submit jobs)
- Kernel (routes and schedules jobs)
- Services (execute jobs)

All contracts are immutable (frozen dataclasses) and serializable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobKind(str, Enum):
    """Canonical job kinds — derived from existing code paths."""

    # User-facing responses
    REPLY_TEXT = "reply_text"
    REPLY_VOICE = "reply_voice"

    # Memory operations
    MEMORY_FACT_EXTRACT = "memory_fact_extract"
    MEMORY_SESSION_SUMMARY = "memory_session_summary"
    MEMORY_ACTION_COMPACT = "memory_action_compact"

    # Notification processing
    NOTIFICATION_DECIDE = "notification_decide"
    NOTIFICATION_COMPOSE = "notification_compose"

    # Tool execution
    TOOL_EXECUTE = "tool_execute"
    SUBAGENT_TASK = "subagent_task"


class JobModality(str, Enum):
    """Modality determines latency requirements and routing."""

    TEXT = "text"  # User-facing textual I/O
    VOICE = "voice"  # User-facing audio-in/audio-out
    SYSTEM = "system"  # Internal/background jobs


class JobPriority(int, Enum):
    """Priority levels for scheduling."""

    INTERACTIVE = 0  # User-facing, low latency required
    BACKGROUND = 1  # Internal jobs, can be deferred


class JobStatus(str, Enum):
    """Status of a kernel job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class KernelRequest:
    """
    A job submission to the kernel.

    All fields are immutable. Create new instances for modifications.
    """

    kind: str  # JobKind value
    modality: str  # JobModality value
    user_id: str
    session_id: str
    payload: dict[str, Any]  # Job-specific data
    priority: int = 0  # JobPriority value
    deadline_ms: int | None = None  # Optional timeout
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""  # Assigned by kernel on submit

    def with_job_id(self, job_id: str) -> "KernelRequest":
        """Return a copy with job_id set."""
        return KernelRequest(
            kind=self.kind,
            modality=self.modality,
            user_id=self.user_id,
            session_id=self.session_id,
            payload=self.payload,
            priority=self.priority,
            deadline_ms=self.deadline_ms,
            request_id=self.request_id,
            job_id=job_id,
        )


@dataclass(frozen=True)
class KernelEvent:
    """
    A streamed event from a running job.

    Events are yielded by the kernel stream() method. Each event is
    independently processable and includes sequence for ordering.
    """

    job_id: str
    stream_type: str  # text_chunk, audio_chunk, tool_result, stream_end, etc.
    payload: dict[str, Any]
    sequence: int  # Monotonic per job
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def text_chunk(
        cls, job_id: str, text: str, sequence: int, role: str = "assistant"
    ) -> "KernelEvent":
        """Create a text_chunk event."""
        return cls(
            job_id=job_id,
            stream_type="text_chunk",
            payload={"text": text, "role": role},
            sequence=sequence,
        )

    @classmethod
    def tool_result(
        cls,
        job_id: str,
        tool_name: str,
        output: str,
        call_id: str,
        error: bool,
        sequence: int,
    ) -> "KernelEvent":
        """Create a tool_result event."""
        return cls(
            job_id=job_id,
            stream_type="tool_result",
            payload={
                "tool_name": tool_name,
                "output": output,
                "call_id": call_id,
                "error": error,
            },
            sequence=sequence,
        )

    @classmethod
    def stream_end(
        cls, job_id: str, sequence: int, finish_reason: str = "stop"
    ) -> "KernelEvent":
        """Create a stream_end event."""
        return cls(
            job_id=job_id,
            stream_type="stream_end",
            payload={"finish_reason": finish_reason},
            sequence=sequence,
        )

    @classmethod
    def status(cls, job_id: str, text: str, sequence: int) -> "KernelEvent":
        """Create a status event (thinking, working, etc.)."""
        return cls(
            job_id=job_id,
            stream_type="status",
            payload={"text": text},
            sequence=sequence,
        )

    @classmethod
    def error(
        cls, job_id: str, message: str, sequence: int, code: str = "unknown"
    ) -> "KernelEvent":
        """Create an error event."""
        return cls(
            job_id=job_id,
            stream_type="error",
            payload={"message": message, "code": code},
            sequence=sequence,
        )


@dataclass(frozen=True)
class KernelResult:
    """
    Terminal result of a kernel job.

    Returned by await_result() when the job completes.
    """

    job_id: str
    status: str  # JobStatus value
    payload: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @classmethod
    def success(cls, job_id: str, payload: dict[str, Any], **metrics) -> "KernelResult":
        """Create a successful result."""
        return cls(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            payload=payload,
            metrics=metrics,
        )

    @classmethod
    def failed(
        cls, job_id: str, error: dict[str, Any], payload: dict[str, Any] | None = None
    ) -> "KernelResult":
        """Create a failed result."""
        return cls(
            job_id=job_id,
            status=JobStatus.FAILED.value,
            payload=payload or {},
            error=error,
        )

    @classmethod
    def canceled(cls, job_id: str) -> "KernelResult":
        """Create a canceled result."""
        return cls(
            job_id=job_id,
            status=JobStatus.CANCELED.value,
            payload={},
        )
