"""
Session Models — data structures for persistent session state.

These models represent the three-level hierarchy:
  Session → Message → MessagePart

Plus the Task Ledger model:
  Task — persistent P↔E communication record

Sessions can be nested (parent_session_id) for sub-agent delegation.
Messages track the conversation turns.
MessageParts track individual pieces within a message (text chunks,
tool calls, tool results) for fine-grained status tracking.
Tasks track work items delegated from P Worker to E Worker.

All models are frozen dataclasses — create new instances for modifications.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Lifecycle status of a session."""

    IDLE = "idle"  # Created but not yet running
    BUSY = "busy"  # Agent loop is actively running
    DONE = "done"  # Agent loop completed successfully
    ERROR = "error"  # Agent loop failed
    CANCELED = "canceled"  # Canceled by user or parent


class PartType(str, Enum):
    """Type of content within a message."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    COMPACTION = "compaction"  # Summarized/compacted content


class PartStatus(str, Enum):
    """Execution status of a message part (mainly for tool calls)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True)
class Session:
    """
    A conversation session — the top-level container.

    Sessions can be nested: a sub-agent gets its own session with
    parent_session_id pointing to the spawning session.
    """

    session_id: str
    status: str = SessionStatus.IDLE.value
    agent_type: str = "default"  # Agent type (default, general, explore, etc.)
    parent_session_id: str | None = None  # For sub-agent sessions
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_status(self, status: str) -> Session:
        """Return a copy with updated status and timestamp."""
        return Session(
            session_id=self.session_id,
            status=status,
            agent_type=self.agent_type,
            parent_session_id=self.parent_session_id,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class Message:
    """
    A single message in a session's conversation history.

    Maps to the OpenAI message format: role + content.
    The content field stores the full message content as a JSON-serializable
    value (string for simple messages, list for multimodal).
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    role: str = "user"  # user, assistant, system, tool
    content: Any = ""  # String or list (multimodal) or None (tool_calls only)
    tool_calls: list[dict[str, Any]] | None = (
        None  # For assistant messages with tool calls
    )
    tool_call_id: str | None = None  # For tool result messages
    sequence: int = 0  # Ordering within session
    created_at: float = field(default_factory=time.time)

    def to_openai_message(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message dict."""
        msg: dict[str, Any] = {"role": self.role}

        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id or ""
            msg["content"] = self.content or ""
        elif self.role == "assistant" and self.tool_calls:
            msg["content"] = self.content
            msg["tool_calls"] = self.tool_calls
        else:
            msg["content"] = self.content

        return msg


@dataclass(frozen=True)
class MessagePart:
    """
    A fine-grained piece within a message.

    Used for tracking individual tool calls, text chunks, and status
    updates within a single assistant message. Enables real-time
    progress tracking and partial result display.
    """

    part_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str = ""
    part_type: str = PartType.TEXT.value
    content: dict[str, Any] = field(default_factory=dict)
    status: str = PartStatus.COMPLETED.value
    sequence: int = 0
    created_at: float = field(default_factory=time.time)


# ─── Task Ledger Models ──────────────────────────────────────────


class TaskStatus(str, Enum):
    """Lifecycle status of a task in the Task Ledger.

    Transitions: pending → running → complete | error
    No other transitions are valid. A failed task is retried by
    creating a new task, not by resetting status.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class TaskType(str, Enum):
    """Type of work represented by a task."""

    MEMORY_EXTRACT = "memory_extract"  # Post-turn memory extraction
    SUB_AGENT = "sub_agent"  # Delegated sub-agent work
    SCHEDULED = "scheduled"  # Cron / scheduled work
    PROACTIVE_CHECK = "proactive_check"  # Proactive engine check
    TOOL_CALL = "tool_call"  # P-worker tool call audit record
    DELEGATION = "delegation"  # P-worker to E-worker delegation record


@dataclass(frozen=True)
class Task:
    """
    A work item in the Task Ledger — the P↔E communication record.

    The Task Ledger is the single communication channel between the
    P Worker and E Worker (Requirements.md §2.2.1):
    - P Worker creates tasks (pending), reads status/result
    - E Worker picks up pending tasks, sets running, sets complete/error
    - LLM reads the ledger via the check_task tool

    Tasks are never deleted. They form a complete audit trail of all
    work the agent has done.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""  # Which session spawned this task
    type: str = TaskType.SUB_AGENT.value
    status: str = TaskStatus.PENDING.value
    priority: str = "normal"  # high / normal / low
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None  # NULL until complete
    error: str | None = None  # NULL unless status is error
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None  # NULL until running
    completed_at: float | None = None  # NULL until complete/error
