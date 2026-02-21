"""
Session Models — data structures for persistent session state.

These models represent the three-level hierarchy:
  Session → Message → MessagePart

Sessions can be nested (parent_session_id) for sub-agent delegation.
Messages track the conversation turns.
MessageParts track individual pieces within a message (text chunks,
tool calls, tool results) for fine-grained status tracking.

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
