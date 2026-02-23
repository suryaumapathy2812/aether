"""
Session management — persistent session state for agent loops.

Provides durable storage for sessions, messages, message parts,
and the Task Ledger so that agent work survives restarts and
sub-agents get their own independent conversation histories.

Key components:
- SessionStore: SQLite-backed CRUD for sessions, messages, parts, tasks
- TaskLedger: P↔E communication channel wrapping SessionStore task ops
- SessionLoop: Outer agent loop (load → LLM → persist → repeat)
- SessionCompactor: Context window compaction
"""

from aether.session.compaction import SessionCompactor
from aether.session.ledger import TaskLedger
from aether.session.loop import SessionLoop
from aether.session.models import (
    Message,
    MessagePart,
    PartStatus,
    PartType,
    Session,
    SessionStatus,
    Task,
    TaskStatus,
    TaskType,
)
from aether.session.store import SessionStore

__all__ = [
    "Session",
    "SessionStatus",
    "Message",
    "MessagePart",
    "PartType",
    "PartStatus",
    "Task",
    "TaskStatus",
    "TaskType",
    "TaskLedger",
    "SessionStore",
    "SessionLoop",
    "SessionCompactor",
]
