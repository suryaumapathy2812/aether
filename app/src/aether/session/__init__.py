"""
Session management — persistent session state for agent loops.

Provides durable storage for sessions, messages, and message parts
so that agent work survives restarts and sub-agents get their own
independent conversation histories.

Key components:
- SessionStore: SQLite-backed CRUD for sessions, messages, parts
- SessionLoop: Outer agent loop (load → LLM → persist → repeat)
- SessionCompactor: Context window compaction (Phase 6 stub)
"""

from aether.session.compaction import SessionCompactor
from aether.session.loop import SessionLoop
from aether.session.models import (
    Message,
    MessagePart,
    PartStatus,
    PartType,
    Session,
    SessionStatus,
)
from aether.session.store import SessionStore

__all__ = [
    "Session",
    "SessionStatus",
    "Message",
    "MessagePart",
    "PartType",
    "PartStatus",
    "SessionStore",
    "SessionLoop",
    "SessionCompactor",
]
