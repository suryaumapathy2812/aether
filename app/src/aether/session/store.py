"""
Session Store — SQLite-backed persistent session state.

Provides CRUD operations for sessions, messages, and message parts.
Uses the same aiosqlite pattern as MemoryStore but with its own DB file
to keep concerns separated.

Usage:
    store = SessionStore()
    await store.start()

    session = await store.create_session("my-session")
    await store.add_message(session.session_id, role="user", content="hello")
    messages = await store.get_messages(session.session_id)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

import aether.core.config as config_module
from aether.session.models import (
    Message,
    MessagePart,
    PartStatus,
    PartType,
    Session,
    SessionStatus,
)

logger = logging.getLogger(__name__)


class SessionStore:
    """
    SQLite-backed session persistence.

    Three tables:
    - sessions: top-level session containers
    - messages: conversation turns within a session
    - message_parts: fine-grained pieces within a message (tool calls, text chunks)

    Thread-safe via aiosqlite. Single writer, multiple readers.
    """

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            # Default: same directory as memory DB
            memory_db = Path(config_module.config.memory.db_path)
            db_path = memory_db.parent / "aether_sessions.db"
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def start(self) -> None:
        """Initialize the database and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'idle',
                agent_type TEXT NOT NULL DEFAULT 'default',
                parent_session_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                sequence INTEGER NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS message_parts (
                part_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                part_type TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'completed',
                sequence INTEGER NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(message_id)
            )
        """)

        # Indexes for common queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, sequence)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_parent
            ON sessions(parent_session_id)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_status
            ON sessions(status)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_parts_message
            ON message_parts(message_id, sequence)
        """)

        await self._db.commit()
        logger.info("SessionStore started (db=%s)", self.db_path)

    async def stop(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ─── Session CRUD ─────────────────────────────────────────────

    async def create_session(
        self,
        session_id: str | None = None,
        agent_type: str = "default",
        parent_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session. Returns the Session object."""
        assert self._db is not None, "SessionStore not started"

        session_id = session_id or str(uuid.uuid4())
        now = time.time()
        meta_json = json.dumps(metadata or {})

        await self._db.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, status, agent_type, parent_session_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                SessionStatus.IDLE.value,
                agent_type,
                parent_session_id,
                now,
                now,
                meta_json,
            ),
        )
        await self._db.commit()

        return Session(
            session_id=session_id,
            status=SessionStatus.IDLE.value,
            agent_type=agent_type,
            parent_session_id=parent_session_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID. Returns None if not found."""
        assert self._db is not None, "SessionStore not started"

        async with self._db.execute(
            "SELECT session_id, status, agent_type, parent_session_id, created_at, updated_at, metadata FROM sessions WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return Session(
                session_id=row[0],
                status=row[1],
                agent_type=row[2],
                parent_session_id=row[3],
                created_at=row[4],
                updated_at=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
            )

    async def update_session_status(self, session_id: str, status: str) -> None:
        """Update a session's status."""
        assert self._db is not None, "SessionStore not started"

        await self._db.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
            (status, time.time(), session_id),
        )
        await self._db.commit()

    async def get_child_sessions(self, parent_session_id: str) -> list[Session]:
        """Get all child sessions for a parent session."""
        assert self._db is not None, "SessionStore not started"

        sessions = []
        async with self._db.execute(
            "SELECT session_id, status, agent_type, parent_session_id, created_at, updated_at, metadata "
            "FROM sessions WHERE parent_session_id = ? ORDER BY created_at",
            (parent_session_id,),
        ) as cursor:
            async for row in cursor:
                sessions.append(
                    Session(
                        session_id=row[0],
                        status=row[1],
                        agent_type=row[2],
                        parent_session_id=row[3],
                        created_at=row[4],
                        updated_at=row[5],
                        metadata=json.loads(row[6]) if row[6] else {},
                    )
                )
        return sessions

    async def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions, optionally filtered by status."""
        assert self._db is not None, "SessionStore not started"

        if status:
            query = "SELECT session_id, status, agent_type, parent_session_id, created_at, updated_at, metadata FROM sessions WHERE status = ? ORDER BY updated_at DESC LIMIT ?"
            params = (status, limit)
        else:
            query = "SELECT session_id, status, agent_type, parent_session_id, created_at, updated_at, metadata FROM sessions ORDER BY updated_at DESC LIMIT ?"
            params = (limit,)

        sessions = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                sessions.append(
                    Session(
                        session_id=row[0],
                        status=row[1],
                        agent_type=row[2],
                        parent_session_id=row[3],
                        created_at=row[4],
                        updated_at=row[5],
                        metadata=json.loads(row[6]) if row[6] else {},
                    )
                )
        return sessions

    # ─── Message CRUD ─────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: Any = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        message_id: str | None = None,
    ) -> Message:
        """Add a message to a session. Auto-increments sequence."""
        assert self._db is not None, "SessionStore not started"

        message_id = message_id or str(uuid.uuid4())
        now = time.time()

        # Get next sequence number
        async with self._db.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            sequence = row[0] if row else 0

        # Serialize content and tool_calls
        content_json = json.dumps(content) if content is not None else None
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        await self._db.execute(
            """
            INSERT INTO messages
                (message_id, session_id, role, content, tool_calls, tool_call_id, sequence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                role,
                content_json,
                tool_calls_json,
                tool_call_id,
                sequence,
                now,
            ),
        )
        await self._db.commit()

        # Update session timestamp
        await self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        await self._db.commit()

        return Message(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            sequence=sequence,
            created_at=now,
        )

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get all messages for a session, ordered by sequence."""
        assert self._db is not None, "SessionStore not started"

        if limit:
            query = (
                "SELECT message_id, session_id, role, content, tool_calls, tool_call_id, sequence, created_at "
                "FROM messages WHERE session_id = ? ORDER BY sequence DESC LIMIT ?"
            )
            params: tuple = (session_id, limit)
        else:
            query = (
                "SELECT message_id, session_id, role, content, tool_calls, tool_call_id, sequence, created_at "
                "FROM messages WHERE session_id = ? ORDER BY sequence"
            )
            params = (session_id,)

        messages = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                content = json.loads(row[3]) if row[3] is not None else None
                tool_calls = json.loads(row[4]) if row[4] else None
                messages.append(
                    Message(
                        message_id=row[0],
                        session_id=row[1],
                        role=row[2],
                        content=content,
                        tool_calls=tool_calls,
                        tool_call_id=row[5],
                        sequence=row[6],
                        created_at=row[7],
                    )
                )

        # If we used DESC LIMIT, reverse to get chronological order
        if limit:
            messages.reverse()

        return messages

    async def get_messages_as_openai(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages in OpenAI-compatible format (for LLM calls)."""
        messages = await self.get_messages(session_id, limit=limit)
        return [m.to_openai_message() for m in messages]

    async def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        assert self._db is not None, "SessionStore not started"

        async with self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def delete_messages_before(
        self,
        session_id: str,
        sequence: int,
    ) -> int:
        """Delete messages with sequence < given value. Used for compaction."""
        assert self._db is not None, "SessionStore not started"

        async with self._db.execute(
            "DELETE FROM messages WHERE session_id = ? AND sequence < ?",
            (session_id, sequence),
        ) as cursor:
            deleted = cursor.rowcount

        await self._db.commit()
        return deleted or 0

    # ─── Message Parts CRUD ───────────────────────────────────────

    async def add_part(
        self,
        message_id: str,
        part_type: str,
        content: dict[str, Any] | None = None,
        status: str = PartStatus.COMPLETED.value,
        part_id: str | None = None,
    ) -> MessagePart:
        """Add a part to a message."""
        assert self._db is not None, "SessionStore not started"

        part_id = part_id or str(uuid.uuid4())
        now = time.time()
        content = content or {}

        # Get next sequence
        async with self._db.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM message_parts WHERE message_id = ?",
            (message_id,),
        ) as cursor:
            row = await cursor.fetchone()
            sequence = row[0] if row else 0

        await self._db.execute(
            """
            INSERT INTO message_parts
                (part_id, message_id, part_type, content, status, sequence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                part_id,
                message_id,
                part_type,
                json.dumps(content),
                status,
                sequence,
                now,
            ),
        )
        await self._db.commit()

        return MessagePart(
            part_id=part_id,
            message_id=message_id,
            part_type=part_type,
            content=content,
            status=status,
            sequence=sequence,
            created_at=now,
        )

    async def update_part_status(
        self,
        part_id: str,
        status: str,
        content: dict[str, Any] | None = None,
    ) -> None:
        """Update a part's status and optionally its content."""
        assert self._db is not None, "SessionStore not started"

        if content is not None:
            await self._db.execute(
                "UPDATE message_parts SET status = ?, content = ? WHERE part_id = ?",
                (status, json.dumps(content), part_id),
            )
        else:
            await self._db.execute(
                "UPDATE message_parts SET status = ? WHERE part_id = ?",
                (status, part_id),
            )
        await self._db.commit()

    async def get_parts(self, message_id: str) -> list[MessagePart]:
        """Get all parts for a message, ordered by sequence."""
        assert self._db is not None, "SessionStore not started"

        parts = []
        async with self._db.execute(
            "SELECT part_id, message_id, part_type, content, status, sequence, created_at "
            "FROM message_parts WHERE message_id = ? ORDER BY sequence",
            (message_id,),
        ) as cursor:
            async for row in cursor:
                parts.append(
                    MessagePart(
                        part_id=row[0],
                        message_id=row[1],
                        part_type=row[2],
                        content=json.loads(row[3]) if row[3] else {},
                        status=row[4],
                        sequence=row[5],
                        created_at=row[6],
                    )
                )
        return parts

    # ─── Convenience ──────────────────────────────────────────────

    async def ensure_session(
        self,
        session_id: str,
        agent_type: str = "default",
    ) -> Session:
        """Get or create a session. Idempotent."""
        session = await self.get_session(session_id)
        if session is None:
            session = await self.create_session(
                session_id=session_id,
                agent_type=agent_type,
            )
        return session

    async def append_user_message(
        self,
        session_id: str,
        content: str,
    ) -> Message:
        """Convenience: add a user message to a session."""
        return await self.add_message(session_id, role="user", content=content)

    async def append_assistant_message(
        self,
        session_id: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Convenience: add an assistant message to a session."""
        return await self.add_message(
            session_id,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    async def append_tool_result(
        self,
        session_id: str,
        tool_call_id: str,
        content: str,
    ) -> Message:
        """Convenience: add a tool result message to a session."""
        return await self.add_message(
            session_id,
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    async def get_last_assistant_text(self, session_id: str) -> str | None:
        """Get the text content of the last assistant message."""
        assert self._db is not None, "SessionStore not started"

        async with self._db.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY sequence DESC LIMIT 1",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                content = json.loads(row[0])
                return content if isinstance(content, str) else None
        return None
