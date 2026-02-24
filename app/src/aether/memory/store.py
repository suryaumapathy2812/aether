"""
Aether Memory Store — SQLite + OpenAI embeddings.

v0.03: Two-tier memory (conversations + facts)
v0.07: Four-tier memory:
1. conversations — raw exchanges (user said X, assistant said Y)
2. facts — extracted knowledge ("user's name is Surya", "user prefers Python")
3. actions — tool calls and results ("created hello-world/main.py at 3pm Tuesday")
4. sessions — session summaries for cross-session continuity
v0.08: Six-tier memory (three-bucket model per Requirements.md §7.1):
5. memories — episodic, behavioral, emotional context
6. decisions — learned rules about agent behavior (preferences, notifications, workflows)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
from openai import AsyncOpenAI

import aether.core.config as config_module

logger = logging.getLogger(__name__)

# Maximum rows to load per table during similarity search.
# Caps the O(N) scan to a bounded set of recent items.
# For true vector search, migrate to pgvector in the Go rewrite.
SEARCH_LIMIT_PER_TABLE = 200


class MemoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path(config_module.config.memory.db_path)
        self.openai: AsyncOpenAI | None = None
        self._openai_base_url: str | None = None
        self._openai_api_key: str | None = None
        self._db: aiosqlite.Connection | None = None

    async def start(self) -> None:
        """Initialize the database and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                embedding TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        # Facts table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                fact_key TEXT,
                embedding TEXT NOT NULL,
                source_conversation_id INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # v0.07: Actions table — tool calls and results
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                arguments TEXT NOT NULL,
                output TEXT NOT NULL,
                error BOOLEAN DEFAULT FALSE,
                embedding TEXT NOT NULL,
                session_id TEXT,
                timestamp REAL NOT NULL
            )
        """)

        # v0.07: Sessions table — cross-session continuity
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL NOT NULL,
                turns INTEGER NOT NULL,
                tools_used TEXT
            )
        """)

        # v0.08: Memories table — episodic, behavioral, emotional context
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory TEXT NOT NULL,
                memory_key TEXT,
                category TEXT NOT NULL DEFAULT 'episodic',
                embedding TEXT NOT NULL,
                source_conversation_id INTEGER,
                created_at REAL NOT NULL,
                expires_at REAL,
                confidence REAL NOT NULL DEFAULT 1.0
            )
        """)
        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_memory_key_unique ON memories(memory_key)"
        )

        # v0.08: Decisions table — learned rules about agent behavior
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision TEXT NOT NULL,
                decision_key TEXT,
                category TEXT NOT NULL DEFAULT 'preference',
                embedding TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'extracted',
                source_conversation_id INTEGER,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0
            )
        """)
        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_decision_key_unique ON decisions(decision_key)"
        )

        # v0.09: Notifications table — durable queue for proactive engine
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                delivery_type TEXT NOT NULL DEFAULT 'surface',
                status TEXT NOT NULL DEFAULT 'pending',
                source TEXT NOT NULL DEFAULT 'proactive',
                deliver_at REAL,
                delivered_at REAL,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at REAL,
                next_retry_at REAL,
                last_error TEXT,
                expires_at REAL,
                created_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_notifications_deliver_at ON notifications(deliver_at)"
        )

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                plugin TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                decision TEXT,
                delivery_type TEXT,
                notification_id INTEGER,
                error TEXT,
                payload TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_events_created_at ON proactive_events(created_at DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_events_status ON proactive_events(status)"
        )

        # Performance indexes for search queries
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_updated_at ON facts(updated_at DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_ended_at ON sessions(ended_at DESC)"
        )

        await self._db.commit()

        await self._migrate_fact_keys()
        await self._migrate_notification_columns()

        # Compact old actions on startup
        await self._compact_old_actions()

        logger.info(f"Memory store initialized at {self.db_path}")

    async def _migrate_notification_columns(self) -> None:
        if not self._db:
            return
        cursor = await self._db.execute("PRAGMA table_info(notifications)")
        columns = {row[1] for row in await cursor.fetchall()}

        if "delivery_attempts" not in columns:
            await self._db.execute(
                "ALTER TABLE notifications ADD COLUMN delivery_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "last_attempt_at" not in columns:
            await self._db.execute(
                "ALTER TABLE notifications ADD COLUMN last_attempt_at REAL"
            )
        if "next_retry_at" not in columns:
            await self._db.execute(
                "ALTER TABLE notifications ADD COLUMN next_retry_at REAL"
            )
        if "last_error" not in columns:
            await self._db.execute(
                "ALTER TABLE notifications ADD COLUMN last_error TEXT"
            )

        # Create index after schema migration so older DBs without
        # next_retry_at don't fail startup before ALTER TABLE runs.
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_notifications_next_retry_at ON notifications(next_retry_at)"
        )
        await self._db.commit()

    async def stop(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def add(self, user_message: str, assistant_message: str) -> None:
        """Store a conversation turn (raw storage only).

        Fact extraction is handled separately via the Task Ledger
        (MEMORY_EXTRACT task type) to avoid duplicate extraction paths.
        See AgentCore._submit_memory_extraction().
        """
        if not self._db:
            raise RuntimeError("Memory store not started")

        # Store the raw conversation
        combined = f"User: {user_message}\nAssistant: {assistant_message}"
        embedding = await self._embed(combined)

        cursor = await self._db.execute(
            "INSERT INTO conversations (user_message, assistant_message, embedding, timestamp) VALUES (?, ?, ?, ?)",
            (user_message, assistant_message, json.dumps(embedding), time.time()),
        )
        await self._db.commit()
        logger.debug(f"Stored conversation: {user_message[:50]}...")

    async def _store_fact(self, fact: str, conv_id: int) -> None:
        """Store a fact, updating if a similar one exists."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        fact = fact.strip()
        fact_key = self._canonicalize_fact(fact)
        if not fact_key:
            return

        now = time.time()

        try:
            cursor = await self._db.execute(
                "SELECT id FROM facts WHERE fact_key = ? LIMIT 1", (fact_key,)
            )
            row = await cursor.fetchone()

            if row:
                await self._db.execute(
                    """UPDATE facts
                       SET updated_at = ?, source_conversation_id = ?
                       WHERE fact_key = ?""",
                    (now, conv_id, fact_key),
                )
                await self._db.commit()
                return

            embedding = await self._embed(fact)
            await self._db.execute(
                """INSERT INTO facts (fact, fact_key, embedding, source_conversation_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_key) DO UPDATE SET updated_at=excluded.updated_at, source_conversation_id=excluded.source_conversation_id""",
                (fact, fact_key, json.dumps(embedding), conv_id, now, now),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug(f"Fact store error (may be duplicate): {e}")

    @staticmethod
    def _canonicalize_fact(fact: str) -> str:
        lowered = fact.strip().lower().replace("\u2019", "'")
        tokens = re.findall(r"[a-z0-9]+", lowered)
        return " ".join(tokens)

    async def _migrate_fact_keys(self) -> None:
        if not self._db:
            return

        cursor = await self._db.execute("PRAGMA table_info(facts)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "fact_key" not in columns:
            await self._db.execute("ALTER TABLE facts ADD COLUMN fact_key TEXT")

        cursor = await self._db.execute(
            "SELECT id, fact FROM facts ORDER BY updated_at DESC, id DESC"
        )
        rows = await cursor.fetchall()
        if not rows:
            await self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_fact_key_unique ON facts(fact_key)"
            )
            await self._db.commit()
            return

        seen_keys: dict[str, int] = {}
        delete_ids: list[int] = []
        for row_id, fact in rows:
            key = self._canonicalize_fact(fact)
            if not key:
                delete_ids.append(row_id)
                continue
            if key in seen_keys:
                delete_ids.append(row_id)
                continue
            seen_keys[key] = row_id
            await self._db.execute(
                "UPDATE facts SET fact_key = ? WHERE id = ?", (key, row_id)
            )

        if delete_ids:
            placeholders = ",".join("?" * len(delete_ids))
            await self._db.execute(
                f"DELETE FROM facts WHERE id IN ({placeholders})",
                delete_ids,
            )

        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_fact_key_unique ON facts(fact_key)"
        )
        await self._db.commit()

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search for relevant memories using cosine similarity."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        query_embedding = await self._embed(query)
        query_vec = np.array(query_embedding)

        results = []

        # Search conversations
        cursor = await self._db.execute(
            "SELECT id, user_message, assistant_message, embedding, timestamp FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (SEARCH_LIMIT_PER_TABLE,),
        )
        rows = await cursor.fetchall()

        for row in rows:
            stored_vec = np.array(json.loads(row[3]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "conversation",
                        "user_message": row[1],
                        "assistant_message": row[2],
                        "similarity": similarity,
                        "timestamp": row[4],
                    }
                )

        # Search facts (these get boosted because they're distilled knowledge)
        cursor = await self._db.execute(
            "SELECT id, fact, embedding, created_at FROM facts ORDER BY updated_at DESC LIMIT ?",
            (SEARCH_LIMIT_PER_TABLE,),
        )
        fact_rows = await cursor.fetchall()

        for row in fact_rows:
            stored_vec = np.array(json.loads(row[2]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            # Facts get a 0.1 boost — they're more valuable than raw conversations
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "fact",
                        "fact": row[1],
                        "similarity": similarity + 0.1,
                        "timestamp": row[3],
                    }
                )

        # Search actions (tool calls — what Aether *did*)
        cursor = await self._db.execute(
            "SELECT id, tool_name, arguments, output, error, embedding, timestamp FROM actions ORDER BY timestamp DESC LIMIT ?",
            (SEARCH_LIMIT_PER_TABLE,),
        )
        action_rows = await cursor.fetchall()

        for row in action_rows:
            stored_vec = np.array(json.loads(row[5]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "action",
                        "tool_name": row[1],
                        "arguments": row[2],
                        "output": row[3][:200],  # Preview only
                        "error": bool(row[4]),
                        "similarity": similarity
                        + 0.05,  # Slight boost over conversations
                        "timestamp": row[6],
                    }
                )

        # Search memories (episodic/behavioral/emotional — same boost as actions)
        cursor = await self._db.execute(
            "SELECT id, memory, category, embedding, confidence, created_at, expires_at FROM memories ORDER BY created_at DESC LIMIT ?",
            (SEARCH_LIMIT_PER_TABLE,),
        )
        memory_rows = await cursor.fetchall()

        now = time.time()
        for row in memory_rows:
            # Skip expired memories
            if row[6] is not None and row[6] < now:
                continue
            stored_vec = np.array(json.loads(row[3]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            # Memories get a +0.05 boost (same as actions)
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "memory",
                        "memory": row[1],
                        "category": row[2],
                        "confidence": row[4],
                        "similarity": similarity + 0.05,
                        "timestamp": row[5],
                    }
                )

        # Search decisions (most valuable — directly influence behavior, +0.15 boost)
        cursor = await self._db.execute(
            "SELECT id, decision, category, source, embedding, confidence, updated_at "
            "FROM decisions WHERE active = TRUE ORDER BY updated_at DESC LIMIT ?",
            (SEARCH_LIMIT_PER_TABLE,),
        )
        decision_rows = await cursor.fetchall()

        for row in decision_rows:
            stored_vec = np.array(json.loads(row[4]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            # Decisions get a +0.15 boost — they're the most valuable
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "decision",
                        "decision": row[1],
                        "category": row[2],
                        "source": row[3],
                        "confidence": row[5],
                        "similarity": similarity + 0.15,
                        "timestamp": row[6],
                    }
                )

        # Search session summaries
        cursor = await self._db.execute(
            "SELECT id, summary, embedding, ended_at FROM sessions ORDER BY ended_at DESC LIMIT ?",
            (SEARCH_LIMIT_PER_TABLE,),
        )
        session_rows = await cursor.fetchall()

        for row in session_rows:
            stored_vec = np.array(json.loads(row[2]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "session",
                        "summary": row[1],
                        "similarity": similarity + 0.05,
                        "timestamp": row[3],
                    }
                )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    async def get_facts(self) -> list[str]:
        """Get all stored facts."""
        if not self._db:
            return []

        cursor = await self._db.execute(
            "SELECT fact FROM facts ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    # --- Memory storage (v0.08: three-bucket model) ---

    async def store_memory(
        self,
        memory: str,
        category: str = "episodic",
        conv_id: int = 0,
        expires_at: float | None = None,
    ) -> None:
        """Store a memory with deduplication (same pattern as _store_fact).

        Categories: episodic, behavioral, emotional.
        """
        if not self._db:
            raise RuntimeError("Memory store not started")

        memory = memory.strip()
        memory_key = self._canonicalize_fact(memory)
        if not memory_key:
            return

        now = time.time()

        try:
            # Check for existing memory with same key — update timestamp if found
            cursor = await self._db.execute(
                "SELECT id FROM memories WHERE memory_key = ? LIMIT 1", (memory_key,)
            )
            row = await cursor.fetchone()

            if row:
                await self._db.execute(
                    """UPDATE memories
                       SET source_conversation_id = ?, confidence = 1.0
                       WHERE memory_key = ?""",
                    (conv_id, memory_key),
                )
                await self._db.commit()
                return

            embedding = await self._embed(memory)
            await self._db.execute(
                """INSERT INTO memories
                   (memory, memory_key, category, embedding, source_conversation_id, created_at, expires_at, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(memory_key) DO UPDATE SET
                       source_conversation_id=excluded.source_conversation_id,
                       confidence=1.0""",
                (
                    memory,
                    memory_key,
                    category,
                    json.dumps(embedding),
                    conv_id,
                    now,
                    expires_at,
                    1.0,
                ),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug(f"Memory store error (may be duplicate): {e}")

    async def store_decision(
        self,
        decision: str,
        category: str = "preference",
        source: str = "extracted",
        conv_id: int = 0,
    ) -> None:
        """Store a decision with deduplication. Newer decisions supersede older ones on same topic.

        Categories: preference, behavior, notification, workflow.
        Source: 'extracted' (LLM noticed pattern) or 'explicit' (user said it).
        """
        if not self._db:
            raise RuntimeError("Memory store not started")

        decision = decision.strip()
        decision_key = self._canonicalize_fact(decision)
        if not decision_key:
            return

        now = time.time()

        try:
            # Check for existing decision with same key — update if found
            cursor = await self._db.execute(
                "SELECT id FROM decisions WHERE decision_key = ? LIMIT 1",
                (decision_key,),
            )
            row = await cursor.fetchone()

            if row:
                # Newer decision supersedes: update text, timestamp, reactivate
                embedding = await self._embed(decision)
                await self._db.execute(
                    """UPDATE decisions
                       SET decision = ?, embedding = ?, source = ?, source_conversation_id = ?,
                           active = TRUE, updated_at = ?, confidence = 1.0
                       WHERE decision_key = ?""",
                    (
                        decision,
                        json.dumps(embedding),
                        source,
                        conv_id,
                        now,
                        decision_key,
                    ),
                )
                await self._db.commit()
                return

            embedding = await self._embed(decision)
            await self._db.execute(
                """INSERT INTO decisions
                   (decision, decision_key, category, embedding, source, source_conversation_id,
                    active, created_at, updated_at, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(decision_key) DO UPDATE SET
                       decision=excluded.decision, embedding=excluded.embedding,
                       source=excluded.source, source_conversation_id=excluded.source_conversation_id,
                       active=TRUE, updated_at=excluded.updated_at, confidence=1.0""",
                (
                    decision,
                    decision_key,
                    category,
                    json.dumps(embedding),
                    source,
                    conv_id,
                    True,
                    now,
                    now,
                    1.0,
                ),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug(f"Decision store error (may be duplicate): {e}")

    async def get_memories(
        self, category: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Get stored memories, optionally filtered by category."""
        if not self._db:
            return []

        if category:
            cursor = await self._db.execute(
                "SELECT id, memory, category, confidence, created_at, expires_at "
                "FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT id, memory, category, confidence, created_at, expires_at "
                "FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )

        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "memory": r[1],
                "category": r[2],
                "confidence": r[3],
                "created_at": r[4],
                "expires_at": r[5],
            }
            for r in rows
        ]

    async def get_decisions(
        self, category: str | None = None, active_only: bool = True
    ) -> list[dict]:
        """Get stored decisions, optionally filtered by category."""
        if not self._db:
            return []

        conditions = []
        params: list[Any] = []

        if active_only:
            conditions.append("active = ?")
            params.append(True)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        cursor = await self._db.execute(
            f"SELECT id, decision, category, source, active, confidence, created_at, updated_at "
            f"FROM decisions {where_clause} ORDER BY updated_at DESC",
            params,
        )

        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "decision": r[1],
                "category": r[2],
                "source": r[3],
                "active": bool(r[4]),
                "confidence": r[5],
                "created_at": r[6],
                "updated_at": r[7],
            }
            for r in rows
        ]

    async def deactivate_decision(self, decision_id: int) -> None:
        """Soft-deactivate a decision (user override)."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            "UPDATE decisions SET active = FALSE, updated_at = ? WHERE id = ?",
            (time.time(), decision_id),
        )
        await self._db.commit()
        logger.info(f"Deactivated decision {decision_id}")

    async def store_preference(self, fact: str) -> None:
        """Store a notification preference as a searchable fact."""
        await self._store_fact(fact, conv_id=0)

    async def get_recent(self, limit: int = 10) -> list[dict]:
        """Get the most recent conversations."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        cursor = await self._db.execute(
            "SELECT id, user_message, assistant_message, timestamp FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "user_message": r[1],
                "assistant_message": r[2],
                "timestamp": r[3],
            }
            for r in rows
        ]

    async def get_last_session_end(self) -> float | None:
        """Get the end timestamp of the most recent session, or None if no sessions."""
        if not self._db:
            return None

        cursor = await self._db.execute(
            "SELECT ended_at FROM sessions ORDER BY ended_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    # --- Action memory (v0.07) ---

    async def add_action(
        self,
        tool_name: str,
        arguments: dict,
        output: str,
        error: bool = False,
        session_id: str | None = None,
    ) -> None:
        """Store a tool call and its result."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        # Truncate output to keep DB manageable
        max_chars = config_module.config.memory.action_output_max_chars
        truncated_output = output[:max_chars]

        # Embed a human-readable summary for search
        args_summary = ", ".join(f"{k}={v}" for k, v in arguments.items())
        embed_text = f"Used {tool_name}({args_summary}): {truncated_output[:200]}"
        embedding = await self._embed(embed_text)

        await self._db.execute(
            """INSERT INTO actions
               (tool_name, arguments, output, error, embedding, session_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                tool_name,
                json.dumps(arguments),
                truncated_output,
                error,
                json.dumps(embedding),
                session_id,
                time.time(),
            ),
        )
        await self._db.commit()
        logger.info(f"Stored action: {tool_name}({args_summary[:60]})")

    # --- Session summaries (v0.07) ---

    async def add_session_summary(
        self,
        session_id: str,
        summary: str,
        started_at: float,
        ended_at: float,
        turns: int,
        tools_used: list[str] | None = None,
    ) -> None:
        """Store a session summary for cross-session continuity."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        embedding = await self._embed(summary)

        await self._db.execute(
            """INSERT INTO sessions
               (session_id, summary, embedding, started_at, ended_at, turns, tools_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                summary,
                json.dumps(embedding),
                started_at,
                ended_at,
                turns,
                json.dumps(tools_used or []),
            ),
        )
        await self._db.commit()
        logger.info(f"Stored session summary: {summary[:80]}...")

    async def get_session_summaries(self, limit: int | None = None) -> list[dict]:
        """Get the most recent session summaries."""
        if not self._db:
            return []

        limit = limit or config_module.config.memory.session_summary_limit
        cursor = await self._db.execute(
            "SELECT session_id, summary, started_at, ended_at, turns, tools_used "
            "FROM sessions ORDER BY ended_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": r[0],
                "summary": r[1],
                "started_at": r[2],
                "ended_at": r[3],
                "turns": r[4],
                "tools_used": json.loads(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    # --- Notification queue (v0.09: proactive engine) ---

    async def queue_notification(
        self,
        text: str,
        delivery_type: str = "surface",
        deliver_at: float | None = None,
        source: str = "proactive",
        metadata: dict | None = None,
    ) -> int:
        """Queue a notification for delivery. Returns the notification ID.

        Args:
            text: Notification text to deliver to the user.
            delivery_type: One of suppress/queue/nudge/surface/interrupt.
            deliver_at: Unix timestamp for scheduled delivery. None = deliver immediately.
            source: Origin of the notification (proactive/scheduled/user).
            metadata: Optional JSON-serializable context (source event, etc.).

        Returns:
            The notification row ID.
        """
        if not self._db:
            raise RuntimeError("Memory store not started")

        now = time.time()
        # Default expiry: 4 hours from creation
        expires_at = now + (4 * 3600)

        cursor = await self._db.execute(
            """INSERT INTO notifications
               (text, delivery_type, status, source, deliver_at, expires_at, created_at, metadata)
               VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)""",
            (
                text,
                delivery_type,
                source,
                deliver_at,
                expires_at,
                now,
                json.dumps(metadata or {}),
            ),
        )
        await self._db.commit()
        notification_id = cursor.lastrowid or 0
        logger.debug("Queued notification %d: %s", notification_id, text[:80])
        return notification_id

    async def get_pending_notifications(self, now: float | None = None) -> list[dict]:
        """Get notifications ready for delivery.

        Returns notifications where:
        - status = 'pending'
        - deliver_at is NULL (immediate) or deliver_at <= now
        - not expired (expires_at is NULL or expires_at > now)
        """
        if not self._db:
            return []

        now = now or time.time()

        cursor = await self._db.execute(
            """SELECT id, text, delivery_type, source, deliver_at, expires_at, created_at, metadata
               FROM notifications
               WHERE status = 'pending'
                  AND (deliver_at IS NULL OR deliver_at <= ?)
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at ASC""",
            (now, now, now),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "delivery_type": r[2],
                "source": r[3],
                "deliver_at": r[4],
                "expires_at": r[5],
                "created_at": r[6],
                "metadata": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    async def mark_delivery_attempt(self, notification_id: int) -> None:
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            """UPDATE notifications
               SET delivery_attempts = COALESCE(delivery_attempts, 0) + 1,
                   last_attempt_at = ?
               WHERE id = ?""",
            (time.time(), notification_id),
        )
        await self._db.commit()

    async def mark_delivery_error(self, notification_id: int, error: str) -> None:
        if not self._db:
            raise RuntimeError("Memory store not started")

        now = time.time()
        cursor = await self._db.execute(
            "SELECT delivery_attempts FROM notifications WHERE id = ?",
            (notification_id,),
        )
        row = await cursor.fetchone()
        attempts = int(row[0] if row and row[0] is not None else 1)
        backoff_s = min(2 ** max(0, attempts - 1), 300)

        await self._db.execute(
            """UPDATE notifications
               SET status = 'pending',
                   last_error = ?,
                   next_retry_at = ?
               WHERE id = ?""",
            (error[:500], now + backoff_s, notification_id),
        )
        await self._db.commit()

    async def mark_delivered(self, notification_id: int) -> None:
        """Mark a notification as delivered."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            """UPDATE notifications
               SET status = 'delivered', delivered_at = ?, next_retry_at = NULL, last_error = NULL
               WHERE id = ?""",
            (time.time(), notification_id),
        )
        await self._db.commit()

    async def record_proactive_event(
        self,
        *,
        event_id: str,
        plugin: str,
        event_type: str,
        payload: dict[str, Any],
        status: str = "ingested",
    ) -> int:
        if not self._db:
            raise RuntimeError("Memory store not started")

        now = time.time()
        cursor = await self._db.execute(
            """INSERT INTO proactive_events
               (event_id, plugin, event_type, status, payload, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                plugin,
                event_type,
                status,
                json.dumps(payload),
                now,
                now,
            ),
        )
        await self._db.commit()
        return int(cursor.lastrowid or 0)

    async def update_proactive_event(
        self,
        row_id: int,
        *,
        status: str,
        decision: str | None = None,
        delivery_type: str | None = None,
        notification_id: int | None = None,
        error: str | None = None,
    ) -> None:
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            """UPDATE proactive_events
               SET status = ?, decision = ?, delivery_type = ?, notification_id = ?,
                   error = ?, updated_at = ?
               WHERE id = ?""",
            (
                status,
                decision,
                delivery_type,
                notification_id,
                error,
                time.time(),
                row_id,
            ),
        )
        await self._db.commit()

    async def get_reliability_snapshot(self) -> dict[str, Any]:
        if not self._db:
            return {}

        now = time.time()
        pending_cursor = await self._db.execute(
            "SELECT COUNT(*) FROM notifications WHERE status = 'pending'"
        )
        pending_row = await pending_cursor.fetchone()
        pending_count = int(pending_row[0] if pending_row else 0)

        retry_cursor = await self._db.execute(
            "SELECT COUNT(*) FROM notifications WHERE status = 'pending' AND next_retry_at IS NOT NULL"
        )
        retry_row = await retry_cursor.fetchone()
        retry_count = int(retry_row[0] if retry_row else 0)

        oldest_cursor = await self._db.execute(
            "SELECT MIN(created_at) FROM notifications WHERE status = 'pending'"
        )
        oldest_row = await oldest_cursor.fetchone()
        oldest = oldest_row[0] if oldest_row else None
        oldest_age_s = max(0.0, now - float(oldest)) if oldest is not None else 0.0

        event_cursor = await self._db.execute(
            "SELECT status, COUNT(*) FROM proactive_events GROUP BY status"
        )
        event_rows = await event_cursor.fetchall()
        events_by_status = {str(k): int(v) for k, v in event_rows}

        return {
            "pending_notifications": pending_count,
            "pending_with_retry": retry_count,
            "oldest_pending_age_s": oldest_age_s,
            "proactive_events": events_by_status,
        }

    async def get_notifications(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self._db:
            return []
        cursor = await self._db.execute(
            """SELECT id, text, delivery_type, status, source, deliver_at, delivered_at,
                      delivery_attempts, last_attempt_at, next_retry_at, last_error,
                      expires_at, created_at, metadata
               FROM notifications ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "delivery_type": r[2],
                "status": r[3],
                "source": r[4],
                "deliver_at": r[5],
                "delivered_at": r[6],
                "delivery_attempts": r[7],
                "last_attempt_at": r[8],
                "next_retry_at": r[9],
                "last_error": r[10],
                "expires_at": r[11],
                "created_at": r[12],
                "metadata": json.loads(r[13]) if r[13] else {},
            }
            for r in rows
        ]

    async def export_snapshot(self) -> dict[str, Any]:
        if not self._db:
            return {}

        facts = await self.get_facts()
        memories = await self.get_memories(limit=10000)
        decisions = await self.get_decisions(active_only=False)
        conversations = await self.get_recent(limit=10000)
        sessions = await self.get_session_summaries(limit=10000)
        notifications = await self.get_notifications(limit=10000)

        proactive_cursor = await self._db.execute(
            """SELECT id, event_id, plugin, event_type, status, decision, delivery_type,
                      notification_id, error, payload, created_at, updated_at
               FROM proactive_events ORDER BY created_at DESC LIMIT 10000"""
        )
        proactive_rows = await proactive_cursor.fetchall()
        proactive_events = [
            {
                "id": r[0],
                "event_id": r[1],
                "plugin": r[2],
                "event_type": r[3],
                "status": r[4],
                "decision": r[5],
                "delivery_type": r[6],
                "notification_id": r[7],
                "error": r[8],
                "payload": json.loads(r[9]) if r[9] else {},
                "created_at": r[10],
                "updated_at": r[11],
            }
            for r in proactive_rows
        ]

        return {
            "facts": facts,
            "memories": memories,
            "decisions": decisions,
            "conversations": conversations,
            "sessions": sessions,
            "notifications": notifications,
            "proactive_events": proactive_events,
        }

    async def mark_dismissed(self, notification_id: int) -> None:
        """Mark a notification as dismissed (user feedback — feeds into learning loop)."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            "UPDATE notifications SET status = 'dismissed' WHERE id = ?",
            (notification_id,),
        )
        await self._db.commit()

    async def mark_snoozed(self, notification_id: int) -> None:
        """Mark a notification as snoozed by the user."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            "UPDATE notifications SET status = 'snoozed' WHERE id = ?",
            (notification_id,),
        )
        await self._db.commit()

    async def mark_muted(self, notification_id: int) -> None:
        """Mark a notification source as muted for this notification instance."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        await self._db.execute(
            "UPDATE notifications SET status = 'muted' WHERE id = ?",
            (notification_id,),
        )
        await self._db.commit()

    async def expire_old_notifications(self, max_age_hours: float = 4.0) -> int:
        """Expire notifications older than max_age_hours. Returns count expired.

        Marks pending notifications as 'expired' if their expires_at has passed
        or if they are older than max_age_hours (fallback for notifications
        without an explicit expires_at).
        """
        if not self._db:
            return 0

        now = time.time()
        cutoff = now - (max_age_hours * 3600)

        cursor = await self._db.execute(
            """UPDATE notifications
               SET status = 'expired'
               WHERE status = 'pending'
                 AND (
                     (expires_at IS NOT NULL AND expires_at <= ?)
                     OR (expires_at IS NULL AND created_at <= ?)
                 )""",
            (now, cutoff),
        )
        await self._db.commit()
        count = cursor.rowcount or 0
        if count:
            logger.info("Expired %d old notifications", count)
        return count

    # --- Action compaction (v0.07) ---

    async def _compact_old_actions(self) -> None:
        """Summarize old actions into facts and delete the raw rows.

        Runs on startup. Actions older than action_retention_days get
        summarized by the LLM and stored as facts.
        """
        if not self._db:
            return

        cutoff = time.time() - (
            config_module.config.memory.action_retention_days * 86400
        )

        cursor = await self._db.execute(
            "SELECT id, tool_name, arguments, output, error, timestamp FROM actions WHERE timestamp < ?",
            (cutoff,),
        )
        old_actions = list(await cursor.fetchall())

        if not old_actions:
            return

        logger.info(f"Compacting {len(old_actions)} old actions into facts")

        # Build a summary of old actions for the LLM
        action_lines = []
        action_ids = []
        for row in old_actions:
            action_ids.append(row[0])
            args = json.loads(row[2])
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            status = "failed" if row[4] else "succeeded"
            action_lines.append(f"- {row[1]}({args_str}) → {status}: {row[3][:100]}")

        if not action_lines:
            return

        actions_text = "\n".join(action_lines[:50])  # Cap at 50 for prompt size

        try:
            response = await self._get_openai_client().chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarize these past tool actions into 2-5 key facts about what was accomplished. "
                            "Focus on what was created, modified, or discovered. "
                            "Return a JSON array of fact strings.\n\n"
                            f"Actions:\n{actions_text}\n\nFacts (JSON array):"
                        ),
                    }
                ],
                max_tokens=300,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if content:
                content = content.strip()
                if content.startswith("["):
                    facts = json.loads(content)
                else:
                    import re

                    match = re.search(r"\[.*\]", content, re.DOTALL)
                    facts = json.loads(match.group()) if match else []

                for fact in facts:
                    if isinstance(fact, str) and fact.strip():
                        await self._store_fact(fact.strip(), 0)

                logger.info(
                    f"Compacted {len(old_actions)} actions into {len(facts)} facts"
                )

            # Delete the old action rows
            placeholders = ",".join("?" * len(action_ids))
            await self._db.execute(
                f"DELETE FROM actions WHERE id IN ({placeholders})",
                action_ids,
            )
            await self._db.commit()

        except Exception as e:
            logger.error(f"Action compaction failed: {e}")

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text via OpenRouter.

        Embedding model is prefixed with 'openai/' for OpenRouter routing
        (e.g. 'text-embedding-3-small' → 'openai/text-embedding-3-small').
        """
        client = self._get_openai_client()
        model = config_module.config.memory.embedding_model
        if "/" not in model:
            model = f"openai/{model}"

        response = await client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding

    def _get_openai_client(self) -> AsyncOpenAI:
        """Return an OpenAI client for embeddings and internal LLM calls.

        Routes through OpenRouter (same as LLM) using OPENROUTER_API_KEY.
        This gives access to any embedding model available on OpenRouter.
        """
        current_api_key = config_module.config.llm.api_key or ""

        # Test hook: if a client was injected directly, prefer it.
        if (
            self.openai is not None
            and self._openai_base_url is None
            and self._openai_api_key is None
        ):
            return self.openai

        if self.openai and self._openai_api_key == current_api_key:
            return self.openai

        self.openai = AsyncOpenAI(
            api_key=current_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._openai_base_url = "https://openrouter.ai/api/v1"
        self._openai_api_key = current_api_key
        return self.openai
