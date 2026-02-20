"""Tests for the memory store — v0.07 four-tier memory system.

Covers: conversations, facts, actions, sessions, search, and compaction.
All OpenAI calls (embeddings + chat) are mocked — no API keys needed.
"""

from __future__ import annotations

import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import numpy as np
import pytest
import pytest_asyncio

from aether.memory.store import MemoryStore


# --- Helpers ---


def _fake_embedding(text: str) -> list[float]:
    """Deterministic 1536-dim embedding from text hash. Normalized to unit length."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(1536).astype(float)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


# A constant unit vector — all items embedded with this will have similarity 1.0
_CONSTANT_EMBEDDING = [1.0 / (1536**0.5)] * 1536


def _mock_embedding_response(text: str):
    """Build a mock OpenAI embeddings.create() return value."""
    embedding = _fake_embedding(text)
    data_item = MagicMock()
    data_item.embedding = embedding
    response = MagicMock()
    response.data = [data_item]
    return response


def _mock_embedding_response_constant():
    """Return a constant embedding — all items will have cosine similarity 1.0."""
    data_item = MagicMock()
    data_item.embedding = list(_CONSTANT_EMBEDDING)
    response = MagicMock()
    response.data = [data_item]
    return response


def _mock_chat_response(content: str):
    """Build a mock OpenAI chat.completions.create() return value."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# --- Fixtures ---


@pytest_asyncio.fixture
async def store(tmp_path):
    """Create a MemoryStore with mocked OpenAI, backed by a real temp SQLite DB."""
    db_path = tmp_path / "test_memory.db"

    s = MemoryStore(db_path=db_path)

    # Mock OpenAI embeddings — use deterministic fake embeddings
    s.openai = MagicMock()
    s.openai.embeddings.create = AsyncMock(
        side_effect=lambda model, input: _mock_embedding_response(input)
    )
    # Mock OpenAI chat — default: return empty facts
    s.openai.chat.completions.create = AsyncMock(return_value=_mock_chat_response("[]"))

    await s.start()
    yield s
    await s.stop()


# --- Core Operations ---


class TestMemoryStoreCore:
    @pytest.mark.asyncio
    async def test_start_creates_tables(self, store):
        """All four tables should exist after start()."""
        cursor = await store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        assert "conversations" in tables
        assert "facts" in tables
        assert "actions" in tables
        assert "sessions" in tables

    @pytest.mark.asyncio
    async def test_add_conversation(self, store):
        """add() stores a conversation row."""
        await store.add("Hello", "Hi there!")

        cursor = await store._db.execute(
            "SELECT user_message, assistant_message FROM conversations"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Hello"
        assert rows[0][1] == "Hi there!"

    @pytest.mark.asyncio
    async def test_add_extracts_facts(self, store):
        """add() extracts facts from the LLM response and stores them."""
        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(
                '["User likes Python", "User lives in Chennai"]'
            )
        )

        await store.add("I love Python and I live in Chennai", "That's great!")

        facts = await store.get_facts()
        assert len(facts) == 2
        assert "User likes Python" in facts
        assert "User lives in Chennai" in facts

    @pytest.mark.asyncio
    async def test_get_facts_order(self, store):
        """get_facts() returns facts in reverse chronological order."""
        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response('["Fact A"]')
        )
        await store.add("msg1", "reply1")

        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response('["Fact B"]')
        )
        await store.add("msg2", "reply2")

        facts = await store.get_facts()
        # Most recently updated first
        assert facts[0] == "Fact B"
        assert facts[1] == "Fact A"

    @pytest.mark.asyncio
    async def test_store_fact_deduplicates_case_and_punctuation(self, store):
        """_store_fact keeps one row for canonical duplicates."""
        await store._store_fact("User's name is Surya.", conv_id=1)
        await store._store_fact("user's   name is SURYA", conv_id=2)

        cursor = await store._db.execute("SELECT fact FROM facts")
        rows = await cursor.fetchall()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_extract_facts_deduplicates_within_single_turn(self, store):
        """Extractor deduplicates repeated fact variants in one response."""
        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(
                '["User likes Python", "user likes python.", "User likes Python"]'
            )
        )

        await store.add("I like Python", "Nice!")

        facts = await store.get_facts()
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_start_migrates_legacy_facts_to_fact_key(self, tmp_path):
        """Legacy facts table is backfilled and deduplicated by canonical fact key."""
        db_path = tmp_path / "legacy_memory.db"
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                """
                CREATE TABLE facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL,
                    source_conversation_id INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            embedding = json.dumps(_fake_embedding("fact"))
            now = time.time()
            await db.execute(
                "INSERT INTO facts (fact, embedding, source_conversation_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("User prefers Python.", embedding, 1, now - 2, now - 2),
            )
            await db.execute(
                "INSERT INTO facts (fact, embedding, source_conversation_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("user prefers python", embedding, 2, now - 1, now - 1),
            )
            await db.commit()

        s = MemoryStore(db_path=db_path)
        s.openai = MagicMock()
        s.openai.embeddings.create = AsyncMock(
            side_effect=lambda model, input: _mock_embedding_response(input)
        )
        s.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response("[]")
        )

        await s.start()
        cursor = await s._db.execute("SELECT COUNT(*) FROM facts")
        count = (await cursor.fetchone())[0]
        assert count == 1

        cursor = await s._db.execute("SELECT fact_key FROM facts")
        fact_key = (await cursor.fetchone())[0]
        assert fact_key == "user prefers python"
        await s.stop()

    @pytest.mark.asyncio
    async def test_get_recent(self, store):
        """get_recent() returns conversations in reverse chronological order."""
        await store.add("first", "reply1")
        await store.add("second", "reply2")
        await store.add("third", "reply3")

        recent = await store.get_recent(limit=2)
        assert len(recent) == 2
        assert recent[0]["user_message"] == "third"
        assert recent[1]["user_message"] == "second"


# --- Action Memory ---


class TestActionMemory:
    @pytest.mark.asyncio
    async def test_add_action(self, store):
        """add_action() stores a tool call with all fields."""
        await store.add_action(
            tool_name="write_file",
            arguments={"path": "main.py", "content": "print('hello')"},
            output="File written successfully",
            session_id="sess-123",
        )

        cursor = await store._db.execute(
            "SELECT tool_name, arguments, output, error, session_id FROM actions"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "write_file"
        assert json.loads(rows[0][1]) == {
            "path": "main.py",
            "content": "print('hello')",
        }
        assert rows[0][2] == "File written successfully"
        assert rows[0][3] == 0  # no error
        assert rows[0][4] == "sess-123"

    @pytest.mark.asyncio
    async def test_add_action_truncates_output(self, store):
        """Output longer than action_output_max_chars gets truncated."""
        long_output = "x" * 5000
        await store.add_action(
            tool_name="run_command",
            arguments={"command": "cat bigfile"},
            output=long_output,
        )

        cursor = await store._db.execute("SELECT output FROM actions")
        rows = await cursor.fetchall()
        assert len(rows[0][0]) == 1000  # config default

    @pytest.mark.asyncio
    async def test_add_action_with_error(self, store):
        """Error flag is stored correctly."""
        await store.add_action(
            tool_name="run_command",
            arguments={"command": "rm -rf /"},
            output="Permission denied",
            error=True,
        )

        cursor = await store._db.execute("SELECT error FROM actions")
        rows = await cursor.fetchall()
        assert rows[0][0] == 1  # True stored as 1


# --- Session Summaries ---


class TestSessionSummaries:
    @pytest.mark.asyncio
    async def test_add_session_summary(self, store):
        """Stores and retrieves a session summary."""
        now = time.time()
        await store.add_session_summary(
            session_id="sess-abc",
            summary="User built a Flask app with auth",
            started_at=now - 3600,
            ended_at=now,
            turns=15,
            tools_used=["write_file", "run_command"],
        )

        summaries = await store.get_session_summaries()
        assert len(summaries) == 1
        s = summaries[0]
        assert s["session_id"] == "sess-abc"
        assert s["summary"] == "User built a Flask app with auth"
        assert s["turns"] == 15
        assert "write_file" in s["tools_used"]

    @pytest.mark.asyncio
    async def test_get_session_summaries_limit(self, store):
        """Respects the limit parameter."""
        now = time.time()
        for i in range(5):
            await store.add_session_summary(
                session_id=f"sess-{i}",
                summary=f"Session {i} summary",
                started_at=now - 3600 + i,
                ended_at=now + i,
                turns=i + 1,
            )

        summaries = await store.get_session_summaries(limit=2)
        assert len(summaries) == 2
        # Most recent first
        assert summaries[0]["session_id"] == "sess-4"
        assert summaries[1]["session_id"] == "sess-3"

    @pytest.mark.asyncio
    async def test_get_session_summaries_empty(self, store):
        """Returns empty list when no summaries exist."""
        summaries = await store.get_session_summaries()
        assert summaries == []


# --- Search ---
#
# Search tests use a constant embedding so all items have cosine similarity 1.0
# with the query. This isolates the search logic from embedding quality.


class TestSearch:
    @staticmethod
    def _use_constant_embeddings(store):
        """Switch the store to return identical embeddings for all texts."""
        store.openai.embeddings.create = AsyncMock(
            side_effect=lambda model, input: _mock_embedding_response_constant()
        )

    @pytest.mark.asyncio
    async def test_search_returns_conversations(self, store):
        """Search finds stored conversations."""
        self._use_constant_embeddings(store)
        await store.add("Tell me about Python", "Python is a great language")

        results = await store.search("Python programming")
        conv_results = [r for r in results if r["type"] == "conversation"]
        assert len(conv_results) >= 1
        assert conv_results[0]["user_message"] == "Tell me about Python"

    @pytest.mark.asyncio
    async def test_search_returns_facts(self, store):
        """Search finds stored facts with boost."""
        self._use_constant_embeddings(store)
        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response('["User prefers dark mode"]')
        )
        await store.add("I prefer dark mode", "Noted!")

        results = await store.search("dark mode preference")
        fact_results = [r for r in results if r["type"] == "fact"]
        assert len(fact_results) >= 1
        assert fact_results[0]["fact"] == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_search_returns_actions(self, store):
        """Search finds stored actions."""
        self._use_constant_embeddings(store)
        await store.add_action(
            tool_name="write_file",
            arguments={"path": "app.py"},
            output="Created app.py",
        )

        results = await store.search("write file app.py")
        action_results = [r for r in results if r["type"] == "action"]
        assert len(action_results) >= 1
        assert action_results[0]["tool_name"] == "write_file"

    @pytest.mark.asyncio
    async def test_search_returns_sessions(self, store):
        """Search finds stored session summaries."""
        self._use_constant_embeddings(store)
        now = time.time()
        await store.add_session_summary(
            session_id="sess-xyz",
            summary="Built a REST API with FastAPI",
            started_at=now - 3600,
            ended_at=now,
            turns=20,
        )

        results = await store.search("FastAPI REST API")
        session_results = [r for r in results if r["type"] == "session"]
        assert len(session_results) >= 1
        assert "FastAPI" in session_results[0]["summary"]

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, store):
        """Search returns at most `limit` results."""
        self._use_constant_embeddings(store)
        for i in range(10):
            await store.add(f"Message {i}", f"Reply {i}")

        results = await store.search("message", limit=3)
        assert len(results) <= 3


# --- Action Compaction ---


class TestActionCompaction:
    @pytest.mark.asyncio
    async def test_compact_old_actions(self, store):
        """Old actions get summarized into facts and deleted."""
        # Insert an action with a timestamp older than retention period
        old_timestamp = time.time() - (8 * 86400)  # 8 days ago
        embedding = _fake_embedding("old action")

        await store._db.execute(
            """INSERT INTO actions
               (tool_name, arguments, output, error, embedding, session_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "write_file",
                json.dumps({"path": "old.py"}),
                "Created old.py",
                False,
                json.dumps(embedding),
                "sess-old",
                old_timestamp,
            ),
        )
        await store._db.commit()

        # Mock the LLM to return compaction facts
        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(
                '["Created old.py file in a past session"]'
            )
        )

        await store._compact_old_actions()

        # Old action should be deleted
        cursor = await store._db.execute("SELECT COUNT(*) FROM actions")
        count = (await cursor.fetchone())[0]
        assert count == 0

        # Fact should be created
        facts = await store.get_facts()
        assert any("old.py" in f for f in facts)

    @pytest.mark.asyncio
    async def test_compact_no_old_actions(self, store):
        """Nothing happens when no actions are old enough."""
        # Insert a recent action
        await store.add_action(
            tool_name="read_file",
            arguments={"path": "new.py"},
            output="file contents",
        )

        # Reset the mock to track calls
        store.openai.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response("[]")
        )

        await store._compact_old_actions()

        # Action should still be there
        cursor = await store._db.execute("SELECT COUNT(*) FROM actions")
        count = (await cursor.fetchone())[0]
        assert count == 1

        # LLM should NOT have been called for compaction
        store.openai.chat.completions.create.assert_not_called()
