"""Tests for SessionCompactor — context window compaction."""

import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from aether.llm.contracts import LLMEventEnvelope, LLMEventType, LLMRequestEnvelope
from aether.session.compaction import (
    CHARS_PER_TOKEN,
    SessionCompactor,
)
from aether.session.store import SessionStore


@pytest_asyncio.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        s = SessionStore(db_path=db_path)
        await s.start()
        yield s
        await s.stop()


def _make_mock_llm_core(summary_text: str = "Summary of conversation."):
    """Mock LLMCore that returns a summary."""

    async def generate_with_tools(envelope: LLMRequestEnvelope) -> AsyncGenerator:
        yield LLMEventEnvelope.text_chunk("req", "job", summary_text, sequence=0)
        yield LLMEventEnvelope.stream_end("req", "job", sequence=1)

    mock = MagicMock()
    mock.generate_with_tools = generate_with_tools
    return mock


# ─── needs_compaction Tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_needs_compaction_false_when_few_messages(store):
    compactor = SessionCompactor(store, max_context_tokens=1000)
    await store.create_session("test-1")
    await store.append_user_message("test-1", "hello")
    await store.append_assistant_message("test-1", "hi")

    assert await compactor.needs_compaction("test-1") is False


@pytest.mark.asyncio
async def test_needs_compaction_true_when_over_budget(store):
    compactor = SessionCompactor(store, max_context_tokens=10)  # Very low budget
    await store.create_session("test-1")

    # Add enough messages to exceed budget
    for i in range(20):
        await store.append_user_message("test-1", f"Message {i} " * 50)
        await store.append_assistant_message("test-1", f"Reply {i} " * 50)

    assert await compactor.needs_compaction("test-1") is True


@pytest.mark.asyncio
async def test_needs_compaction_false_when_under_budget(store):
    compactor = SessionCompactor(store, max_context_tokens=100_000)
    await store.create_session("test-1")
    await store.append_user_message("test-1", "short message")
    await store.append_assistant_message("test-1", "short reply")

    assert await compactor.needs_compaction("test-1") is False


# ─── compact Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_with_llm(store):
    """Compaction with LLM summarization."""
    llm = _make_mock_llm_core("Key points: user asked about X, assistant explained Y.")
    compactor = SessionCompactor(store, llm_core=llm, preserve_recent=2)

    await store.create_session("test-1")
    for i in range(10):
        await store.append_user_message("test-1", f"Question {i}")
        await store.append_assistant_message("test-1", f"Answer {i}")

    before_count = await store.get_message_count("test-1")
    assert before_count == 20

    result = await compactor.compact("test-1")
    assert result is True

    # Should have fewer messages now (recent preserved + summary)
    after_count = await store.get_message_count("test-1")
    assert after_count < before_count

    # Summary message should exist
    messages = await store.get_messages("test-1")
    has_summary = any(
        "[Compacted conversation summary]" in str(m.content or "") for m in messages
    )
    assert has_summary


@pytest.mark.asyncio
async def test_compact_without_llm(store):
    """Compaction with simple truncation fallback."""
    compactor = SessionCompactor(store, llm_core=None, preserve_recent=2)

    await store.create_session("test-1")
    for i in range(10):
        await store.append_user_message("test-1", f"Question {i}")
        await store.append_assistant_message("test-1", f"Answer {i}")

    result = await compactor.compact("test-1")
    assert result is True


@pytest.mark.asyncio
async def test_compact_too_few_messages(store):
    """Compaction skipped when too few messages."""
    compactor = SessionCompactor(store, preserve_recent=6)

    await store.create_session("test-1")
    await store.append_user_message("test-1", "hello")
    await store.append_assistant_message("test-1", "hi")

    result = await compactor.compact("test-1")
    assert result is False


@pytest.mark.asyncio
async def test_compact_preserves_recent_messages(store):
    """Recent messages are preserved after compaction."""
    llm = _make_mock_llm_core("Summary.")
    compactor = SessionCompactor(store, llm_core=llm, preserve_recent=4)

    await store.create_session("test-1")
    for i in range(10):
        await store.append_user_message("test-1", f"Q{i}")
        await store.append_assistant_message("test-1", f"A{i}")

    await compactor.compact("test-1")

    messages = await store.get_messages("test-1")
    # Recent messages should still be there
    contents = [str(m.content or "") for m in messages]
    # The last few original messages should be preserved
    assert any("Q9" in c for c in contents)
    assert any("A9" in c for c in contents)


@pytest.mark.asyncio
async def test_compact_handles_tool_calls(store):
    """Compaction handles messages with tool calls."""
    llm = _make_mock_llm_core("Summary with tool usage.")
    compactor = SessionCompactor(store, llm_core=llm, preserve_recent=2)

    await store.create_session("test-1")
    await store.append_user_message("test-1", "Read the file")
    await store.add_message(
        "test-1",
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    )
    await store.add_message(
        "test-1",
        role="tool",
        content="file contents here",
        tool_call_id="call-1",
    )
    await store.append_assistant_message("test-1", "Here are the contents.")
    # Add more to exceed preserve_recent
    for i in range(5):
        await store.append_user_message("test-1", f"Follow up {i}")
        await store.append_assistant_message("test-1", f"Reply {i}")

    result = await compactor.compact("test-1")
    assert result is True
