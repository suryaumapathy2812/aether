"""Tests for SessionStore — persistent session state."""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from aether.session.models import (
    Message,
    MessagePart,
    PartStatus,
    PartType,
    Session,
    SessionStatus,
)
from aether.session.store import SessionStore


@pytest_asyncio.fixture
async def store():
    """Create a SessionStore with a temp DB for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        s = SessionStore(db_path=db_path)
        await s.start()
        yield s
        await s.stop()


# ─── Session CRUD ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_session(store: SessionStore):
    session = await store.create_session("test-1")
    assert session.session_id == "test-1"
    assert session.status == SessionStatus.IDLE.value
    assert session.agent_type == "default"
    assert session.parent_session_id is None


@pytest.mark.asyncio
async def test_create_session_with_parent(store: SessionStore):
    parent = await store.create_session("parent-1")
    child = await store.create_session(
        "child-1", agent_type="explore", parent_session_id="parent-1"
    )
    assert child.parent_session_id == "parent-1"
    assert child.agent_type == "explore"


@pytest.mark.asyncio
async def test_get_session(store: SessionStore):
    await store.create_session("test-1", agent_type="general")
    session = await store.get_session("test-1")
    assert session is not None
    assert session.session_id == "test-1"
    assert session.agent_type == "general"


@pytest.mark.asyncio
async def test_get_session_not_found(store: SessionStore):
    session = await store.get_session("nonexistent")
    assert session is None


@pytest.mark.asyncio
async def test_update_session_status(store: SessionStore):
    await store.create_session("test-1")
    await store.update_session_status("test-1", SessionStatus.BUSY.value)
    session = await store.get_session("test-1")
    assert session is not None
    assert session.status == SessionStatus.BUSY.value


@pytest.mark.asyncio
async def test_get_child_sessions(store: SessionStore):
    await store.create_session("parent-1")
    await store.create_session("child-1", parent_session_id="parent-1")
    await store.create_session("child-2", parent_session_id="parent-1")
    await store.create_session("other-1")  # Not a child

    children = await store.get_child_sessions("parent-1")
    assert len(children) == 2
    assert {c.session_id for c in children} == {"child-1", "child-2"}


@pytest.mark.asyncio
async def test_list_sessions(store: SessionStore):
    await store.create_session("s1")
    await store.create_session("s2")
    await store.create_session("s3")

    sessions = await store.list_sessions()
    assert len(sessions) == 3


@pytest.mark.asyncio
async def test_list_sessions_by_status(store: SessionStore):
    await store.create_session("s1")
    await store.create_session("s2")
    await store.update_session_status("s2", SessionStatus.BUSY.value)

    idle = await store.list_sessions(status=SessionStatus.IDLE.value)
    assert len(idle) == 1
    assert idle[0].session_id == "s1"

    busy = await store.list_sessions(status=SessionStatus.BUSY.value)
    assert len(busy) == 1
    assert busy[0].session_id == "s2"


@pytest.mark.asyncio
async def test_ensure_session_creates(store: SessionStore):
    session = await store.ensure_session("new-session")
    assert session.session_id == "new-session"
    assert session.status == SessionStatus.IDLE.value


@pytest.mark.asyncio
async def test_ensure_session_idempotent(store: SessionStore):
    s1 = await store.ensure_session("test-1")
    await store.update_session_status("test-1", SessionStatus.BUSY.value)
    s2 = await store.ensure_session("test-1")
    assert s2.status == SessionStatus.BUSY.value  # Not overwritten


# ─── Message CRUD ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_and_get_messages(store: SessionStore):
    await store.create_session("test-1")
    await store.add_message("test-1", role="user", content="hello")
    await store.add_message("test-1", role="assistant", content="hi there")

    messages = await store.get_messages("test-1")
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "hello"
    assert messages[0].sequence == 0
    assert messages[1].role == "assistant"
    assert messages[1].content == "hi there"
    assert messages[1].sequence == 1


@pytest.mark.asyncio
async def test_message_sequence_auto_increments(store: SessionStore):
    await store.create_session("test-1")
    m1 = await store.add_message("test-1", role="user", content="first")
    m2 = await store.add_message("test-1", role="assistant", content="second")
    m3 = await store.add_message("test-1", role="user", content="third")

    assert m1.sequence == 0
    assert m2.sequence == 1
    assert m3.sequence == 2


@pytest.mark.asyncio
async def test_add_message_with_tool_calls(store: SessionStore):
    await store.create_session("test-1")
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "test.py"}'},
        }
    ]
    await store.add_message(
        "test-1", role="assistant", content=None, tool_calls=tool_calls
    )

    messages = await store.get_messages("test-1")
    assert len(messages) == 1
    assert messages[0].tool_calls == tool_calls
    assert messages[0].content is None


@pytest.mark.asyncio
async def test_add_tool_result_message(store: SessionStore):
    await store.create_session("test-1")
    await store.add_message(
        "test-1", role="tool", content="file contents here", tool_call_id="call_123"
    )

    messages = await store.get_messages("test-1")
    assert len(messages) == 1
    assert messages[0].role == "tool"
    assert messages[0].tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_get_messages_with_limit(store: SessionStore):
    await store.create_session("test-1")
    for i in range(10):
        await store.add_message("test-1", role="user", content=f"msg {i}")

    # Get last 3 messages
    messages = await store.get_messages("test-1", limit=3)
    assert len(messages) == 3
    # Should be the LAST 3, in chronological order
    assert messages[0].content == "msg 7"
    assert messages[1].content == "msg 8"
    assert messages[2].content == "msg 9"


@pytest.mark.asyncio
async def test_get_messages_as_openai(store: SessionStore):
    await store.create_session("test-1")
    await store.add_message("test-1", role="user", content="hello")
    await store.add_message("test-1", role="assistant", content="hi")

    openai_msgs = await store.get_messages_as_openai("test-1")
    assert len(openai_msgs) == 2
    assert openai_msgs[0] == {"role": "user", "content": "hello"}
    assert openai_msgs[1] == {"role": "assistant", "content": "hi"}


@pytest.mark.asyncio
async def test_get_messages_as_openai_with_tool_calls(store: SessionStore):
    await store.create_session("test-1")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        }
    ]
    await store.add_message(
        "test-1", role="assistant", content="let me check", tool_calls=tool_calls
    )
    await store.add_message(
        "test-1", role="tool", content="result", tool_call_id="call_1"
    )

    openai_msgs = await store.get_messages_as_openai("test-1")
    assert openai_msgs[0]["role"] == "assistant"
    assert openai_msgs[0]["tool_calls"] == tool_calls
    assert openai_msgs[1]["role"] == "tool"
    assert openai_msgs[1]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_get_message_count(store: SessionStore):
    await store.create_session("test-1")
    assert await store.get_message_count("test-1") == 0

    await store.add_message("test-1", role="user", content="hello")
    assert await store.get_message_count("test-1") == 1

    await store.add_message("test-1", role="assistant", content="hi")
    assert await store.get_message_count("test-1") == 2


@pytest.mark.asyncio
async def test_delete_messages_before(store: SessionStore):
    await store.create_session("test-1")
    for i in range(5):
        await store.add_message("test-1", role="user", content=f"msg {i}")

    # Delete messages with sequence < 3 (keeps msg 3 and msg 4)
    deleted = await store.delete_messages_before("test-1", sequence=3)
    assert deleted == 3

    remaining = await store.get_messages("test-1")
    assert len(remaining) == 2
    assert remaining[0].content == "msg 3"
    assert remaining[1].content == "msg 4"


# ─── Convenience Methods ──────────────────────────────────────


@pytest.mark.asyncio
async def test_append_user_message(store: SessionStore):
    await store.create_session("test-1")
    msg = await store.append_user_message("test-1", "hello world")
    assert msg.role == "user"
    assert msg.content == "hello world"


@pytest.mark.asyncio
async def test_append_assistant_message(store: SessionStore):
    await store.create_session("test-1")
    msg = await store.append_assistant_message("test-1", "hi there")
    assert msg.role == "assistant"
    assert msg.content == "hi there"


@pytest.mark.asyncio
async def test_append_tool_result(store: SessionStore):
    await store.create_session("test-1")
    msg = await store.append_tool_result("test-1", "call_1", "result text")
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"
    assert msg.content == "result text"


@pytest.mark.asyncio
async def test_get_last_assistant_text(store: SessionStore):
    await store.create_session("test-1")
    await store.append_user_message("test-1", "hello")
    await store.append_assistant_message("test-1", "first reply")
    await store.append_user_message("test-1", "another question")
    await store.append_assistant_message("test-1", "second reply")

    last = await store.get_last_assistant_text("test-1")
    assert last == "second reply"


@pytest.mark.asyncio
async def test_get_last_assistant_text_empty(store: SessionStore):
    await store.create_session("test-1")
    last = await store.get_last_assistant_text("test-1")
    assert last is None


# ─── Message Parts ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_and_get_parts(store: SessionStore):
    await store.create_session("test-1")
    msg = await store.add_message("test-1", role="assistant", content="working...")

    part1 = await store.add_part(
        msg.message_id,
        part_type=PartType.TEXT.value,
        content={"text": "Let me check that."},
    )
    part2 = await store.add_part(
        msg.message_id,
        part_type=PartType.TOOL_CALL.value,
        content={"tool_name": "read_file", "arguments": {"path": "test.py"}},
        status=PartStatus.RUNNING.value,
    )

    parts = await store.get_parts(msg.message_id)
    assert len(parts) == 2
    assert parts[0].part_type == PartType.TEXT.value
    assert parts[0].content == {"text": "Let me check that."}
    assert parts[1].part_type == PartType.TOOL_CALL.value
    assert parts[1].status == PartStatus.RUNNING.value


@pytest.mark.asyncio
async def test_update_part_status(store: SessionStore):
    await store.create_session("test-1")
    msg = await store.add_message("test-1", role="assistant", content=None)
    part = await store.add_part(
        msg.message_id,
        part_type=PartType.TOOL_CALL.value,
        content={"tool_name": "read_file"},
        status=PartStatus.RUNNING.value,
    )

    await store.update_part_status(
        part.part_id,
        PartStatus.COMPLETED.value,
        content={"tool_name": "read_file", "output": "file contents"},
    )

    parts = await store.get_parts(msg.message_id)
    assert parts[0].status == PartStatus.COMPLETED.value
    assert parts[0].content["output"] == "file contents"


# ─── Multi-Session Isolation ──────────────────────────────────


@pytest.mark.asyncio
async def test_sessions_are_isolated(store: SessionStore):
    await store.create_session("session-a")
    await store.create_session("session-b")

    await store.append_user_message("session-a", "hello from A")
    await store.append_user_message("session-b", "hello from B")
    await store.append_user_message("session-b", "second from B")

    msgs_a = await store.get_messages("session-a")
    msgs_b = await store.get_messages("session-b")

    assert len(msgs_a) == 1
    assert len(msgs_b) == 2
    assert msgs_a[0].content == "hello from A"
    assert msgs_b[0].content == "hello from B"


# ─── Full Conversation Flow ──────────────────────────────────


@pytest.mark.asyncio
async def test_full_conversation_flow(store: SessionStore):
    """Simulate a complete conversation with tool calls."""
    session = await store.create_session("conv-1")

    # User asks a question
    await store.append_user_message("conv-1", "What files are in the project?")

    # Assistant responds with a tool call
    tool_calls = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "list_directory", "arguments": '{"path": "."}'},
        }
    ]
    await store.add_message(
        "conv-1", role="assistant", content="Let me check.", tool_calls=tool_calls
    )

    # Tool result comes back
    await store.append_tool_result("conv-1", "call_abc", "main.py\nREADME.md\ntests/")

    # Assistant gives final response
    await store.append_assistant_message(
        "conv-1",
        "The project has main.py, README.md, and a tests directory.",
    )

    # Verify the full conversation
    messages = await store.get_messages("conv-1")
    assert len(messages) == 4
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[1].tool_calls is not None
    assert messages[2].role == "tool"
    assert messages[3].role == "assistant"
    assert messages[3].tool_calls is None

    # Verify OpenAI format
    openai_msgs = await store.get_messages_as_openai("conv-1")
    assert len(openai_msgs) == 4
    assert openai_msgs[2]["tool_call_id"] == "call_abc"


# ─── Model Tests ──────────────────────────────────────────────


def test_session_with_status():
    session = Session(session_id="test", status="idle")
    updated = session.with_status("busy")
    assert updated.status == "busy"
    assert updated.session_id == "test"
    assert session.status == "idle"  # Original unchanged (frozen)


def test_message_to_openai_user():
    msg = Message(role="user", content="hello")
    assert msg.to_openai_message() == {"role": "user", "content": "hello"}


def test_message_to_openai_assistant_with_tools():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        }
    ]
    msg = Message(role="assistant", content="checking", tool_calls=tool_calls)
    openai = msg.to_openai_message()
    assert openai["role"] == "assistant"
    assert openai["tool_calls"] == tool_calls
    assert openai["content"] == "checking"


def test_message_to_openai_tool_result():
    msg = Message(role="tool", content="result", tool_call_id="call_1")
    openai = msg.to_openai_message()
    assert openai["role"] == "tool"
    assert openai["tool_call_id"] == "call_1"
    assert openai["content"] == "result"
