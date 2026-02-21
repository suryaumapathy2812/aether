"""Tests for SubAgentManager — spawn, monitor, cancel sub-agents."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from aether.agents.manager import SubAgentManager
from aether.kernel.event_bus import EventBus
from aether.llm.contracts import LLMEventEnvelope, LLMEventType, LLMRequestEnvelope
from aether.session.models import SessionStatus
from aether.session.store import SessionStore
from aether.tools.check_task import CheckTaskTool
from aether.tools.spawn_task import SpawnTaskTool


# ─── Fixtures ─────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        s = SessionStore(db_path=db_path)
        await s.start()
        yield s
        await s.stop()


@pytest.fixture
def event_bus():
    return EventBus()


def _make_text_events(text: str) -> list[LLMEventEnvelope]:
    return [
        LLMEventEnvelope.text_chunk("req", "job", text, sequence=0),
        LLMEventEnvelope.stream_end("req", "job", sequence=1),
    ]


def _make_mock_llm_core(text: str = "Task completed."):
    """Mock LLMCore that always returns a simple text response."""

    async def generate_with_tools(envelope: LLMRequestEnvelope) -> AsyncGenerator:
        for event in _make_text_events(text):
            yield event

    mock = MagicMock()
    mock.generate_with_tools = generate_with_tools
    return mock


def _make_mock_context_builder():
    async def build(user_message, session, enabled_plugins=None):
        return LLMRequestEnvelope(messages=list(session.history), tools=[])

    mock = AsyncMock()
    mock.build = build
    return mock


@pytest.fixture
def manager(store, event_bus):
    return SubAgentManager(
        session_store=store,
        llm_core=_make_mock_llm_core(),
        context_builder=_make_mock_context_builder(),
        event_bus=event_bus,
        max_iterations=10,
        max_duration=30,
    )


# ─── SubAgentManager Tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_returns_child_id(store, manager):
    parent = await store.create_session("parent-1")
    child_id = await manager.spawn("Do something", parent_session_id="parent-1")

    assert child_id.startswith("sub-")
    assert len(child_id) > 4


@pytest.mark.asyncio
async def test_spawn_creates_child_session(store, manager):
    await store.create_session("parent-1")
    child_id = await manager.spawn("Do something", parent_session_id="parent-1")

    child = await store.get_session(child_id)
    assert child is not None
    assert child.parent_session_id == "parent-1"
    assert child.agent_type == "general"


@pytest.mark.asyncio
async def test_spawn_adds_user_message(store, manager):
    await store.create_session("parent-1")
    child_id = await manager.spawn("Analyze the code", parent_session_id="parent-1")

    # Wait for task to complete
    await asyncio.sleep(0.1)

    messages = await store.get_messages_as_openai(child_id)
    assert len(messages) >= 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Analyze the code"


@pytest.mark.asyncio
async def test_spawn_completes_and_produces_result(store, manager):
    await store.create_session("parent-1")
    child_id = await manager.spawn("Do something", parent_session_id="parent-1")

    # Wait for completion
    await asyncio.sleep(0.2)

    result = await manager.get_result(child_id)
    assert result == "Task completed."


@pytest.mark.asyncio
async def test_get_status_running(store, event_bus):
    """Status shows running while task is active."""

    # Use a slow LLM to keep the task running
    async def slow_generate(envelope):
        await asyncio.sleep(1.0)
        for event in _make_text_events("Done"):
            yield event

    slow_llm = MagicMock()
    slow_llm.generate_with_tools = slow_generate

    mgr = SubAgentManager(
        session_store=store,
        llm_core=slow_llm,
        context_builder=_make_mock_context_builder(),
        event_bus=event_bus,
    )

    await store.create_session("parent-1")
    child_id = await mgr.spawn("Slow task", parent_session_id="parent-1")

    status = await mgr.get_status(child_id)
    assert status["running"] is True

    # Cancel to clean up
    await mgr.cancel(child_id)


@pytest.mark.asyncio
async def test_get_status_completed(store, manager):
    await store.create_session("parent-1")
    child_id = await manager.spawn("Quick task", parent_session_id="parent-1")

    await asyncio.sleep(0.2)

    status = await manager.get_status(child_id)
    assert status["status"] == SessionStatus.DONE.value
    assert status["running"] is False


@pytest.mark.asyncio
async def test_get_status_not_found(manager):
    status = await manager.get_status("nonexistent")
    assert status["status"] == "not_found"


@pytest.mark.asyncio
async def test_cancel_running_task(store, event_bus):
    async def slow_generate(envelope):
        await asyncio.sleep(5.0)
        for event in _make_text_events("Done"):
            yield event

    slow_llm = MagicMock()
    slow_llm.generate_with_tools = slow_generate

    mgr = SubAgentManager(
        session_store=store,
        llm_core=slow_llm,
        context_builder=_make_mock_context_builder(),
        event_bus=event_bus,
    )

    await store.create_session("parent-1")
    child_id = await mgr.spawn("Long task", parent_session_id="parent-1")

    canceled = await mgr.cancel(child_id)
    assert canceled is True


@pytest.mark.asyncio
async def test_cancel_nonexistent(manager):
    canceled = await manager.cancel("nonexistent")
    assert canceled is False


@pytest.mark.asyncio
async def test_list_children(store, manager):
    await store.create_session("parent-1")
    child1 = await manager.spawn("Task 1", parent_session_id="parent-1")
    child2 = await manager.spawn("Task 2", parent_session_id="parent-1")

    await asyncio.sleep(0.2)

    children = await manager.list_children("parent-1")
    child_ids = [c["session_id"] for c in children]
    assert child1 in child_ids
    assert child2 in child_ids


@pytest.mark.asyncio
async def test_active_count(store, event_bus):
    async def slow_generate(envelope):
        await asyncio.sleep(2.0)
        for event in _make_text_events("Done"):
            yield event

    slow_llm = MagicMock()
    slow_llm.generate_with_tools = slow_generate

    mgr = SubAgentManager(
        session_store=store,
        llm_core=slow_llm,
        context_builder=_make_mock_context_builder(),
        event_bus=event_bus,
    )

    await store.create_session("parent-1")
    await mgr.spawn("Task 1", parent_session_id="parent-1")
    await mgr.spawn("Task 2", parent_session_id="parent-1")

    assert mgr.active_count() == 2

    await mgr.cancel_all()


@pytest.mark.asyncio
async def test_cancel_all(store, event_bus):
    async def slow_generate(envelope):
        await asyncio.sleep(2.0)
        for event in _make_text_events("Done"):
            yield event

    slow_llm = MagicMock()
    slow_llm.generate_with_tools = slow_generate

    mgr = SubAgentManager(
        session_store=store,
        llm_core=slow_llm,
        context_builder=_make_mock_context_builder(),
        event_bus=event_bus,
    )

    await store.create_session("parent-1")
    await mgr.spawn("Task 1", parent_session_id="parent-1")
    await mgr.spawn("Task 2", parent_session_id="parent-1")

    canceled = await mgr.cancel_all()
    assert canceled == 2


@pytest.mark.asyncio
async def test_spawn_with_agent_type(store, manager):
    await store.create_session("parent-1")
    child_id = await manager.spawn(
        "Explore the code",
        parent_session_id="parent-1",
        agent_type="explore",
    )

    child = await store.get_session(child_id)
    assert child is not None
    assert child.agent_type == "explore"


# ─── SpawnTaskTool Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_task_tool(store, manager):
    await store.create_session("parent-1")
    tool = SpawnTaskTool(manager, parent_session_id="parent-1")

    result = await tool.execute(prompt="Do something")
    assert not result.error
    assert "Task spawned" in result.output
    assert "sub-" in result.output


@pytest.mark.asyncio
async def test_spawn_task_tool_with_agent_type(store, manager):
    await store.create_session("parent-1")
    tool = SpawnTaskTool(manager, parent_session_id="parent-1")

    result = await tool.execute(prompt="Explore code", agent_type="explore")
    assert not result.error
    assert "Task spawned" in result.output


# ─── CheckTaskTool Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_check_task_tool_completed(store, manager):
    await store.create_session("parent-1")
    child_id = await manager.spawn("Quick task", parent_session_id="parent-1")

    await asyncio.sleep(0.2)

    tool = CheckTaskTool(manager)
    result = await tool.execute(task_id=child_id)
    assert not result.error
    assert "completed" in result.output
    assert "Task completed." in result.output


@pytest.mark.asyncio
async def test_check_task_tool_not_found(manager):
    tool = CheckTaskTool(manager)
    result = await tool.execute(task_id="nonexistent")
    assert result.error
    assert "not found" in result.output


@pytest.mark.asyncio
async def test_check_task_tool_running(store, event_bus):
    async def slow_generate(envelope):
        await asyncio.sleep(2.0)
        for event in _make_text_events("Done"):
            yield event

    slow_llm = MagicMock()
    slow_llm.generate_with_tools = slow_generate

    mgr = SubAgentManager(
        session_store=store,
        llm_core=slow_llm,
        context_builder=_make_mock_context_builder(),
        event_bus=event_bus,
    )

    await store.create_session("parent-1")
    child_id = await mgr.spawn("Slow task", parent_session_id="parent-1")

    tool = CheckTaskTool(mgr)
    result = await tool.execute(task_id=child_id)
    assert not result.error
    assert "still running" in result.output

    await mgr.cancel(child_id)
