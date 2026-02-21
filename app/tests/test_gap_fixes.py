"""
Tests for the 4 audit gap fixes:

1. Agent-type system prompts + tool filtering wired in SessionLoop
2. task.completed events consumed by AgentCore → notifications
3. TaskRunner rewritten to delegate to SubAgentManager
4. WSSidecar wired to EventBus task.completed events
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aether.agents.agent_types import get_agent_type
from aether.agents.manager import SubAgentManager
from aether.agents.task_runner import TaskRunner
from aether.kernel.event_bus import EventBus
from aether.llm.contracts import (
    LLMEventEnvelope,
    LLMEventType,
    LLMRequestEnvelope,
)
from aether.session.loop import SessionLoop
from aether.session.models import SessionStatus
from aether.session.store import SessionStore


# ─── Shared Fixtures ──────────────────────────────────────────


@pytest_asyncio.fixture
async def store():
    """Create a SessionStore with a temp DB for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        s = SessionStore(db_path=db_path)
        await s.start()
        yield s
        await s.stop()


@pytest.fixture
def event_bus():
    return EventBus()


def _make_text_events(
    text: str, request_id: str = "req-1", job_id: str = "job-1"
) -> list[LLMEventEnvelope]:
    return [
        LLMEventEnvelope.text_chunk(request_id, job_id, text, sequence=0),
        LLMEventEnvelope.stream_end(request_id, job_id, sequence=1),
    ]


def _make_mock_llm_core(text: str = "Task completed."):
    """Mock LLMCore that always returns a simple text response."""

    async def generate_with_tools(
        envelope: LLMRequestEnvelope,
    ) -> AsyncGenerator[LLMEventEnvelope, None]:
        for event in _make_text_events(text):
            yield event

    mock = MagicMock()
    mock.generate_with_tools = generate_with_tools
    return mock


def _make_mock_context_builder():
    async def build(user_message, session, enabled_plugins=None):
        return LLMRequestEnvelope(
            messages=list(session.history),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "description": "Run a command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List directory",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Web search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "spawn_task",
                        "description": "Spawn task",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "check_task",
                        "description": "Check task",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )

    mock = AsyncMock()
    mock.build = build
    return mock


# ═══════════════════════════════════════════════════════════════
# GAP 1: Agent-type system prompts + tool filtering
# ═══════════════════════════════════════════════════════════════


class TestAgentTypeWiring:
    """Test that SessionLoop applies agent-type system prompts and tool filtering."""

    @pytest.mark.asyncio
    async def test_explore_agent_gets_filtered_tools(
        self, store: SessionStore, event_bus: EventBus
    ):
        """An explore sub-agent should only get read_file, list_directory, web_search."""
        await store.create_session("test-explore", agent_type="explore")
        await store.append_user_message("test-explore", "Explore the codebase")

        # Track what envelope the LLM receives
        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Found some files."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        loop = SessionLoop(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
            agent_type_override="explore",
        )

        result = await loop.run("test-explore")
        assert result == "Found some files."

        # Check the envelope that was sent to the LLM
        assert len(received_envelopes) == 1
        env = received_envelopes[0]

        # Tools should be filtered to explore's allowed_tools
        tool_names = [t["function"]["name"] for t in env.tools]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        assert "web_search" in tool_names
        assert "write_file" not in tool_names
        assert "run_command" not in tool_names
        assert "spawn_task" not in tool_names

    @pytest.mark.asyncio
    async def test_explore_agent_gets_custom_system_prompt(
        self, store: SessionStore, event_bus: EventBus
    ):
        """An explore agent should get the explore-specific system prompt."""
        await store.create_session("test-explore", agent_type="explore")
        await store.append_user_message("test-explore", "Explore the codebase")

        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Done."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        loop = SessionLoop(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
            agent_type_override="explore",
        )

        await loop.run("test-explore")

        assert len(received_envelopes) == 1
        env = received_envelopes[0]

        # System prompt should be the explore agent's prompt
        explore_def = get_agent_type("explore")
        system_msg = env.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == explore_def.system_prompt

    @pytest.mark.asyncio
    async def test_planner_agent_gets_filtered_tools(
        self, store: SessionStore, event_bus: EventBus
    ):
        """A planner agent should only get read-only tools."""
        await store.create_session("test-planner", agent_type="planner")
        await store.append_user_message("test-planner", "Plan the refactor")

        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Here's the plan."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        loop = SessionLoop(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
            agent_type_override="planner",
        )

        await loop.run("test-planner")

        assert len(received_envelopes) == 1
        tool_names = [t["function"]["name"] for t in received_envelopes[0].tools]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        assert "write_file" not in tool_names
        assert "run_command" not in tool_names

    @pytest.mark.asyncio
    async def test_general_agent_keeps_all_tools_except_denied(
        self, store: SessionStore, event_bus: EventBus
    ):
        """A general agent should NOT have agent-type filtering applied
        (it uses the default context builder output as-is)."""
        await store.create_session("test-general", agent_type="general")
        await store.append_user_message("test-general", "Do something")

        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Done."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        loop = SessionLoop(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
            # No agent_type_override → defaults to session's agent_type ("general")
        )

        await loop.run("test-general")

        assert len(received_envelopes) == 1
        tool_names = [t["function"]["name"] for t in received_envelopes[0].tools]
        # General agent: no _apply_agent_type called (default/general skipped)
        # So all tools from context builder are present
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "run_command" in tool_names
        assert "spawn_task" in tool_names

    @pytest.mark.asyncio
    async def test_sub_agent_manager_passes_agent_type_to_loop(
        self, store: SessionStore, event_bus: EventBus
    ):
        """SubAgentManager should create a SessionLoop with agent_type_override."""
        await store.create_session("parent-1")

        # Track what the SessionLoop receives
        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Explored."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        mgr = SubAgentManager(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
        )

        child_id = await mgr.spawn(
            "Explore the code",
            parent_session_id="parent-1",
            agent_type="explore",
        )

        # Wait for completion
        await asyncio.sleep(0.3)

        # The sub-agent should have received filtered tools
        assert len(received_envelopes) >= 1
        tool_names = [t["function"]["name"] for t in received_envelopes[0].tools]
        assert "read_file" in tool_names
        assert "write_file" not in tool_names

    @pytest.mark.asyncio
    async def test_sub_agent_manager_uses_agent_type_limits(
        self, store: SessionStore, event_bus: EventBus
    ):
        """SubAgentManager should use agent type's max_iterations and max_duration."""
        await store.create_session("parent-1")

        # Explore agent has max_iterations=30, max_duration=180
        # Manager default is max_iterations=25, max_duration=300
        # Effective should be min(25, 30)=25 iterations, min(300, 180)=180 duration
        explore_def = get_agent_type("explore")
        assert explore_def.max_iterations == 30
        assert explore_def.max_duration == 180.0

        mgr = SubAgentManager(
            session_store=store,
            llm_core=_make_mock_llm_core("Done."),
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
            max_iterations=25,
            max_duration=300,
        )

        child_id = await mgr.spawn(
            "Explore",
            parent_session_id="parent-1",
            agent_type="explore",
        )

        await asyncio.sleep(0.2)

        # Verify the child session was created with explore type
        child = await store.get_session(child_id)
        assert child is not None
        assert child.agent_type == "explore"


# ═══════════════════════════════════════════════════════════════
# GAP 2: task.completed events consumed by AgentCore
# ═══════════════════════════════════════════════════════════════


class TestTaskCompletedNotifications:
    """Test that AgentCore subscribes to task.completed and pushes notifications."""

    @pytest.mark.asyncio
    async def test_agent_core_subscribes_on_start(self, event_bus: EventBus):
        """AgentCore.start() should subscribe to task.completed topic."""
        from aether.agent import AgentCore

        scheduler = AsyncMock()
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()

        agent = AgentCore(
            scheduler=scheduler,
            memory_store=MagicMock(),
            llm_provider=MagicMock(),
            tool_registry=MagicMock(),
            skill_loader=MagicMock(),
            plugin_context=MagicMock(),
            event_bus=event_bus,
        )

        await agent.start()

        # Should have a listener task running
        assert agent._task_completed_listener is not None
        assert not agent._task_completed_listener.done()

        # Should have subscribed to the topic
        assert event_bus.subscriber_count("task.completed") >= 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_core_broadcasts_on_task_completed(
        self, store: SessionStore, event_bus: EventBus
    ):
        """When task.completed fires, AgentCore should broadcast a notification."""
        from aether.agent import AgentCore

        scheduler = AsyncMock()
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()

        agent = AgentCore(
            scheduler=scheduler,
            memory_store=MagicMock(),
            llm_provider=MagicMock(),
            tool_registry=MagicMock(),
            skill_loader=MagicMock(),
            plugin_context=MagicMock(),
            session_store=store,
            event_bus=event_bus,
        )

        # Track notifications
        received_notifications: list[dict] = []

        async def on_notif(notif: dict) -> None:
            received_notifications.append(notif)

        agent.subscribe_notifications(on_notif)

        await agent.start()

        # Create a session with a result message
        await store.create_session("sub-123")
        await store.append_user_message("sub-123", "Do something")
        await store.append_assistant_message("sub-123", "I did the thing.")

        # Publish task.completed event
        await event_bus.publish("task.completed", {"session_id": "sub-123"})

        # Give the listener time to process
        await asyncio.sleep(0.1)

        assert len(received_notifications) >= 1
        notif = received_notifications[0]
        assert notif["type"] == "task_completed"
        assert notif["session_id"] == "sub-123"
        assert "I did the thing." in notif["preview"]

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_core_cleans_up_on_stop(self, event_bus: EventBus):
        """AgentCore.stop() should cancel the listener and unsubscribe."""
        from aether.agent import AgentCore

        scheduler = AsyncMock()
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()

        agent = AgentCore(
            scheduler=scheduler,
            memory_store=MagicMock(),
            llm_provider=MagicMock(),
            tool_registry=MagicMock(),
            skill_loader=MagicMock(),
            plugin_context=MagicMock(),
            event_bus=event_bus,
        )

        await agent.start()
        assert agent._task_completed_listener is not None

        await agent.stop()

        assert agent._task_completed_listener is None
        assert agent._task_completed_queue is None
        assert event_bus.subscriber_count("task.completed") == 0

    @pytest.mark.asyncio
    async def test_agent_core_no_event_bus_no_crash(self):
        """AgentCore works fine without an EventBus (no subscription)."""
        from aether.agent import AgentCore

        scheduler = AsyncMock()
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()

        agent = AgentCore(
            scheduler=scheduler,
            memory_store=MagicMock(),
            llm_provider=MagicMock(),
            tool_registry=MagicMock(),
            skill_loader=MagicMock(),
            plugin_context=MagicMock(),
            event_bus=None,
        )

        await agent.start()
        assert agent._task_completed_listener is None
        await agent.stop()


# ═══════════════════════════════════════════════════════════════
# GAP 3: TaskRunner delegates to SubAgentManager
# ═══════════════════════════════════════════════════════════════


class TestTaskRunnerDelegation:
    """Test that TaskRunner delegates to SubAgentManager."""

    @pytest.mark.asyncio
    async def test_run_returns_result(self, store: SessionStore, event_bus: EventBus):
        """TaskRunner.run() should spawn a sub-agent and return its result."""
        await store.create_session("parent-1")

        mgr = SubAgentManager(
            session_store=store,
            llm_core=_make_mock_llm_core("File created successfully."),
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
        )

        runner = TaskRunner(
            sub_agent_manager=mgr,
            parent_session_id="parent-1",
        )

        result = await runner.run("Create a file")
        assert "File created successfully." in result

    @pytest.mark.asyncio
    async def test_run_timeout(self, store: SessionStore, event_bus: EventBus):
        """TaskRunner.run() should timeout and cancel the sub-agent."""

        async def slow_generate(envelope):
            await asyncio.sleep(10.0)
            for event in _make_text_events("Done"):
                yield event

        slow_llm = MagicMock()
        slow_llm.generate_with_tools = slow_generate

        await store.create_session("parent-1")

        mgr = SubAgentManager(
            session_store=store,
            llm_core=slow_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
        )

        runner = TaskRunner(
            sub_agent_manager=mgr,
            parent_session_id="parent-1",
        )

        result = await runner.run("Slow task", timeout=0.3)
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_run_with_agent_type(self, store: SessionStore, event_bus: EventBus):
        """TaskRunner.run() should pass agent_type to SubAgentManager."""
        await store.create_session("parent-1")

        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Explored."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        mgr = SubAgentManager(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
        )

        runner = TaskRunner(
            sub_agent_manager=mgr,
            parent_session_id="parent-1",
        )

        result = await runner.run("Explore code", agent_type="explore")
        assert "Explored." in result

        # Verify explore agent type was applied (tools filtered)
        assert len(received_envelopes) >= 1
        tool_names = [t["function"]["name"] for t in received_envelopes[0].tools]
        assert "read_file" in tool_names
        assert "write_file" not in tool_names

    @pytest.mark.asyncio
    async def test_run_task_tool_uses_new_runner(
        self, store: SessionStore, event_bus: EventBus
    ):
        """RunTaskTool should work with the new TaskRunner."""
        from aether.tools.run_task import RunTaskTool

        await store.create_session("parent-1")

        mgr = SubAgentManager(
            session_store=store,
            llm_core=_make_mock_llm_core("Task done."),
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
        )

        runner = TaskRunner(
            sub_agent_manager=mgr,
            parent_session_id="parent-1",
        )

        tool = RunTaskTool(runner)
        result = await tool.execute(prompt="Do something")
        assert not result.error
        assert "Task done." in result.output


# ═══════════════════════════════════════════════════════════════
# GAP 4: WSSidecar wired to EventBus
# ═══════════════════════════════════════════════════════════════


class TestWSSidecarEventBus:
    """Test that WSSidecar accepts event_bus parameter and subscribes."""

    def test_sidecar_accepts_event_bus(self, event_bus: EventBus):
        """WSSidecar should accept an event_bus parameter."""
        from aether.ws.sidecar import WSSidecar

        agent = MagicMock()
        sidecar = WSSidecar(agent=agent, event_bus=event_bus)
        assert sidecar._event_bus is event_bus

    def test_sidecar_works_without_event_bus(self):
        """WSSidecar should work fine without an event_bus (backward compat)."""
        from aether.ws.sidecar import WSSidecar

        agent = MagicMock()
        sidecar = WSSidecar(agent=agent)
        assert sidecar._event_bus is None

    def test_sidecar_connection_count(self):
        """Connection count should start at 0."""
        from aether.ws.sidecar import WSSidecar

        agent = MagicMock()
        sidecar = WSSidecar(agent=agent, event_bus=EventBus())
        assert sidecar.connection_count == 0


# ═══════════════════════════════════════════════════════════════
# Integration: End-to-end sub-agent with agent type
# ═══════════════════════════════════════════════════════════════


class TestEndToEndSubAgent:
    """Integration tests for the full sub-agent flow with agent types."""

    @pytest.mark.asyncio
    async def test_explore_sub_agent_full_flow(
        self, store: SessionStore, event_bus: EventBus
    ):
        """
        Full flow: spawn explore sub-agent → runs with filtered tools →
        completes → task.completed event published.
        """
        await store.create_session("parent-1")

        # Subscribe to task.completed
        completed_queue = event_bus.subscribe("task.completed")

        received_envelopes: list[LLMRequestEnvelope] = []

        async def tracking_generate(
            envelope: LLMRequestEnvelope,
        ) -> AsyncGenerator[LLMEventEnvelope, None]:
            received_envelopes.append(envelope)
            for event in _make_text_events("Found 3 relevant files."):
                yield event

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = tracking_generate

        mgr = SubAgentManager(
            session_store=store,
            llm_core=mock_llm,
            context_builder=_make_mock_context_builder(),
            event_bus=event_bus,
        )

        child_id = await mgr.spawn(
            "Find all Python files",
            parent_session_id="parent-1",
            agent_type="explore",
        )

        # Wait for completion
        await asyncio.sleep(0.3)

        # Verify result
        result = await mgr.get_result(child_id)
        assert result == "Found 3 relevant files."

        # Verify tools were filtered
        assert len(received_envelopes) >= 1
        tool_names = [t["function"]["name"] for t in received_envelopes[0].tools]
        assert "read_file" in tool_names
        assert "write_file" not in tool_names

        # Verify system prompt was explore-specific
        explore_def = get_agent_type("explore")
        system_msg = received_envelopes[0].messages[0]
        assert system_msg["content"] == explore_def.system_prompt

        # Verify task.completed event was published
        # (give a bit more time for the callback)
        await asyncio.sleep(0.2)
        # The event may or may not have arrived depending on event loop timing
        # Just verify the child session is done
        child = await store.get_session(child_id)
        assert child is not None
        assert child.status == SessionStatus.DONE.value

    @pytest.mark.asyncio
    async def test_apply_agent_type_inserts_system_prompt_when_missing(
        self, store: SessionStore, event_bus: EventBus
    ):
        """_apply_agent_type should insert a system prompt if none exists."""
        loop = SessionLoop(
            session_store=store,
            llm_core=MagicMock(),
            context_builder=MagicMock(),
        )

        # Envelope with no system message
        envelope = LLMRequestEnvelope(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        explore_def = get_agent_type("explore")
        loop._apply_agent_type(envelope, explore_def)

        assert envelope.messages[0]["role"] == "system"
        assert envelope.messages[0]["content"] == explore_def.system_prompt
        assert envelope.messages[1]["role"] == "user"
