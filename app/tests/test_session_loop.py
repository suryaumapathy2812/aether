"""Tests for SessionLoop — the outer agent loop."""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aether.kernel.event_bus import EventBus
from aether.llm.contracts import (
    LLMEventEnvelope,
    LLMEventType,
    LLMRequestEnvelope,
)
from aether.session.compaction import SessionCompactor
from aether.session.loop import SessionLoop
from aether.session.models import SessionStatus
from aether.session.store import SessionStore


# ─── Fixtures ─────────────────────────────────────────────────


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
    """Create a sequence of LLM events for a simple text response."""
    return [
        LLMEventEnvelope.text_chunk(request_id, job_id, text, sequence=0),
        LLMEventEnvelope.stream_end(request_id, job_id, sequence=1),
    ]


def _make_tool_then_text_events(
    tool_name: str = "read_file",
    tool_args: dict | None = None,
    call_id: str = "call-1",
    final_text: str = "Done!",
    request_id: str = "req-1",
    job_id: str = "job-1",
) -> list[LLMEventEnvelope]:
    """Create events for: tool_call → tool_result → text → end."""
    return [
        LLMEventEnvelope.tool_call(
            request_id, job_id, tool_name, tool_args or {}, call_id, sequence=0
        ),
        LLMEventEnvelope.tool_result(
            request_id, job_id, tool_name, "result", call_id, False, sequence=1
        ),
        LLMEventEnvelope.text_chunk(request_id, job_id, final_text, sequence=2),
        LLMEventEnvelope.stream_end(request_id, job_id, sequence=3),
    ]


def _make_mock_llm_core(event_sequences: list[list[LLMEventEnvelope]]):
    """
    Create a mock LLMCore that yields different event sequences on each call.

    Also mutates the envelope's messages list to simulate what the real
    LLMCore.generate_with_tools() does (appending assistant + tool messages).
    """
    call_count = 0

    async def generate_with_tools(
        envelope: LLMRequestEnvelope,
    ) -> AsyncGenerator[LLMEventEnvelope, None]:
        nonlocal call_count
        idx = min(call_count, len(event_sequences) - 1)
        events = event_sequences[idx]
        call_count += 1

        # Simulate LLMCore mutating the envelope's messages
        # When there are tool calls, LLMCore appends:
        #   1. Assistant message with tool_calls
        #   2. Tool result messages
        has_tool_calls = any(
            e.event_type == LLMEventType.TOOL_CALL.value for e in events
        )
        if has_tool_calls:
            tool_call_events = [
                e for e in events if e.event_type == LLMEventType.TOOL_CALL.value
            ]
            tool_result_events = [
                e for e in events if e.event_type == LLMEventType.TOOL_RESULT.value
            ]

            # Append assistant message with tool_calls
            envelope.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.payload["call_id"],
                            "type": "function",
                            "function": {
                                "name": tc.payload["tool_name"],
                                "arguments": "{}",
                            },
                        }
                        for tc in tool_call_events
                    ],
                }
            )

            # Append tool result messages
            for tr in tool_result_events:
                envelope.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.payload["call_id"],
                        "content": tr.payload["output"],
                    }
                )

        for event in events:
            yield event

    mock = MagicMock()
    mock.generate_with_tools = generate_with_tools
    return mock


def _make_mock_context_builder(envelope: LLMRequestEnvelope | None = None):
    """Create a mock ContextBuilder that returns a fixed envelope."""
    mock = AsyncMock()

    async def build(
        user_message: str,
        session: Any,
        enabled_plugins: list[str] | None = None,
    ) -> LLMRequestEnvelope:
        # Return a fresh envelope each time with the session's messages
        # The SessionLoop passes messages via session_state.history
        return envelope or LLMRequestEnvelope(
            messages=list(session.history),
            tools=[],
        )

    mock.build = build
    return mock


# ─── Exit Condition Tests ─────────────────────────────────────


class TestShouldExit:
    """Test the _should_exit() method directly."""

    def _make_loop(self):
        return SessionLoop(
            session_store=MagicMock(),
            llm_core=MagicMock(),
            context_builder=MagicMock(),
        )

    def test_empty_messages(self):
        loop = self._make_loop()
        assert loop._should_exit([]) is False

    def test_user_message_last(self):
        loop = self._make_loop()
        messages = [{"role": "user", "content": "hello"}]
        assert loop._should_exit(messages) is False

    def test_assistant_text_no_tools(self):
        loop = self._make_loop()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert loop._should_exit(messages) is True

    def test_assistant_with_tool_calls(self):
        loop = self._make_loop()
        messages = [
            {"role": "user", "content": "read file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "function": {"name": "read_file"}}],
            },
        ]
        assert loop._should_exit(messages) is False

    def test_tool_result_last(self):
        loop = self._make_loop()
        messages = [
            {"role": "user", "content": "read file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file contents"},
        ]
        assert loop._should_exit(messages) is False

    def test_system_message_last(self):
        loop = self._make_loop()
        messages = [{"role": "system", "content": "You are helpful."}]
        assert loop._should_exit(messages) is False

    def test_assistant_empty_content_no_tools(self):
        """Assistant with empty string content and no tools — should NOT exit."""
        loop = self._make_loop()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        # Empty string is falsy, so has_content is False → don't exit
        assert loop._should_exit(messages) is False


# ─── Helper Tests ─────────────────────────────────────────────


class TestExtractLastUserText:
    def _make_loop(self):
        return SessionLoop(
            session_store=MagicMock(),
            llm_core=MagicMock(),
            context_builder=MagicMock(),
        )

    def test_string_content(self):
        loop = self._make_loop()
        messages = [{"role": "user", "content": "hello world"}]
        assert loop._extract_last_user_text(messages) == "hello world"

    def test_list_content(self):
        loop = self._make_loop()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        assert loop._extract_last_user_text(messages) == "look at this"

    def test_multiple_users_returns_last(self):
        loop = self._make_loop()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert loop._extract_last_user_text(messages) == "second"

    def test_no_user_messages(self):
        loop = self._make_loop()
        messages = [{"role": "system", "content": "system prompt"}]
        assert loop._extract_last_user_text(messages) == ""

    def test_empty_messages(self):
        loop = self._make_loop()
        assert loop._extract_last_user_text([]) == ""


# ─── Integration Tests (real SessionStore + EventBus) ─────────


@pytest.mark.asyncio
async def test_simple_text_response(store: SessionStore, event_bus: EventBus):
    """User sends message → LLM responds with text → loop exits."""
    session = await store.create_session("test-session")
    await store.append_user_message("test-session", "What is 2+2?")

    llm_core = _make_mock_llm_core([_make_text_events("4")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    result = await loop.run("test-session")

    assert result == "4"

    # Session should be marked done
    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.DONE.value


@pytest.mark.asyncio
async def test_tool_call_flow(store: SessionStore, event_bus: EventBus):
    """User message → LLM calls tool → processes result → responds with text."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Read the config file")

    # First call: tool call. Second call: text response.
    llm_core = _make_mock_llm_core(
        [
            _make_tool_then_text_events(
                tool_name="read_file",
                tool_args={"path": "config.yaml"},
                final_text="",  # Text comes with tool calls but is empty
            ),
            _make_text_events("The config file contains your settings."),
        ]
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    result = await loop.run("test-session")

    assert result == "The config file contains your settings."

    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.DONE.value


@pytest.mark.asyncio
async def test_multi_tool_iterations(store: SessionStore, event_bus: EventBus):
    """LLM calls tools across multiple iterations before final response."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Find and read the file")

    llm_core = _make_mock_llm_core(
        [
            # Iteration 1: search for file
            _make_tool_then_text_events(
                tool_name="search",
                call_id="call-1",
                final_text="",
            ),
            # Iteration 2: read the file
            _make_tool_then_text_events(
                tool_name="read_file",
                call_id="call-2",
                final_text="",
            ),
            # Iteration 3: final text response
            _make_text_events("Found and read the file. Here are the contents."),
        ]
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    result = await loop.run("test-session")

    assert result == "Found and read the file. Here are the contents."


@pytest.mark.asyncio
async def test_abort_stops_loop(store: SessionStore, event_bus: EventBus):
    """Setting the abort event stops the loop."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Do something long")

    # LLM that would keep calling tools forever
    llm_core = _make_mock_llm_core(
        [
            _make_tool_then_text_events(
                tool_name="long_task", call_id="call-n", final_text=""
            )
        ]
        * 50
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    abort = asyncio.Event()

    async def abort_after_delay():
        await asyncio.sleep(0.05)
        abort.set()

    task = asyncio.create_task(abort_after_delay())
    result = await loop.run("test-session", abort=abort)
    await task

    # Loop should have stopped — result may be None since we aborted
    # Session should still be marked done (not error)
    session = await store.get_session("test-session")
    assert session is not None
    assert session.status in (SessionStatus.DONE.value, SessionStatus.CANCELED.value)


@pytest.mark.asyncio
async def test_max_iterations_limit(store: SessionStore, event_bus: EventBus):
    """Loop stops when max iterations is reached."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Keep going")

    # LLM always calls tools — never gives final text
    llm_core = _make_mock_llm_core(
        [_make_tool_then_text_events(tool_name="tool", call_id="call-n", final_text="")]
        * 10
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
        max_iterations=3,  # Low limit for testing
    )

    result = await loop.run("test-session")

    # Should have stopped at 3 iterations without a final text
    assert result is None

    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.DONE.value


@pytest.mark.asyncio
async def test_max_duration_limit(store: SessionStore, event_bus: EventBus):
    """Loop stops when max duration is exceeded."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Keep going")

    # LLM always calls tools
    llm_core = _make_mock_llm_core(
        [_make_tool_then_text_events(tool_name="tool", call_id="call-n", final_text="")]
        * 50
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
        max_iterations=50,
        max_duration=0.01,  # 10ms — will expire almost immediately
    )

    result = await loop.run("test-session")

    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.DONE.value


@pytest.mark.asyncio
async def test_empty_messages_exits(store: SessionStore, event_bus: EventBus):
    """Loop exits gracefully when session has no messages."""
    await store.create_session("test-session")
    # No messages added

    llm_core = _make_mock_llm_core([_make_text_events("should not reach")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    result = await loop.run("test-session")

    assert result is None

    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.DONE.value


@pytest.mark.asyncio
async def test_events_published_to_bus(store: SessionStore, event_bus: EventBus):
    """LLM events are forwarded to the EventBus."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    llm_core = _make_mock_llm_core([_make_text_events("Hi!")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    # Subscribe to session events
    event_queue = event_bus.subscribe("session.test-session.event")
    status_queue = event_bus.subscribe("session.test-session.status")

    result = await loop.run("test-session")

    assert result == "Hi!"

    # Collect events (the loop publishes end sentinel)
    events = [e async for e in event_bus.listen(event_queue)]
    statuses = [e async for e in event_bus.listen(status_queue)]

    # Should have text_chunk + stream_end events
    assert len(events) >= 2
    event_types = [e["type"] for e in events]
    assert LLMEventType.TEXT_CHUNK.value in event_types
    assert LLMEventType.STREAM_END.value in event_types

    # Should have busy → done status transitions
    status_values = [s["status"] for s in statuses]
    assert "busy" in status_values
    assert SessionStatus.DONE.value in status_values


@pytest.mark.asyncio
async def test_status_transitions(store: SessionStore, event_bus: EventBus):
    """Session status transitions: idle → busy → done."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    # Verify initial status
    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.IDLE.value

    llm_core = _make_mock_llm_core([_make_text_events("Hi!")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    await loop.run("test-session")

    # Final status should be done
    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.DONE.value


@pytest.mark.asyncio
async def test_error_sets_error_status(store: SessionStore, event_bus: EventBus):
    """Exceptions during the loop set session status to error."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    # LLM that raises an exception
    async def exploding_generate(envelope):
        raise RuntimeError("LLM exploded")
        yield  # Make it an async generator  # noqa: E501

    mock_llm = MagicMock()
    mock_llm.generate_with_tools = exploding_generate
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=mock_llm,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    result = await loop.run("test-session")

    assert result is None

    session = await store.get_session("test-session")
    assert session is not None
    assert session.status == SessionStatus.ERROR.value


@pytest.mark.asyncio
async def test_error_event_published(store: SessionStore, event_bus: EventBus):
    """Error events are published to the EventBus."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    async def exploding_generate(envelope):
        raise RuntimeError("boom")
        yield  # noqa: E501

    mock_llm = MagicMock()
    mock_llm.generate_with_tools = exploding_generate
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=mock_llm,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    event_queue = event_bus.subscribe("session.test-session.event")

    await loop.run("test-session")

    events = [e async for e in event_bus.listen(event_queue)]
    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 1
    assert "boom" in error_events[0]["payload"]["message"]


@pytest.mark.asyncio
async def test_no_event_bus(store: SessionStore):
    """Loop works fine without an EventBus (None)."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    llm_core = _make_mock_llm_core([_make_text_events("Hi!")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=None,  # No event bus
    )

    result = await loop.run("test-session")

    assert result == "Hi!"


@pytest.mark.asyncio
async def test_messages_persisted_after_text_response(
    store: SessionStore, event_bus: EventBus
):
    """After a text response, the assistant message is persisted."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    llm_core = _make_mock_llm_core([_make_text_events("Hi there!")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    await loop.run("test-session")

    messages = await store.get_messages_as_openai("test-session")
    assert len(messages) == 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_tool_messages_persisted(store: SessionStore, event_bus: EventBus):
    """Tool call and result messages are persisted to the store."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Read config")

    llm_core = _make_mock_llm_core(
        [
            _make_tool_then_text_events(
                tool_name="read_file",
                tool_args={"path": "config.yaml"},
                call_id="call-1",
                final_text="",
            ),
            _make_text_events("Here are the config contents."),
        ]
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    await loop.run("test-session")

    messages = await store.get_messages_as_openai("test-session")
    roles = [m["role"] for m in messages]

    # Should have: user, assistant (tool_calls), tool (result), assistant (final text)
    assert roles[0] == "user"
    assert "assistant" in roles
    assert "tool" in roles


@pytest.mark.asyncio
async def test_compaction_triggered(store: SessionStore, event_bus: EventBus):
    """When compactor says needs_compaction, compact() is called."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    llm_core = _make_mock_llm_core([_make_text_events("Hi!")])
    context_builder = _make_mock_context_builder()

    # Compactor that says compaction needed on first check, then not
    compactor = AsyncMock(spec=SessionCompactor)
    call_count = 0

    async def needs_compaction_side_effect(session_id):
        nonlocal call_count
        call_count += 1
        return call_count == 1  # Only first call returns True

    compactor.needs_compaction = AsyncMock(side_effect=needs_compaction_side_effect)
    compactor.compact = AsyncMock(return_value=True)

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
        compactor=compactor,
    )

    result = await loop.run("test-session")

    # compact() should have been called once
    compactor.compact.assert_awaited_once_with("test-session")
    # Loop should still complete
    assert result == "Hi!"


@pytest.mark.asyncio
async def test_tool_call_with_final_text(store: SessionStore, event_bus: EventBus):
    """When tool calls AND text come in the same iteration, both are persisted."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Read and summarize")

    # Tool call with accompanying text in same response
    llm_core = _make_mock_llm_core(
        [
            _make_tool_then_text_events(
                tool_name="read_file",
                call_id="call-1",
                final_text="I read the file and here's the summary.",
            ),
        ]
    )
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    result = await loop.run("test-session")

    assert result == "I read the file and here's the summary."


@pytest.mark.asyncio
async def test_session_end_sentinel_published(store: SessionStore, event_bus: EventBus):
    """The loop publishes an end sentinel to the event topic when done."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    llm_core = _make_mock_llm_core([_make_text_events("Hi!")])
    context_builder = _make_mock_context_builder()

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    event_queue = event_bus.subscribe("session.test-session.event")

    await loop.run("test-session")

    # listen() should terminate (end sentinel was published)
    events = [e async for e in event_bus.listen(event_queue)]
    # If we got here, the end sentinel worked
    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_enabled_plugins_passed_through(store: SessionStore, event_bus: EventBus):
    """enabled_plugins parameter is passed to context builder."""
    await store.create_session("test-session")
    await store.append_user_message("test-session", "Hello")

    llm_core = _make_mock_llm_core([_make_text_events("Hi!")])

    # Track what plugins were passed
    received_plugins = []

    async def tracking_build(user_message, session, enabled_plugins=None):
        received_plugins.append(enabled_plugins)
        return LLMRequestEnvelope(messages=list(session.history), tools=[])

    context_builder = AsyncMock()
    context_builder.build = tracking_build

    loop = SessionLoop(
        session_store=store,
        llm_core=llm_core,
        context_builder=context_builder,
        event_bus=event_bus,
    )

    await loop.run("test-session", enabled_plugins=["gmail", "calendar"])

    assert len(received_plugins) >= 1
    assert received_plugins[0] == ["gmail", "calendar"]
