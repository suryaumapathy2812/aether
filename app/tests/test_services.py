"""
Tests for service layer: ReplyService, MemoryService, NotificationService, ToolService.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aether.llm.contracts import (
    LLMEventEnvelope,
    LLMEventType,
    LLMRequestEnvelope,
    ToolResult,
)
from aether.services.memory_service import MemoryService
from aether.services.notification_service import NotificationService
from aether.services.reply_service import ReplyService
from aether.services.tool_service import ToolService


# ─── Helpers ─────────────────────────────────────────────────────


def _make_llm_core(*events: LLMEventEnvelope) -> AsyncMock:
    """Create a mock LLMCore that yields the given events."""
    llm_core = AsyncMock()

    async def fake_generate_with_tools(envelope):
        for event in events:
            yield event

    llm_core.generate_with_tools = fake_generate_with_tools
    return llm_core


def _text_chunk(text: str, seq: int = 1) -> LLMEventEnvelope:
    return LLMEventEnvelope.text_chunk(
        request_id="req-1", job_id="job-1", text=text, sequence=seq
    )


def _stream_end(seq: int = 2) -> LLMEventEnvelope:
    return LLMEventEnvelope.stream_end(request_id="req-1", job_id="job-1", sequence=seq)


# ─── ReplyService ────────────────────────────────────────────────


class TestReplyService:
    """Test ReplyService response generation."""

    @pytest.mark.asyncio
    async def test_generate_reply_yields_events(self):
        """ReplyService yields LLM events and stores to memory."""
        llm_core = _make_llm_core(
            _text_chunk("Hello there!"),
            _stream_end(),
        )
        context_builder = AsyncMock()
        context_builder.build = AsyncMock(
            return_value=LLMRequestEnvelope(kind="reply_text", modality="text")
        )
        memory_store = AsyncMock()
        memory_store.add = AsyncMock()

        service = ReplyService(llm_core, context_builder, memory_store)

        from aether.llm.context_builder import SessionState

        session = SessionState(session_id="s1", user_id="u1", mode="text", history=[])

        events = []
        async for event in service.generate_reply("Hi", session):
            events.append(event)

        assert len(events) == 2
        assert events[0].event_type == LLMEventType.TEXT_CHUNK.value
        assert events[0].payload["text"] == "Hello there!"
        assert events[1].event_type == LLMEventType.STREAM_END.value

        # Memory should be stored
        memory_store.add.assert_called_once_with("Hi", "Hello there!")

    @pytest.mark.asyncio
    async def test_generate_reply_empty_response_no_memory_store(self):
        """ReplyService doesn't store empty responses to memory."""
        llm_core = _make_llm_core(_stream_end(seq=1))
        context_builder = AsyncMock()
        context_builder.build = AsyncMock(
            return_value=LLMRequestEnvelope(kind="reply_text", modality="text")
        )
        memory_store = AsyncMock()

        service = ReplyService(llm_core, context_builder, memory_store)

        from aether.llm.context_builder import SessionState

        session = SessionState(session_id="s1", user_id="u1", mode="text", history=[])

        events = []
        async for event in service.generate_reply("Hi", session):
            events.append(event)

        assert len(events) == 1
        memory_store.add.assert_not_called()


# ─── MemoryService ───────────────────────────────────────────────


class TestMemoryService:
    """Test MemoryService background operations."""

    @pytest.mark.asyncio
    async def test_extract_facts_parses_json(self):
        """MemoryService extracts facts from LLM response."""
        llm_core = _make_llm_core(
            _text_chunk('["User likes Python", "User lives in SF"]'),
            _stream_end(),
        )
        memory_store = AsyncMock()
        memory_store._store_fact = AsyncMock()

        service = MemoryService(llm_core, memory_store)
        facts = await service.extract_facts("I love Python", "That's great!")

        assert len(facts) == 2
        assert "User likes Python" in facts
        assert "User lives in SF" in facts
        assert memory_store._store_fact.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_facts_empty_response(self):
        """MemoryService handles empty fact extraction."""
        llm_core = _make_llm_core(
            _text_chunk("[]"),
            _stream_end(),
        )
        memory_store = AsyncMock()

        service = MemoryService(llm_core, memory_store)
        facts = await service.extract_facts("Hello", "Hi there")

        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_invalid_json(self):
        """MemoryService handles invalid JSON gracefully."""
        llm_core = _make_llm_core(
            _text_chunk("not valid json"),
            _stream_end(),
        )
        memory_store = AsyncMock()

        service = MemoryService(llm_core, memory_store)
        facts = await service.extract_facts("Hello", "Hi there")

        assert facts == []

    @pytest.mark.asyncio
    async def test_summarize_session(self):
        """MemoryService summarizes a session."""
        llm_core = _make_llm_core(
            _text_chunk("User asked about Python and created a project."),
            _stream_end(),
        )
        memory_store = AsyncMock()
        memory_store.add_session_summary = AsyncMock()

        service = MemoryService(llm_core, memory_store)
        summary = await service.summarize_session(
            session_id="s1",
            conversation_history=[
                {"role": "user", "content": "Help me with Python"},
                {"role": "assistant", "content": "Sure, let me help."},
            ],
            started_at=1000.0,
            turn_count=1,
        )

        assert "Python" in summary
        memory_store.add_session_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_session_empty_history(self):
        """MemoryService returns empty for empty history."""
        llm_core = _make_llm_core()
        memory_store = AsyncMock()

        service = MemoryService(llm_core, memory_store)
        summary = await service.summarize_session(
            session_id="s1",
            conversation_history=[],
            started_at=1000.0,
            turn_count=0,
        )

        assert summary == ""

    @pytest.mark.asyncio
    async def test_compact_actions(self):
        """MemoryService delegates compaction to MemoryStore."""
        llm_core = AsyncMock()
        memory_store = AsyncMock()
        memory_store._compact_old_actions = AsyncMock()

        service = MemoryService(llm_core, memory_store)
        await service.compact_actions()

        memory_store._compact_old_actions.assert_called_once()

    def test_parse_facts_with_markdown_wrapper(self):
        """MemoryService parses facts wrapped in markdown."""
        llm_core = AsyncMock()
        memory_store = AsyncMock()
        service = MemoryService(llm_core, memory_store)

        facts = service._parse_facts('```json\n["fact one", "fact two"]\n```')
        assert len(facts) == 2


# ─── NotificationService ────────────────────────────────────────


class TestNotificationService:
    """Test NotificationService event classification."""

    @pytest.mark.asyncio
    async def test_classify_returns_valid_decision(self):
        """NotificationService classifies events correctly."""
        llm_core = _make_llm_core(
            _text_chunk("surface"),
            _stream_end(),
        )
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(return_value=[])

        service = NotificationService(llm_core, memory_store)

        # Create a mock PluginEvent
        event = MagicMock()
        event.plugin = "gmail"
        event.event_type = "new_email"
        event.sender = {"name": "Mom", "email": "mom@example.com"}
        event.summary = "Mom sent you an email about dinner"
        event.urgency = "high"
        event.requires_action = False

        decision = await service.process_event(event)

        assert decision.action == "surface"
        assert decision.notification  # Should have notification text
        assert decision.event is event

    @pytest.mark.asyncio
    async def test_classify_fallback_on_invalid_response(self):
        """NotificationService falls back to 'archive' on invalid LLM response."""
        llm_core = _make_llm_core(
            _text_chunk("maybe_surface_or_not"),
            _stream_end(),
        )
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(return_value=[])

        service = NotificationService(llm_core, memory_store)

        event = MagicMock()
        event.plugin = "slack"
        event.event_type = "message"
        event.sender = {"name": "Bot"}
        event.summary = "Daily standup reminder"
        event.urgency = "low"
        event.requires_action = False

        decision = await service.process_event(event)
        assert decision.action == "archive"


# ─── ToolService ─────────────────────────────────────────────────


class TestToolService:
    """Test ToolService execution coordination."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """ToolService executes a tool successfully."""
        orchestrator = AsyncMock()
        orchestrator.execute = AsyncMock(
            return_value=ToolResult.success(
                tool_name="read_file",
                output="file contents",
                call_id="call-1",
            )
        )

        service = ToolService(orchestrator)
        result = await service.execute(
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            call_id="call-1",
        )

        assert result.tool_name == "read_file"
        assert result.output == "file contents"
        assert result.error is False

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """ToolService handles tool execution failure."""
        orchestrator = AsyncMock()
        orchestrator.execute = AsyncMock(side_effect=RuntimeError("Tool crashed"))

        service = ToolService(orchestrator)
        result = await service.execute(
            tool_name="bad_tool",
            arguments={},
            call_id="call-2",
        )

        assert result.error is True
        assert "Tool crashed" in result.output

    @pytest.mark.asyncio
    async def test_execute_batch(self):
        """ToolService executes a batch of tools."""
        orchestrator = AsyncMock()
        orchestrator.execute = AsyncMock(
            return_value=ToolResult.success(tool_name="test", output="ok", call_id="c")
        )

        service = ToolService(orchestrator)
        results = await service.execute_batch(
            [
                {"tool_name": "tool_a", "arguments": {}, "call_id": "c1"},
                {"tool_name": "tool_b", "arguments": {}, "call_id": "c2"},
            ]
        )

        assert len(results) == 2
        assert all(not r.error for r in results)
