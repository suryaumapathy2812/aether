"""Tests for notification preference lookup — verifies the r['content'] bug fix.

Both NotificationService._get_preferences and EventProcessor._get_preferences
must use the correct field keys (r['fact'], r['memory'], r['decision'],
r['tool_name'], r['summary']) instead of the old r['content'] which would
raise KeyError.

Covers:
- NotificationService: fact, memory, decision, action, session types
- EventProcessor: fact, memory, decision, action, session types
- Edge case: empty results, exception handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from aether.plugins.event import PluginEvent
from aether.services.notification_service import NotificationService
from aether.processors.event import EventProcessor


# ─── Helpers ─────────────────────────────────────────────────────


def _make_plugin_event() -> PluginEvent:
    """Create a real PluginEvent for testing."""
    return PluginEvent(
        plugin="gmail",
        event_type="new_email",
        source_id="msg-123",
        summary="Meeting tomorrow",
        sender={"name": "Alice", "email": "alice@example.com"},
        urgency="medium",
        requires_action=False,
    )


# ─── NotificationService._get_preferences ───────────────────────


class TestNotificationServicePreferences:
    """Test NotificationService._get_preferences with all memory types."""

    @pytest.mark.asyncio
    async def test_get_preferences_handles_fact_type(self):
        """Positive: fact results use r['fact'] not r['content'].

        Objective: verify that the bug fix correctly reads the 'fact' key
        from fact-type search results.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "fact",
                    "fact": "User ignores newsletters",
                    "similarity": 0.9,
                    "timestamp": 0,
                },
            ]
        )

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "User ignores newsletters" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_memory_type(self):
        """Positive: memory results use r['memory'].

        Objective: verify that memory-type results are read correctly.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "memory",
                    "memory": "User was annoyed by spam",
                    "category": "behavioral",
                    "similarity": 0.8,
                    "timestamp": 0,
                },
            ]
        )

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "User was annoyed by spam" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_decision_type(self):
        """Positive: decision results use r['decision'].

        Objective: verify that decision-type results are formatted with
        the 'Rule:' prefix.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "decision",
                    "decision": "Don't notify about newsletters",
                    "category": "notification",
                    "similarity": 0.95,
                    "timestamp": 0,
                },
            ]
        )

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "Don't notify about newsletters" in prefs
        assert "Rule:" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_action_type(self):
        """Positive: action results use r['tool_name'].

        Objective: verify that action-type results include the tool name.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "action",
                    "tool_name": "archive_email",
                    "arguments": "{}",
                    "output": "done",
                    "similarity": 0.7,
                    "timestamp": 0,
                },
            ]
        )

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "archive_email" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_session_type(self):
        """Positive: session results use r['summary'].

        Objective: verify that session-type results include the summary.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "session",
                    "summary": "Discussed email filtering rules",
                    "similarity": 0.6,
                    "timestamp": 0,
                },
            ]
        )

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "Discussed email filtering rules" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_no_results(self):
        """Negative: no search results returns default message.

        Objective: verify graceful handling when memory has no matches.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(return_value=[])

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "No specific preferences found" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_search_exception(self):
        """Negative: search exception returns default message.

        Objective: verify that a database error doesn't crash the service.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(side_effect=Exception("DB down"))

        llm_core = AsyncMock()
        service = NotificationService(llm_core, memory_store)
        event = _make_plugin_event()

        prefs = await service._get_preferences(event)
        assert "No specific preferences found" in prefs


# ─── EventProcessor._get_preferences ────────────────────────────


class TestEventProcessorPreferences:
    """Test EventProcessor._get_preferences with all memory types."""

    @pytest.mark.asyncio
    async def test_get_preferences_handles_fact_type(self):
        """Positive: EventProcessor uses r['fact'] for fact-type results.

        Objective: verify the same bug fix applies to EventProcessor.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "fact",
                    "fact": "User prefers quiet hours after 10pm",
                    "similarity": 0.9,
                    "timestamp": 0,
                },
            ]
        )

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "User prefers quiet hours after 10pm" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_decision_type(self):
        """Positive: EventProcessor handles decision type correctly.

        Objective: verify decision-type results use r['decision'] with
        'Rule:' prefix.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "decision",
                    "decision": "Mute Slack after 9pm",
                    "category": "notification",
                    "similarity": 0.9,
                    "timestamp": 0,
                },
            ]
        )

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "Mute Slack after 9pm" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_memory_type(self):
        """Positive: EventProcessor handles memory type correctly.

        Objective: verify memory-type results use r['memory'].
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "memory",
                    "memory": "User gets frustrated by marketing emails",
                    "category": "behavioral",
                    "similarity": 0.85,
                    "timestamp": 0,
                },
            ]
        )

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "User gets frustrated by marketing emails" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_action_type(self):
        """Positive: EventProcessor handles action type correctly.

        Objective: verify action-type results use r['tool_name'].
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "action",
                    "tool_name": "mute_thread",
                    "arguments": "{}",
                    "output": "muted",
                    "similarity": 0.7,
                    "timestamp": 0,
                },
            ]
        )

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "mute_thread" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_handles_session_type(self):
        """Positive: EventProcessor handles session type correctly.

        Objective: verify session-type results use r['summary'].
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "session",
                    "summary": "Set up notification filters",
                    "similarity": 0.6,
                    "timestamp": 0,
                },
            ]
        )

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "Set up notification filters" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_mixed_types(self):
        """Positive: multiple result types are all included.

        Objective: verify that a mix of types all appear in the output.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "fact",
                    "fact": "User is a morning person",
                    "similarity": 0.9,
                    "timestamp": 0,
                },
                {
                    "type": "decision",
                    "decision": "No notifications before 7am",
                    "category": "notification",
                    "similarity": 0.85,
                    "timestamp": 0,
                },
            ]
        )

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "User is a morning person" in prefs
        assert "No notifications before 7am" in prefs

    @pytest.mark.asyncio
    async def test_get_preferences_search_exception(self):
        """Negative: search exception returns default message.

        Objective: verify that EventProcessor handles errors gracefully.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(side_effect=Exception("Connection lost"))

        llm_provider = AsyncMock()
        processor = EventProcessor(llm_provider, memory_store)
        event = _make_plugin_event()

        prefs = await processor._get_preferences(event)
        assert "No specific preferences found" in prefs
