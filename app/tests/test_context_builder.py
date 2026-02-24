"""Tests for ContextBuilder — decision injection and memory retrieval.

Covers:
- _retrieve_decisions: active decisions formatted as system section, None when empty
- _retrieve_memory: all memory types (fact, memory, decision, action, session) formatted
- Edge cases: no memory store, empty results
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aether.llm.context_builder import ContextBuilder


class TestRetrieveDecisions:
    """Test ContextBuilder._retrieve_decisions."""

    @pytest.mark.asyncio
    async def test_decisions_injected_as_system_message(self):
        """Positive: active decisions are formatted as a 'learned rules' section.

        Objective: verify that decisions returned by memory_store.get_decisions
        are formatted into a prompt section with the correct header and bullet
        points, ready for injection as a system message.
        """
        memory_store = AsyncMock()
        memory_store.get_decisions = AsyncMock(
            return_value=[
                {
                    "id": 1,
                    "decision": "Don't notify after 9pm",
                    "category": "notification",
                    "source": "extracted",
                    "active": True,
                    "confidence": 1.0,
                    "created_at": 0,
                    "updated_at": 0,
                },
            ]
        )

        builder = ContextBuilder(memory_store=memory_store)
        section = await builder._retrieve_decisions()

        assert section is not None
        assert "Don't notify after 9pm" in section
        assert "learned rules" in section.lower()

    @pytest.mark.asyncio
    async def test_multiple_decisions_all_included(self):
        """Positive: all active decisions appear as bullet points.

        Objective: verify that every decision is included, not just the first.
        """
        memory_store = AsyncMock()
        memory_store.get_decisions = AsyncMock(
            return_value=[
                {"decision": "Prefer bullet points"},
                {"decision": "Use casual tone"},
                {"decision": "No emojis in code"},
            ]
        )

        builder = ContextBuilder(memory_store=memory_store)
        section = await builder._retrieve_decisions()

        assert section is not None
        assert "- Prefer bullet points" in section
        assert "- Use casual tone" in section
        assert "- No emojis in code" in section

    @pytest.mark.asyncio
    async def test_no_decisions_returns_none(self):
        """Negative: no active decisions returns None (section omitted from prompt).

        Objective: verify that an empty decision list results in None so the
        section is cleanly omitted rather than injecting an empty block.
        """
        memory_store = AsyncMock()
        memory_store.get_decisions = AsyncMock(return_value=[])

        builder = ContextBuilder(memory_store=memory_store)
        section = await builder._retrieve_decisions()
        assert section is None

    @pytest.mark.asyncio
    async def test_no_memory_store_decisions_returns_none(self):
        """Negative: no memory store at all returns None.

        Objective: verify graceful handling when memory_store is None.
        """
        builder = ContextBuilder(memory_store=None)
        section = await builder._retrieve_decisions()
        assert section is None

    @pytest.mark.asyncio
    async def test_decisions_retrieval_error_returns_none(self):
        """Negative: exception during get_decisions returns None (logged, not raised).

        Objective: verify that a database error doesn't crash the context builder.
        """
        memory_store = AsyncMock()
        memory_store.get_decisions = AsyncMock(side_effect=Exception("DB timeout"))

        builder = ContextBuilder(memory_store=memory_store)
        section = await builder._retrieve_decisions()
        assert section is None


class TestRetrieveMemory:
    """Test ContextBuilder._retrieve_memory."""

    @pytest.mark.asyncio
    async def test_memory_retrieval_handles_all_types(self):
        """Positive: all memory types are correctly formatted.

        Objective: verify that fact, memory, decision, action, and session
        results are each formatted with their correct prefix and field key.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "fact",
                    "fact": "User's name is Surya",
                    "similarity": 0.9,
                    "timestamp": 0,
                },
                {
                    "type": "memory",
                    "memory": "User was stressed about deadline",
                    "category": "episodic",
                    "similarity": 0.8,
                    "timestamp": 0,
                },
                {
                    "type": "decision",
                    "decision": "Prefer bullet points",
                    "category": "preference",
                    "source": "extracted",
                    "similarity": 0.85,
                    "timestamp": 0,
                },
                {
                    "type": "action",
                    "tool_name": "web_search",
                    "arguments": "{}",
                    "output": "results...",
                    "similarity": 0.7,
                    "timestamp": 0,
                },
                {
                    "type": "session",
                    "summary": "Discussed project plan",
                    "similarity": 0.6,
                    "timestamp": 0,
                },
            ]
        )

        builder = ContextBuilder(memory_store=memory_store)

        # Patch config to avoid loading real config
        with patch("aether.llm.context_builder.config_module") as mock_config:
            mock_config.config.memory.search_limit = 10
            result = await builder._retrieve_memory("test query")

        assert result is not None
        assert "[Known fact]" in result
        assert "User's name is Surya" in result
        assert "[Memory (episodic)]" in result
        assert "User was stressed about deadline" in result
        assert "[Decision (preference)]" in result
        assert "Prefer bullet points" in result
        assert "web_search" in result
        assert "[Previous session]" in result
        assert "Discussed project plan" in result

    @pytest.mark.asyncio
    async def test_conversation_type_formatted(self):
        """Positive: conversation type uses user_message and assistant_message.

        Objective: verify the conversation memory type is formatted correctly.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(
            return_value=[
                {
                    "type": "conversation",
                    "user_message": "What's the weather?",
                    "assistant_message": "It's sunny today.",
                    "similarity": 0.7,
                    "timestamp": 0,
                },
            ]
        )

        builder = ContextBuilder(memory_store=memory_store)

        with patch("aether.llm.context_builder.config_module") as mock_config:
            mock_config.config.memory.search_limit = 10
            result = await builder._retrieve_memory("weather")

        assert result is not None
        assert "[Previous conversation]" in result
        assert "What's the weather?" in result
        assert "It's sunny today." in result

    @pytest.mark.asyncio
    async def test_empty_search_results_returns_none(self):
        """Negative: no search results returns None.

        Objective: verify that empty results don't produce an empty string.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(return_value=[])

        builder = ContextBuilder(memory_store=memory_store)

        with patch("aether.llm.context_builder.config_module") as mock_config:
            mock_config.config.memory.search_limit = 10
            result = await builder._retrieve_memory("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_memory_store_raises_assertion(self):
        """Negative: calling _retrieve_memory with no memory store raises AssertionError.

        Objective: verify that _retrieve_memory enforces the precondition
        that memory_store is not None. The public build() method guards
        against this by checking before calling _retrieve_memory.
        """
        builder = ContextBuilder(memory_store=None)
        with pytest.raises(AssertionError):
            await builder._retrieve_memory("test")

    @pytest.mark.asyncio
    async def test_memory_search_error_returns_none(self):
        """Negative: exception during search returns None (logged, not raised).

        Objective: verify that a search failure doesn't crash the builder.
        """
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(side_effect=Exception("Embedding service down"))

        builder = ContextBuilder(memory_store=memory_store)

        with patch("aether.llm.context_builder.config_module") as mock_config:
            mock_config.config.memory.search_limit = 10
            result = await builder._retrieve_memory("test")

        assert result is None
