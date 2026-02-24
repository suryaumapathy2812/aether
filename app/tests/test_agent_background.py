from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from aether.agent import AgentCore
from aether.kernel.contracts import KernelEvent


@pytest.mark.asyncio
async def test_ensure_system_session_creates_once() -> None:
    scheduler = AsyncMock()
    memory_store = AsyncMock()
    llm_provider = AsyncMock()
    tool_registry = AsyncMock()
    skill_loader = AsyncMock()
    plugin_context = AsyncMock()
    session_store = AsyncMock()
    task_ledger = AsyncMock()

    agent = AgentCore(
        scheduler=scheduler,
        memory_store=memory_store,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        skill_loader=skill_loader,
        plugin_context=plugin_context,
        session_store=session_store,
        task_ledger=task_ledger,
    )

    await agent._ensure_system_session()
    await agent._ensure_system_session()

    session_store.create_session.assert_awaited_once_with(
        session_id="system",
        agent_type="system",
        metadata={"kind": "internal"},
    )


@pytest.mark.asyncio
async def test_generate_text_reply_session_requires_p_worker_handler() -> None:
    scheduler = AsyncMock()
    memory_store = AsyncMock()
    llm_provider = AsyncMock()
    tool_registry = AsyncMock()
    skill_loader = AsyncMock()
    plugin_context = AsyncMock()
    session_store = AsyncMock()
    task_ledger = AsyncMock()

    agent = AgentCore(
        scheduler=scheduler,
        memory_store=memory_store,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        skill_loader=skill_loader,
        plugin_context=plugin_context,
        session_store=session_store,
        task_ledger=task_ledger,
    )

    stream = agent.generate_text_reply_session(text="hi", session_id="s-1")
    with pytest.raises(RuntimeError, match="P-worker text handler unavailable"):
        await anext(stream)


@pytest.mark.asyncio
async def test_generate_text_reply_session_uses_p_worker_handler() -> None:
    scheduler = AsyncMock()
    memory_store = AsyncMock()
    llm_provider = AsyncMock()
    tool_registry = AsyncMock()
    skill_loader = AsyncMock()
    plugin_context = AsyncMock()
    session_store = AsyncMock()
    task_ledger = AsyncMock()

    agent = AgentCore(
        scheduler=scheduler,
        memory_store=memory_store,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        skill_loader=skill_loader,
        plugin_context=plugin_context,
        session_store=session_store,
        task_ledger=task_ledger,
    )

    async def _handler(
        text: str,
        session_id: str,
        history: list[dict] | None,
        vision: dict | None,
    ) -> AsyncGenerator[KernelEvent, None]:
        assert text == "hello"
        assert session_id == "s-2"
        assert history is None
        assert vision is None
        yield KernelEvent.text_chunk("job-1", "ok", 1)

    agent.set_text_reply_handler(_handler)

    chunks = []
    async for event in agent.generate_text_reply_session(
        text="hello", session_id="s-2"
    ):
        chunks.append(event.payload.get("text", ""))

    assert chunks == ["ok"]
