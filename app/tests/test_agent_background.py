from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aether.agent import AgentCore


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
