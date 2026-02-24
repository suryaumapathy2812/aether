from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aether.voice.backends.gemini.bridge import DelegationBridge


@pytest.mark.asyncio
async def test_run_delegated_session_marks_ledger_complete() -> None:
    agent_core = AsyncMock()
    agent_core.run_session = AsyncMock(return_value="done")
    task_ledger = AsyncMock()

    bridge = DelegationBridge(
        agent_core=agent_core,
        task_ledger=task_ledger,
        voice_session_id="voice-1",
    )

    await bridge._run_delegated_session(
        ledger_task_id="task-1",
        delegation_session_id="deleg-1",
        prompt="do it",
    )

    task_ledger.set_running.assert_awaited_once_with("task-1")
    task_ledger.set_complete.assert_awaited_once()
    task_ledger.set_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_delegated_session_marks_ledger_error_on_failure() -> None:
    agent_core = AsyncMock()
    agent_core.run_session = AsyncMock(side_effect=RuntimeError("boom"))
    task_ledger = AsyncMock()

    bridge = DelegationBridge(
        agent_core=agent_core,
        task_ledger=task_ledger,
        voice_session_id="voice-1",
    )

    await bridge._run_delegated_session(
        ledger_task_id="task-2",
        delegation_session_id="deleg-2",
        prompt="do it",
    )

    task_ledger.set_running.assert_awaited_once_with("task-2")
    task_ledger.set_complete.assert_not_awaited()
    task_ledger.set_error.assert_awaited_once()
