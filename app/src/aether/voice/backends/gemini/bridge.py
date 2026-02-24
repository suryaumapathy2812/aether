"""Delegation bridge from Gemini P-worker to AgentCore E-worker."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from aether.session.models import TaskType

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.session.ledger import TaskLedger

logger = logging.getLogger(__name__)


class DelegationBridge:
    """Handles `delegate_to_agent` as a non-blocking delegation tool."""

    DESCRIPTION = (
        "Delegate complex tasks to the autonomous reasoning agent. "
        "Use when work requires planning, multiple tool calls, or longer execution."
    )

    PARAMETERS = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Task description for the delegated agent",
            },
            "context": {
                "type": "string",
                "description": "Optional supporting context from the conversation",
            },
        },
        "required": ["task"],
    }

    def __init__(
        self,
        *,
        agent_core: "AgentCore",
        task_ledger: "TaskLedger",
        voice_session_id: str = "voice",
    ) -> None:
        self._agent_core = agent_core
        self._task_ledger = task_ledger
        self._voice_session_id = voice_session_id
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def delegate_to_agent(self, args: dict[str, Any]) -> str:
        task_text = str(args.get("task", "")).strip()
        context = str(args.get("context", "")).strip()

        if not task_text:
            return json.dumps({"ok": False, "error": "Missing required field: task"})

        delegation_session_id = f"delegation-{uuid.uuid4().hex[:8]}"
        prompt = task_text if not context else f"{task_text}\n\nContext:\n{context}"

        ledger_task_id = await self._task_ledger.submit(
            session_id=self._voice_session_id,
            task_type=TaskType.SUB_AGENT.value,
            payload={
                "source": "gemini_delegate_to_agent",
                "delegation_session_id": delegation_session_id,
                "task": task_text,
                "context": context,
            },
            priority="high",
        )

        background_task = asyncio.create_task(
            self._run_delegated_session(
                ledger_task_id=ledger_task_id,
                delegation_session_id=delegation_session_id,
                prompt=prompt,
            ),
            name=f"delegate-to-agent-{ledger_task_id}",
        )
        self._background_tasks.add(background_task)
        background_task.add_done_callback(self._background_tasks.discard)

        return json.dumps(
            {
                "ok": True,
                "status": "accepted",
                "task_id": ledger_task_id,
                "delegation_session_id": delegation_session_id,
                "message": "Delegation accepted and running in background.",
            }
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return await self.delegate_to_agent(args)

    async def _run_delegated_session(
        self,
        *,
        ledger_task_id: str,
        delegation_session_id: str,
        prompt: str,
    ) -> None:
        try:
            await self._task_ledger.set_running(ledger_task_id)
            result = await self._agent_core.run_session(
                session_id=delegation_session_id,
                user_message=prompt,
                background=False,
            )
            await self._task_ledger.set_complete(
                ledger_task_id,
                {
                    "delegation_session_id": delegation_session_id,
                    "result": result or "",
                },
            )
        except Exception as exc:
            logger.exception("Delegated session failed: %s", delegation_session_id)
            try:
                await self._task_ledger.set_error(ledger_task_id, str(exc))
            except Exception:
                logger.debug("Failed to mark delegated task as errored", exc_info=True)
