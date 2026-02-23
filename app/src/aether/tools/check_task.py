"""
Check Task tool — inspect the Task Ledger and sub-agent status.

When a TaskLedger is available, this tool reads directly from the
persistent SQLite-backed ledger. It can query any task type (sub-agents,
memory extractions, scheduled tasks, etc.) — not just sub-agents.

When no TaskLedger is available, falls back to SubAgentManager for
backward compatibility.

The tool name stays 'check_task' for backward compatibility with
existing LLM prompts and tool schemas.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aether.session.models import SessionStatus, TaskStatus
from aether.tools.base import AetherTool, ToolParam, ToolResult

if TYPE_CHECKING:
    from aether.agents.manager import SubAgentManager
    from aether.session.ledger import TaskLedger


class CheckTaskTool(AetherTool):
    name = "check_task"
    description = (
        "Check the status of background tasks. Can query a specific task by ID, "
        "list all tasks for a session, or filter by status (pending/running/complete/error). "
        "Use the task_id returned by spawn_task, or omit it to list recent tasks."
    )
    status_text = "Checking task status..."
    parameters = [
        ToolParam(
            name="task_id",
            type="string",
            description="The task_id returned by spawn_task. Omit to list tasks.",
            required=False,
        ),
        ToolParam(
            name="session_id",
            type="string",
            description="Filter tasks by session ID. Omit to show all.",
            required=False,
        ),
        ToolParam(
            name="status",
            type="string",
            description="Filter by status: pending, running, complete, error",
            required=False,
            enum=["pending", "running", "complete", "error"],
        ),
    ]

    def __init__(
        self,
        sub_agent_manager: "SubAgentManager",
        task_ledger: "TaskLedger | None" = None,
    ):
        self._manager = sub_agent_manager
        self._ledger = task_ledger

    async def execute(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
    ) -> ToolResult:
        try:
            # If we have a TaskLedger, use it for all queries
            if self._ledger is not None:
                return await self._query_ledger(task_id, session_id, status)

            # Fallback: use SubAgentManager directly (backward compat)
            return await self._query_manager(task_id)

        except Exception as e:
            return ToolResult.fail(f"Failed to check task: {e}")

    async def _query_ledger(
        self,
        task_id: str | None,
        session_id: str | None,
        status: str | None,
    ) -> ToolResult:
        """Query the Task Ledger (persistent, all task types)."""
        assert self._ledger is not None

        # Single task lookup
        if task_id is not None:
            task = await self._ledger.get_task(task_id)
            if task is None:
                return ToolResult.fail(f"Task {task_id} not found")

            lines = [
                f"Task: {task.task_id}",
                f"Type: {task.type}",
                f"Status: {task.status}",
                f"Session: {task.session_id}",
                f"Priority: {task.priority}",
            ]

            if task.status == TaskStatus.COMPLETE.value and task.result:
                result_str = task.result.get("result", "(no output)")
                lines.append(f"Result: {result_str}")
            elif task.status == TaskStatus.ERROR.value and task.error:
                lines.append(f"Error: {task.error}")
            elif task.status == TaskStatus.RUNNING.value:
                lines.append("(still running)")
            elif task.status == TaskStatus.PENDING.value:
                lines.append("(waiting to start)")

            return ToolResult.success("\n".join(lines))

        # List tasks with filters
        tasks = await self._ledger.get_tasks(
            session_id=session_id,
            status=status,
            limit=20,
        )

        if not tasks:
            filters = []
            if session_id:
                filters.append(f"session={session_id}")
            if status:
                filters.append(f"status={status}")
            filter_str = f" (filters: {', '.join(filters)})" if filters else ""
            return ToolResult.success(f"No tasks found{filter_str}")

        lines = [f"Found {len(tasks)} task(s):"]
        for t in tasks:
            summary = f"  [{t.status}] {t.task_id} — {t.type}"
            if t.status == TaskStatus.COMPLETE.value and t.result:
                preview = str(t.result.get("result", ""))[:100]
                summary += f" → {preview}"
            elif t.status == TaskStatus.ERROR.value and t.error:
                summary += f" ✗ {t.error[:80]}"
            lines.append(summary)

        return ToolResult.success("\n".join(lines))

    async def _query_manager(self, task_id: str | None) -> ToolResult:
        """Fallback: query SubAgentManager directly (no ledger)."""
        if task_id is None:
            return ToolResult.fail(
                "task_id is required when Task Ledger is not available"
            )

        status = await self._manager.get_status(task_id)

        if status["status"] == "not_found":
            return ToolResult.fail(f"Task {task_id} not found")

        if status["status"] == SessionStatus.DONE.value:
            result = await self._manager.get_result(task_id)
            return ToolResult.success(
                f"Task {task_id} completed.\nResult: {result or '(no output)'}"
            )

        if status["status"] == SessionStatus.ERROR.value:
            result = await self._manager.get_result(task_id)
            return ToolResult.fail(
                f"Task {task_id} failed.\nLast output: {result or '(no output)'}"
            )

        if status.get("running"):
            return ToolResult.success(
                f"Task {task_id} is still running "
                f"(status: {status['status']}, type: {status['agent_type']})"
            )

        return ToolResult.success(f"Task {task_id} status: {status['status']}")
