"""
Check Task tool — inspect the Task Ledger for task status.

Reads directly from the persistent SQLite-backed Task Ledger.
Can query any task type (sub-agents, memory extractions, scheduled
tasks, etc.) — not just sub-agents.

The tool name stays 'check_task' for backward compatibility with
existing LLM prompts and tool schemas.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aether.session.models import TaskStatus
from aether.tools.base import AetherTool, ToolParam, ToolResult

if TYPE_CHECKING:
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

    def __init__(self, task_ledger: "TaskLedger") -> None:
        self._ledger = task_ledger

    async def execute(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
    ) -> ToolResult:
        try:
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

        except Exception as e:
            return ToolResult.fail(f"Failed to check task: {e}")
