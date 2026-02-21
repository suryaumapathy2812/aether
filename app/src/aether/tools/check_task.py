"""
Check Task tool — monitor sub-agent status and retrieve results.

Used after spawn_task to check if a background task is done
and retrieve its output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aether.session.models import SessionStatus
from aether.tools.base import AetherTool, ToolParam, ToolResult

if TYPE_CHECKING:
    from aether.agents.manager import SubAgentManager


class CheckTaskTool(AetherTool):
    name = "check_task"
    description = (
        "Check the status of a background task spawned with spawn_task. "
        "Returns the task status and result if completed. "
        "Use the task_id returned by spawn_task."
    )
    status_text = "Checking task status..."
    parameters = [
        ToolParam(
            name="task_id",
            type="string",
            description="The task_id returned by spawn_task",
        ),
    ]

    def __init__(self, sub_agent_manager: "SubAgentManager"):
        self._manager = sub_agent_manager

    async def execute(self, task_id: str) -> ToolResult:
        try:
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

        except Exception as e:
            return ToolResult.fail(f"Failed to check task: {e}")
