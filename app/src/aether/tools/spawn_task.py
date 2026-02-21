"""
Spawn Task tool — fire-and-forget sub-agent delegation.

Unlike run_task (which blocks), spawn_task returns immediately with a
task_id. The sub-agent runs in the background. Use check_task to
monitor progress and retrieve results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aether.tools.base import AetherTool, ToolParam, ToolResult

if TYPE_CHECKING:
    from aether.agents.manager import SubAgentManager


class SpawnTaskTool(AetherTool):
    name = "spawn_task"
    description = (
        "Delegate a task to a background worker agent. Returns immediately with a "
        "task_id — the worker runs independently. Use check_task to monitor progress "
        "and retrieve results. Good for long-running work that shouldn't block the "
        "conversation."
    )
    status_text = "Starting background task..."
    parameters = [
        ToolParam(
            name="prompt",
            type="string",
            description="Clear instructions for what the worker should do",
        ),
        ToolParam(
            name="agent_type",
            type="string",
            description="Type of agent to use: general, explore, planner",
            required=False,
            default="general",
            enum=["general", "explore", "planner"],
        ),
    ]

    def __init__(
        self,
        sub_agent_manager: "SubAgentManager",
        parent_session_id: str = "",
    ):
        self._manager = sub_agent_manager
        self._parent_session_id = parent_session_id

    async def execute(self, prompt: str, agent_type: str = "general") -> ToolResult:
        try:
            child_id = await self._manager.spawn(
                prompt=prompt,
                parent_session_id=self._parent_session_id,
                agent_type=agent_type,
            )
            return ToolResult.success(
                f"Task spawned with id: {child_id}. Use check_task to monitor progress."
            )
        except Exception as e:
            return ToolResult.fail(f"Failed to spawn task: {e}")
