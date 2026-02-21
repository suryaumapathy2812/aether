"""
Run Task tool — delegate work to a sub-agent (blocking).

Blocks until the sub-agent finishes and returns the result directly.
Like Claude Code's Task tool — the parent LLM calls it, waits for
the result, and continues.

Uses TaskRunner which delegates to SubAgentManager under the hood,
so sub-agents get full SessionLoop capabilities (persistent state,
agent types, tool filtering, event bus).
"""

from __future__ import annotations

from aether.agents.task_runner import TaskRunner
from aether.tools.base import AetherTool, ToolParam, ToolResult


class RunTaskTool(AetherTool):
    name = "run_task"
    description = (
        "Delegate a complex task to a worker agent. The worker has its own tool access "
        "and runs independently. Use this for multi-step tasks like creating project "
        "structures, running multiple commands, or any work that needs several tool calls. "
        "Blocks until the worker finishes (max 60s) and returns the result."
    )
    status_text = "Working on it..."
    parameters = [
        ToolParam(
            name="prompt",
            type="string",
            description="Clear instructions for what the worker should do",
        ),
    ]

    def __init__(self, task_runner: TaskRunner):
        self.task_runner = task_runner

    async def execute(self, prompt: str) -> ToolResult:
        result = await self.task_runner.run(prompt)
        return ToolResult.success(result)
