"""
Task Runner — sub-agents lite.

Runs a sub-agent with its own LLM call, tool access, and agentic loop.
Blocks until the sub-agent finishes and returns the result.

Sub-agents get a filtered tool registry — no access to run_task
(prevents recursive spawning).

Max 60s timeout per task, max 5 tool loop iterations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from aether.providers.base import LLMProvider
from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
MAX_ITERATIONS = 5  # Tool loop iterations per task


class TaskRunner:
    """
    Runs sub-agent tasks with tool access.

    Usage:
        runner = TaskRunner(llm_provider, tool_registry)
        result = await runner.run("Create a project structure with 5 files")
    """

    def __init__(
        self, provider: LLMProvider, tool_registry: ToolRegistry | None = None
    ):
        self.provider = provider
        # Filter out task tools — sub-agents can't spawn more sub-agents
        if tool_registry:
            self.tool_registry = tool_registry.without(
                "run_task", "spawn_task", "check_task"
            )
        else:
            self.tool_registry = None

    async def run(self, prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Run a sub-agent task. Blocks until complete. Returns the result text."""
        start = time.time()
        logger.info(f"Running sub-agent task: {prompt[:80]}")

        try:
            result = await asyncio.wait_for(
                self._execute(prompt),
                timeout=timeout,
            )
            duration = time.time() - start
            logger.info(f"Sub-agent task completed in {duration:.1f}s")
            return result

        except asyncio.TimeoutError:
            duration = time.time() - start
            logger.warning(f"Sub-agent task timed out after {duration:.1f}s")
            return f"Task timed out after {timeout}s. Partial work may have been completed."

        except Exception as e:
            logger.error(f"Sub-agent task failed: {e}", exc_info=True)
            return f"Task failed: {e}"

    async def _execute(self, prompt: str) -> str:
        """Run the LLM agentic loop for a task."""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a background worker agent. Complete the given task efficiently. "
                    "Use tools as needed. Be concise in your final response — just report what you did."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        tools_schema = (
            self.tool_registry.to_openai_tools() if self.tool_registry else None
        )
        full_response = ""

        for iteration in range(MAX_ITERATIONS):
            buffer = ""
            pending_tool_calls = []

            async for event in self.provider.generate_stream_with_tools(
                messages,
                tools=tools_schema,
                max_tokens=1000,
                temperature=0.3,
            ):
                if event.type == "token":
                    buffer += event.content
                elif event.type == "tool_calls":
                    pending_tool_calls = event.tool_calls

            full_response += buffer

            if not pending_tool_calls:
                break

            # Execute tools
            if not self.tool_registry:
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": buffer or None,
            }
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in pending_tool_calls
            ]
            messages.append(assistant_msg)

            for tc in pending_tool_calls:
                logger.info(f"Sub-agent tool: {tc.name}({list(tc.arguments.keys())})")
                result = await self.tool_registry.dispatch(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.output,
                    }
                )

            logger.info(
                f"Sub-agent iteration {iteration + 1}: executed {len(pending_tool_calls)} tools"
            )

        return full_response.strip()
