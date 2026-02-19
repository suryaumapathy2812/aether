"""
Tool Service — tool execution coordination.

Wraps ToolOrchestrator with service-level concerns:
- Logging and metrics
- Error handling and fallback
- Batch execution support
- Plugin context resolution

This service is used by the kernel when executing tool_execute jobs
directly (outside the LLM agentic loop).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aether.core.metrics import metrics
from aether.llm.contracts import ToolResult

if TYPE_CHECKING:
    from aether.tools.orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)


class ToolService:
    """
    Tool execution coordination service.

    Wraps ToolOrchestrator with service-level logging, metrics,
    and error handling. Used for standalone tool execution jobs
    (tool_execute kind) that aren't part of an LLM agentic loop.
    """

    def __init__(self, tool_orchestrator: "ToolOrchestrator") -> None:
        self._orchestrator = tool_orchestrator

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str = "",
        plugin_context: dict[str, dict[str, Any]] | None = None,
    ) -> ToolResult:
        """
        Execute a single tool with logging and metrics.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            call_id: Correlation ID for the tool call
            plugin_context: Runtime credentials for plugin tools

        Returns:
            ToolResult with output and error status
        """
        started = time.time()

        try:
            result = await self._orchestrator.execute(
                tool_name=tool_name,
                arguments=arguments,
                call_id=call_id,
                plugin_context=plugin_context,
            )

            elapsed_ms = round((time.time() - started) * 1000)
            status = "error" if result.error else "ok"
            metrics.observe(
                "service.tool.duration_ms", elapsed_ms, labels={"tool": tool_name}
            )
            metrics.inc(
                "service.tool.executed", labels={"tool": tool_name, "status": status}
            )
            logger.info(
                f"Tool {tool_name} executed in {elapsed_ms}ms (status={status})"
            )

            return result

        except Exception as e:
            elapsed_ms = round((time.time() - started) * 1000)
            metrics.observe(
                "service.tool.duration_ms", elapsed_ms, labels={"tool": tool_name}
            )
            metrics.inc(
                "service.tool.executed",
                labels={"tool": tool_name, "status": "exception"},
            )
            logger.error(
                f"Tool {tool_name} failed in {elapsed_ms}ms: {e}",
                exc_info=True,
            )
            return ToolResult.failed(
                tool_name=tool_name,
                error_msg=str(e),
                call_id=call_id,
            )

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        plugin_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tools in sequence with logging.

        Args:
            tool_calls: List of {tool_name, arguments, call_id}
            plugin_context: Runtime credentials for plugin tools

        Returns:
            List of ToolResults in the same order
        """
        started = time.time()
        results = []

        for tc in tool_calls:
            result = await self.execute(
                tool_name=tc["tool_name"],
                arguments=tc["arguments"],
                call_id=tc.get("call_id", ""),
                plugin_context=plugin_context,
            )
            results.append(result)

        elapsed_ms = round((time.time() - started) * 1000)
        ok_count = sum(1 for r in results if not r.error)
        logger.info(
            f"Batch: {len(results)} tools in {elapsed_ms}ms "
            f"({ok_count} ok, {len(results) - ok_count} errors)"
        )

        return results
