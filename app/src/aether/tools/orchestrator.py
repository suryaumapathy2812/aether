"""
Tool Orchestrator — executes tools with proper context injection.

This is the bridge between the LLM Core and the Tool Registry:
1. LLM Core yields ToolCallRequest
2. Tool Orchestrator executes the tool with context injection
3. Plugin tools receive OAuth tokens and config from plugin_context
4. Built-in tools receive working_dir and other runtime context

The orchestrator ensures that tools have the right context for execution,
without the LLM knowing about OAuth tokens or file system paths.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aether.core.config import config
from aether.llm.contracts import ToolResult

if TYPE_CHECKING:
    from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """
    Executes tools with proper context injection.

    Plugin context flow:
    1. Tool is looked up in registry
    2. If tool belongs to a plugin, inject plugin_context[plugin_name]
    3. Tool receives context via safe_execute(context=...)

    This ensures:
    - Plugin tools get OAuth tokens and config
    - Built-in tools get working_dir and other runtime context
    - LLM never sees sensitive tokens (they're not in the prompt)
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the tool orchestrator.

        Args:
            tool_registry: The registry containing all available tools
        """
        self.tool_registry = tool_registry

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str,
        plugin_context: dict[str, dict[str, Any]] | None = None,
    ) -> ToolResult:
        """
        Execute a tool with context injection.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments from the LLM function call
            call_id: OpenAI-style tool call ID for correlation
            plugin_context: Runtime credentials for plugin tools
                Format: {plugin_name: {access_token, config, ...}}

        Returns:
            ToolResult with output, call_id, and error status
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return ToolResult.failed(
                tool_name=tool_name,
                error_msg=f"Unknown tool: {tool_name}",
                call_id=call_id,
            )

        # Determine if this is a plugin tool
        plugin_name = self.tool_registry.get_plugin_for_tool(tool_name)

        # Build context for tool execution
        if plugin_name and plugin_context:
            # Plugin tool: inject OAuth tokens and config
            context = plugin_context.get(plugin_name, {})
        else:
            # Built-in tool: inject working_dir, etc.
            context = {"working_dir": config.server.working_dir}

        logger.info(f"Executing tool: {tool_name} (plugin={plugin_name})")

        # Execute with context
        result = await tool.safe_execute(context=context, **arguments)

        # Create new result with call_id
        return ToolResult(
            tool_name=tool_name,
            output=result.output,
            call_id=call_id,
            error=result.error,
            metadata=result.metadata,
        )

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        plugin_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls in sequence.

        Args:
            tool_calls: List of {tool_name, arguments, call_id}
            plugin_context: Runtime credentials for plugin tools

        Returns:
            List of ToolResults in the same order
        """
        results = []
        for tc in tool_calls:
            result = await self.execute(
                tool_name=tc["tool_name"],
                arguments=tc["arguments"],
                call_id=tc["call_id"],
                plugin_context=plugin_context,
            )
            results.append(result)
        return results
