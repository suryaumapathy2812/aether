"""
Tool Registry — register tools, get schemas, dispatch by name.

Dead simple. Register tools, ask for their schemas in any format,
dispatch calls by tool name.
"""

from __future__ import annotations

import logging
from typing import Any

from aether.tools.base import AetherTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, AetherTool] = {}

    def register(self, tool: AetherTool) -> None:
        """Register a tool. Overwrites if name already exists."""
        if not tool.name:
            raise ValueError(f"Tool must have a name: {tool}")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> AetherTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[AetherTool]:
        """All registered tools."""
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())

    async def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given args."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.fail(f"Unknown tool: {name}")

        logger.info(f"Dispatching tool: {name} with args: {list(args.keys())}")
        return await tool.safe_execute(**args)

    def get_status_text(self, name: str) -> str:
        """Get the status text for a tool (for spinners/voice acknowledge)."""
        tool = self._tools.get(name)
        return tool.status_text if tool else "Working..."

    # --- Schema export for different providers ---

    def to_openai_tools(self) -> list[dict]:
        """Get all tool schemas in OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def without(self, *names: str) -> ToolRegistry:
        """Return a new registry excluding the named tools. Used to restrict sub-agent access."""
        filtered = ToolRegistry()
        for name, tool in self._tools.items():
            if name not in names:
                filtered._tools[name] = tool
        return filtered

    def to_anthropic_tools(self) -> list[dict]:
        """Get all tool schemas in Anthropic tool use format."""
        tools = []
        for tool in self._tools.values():
            schema = tool.to_openai_schema()
            # Anthropic uses a slightly different format
            tools.append(
                {
                    "name": schema["function"]["name"],
                    "description": schema["function"]["description"],
                    "input_schema": schema["function"]["parameters"],
                }
            )
        return tools


# Global default registry
_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Get or create the default global tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry
