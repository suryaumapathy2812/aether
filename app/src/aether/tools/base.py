"""
AetherTool — the base class for all tools.

Inspired by OpenCode's Tool.define pattern: name + description + parameters + execute.
Each tool also has a status_text for the acknowledge pattern in voice mode.

Tools are provider-agnostic. The registry converts them to whatever schema
format the LLM provider needs (OpenAI function calling, Anthropic tools, etc.)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolParam:
    """A single parameter for a tool."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    items: dict | None = None  # For array types


@dataclass
class ToolResult:
    """The result of executing a tool."""
    output: str
    metadata: dict = field(default_factory=dict)
    error: bool = False

    @classmethod
    def success(cls, output: str, **metadata) -> ToolResult:
        return cls(output=output, metadata=metadata)

    @classmethod
    def fail(cls, error_msg: str, **metadata) -> ToolResult:
        return cls(output=error_msg, metadata=metadata, error=True)


class AetherTool(ABC):
    """
    Base class for all Aether tools.

    Subclass this, set the class attributes, implement execute().
    That's it. One tool per file, keep it simple.
    """

    # --- Override these in subclasses ---
    name: str = ""
    description: str = ""
    status_text: str = "Working..."  # Shown as spinner text / spoken as TTS acknowledge
    parameters: list[ToolParam] = []

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Run the tool with the given arguments. Return a ToolResult."""
        ...

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, args: dict) -> dict:
        """Validate and fill defaults. Returns cleaned args."""
        cleaned = {}
        for param in self.parameters:
            if param.name in args:
                cleaned[param.name] = args[param.name]
            elif param.required:
                raise ValueError(f"Missing required parameter: {param.name}")
            elif param.default is not None:
                cleaned[param.name] = param.default
        return cleaned

    async def safe_execute(self, **kwargs) -> ToolResult:
        """Execute with validation and error handling."""
        try:
            cleaned = self.validate_args(kwargs)
            return await self.execute(**cleaned)
        except ValueError as e:
            return ToolResult.fail(f"Invalid arguments: {e}")
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}", exc_info=True)
            return ToolResult.fail(f"Tool error: {e}")

    def __repr__(self) -> str:
        return f"<Tool:{self.name}>"
