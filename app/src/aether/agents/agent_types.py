"""
Agent Type System — specialized agent definitions.

Each agent type defines its behavior: system prompt, allowed tools,
model override, and iteration limits. This enables specialized
sub-agents (explorer = read-only, planner = no execution, etc.)

Usage:
    agent_def = get_agent_type("explore")
    # agent_def.system_prompt, agent_def.allowed_tools, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentTypeDefinition:
    """Definition of a specialized agent type."""

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None = None  # None = all tools allowed
    denied_tools: list[str] = field(default_factory=list)
    model_override: str | None = None  # Override the default model
    max_iterations: int = 25
    max_duration: float = 300.0  # seconds
    can_spawn_sub_agents: bool = False  # Prevent recursive spawning


# ─── Built-in Agent Types ─────────────────────────────────────

GENERAL_AGENT = AgentTypeDefinition(
    name="general",
    description="General-purpose agent with full tool access",
    system_prompt=(
        "You are a background worker agent. Complete the given task efficiently. "
        "Use tools as needed. Be concise in your final response — just report "
        "what you did and the results."
    ),
    denied_tools=["spawn_task", "check_task"],  # No recursive spawning
    max_iterations=25,
    max_duration=300.0,
)

EXPLORE_AGENT = AgentTypeDefinition(
    name="explore",
    description="Read-only codebase exploration agent",
    system_prompt=(
        "You are a codebase exploration agent. Your job is to read and analyze "
        "code, find patterns, and answer questions about the codebase. "
        "You can ONLY read files and list directories — you cannot modify "
        "anything. Be thorough in your exploration and provide detailed findings."
    ),
    allowed_tools=["read_file", "list_directory", "web_search"],
    max_iterations=30,
    max_duration=180.0,
)

PLANNER_AGENT = AgentTypeDefinition(
    name="planner",
    description="Planning agent that creates plans without executing",
    system_prompt=(
        "You are a planning agent. Your job is to analyze a task and create "
        "a detailed plan for how to accomplish it. You can read files to "
        "understand the codebase, but you should NOT execute any changes. "
        "Output a clear, step-by-step plan with specific file paths and "
        "code changes needed."
    ),
    allowed_tools=["read_file", "list_directory", "web_search"],
    max_iterations=20,
    max_duration=120.0,
)

# Registry of all agent types
_AGENT_TYPES: dict[str, AgentTypeDefinition] = {
    "default": GENERAL_AGENT,
    "general": GENERAL_AGENT,
    "explore": EXPLORE_AGENT,
    "planner": PLANNER_AGENT,
}


def get_agent_type(name: str) -> AgentTypeDefinition:
    """Get an agent type definition by name. Falls back to 'general'."""
    agent_type = _AGENT_TYPES.get(name)
    if agent_type is None:
        logger.warning("Unknown agent type '%s', falling back to 'general'", name)
        return GENERAL_AGENT
    return agent_type


def register_agent_type(definition: AgentTypeDefinition) -> None:
    """Register a custom agent type."""
    _AGENT_TYPES[definition.name] = definition
    logger.info("Registered agent type: %s", definition.name)


def list_agent_types() -> list[AgentTypeDefinition]:
    """List all registered agent types."""
    # Deduplicate (default and general point to same object)
    seen = set()
    result = []
    for defn in _AGENT_TYPES.values():
        if defn.name not in seen:
            seen.add(defn.name)
            result.append(defn)
    return result


def get_filtered_tools(
    agent_type: str,
    all_tool_names: list[str],
) -> list[str]:
    """
    Get the list of tool names allowed for an agent type.

    Applies both allowed_tools (whitelist) and denied_tools (blacklist).
    """
    defn = get_agent_type(agent_type)

    if defn.allowed_tools is not None:
        # Whitelist mode: only these tools
        tools = [t for t in all_tool_names if t in defn.allowed_tools]
    else:
        # All tools minus denied
        tools = [t for t in all_tool_names if t not in defn.denied_tools]

    return tools
