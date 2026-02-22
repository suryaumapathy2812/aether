"""Read-skill tool — lets the LLM load the full content of a skill by name.

Built-in tool (always available, no plugin required). Given a skill name,
returns its full markdown content so the LLM can apply its guidance.

Typical flow:
  1. search_skill(query="...") → find relevant skill names
  2. read_skill(name="...") → load full content of the chosen skill
"""

from __future__ import annotations

import logging

from aether.skills.loader import SkillLoader
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)


class ReadSkillTool(AetherTool):
    """Load the full content of a skill by name."""

    name = "read_skill"
    description = (
        "Load the full content of a skill by name. Skills contain detailed guidance "
        "on behavior, tool usage, plugin workflows, and more. Use search_skill first "
        "to find the right skill name, then call this to load it."
    )
    status_text = "Loading skill..."
    parameters = [
        ToolParam(
            name="name",
            type="string",
            description=(
                "The exact skill name to load "
                "(e.g. 'soul', 'tool-calling', 'gmail', 'google-calendar'). "
                "Use search_skill to discover available skill names."
            ),
            required=True,
        ),
    ]

    def __init__(self, skill_loader: SkillLoader) -> None:
        self._skill_loader = skill_loader

    async def execute(self, name: str, **_) -> ToolResult:
        name = name.strip()
        if not name:
            return ToolResult.fail("Skill name cannot be empty.")

        skill = self._skill_loader.get(name)

        if skill is None:
            # Try to be helpful — list what's available
            all_skills = self._skill_loader.all()
            available = ", ".join(s.name for s in all_skills) if all_skills else "none"
            return ToolResult.fail(
                f"Skill '{name}' not found. Available skills: {available}. "
                "Use search_skill to find the right name."
            )

        content = skill.content
        if not content:
            return ToolResult.fail(f"Skill '{name}' exists but has no content.")

        logger.info("LLM loaded skill: %s (%d chars)", name, len(content))
        return ToolResult.success(content)
