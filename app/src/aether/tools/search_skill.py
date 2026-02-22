"""Search-skill tool — lets the LLM find relevant skills by query.

Built-in tool (always available, no plugin required). Searches across all
discovered skills (from app/skills/ and app/plugins/*/SKILL.md) by keyword
overlap and returns matching skill names + descriptions.

The LLM uses this to discover which skill to load before calling read_skill.
"""

from __future__ import annotations

import logging

from aether.skills.loader import SkillLoader
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)


class SearchSkillTool(AetherTool):
    """Search available skills by topic or keyword."""

    name = "search_skill"
    description = (
        "Search your available skills by topic or keyword. Returns matching skill "
        "names and descriptions. Use this when you need guidance on a specific "
        "workflow, plugin, or behavior — then call read_skill to load the full content."
    )
    status_text = "Searching skills..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description=(
                "What to search for. Be descriptive "
                "(e.g. 'gmail push notifications', 'tool calling rules', "
                "'Aether personality', 'google calendar workflow')."
            ),
            required=True,
        ),
    ]

    def __init__(self, skill_loader: SkillLoader) -> None:
        self._skill_loader = skill_loader

    async def execute(self, query: str, **_) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult.fail("Cannot search with an empty query.")

        matches = self._skill_loader.match(query)

        if not matches:
            # Fall back to listing all skills
            all_skills = self._skill_loader.all()
            if not all_skills:
                return ToolResult.success("No skills are currently installed.")
            lines = ["No close matches found. All available skills:"]
            for skill in all_skills:
                lines.append(f"- **{skill.name}**: {skill.description}")
            return ToolResult.success("\n".join(lines))

        lines = [f"Found {len(matches)} matching skill(s):"]
        for skill in matches:
            lines.append(f"- **{skill.name}**: {skill.description}")
        lines.append("\nUse read_skill(name=...) to load the full content of a skill.")

        logger.info("LLM searched skills: %s → %d results", query[:80], len(matches))
        return ToolResult.success("\n".join(lines))
