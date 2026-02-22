"""Create-skill tool — lets the LLM (or user) create a new skill at runtime.

Writes a SKILL.md to app/.skills/{name}/ and registers it live in the
SkillLoader so it's immediately available without a restart.

Name sanitization mirrors the skills CLI (installer.ts sanitizeName):
  - lowercase
  - replace non-alphanumeric (except . and _) with hyphens
  - strip leading/trailing dots and hyphens
  - max 255 chars
  - fallback to 'unnamed-skill' if empty after sanitization
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from aether.skills.loader import Skill, SkillLoader
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# app/.skills/ lives two levels above this file:
# app/src/aether/tools/create_skill.py → app/
_APP_ROOT = Path(__file__).parent.parent.parent.parent
USER_SKILLS_DIR = _APP_ROOT / ".skills"


def _sanitize_name(name: str) -> str:
    """Sanitize a skill name for safe filesystem use (mirrors skills CLI sanitizeName)."""
    sanitized = name.lower()
    sanitized = re.sub(r"[^a-z0-9._]+", "-", sanitized)
    sanitized = re.sub(r"^[.\-]+|[.\-]+$", "", sanitized)
    return sanitized[:255] or "unnamed-skill"


def _build_skill_md(name: str, description: str, content: str) -> str:
    """Render a SKILL.md with YAML frontmatter."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{content.strip()}\n"


class CreateSkillTool(AetherTool):
    """Create a new skill and register it immediately (no restart needed)."""

    name = "create_skill"
    description = (
        "Create a new skill and make it immediately available. "
        "Skills are reusable guidance documents that teach you how to handle "
        "specific workflows, tools, or behaviors. "
        "The skill is saved to disk and registered live — no restart needed."
    )
    status_text = "Creating skill..."
    parameters = [
        ToolParam(
            name="name",
            type="string",
            description=(
                "Short kebab-case name for the skill "
                "(e.g. 'gmail-drafting', 'meeting-notes', 'code-review'). "
                "Will be sanitized automatically."
            ),
            required=True,
        ),
        ToolParam(
            name="description",
            type="string",
            description=(
                "One-sentence description of what this skill covers. "
                "Used for search/discovery — be specific and descriptive."
            ),
            required=True,
        ),
        ToolParam(
            name="content",
            type="string",
            description=(
                "Full markdown body of the skill. "
                "Include step-by-step guidance, rules, examples, and any "
                "tool-calling instructions relevant to this skill."
            ),
            required=True,
        ),
    ]

    def __init__(self, skill_loader: SkillLoader) -> None:
        self._skill_loader = skill_loader

    async def execute(
        self, name: str, description: str, content: str, **_
    ) -> ToolResult:
        name = name.strip()
        description = description.strip()
        content = content.strip()

        if not name:
            return ToolResult.fail("Skill name cannot be empty.")
        if not description:
            return ToolResult.fail("Skill description cannot be empty.")
        if not content:
            return ToolResult.fail("Skill content cannot be empty.")

        safe_name = _sanitize_name(name)

        # Check for collision with existing skills
        existing = self._skill_loader.get(safe_name)
        if existing:
            return ToolResult.fail(
                f"A skill named '{safe_name}' already exists "
                f"(at {existing.location}). "
                "Use a different name or read_skill to view the existing one."
            )

        # Write to disk
        skill_dir = USER_SKILLS_DIR / safe_name
        skill_path = skill_dir / "SKILL.md"

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(
                _build_skill_md(safe_name, description, content),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("Failed to write skill '%s': %s", safe_name, e)
            return ToolResult.fail(f"Failed to write skill to disk: {e}")

        # Register live in skill_loader
        skill = Skill(
            name=safe_name,
            description=description,
            location=str(skill_path),
        )
        self._skill_loader.register(skill)

        logger.info("Created skill '%s' at %s", safe_name, skill_path)
        return ToolResult.success(
            f"Skill **{safe_name}** created successfully.\n"
            f"Location: `{skill_path}`\n"
            f"It's now available via `search_skill` and `read_skill`.",
            skill_name=safe_name,
            location=str(skill_path),
        )
