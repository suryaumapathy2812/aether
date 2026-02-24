"""Remove-skill tool — uninstall a user-installed or user-created skill.

Only skills in app/.skills/ (user-installed via install_skill or created via
create_skill) can be removed.  Built-in skills (app/skills/) and plugin skills
(app/plugins/) are protected and cannot be removed.

The skill's SKILL.md is deleted, its parent directory is cleaned up if empty,
and the skill is unregistered from the SkillLoader so it's no longer available.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from aether.skills.loader import SkillLoader
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# app/.skills/ lives four levels above this file:
# app/src/aether/tools/remove_skill.py → app/
_APP_ROOT = Path(__file__).parent.parent.parent.parent
USER_SKILLS_DIR = _APP_ROOT / ".skills"
BUILTIN_SKILLS_DIR = _APP_ROOT / "skills"
PLUGIN_SKILLS_DIR = _APP_ROOT / "plugins"


class RemoveSkillTool(AetherTool):
    """Remove a user-installed or user-created skill (built-in and plugin skills cannot be removed)."""

    name = "remove_skill"
    description = (
        "Remove a user-installed or user-created skill. "
        "Only skills that were installed via install_skill or created via create_skill "
        "can be removed. Built-in skills and plugin skills are protected. "
        "The skill is deleted from disk and unregistered immediately — no restart needed."
    )
    status_text = "Removing skill..."
    parameters = [
        ToolParam(
            name="name",
            type="string",
            description=(
                "The name of the skill to remove. "
                "Use search_skill to find available skills and their names."
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

        # Look up the skill
        skill = self._skill_loader.get(name)
        if not skill:
            return ToolResult.fail(
                f"No skill named '{name}' found. "
                "Use search_skill to find available skills."
            )

        # Determine the skill's location and check if it's removable
        skill_path = Path(skill.location).resolve()

        builtin_resolved = BUILTIN_SKILLS_DIR.resolve()
        plugin_resolved = PLUGIN_SKILLS_DIR.resolve()
        user_resolved = USER_SKILLS_DIR.resolve()

        if _is_under(skill_path, builtin_resolved):
            return ToolResult.fail(
                f"Cannot remove built-in skills. "
                f"'{name}' is a built-in skill at {skill.location}."
            )

        if _is_under(skill_path, plugin_resolved):
            return ToolResult.fail(
                f"Cannot remove plugin skills. "
                f"'{name}' is a plugin skill at {skill.location}."
            )

        if not _is_under(skill_path, user_resolved):
            return ToolResult.fail(
                f"Cannot remove skill '{name}' — it is not in the user skills "
                f"directory ({USER_SKILLS_DIR}). Location: {skill.location}"
            )

        # Delete the SKILL.md file
        try:
            skill_path.unlink(missing_ok=True)
        except OSError as e:
            logger.error("Failed to delete skill file '%s': %s", skill_path, e)
            return ToolResult.fail(f"Failed to delete skill file: {e}")

        # Remove the parent directory if it's now empty (or remove the whole skill dir)
        skill_dir = skill_path.parent
        try:
            if skill_dir != user_resolved and skill_dir.exists():
                # Remove the skill's directory (e.g. app/.skills/my-skill/)
                shutil.rmtree(str(skill_dir))
        except OSError as e:
            # Non-fatal: the file is already deleted, just log the warning
            logger.warning("Failed to clean up skill directory '%s': %s", skill_dir, e)

        # Unregister from the SkillLoader
        self._skill_loader.unregister(name)

        logger.info("Removed skill '%s' (was at %s)", name, skill.location)
        return ToolResult.success(
            f"Skill **{name}** removed successfully.\n"
            f"Deleted: `{skill.location}`\n"
            f"It is no longer available via search_skill or read_skill.",
            skill_name=name,
            location=skill.location,
        )


def _is_under(path: Path, parent: Path) -> bool:
    """Check if *path* is under *parent* (both should be resolved)."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
