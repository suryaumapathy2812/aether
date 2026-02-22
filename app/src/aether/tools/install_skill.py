"""Install-skill tool — install a skill from the skills.sh marketplace (GitHub).

Accepts the same shorthand format as the skills CLI:
  - owner/repo@skill-name   → installs a specific skill from a repo
  - owner/repo              → installs the root SKILL.md from a repo

Raw GitHub URL pattern:
  https://raw.githubusercontent.com/{owner}/{repo}/main/{skill-name}/SKILL.md
  https://raw.githubusercontent.com/{owner}/{repo}/main/SKILL.md  (root)

The skill is saved to app/.skills/{name}/SKILL.md and registered live.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from aether.skills.loader import Skill, SkillLoader
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

_APP_ROOT = Path(__file__).parent.parent.parent.parent
USER_SKILLS_DIR = _APP_ROOT / ".skills"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
YAML_LINE_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


def _sanitize_name(name: str) -> str:
    """Sanitize a skill name for safe filesystem use (mirrors skills CLI sanitizeName)."""
    sanitized = name.lower()
    sanitized = re.sub(r"[^a-z0-9._]+", "-", sanitized)
    sanitized = re.sub(r"^[.\-]+|[.\-]+$", "", sanitized)
    return sanitized[:255] or "unnamed-skill"


def _parse_source(source: str) -> tuple[str, str, str | None] | None:
    """Parse a skills.sh shorthand source into (owner, repo, skill_name|None).

    Supported formats:
      owner/repo@skill-name  → specific skill in a repo
      owner/repo             → root SKILL.md of a repo

    Returns None if the source cannot be parsed as a GitHub shorthand.
    """
    source = source.strip()

    # owner/repo@skill-name
    at_match = re.match(r"^([^/\s]+)/([^/@\s]+)@([^\s]+)$", source)
    if at_match:
        owner, repo, skill_name = at_match.groups()
        return owner, repo, skill_name

    # owner/repo (no @)
    slash_match = re.match(r"^([^/\s]+)/([^/\s]+)$", source)
    if slash_match and ":" not in source and not source.startswith("."):
        owner, repo = slash_match.groups()
        return owner, repo, None

    return None


def _raw_url(owner: str, repo: str, skill_name: str | None) -> str:
    """Build the raw.githubusercontent.com URL for a SKILL.md."""
    base = f"https://raw.githubusercontent.com/{owner}/{repo}/main"
    if skill_name:
        return f"{base}/{skill_name}/SKILL.md"
    return f"{base}/SKILL.md"


def _parse_frontmatter(raw: str) -> dict[str, str]:
    """Extract name and description from YAML frontmatter."""
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}
    return dict(YAML_LINE_RE.findall(match.group(1)))


class InstallSkillTool(AetherTool):
    """Install a skill from the skills.sh marketplace (GitHub) and register it immediately."""

    name = "install_skill"
    description = (
        "Install a skill from the skills.sh marketplace. "
        "Accepts GitHub shorthand: 'owner/repo@skill-name' for a specific skill, "
        "or 'owner/repo' for the root skill in a repo. "
        "The skill is downloaded and registered immediately — no restart needed. "
        "Use search_marketplace to find skills first."
    )
    status_text = "Installing skill..."
    parameters = [
        ToolParam(
            name="source",
            type="string",
            description=(
                "GitHub shorthand for the skill to install. "
                "Examples: 'vercel-labs/agent-skills@nextjs', 'anthropics/claude-skills@coding'. "
                "Use 'owner/repo@skill-name' for a specific skill or 'owner/repo' for the root skill."
            ),
            required=True,
        ),
    ]

    def __init__(self, skill_loader: SkillLoader) -> None:
        self._skill_loader = skill_loader

    async def execute(self, source: str, **_) -> ToolResult:
        source = source.strip()
        if not source:
            return ToolResult.fail("Source cannot be empty.")

        parsed = _parse_source(source)
        if not parsed:
            return ToolResult.fail(
                f"Cannot parse source '{source}'. "
                "Use 'owner/repo@skill-name' or 'owner/repo' format. "
                "Example: 'vercel-labs/agent-skills@nextjs'"
            )

        owner, repo, skill_name = parsed
        url = _raw_url(owner, repo, skill_name)

        logger.info("Fetching skill from %s", url)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, follow_redirects=True)
        except httpx.RequestError as e:
            logger.error("Network error fetching skill: %s", e)
            return ToolResult.fail(f"Network error fetching skill: {e}")

        if resp.status_code == 404:
            hint = (
                f"Skill '{skill_name}' not found in {owner}/{repo}. "
                if skill_name
                else f"No root SKILL.md found in {owner}/{repo}. "
            )
            return ToolResult.fail(
                f"{hint}"
                "Check the skill name or use search_marketplace to find available skills."
            )

        if resp.status_code != 200:
            return ToolResult.fail(
                f"Failed to fetch skill (HTTP {resp.status_code}): {url}"
            )

        raw_content = resp.text

        # Parse and validate frontmatter
        metadata = _parse_frontmatter(raw_content)
        fm_name = metadata.get("name", "").strip()
        fm_description = metadata.get("description", "").strip()

        if not fm_name or not fm_description:
            return ToolResult.fail(
                f"Invalid SKILL.md at {url}: missing 'name' or 'description' in frontmatter. "
                "This skill cannot be installed."
            )

        safe_name = _sanitize_name(fm_name)

        # Check for collision
        existing = self._skill_loader.get(safe_name)
        if existing:
            return ToolResult.fail(
                f"A skill named '{safe_name}' is already installed "
                f"(at {existing.location}). "
                "Remove it first if you want to reinstall."
            )

        # Write to disk
        skill_dir = USER_SKILLS_DIR / safe_name
        skill_path = skill_dir / "SKILL.md"

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(raw_content, encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write skill '%s': %s", safe_name, e)
            return ToolResult.fail(f"Failed to write skill to disk: {e}")

        # Register live
        skill = Skill(
            name=safe_name,
            description=fm_description,
            location=str(skill_path),
        )
        self._skill_loader.register(skill)

        logger.info("Installed skill '%s' from %s", safe_name, url)
        return ToolResult.success(
            f"Skill **{safe_name}** installed successfully.\n"
            f"Source: `{url}`\n"
            f"Location: `{skill_path}`\n"
            f"Description: {fm_description}\n\n"
            f"Use `read_skill(name='{safe_name}')` to load it.",
            skill_name=safe_name,
            source=source,
            location=str(skill_path),
        )
