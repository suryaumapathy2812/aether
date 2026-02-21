"""
Skill Loader — discover and load SKILL.md files.

Skills are markdown files with YAML frontmatter (name + description).
They teach the LLM how to use tools for specific workflows.

Discovery:
  1. Scan skills/ directory for SKILL.md files
  2. Parse frontmatter (name, description)
  3. Store metadata — body loaded on demand (progressive loading)

Invocation:
  - Description matching (implicit, like Claude Code)
  - The LLM system prompt includes skill descriptions
  - When relevant, the full skill body is injected into context
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Simple YAML frontmatter parser (avoids pyyaml dependency)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
YAML_LINE_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
    location: str  # File path to SKILL.md
    _content: str | None = field(default=None, repr=False)
    plugin_name: str | None = field(default=None)  # Set for plugin-sourced skills

    @property
    def content(self) -> str:
        """Load the full skill body on demand (progressive loading)."""
        if self._content is None:
            try:
                with open(self.location, "r", encoding="utf-8") as f:
                    raw = f.read()
                # Strip frontmatter
                match = FRONTMATTER_RE.match(raw)
                self._content = raw[match.end() :].strip() if match else raw.strip()
            except Exception as e:
                logger.error(f"Failed to load skill content from {self.location}: {e}")
                self._content = ""
        return self._content


class SkillLoader:
    """Discovers and manages skills from the filesystem."""

    def __init__(self, skills_dirs: list[str] | None = None):
        self._skills: dict[str, Skill] = {}
        self._dirs = skills_dirs or []

    def add_directory(self, path: str) -> None:
        """Add a directory to scan for skills."""
        self._dirs.append(path)

    def discover(self) -> int:
        """Scan all configured directories for SKILL.md files. Returns count of skills found."""
        count = 0
        for directory in self._dirs:
            if not os.path.isdir(directory):
                logger.debug(f"Skills directory not found: {directory}")
                continue

            for root, dirs, files in os.walk(directory):
                if "SKILL.md" in files:
                    skill_path = os.path.join(root, "SKILL.md")
                    skill = self._parse_skill(skill_path)
                    if skill:
                        if skill.name in self._skills:
                            logger.warning(f"Duplicate skill name: {skill.name}")
                        self._skills[skill.name] = skill
                        count += 1
                        logger.info(f"Discovered skill: {skill.name} ({skill_path})")

        return count

    def register(self, skill: Skill) -> None:
        """Register a skill directly (e.g. from a plugin's SKILL.md)."""
        if skill.name in self._skills:
            logger.warning(f"Duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        """All discovered skills."""
        return list(self._skills.values())

    def match(self, query: str) -> list[Skill]:
        """Find skills whose description matches the query (simple keyword matching)."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matches = []
        for skill in self._skills.values():
            desc_lower = skill.description.lower()
            desc_words = set(desc_lower.split())
            # Score: how many query words appear in the description
            overlap = len(query_words & desc_words)
            if overlap > 0:
                matches.append((overlap, skill))

        # Sort by relevance (most overlapping words first)
        matches.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in matches]

    def get_system_prompt_section(
        self, enabled_plugins: list[str] | None = None
    ) -> str:
        """Generate a system prompt section listing available skills.

        Plugin-sourced skills are only listed if their plugin is enabled.
        Built-in skills (no plugin_name) are always listed.
        """
        if not self._skills:
            return ""

        enabled_set = set(enabled_plugins) if enabled_plugins else set()
        lines = ["Available skills (use when relevant):"]
        for skill in self._skills.values():
            # Skip plugin skills whose plugin is not enabled
            if skill.plugin_name and skill.plugin_name not in enabled_set:
                continue
            lines.append(f"- **{skill.name}**: {skill.description}")

        # If only the header remains, no skills to show
        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _parse_skill(self, path: str) -> Skill | None:
        """Parse a SKILL.md file. Returns Skill or None if invalid."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            match = FRONTMATTER_RE.match(raw)
            if not match:
                logger.warning(f"No frontmatter in {path}")
                return None

            frontmatter = match.group(1)
            metadata = dict(YAML_LINE_RE.findall(frontmatter))

            name = metadata.get("name", "").strip()
            description = metadata.get("description", "").strip()

            if not name or not description:
                logger.warning(f"Missing name or description in {path}")
                return None

            return Skill(
                name=name,
                description=description,
                location=path,
            )

        except Exception as e:
            logger.error(f"Failed to parse skill: {path}: {e}")
            return None
