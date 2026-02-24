"""Tests for the remove_skill tool.

Covers:
- _is_under helper for path containment checks
- RemoveSkillTool.execute for empty name, nonexistent skill, built-in/plugin
  refusal, and successful user-skill removal with unregister.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aether.skills.loader import Skill, SkillLoader
from aether.tools.remove_skill import RemoveSkillTool, _is_under
import aether.tools.remove_skill as remove_skill_mod


# ─── _is_under helper ───────────────────────────────────────────


class TestIsUnder:
    """Test the _is_under path containment helper."""

    def test_path_under_parent(self):
        """Positive: /foo/bar/baz is under /foo/bar → True."""
        # Objective: verify _is_under returns True when child is inside parent.
        parent = Path("/foo/bar")
        child = Path("/foo/bar/baz/file.md")
        assert _is_under(child, parent) is True

    def test_path_not_under_parent(self):
        """Negative: /foo/bar is NOT under /other → False."""
        # Objective: verify _is_under returns False when child is outside parent.
        parent = Path("/other")
        child = Path("/foo/bar/file.md")
        assert _is_under(child, parent) is False

    def test_path_equal_to_parent(self):
        """Edge: a path is considered 'under' itself (relative_to succeeds)."""
        p = Path("/foo/bar")
        assert _is_under(p, p) is True

    def test_partial_name_overlap_not_under(self):
        """Negative: /foo/bar-extra is NOT under /foo/bar (no false prefix match)."""
        parent = Path("/foo/bar")
        child = Path("/foo/bar-extra/file.md")
        assert _is_under(child, parent) is False


# ─── RemoveSkillTool ─────────────────────────────────────────────


class TestRemoveSkillTool:
    """Test RemoveSkillTool.execute with various skill locations."""

    @pytest.mark.asyncio
    async def test_remove_empty_name(self):
        """Negative: empty name returns failure.

        Objective: verify that a blank skill name is rejected immediately.
        """
        loader = SkillLoader()
        tool = RemoveSkillTool(skill_loader=loader)
        result = await tool.execute(name="   ")
        assert result.error is True
        assert "empty" in result.output.lower()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_skill(self):
        """Negative: removing a skill that doesn't exist returns failure.

        Objective: verify that a missing skill name is handled gracefully.
        """
        loader = SkillLoader()
        tool = RemoveSkillTool(skill_loader=loader)
        result = await tool.execute(name="no-such-skill")
        assert result.error is True
        assert "no skill named" in result.output.lower()

    @pytest.mark.asyncio
    async def test_remove_builtin_skill_refused(self):
        """Negative: cannot remove built-in skills (app/skills/).

        Objective: verify that skills under BUILTIN_SKILLS_DIR are protected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builtin_dir = Path(tmpdir) / "skills"
            builtin_dir.mkdir()
            skill_file = builtin_dir / "soul" / "SKILL.md"
            skill_file.parent.mkdir()
            skill_file.write_text("# Soul skill")

            loader = SkillLoader()
            loader.register(
                Skill(
                    name="soul",
                    description="Core identity skill",
                    location=str(skill_file),
                )
            )

            tool = RemoveSkillTool(skill_loader=loader)

            with (
                patch.object(remove_skill_mod, "BUILTIN_SKILLS_DIR", builtin_dir),
                patch.object(
                    remove_skill_mod, "PLUGIN_SKILLS_DIR", Path(tmpdir) / "plugins"
                ),
                patch.object(
                    remove_skill_mod, "USER_SKILLS_DIR", Path(tmpdir) / ".skills"
                ),
            ):
                result = await tool.execute(name="soul")

            assert result.error is True
            assert "built-in" in result.output.lower()

    @pytest.mark.asyncio
    async def test_remove_plugin_skill_refused(self):
        """Negative: cannot remove plugin skills (app/plugins/).

        Objective: verify that skills under PLUGIN_SKILLS_DIR are protected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "plugins"
            plugin_dir.mkdir()
            skill_file = plugin_dir / "gmail" / "SKILL.md"
            skill_file.parent.mkdir()
            skill_file.write_text("# Gmail skill")

            loader = SkillLoader()
            loader.register(
                Skill(
                    name="gmail-skill",
                    description="Gmail plugin skill",
                    location=str(skill_file),
                )
            )

            tool = RemoveSkillTool(skill_loader=loader)

            with (
                patch.object(
                    remove_skill_mod, "BUILTIN_SKILLS_DIR", Path(tmpdir) / "skills"
                ),
                patch.object(remove_skill_mod, "PLUGIN_SKILLS_DIR", plugin_dir),
                patch.object(
                    remove_skill_mod, "USER_SKILLS_DIR", Path(tmpdir) / ".skills"
                ),
            ):
                result = await tool.execute(name="gmail-skill")

            assert result.error is True
            assert "plugin" in result.output.lower()

    @pytest.mark.asyncio
    async def test_remove_user_skill_success(self):
        """Positive: successfully removes a user-installed skill from .skills/.

        Objective: verify that a skill under USER_SKILLS_DIR is deleted from
        disk and the tool returns success.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / ".skills"
            user_dir.mkdir()
            skill_dir = user_dir / "my-custom"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("# My custom skill")

            loader = SkillLoader()
            loader.register(
                Skill(
                    name="my-custom",
                    description="A user-created skill",
                    location=str(skill_file),
                )
            )

            tool = RemoveSkillTool(skill_loader=loader)

            with (
                patch.object(
                    remove_skill_mod, "BUILTIN_SKILLS_DIR", Path(tmpdir) / "skills"
                ),
                patch.object(
                    remove_skill_mod, "PLUGIN_SKILLS_DIR", Path(tmpdir) / "plugins"
                ),
                patch.object(remove_skill_mod, "USER_SKILLS_DIR", user_dir),
            ):
                result = await tool.execute(name="my-custom")

            assert result.error is False
            assert "removed successfully" in result.output.lower()
            # File should be deleted
            assert not skill_file.exists()
            # Directory should be cleaned up
            assert not skill_dir.exists()

    @pytest.mark.asyncio
    async def test_unregister_called_on_success(self):
        """Positive: SkillLoader.unregister() is called after successful removal.

        Objective: verify that the skill is unregistered from the loader so it
        is no longer discoverable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / ".skills"
            user_dir.mkdir()
            skill_dir = user_dir / "temp-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("# Temp skill")

            loader = SkillLoader()
            loader.register(
                Skill(
                    name="temp-skill",
                    description="Temporary skill",
                    location=str(skill_file),
                )
            )

            # Confirm it's registered
            assert loader.get("temp-skill") is not None

            tool = RemoveSkillTool(skill_loader=loader)

            with (
                patch.object(
                    remove_skill_mod, "BUILTIN_SKILLS_DIR", Path(tmpdir) / "skills"
                ),
                patch.object(
                    remove_skill_mod, "PLUGIN_SKILLS_DIR", Path(tmpdir) / "plugins"
                ),
                patch.object(remove_skill_mod, "USER_SKILLS_DIR", user_dir),
            ):
                result = await tool.execute(name="temp-skill")

            assert result.error is False
            # Skill should be unregistered
            assert loader.get("temp-skill") is None

    @pytest.mark.asyncio
    async def test_remove_skill_outside_all_dirs_refused(self):
        """Negative: skill not in any known directory is refused.

        Objective: verify that a skill whose location is outside builtin,
        plugin, and user dirs cannot be removed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            rogue_file = Path(tmpdir) / "rogue" / "SKILL.md"
            rogue_file.parent.mkdir()
            rogue_file.write_text("# Rogue skill")

            loader = SkillLoader()
            loader.register(
                Skill(
                    name="rogue",
                    description="A rogue skill",
                    location=str(rogue_file),
                )
            )

            tool = RemoveSkillTool(skill_loader=loader)

            with (
                patch.object(
                    remove_skill_mod, "BUILTIN_SKILLS_DIR", Path(tmpdir) / "skills"
                ),
                patch.object(
                    remove_skill_mod, "PLUGIN_SKILLS_DIR", Path(tmpdir) / "plugins"
                ),
                patch.object(
                    remove_skill_mod, "USER_SKILLS_DIR", Path(tmpdir) / ".skills"
                ),
            ):
                result = await tool.execute(name="rogue")

            assert result.error is True
            assert "not in the user skills" in result.output.lower()
