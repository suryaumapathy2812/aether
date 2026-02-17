"""Tests for the skill system."""

import os
import tempfile
import pytest

from aether.skills.loader import SkillLoader


@pytest.fixture
def skills_dir():
    with tempfile.TemporaryDirectory() as d:
        # Create a valid skill
        skill_dir = os.path.join(d, "test-skill")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write("""---
name: test-skill
description: A test skill for unit testing. Handles testing and verification tasks.
---

## Test Skill

This is the body of the test skill.
Use the `run_command` tool to execute tests.
""")

        # Create another skill
        skill_dir2 = os.path.join(d, "web-skill")
        os.makedirs(skill_dir2)
        with open(os.path.join(skill_dir2, "SKILL.md"), "w") as f:
            f.write("""---
name: web-research
description: Search the web and summarize results. Use for research and information gathering.
---

## Web Research

Use `web_search` to find information.
""")

        yield d


class TestSkillLoader:
    def test_discover(self, skills_dir):
        loader = SkillLoader(skills_dirs=[skills_dir])
        count = loader.discover()
        assert count == 2

    def test_get_skill(self, skills_dir):
        loader = SkillLoader(skills_dirs=[skills_dir])
        loader.discover()
        skill = loader.get("test-skill")
        assert skill is not None
        assert skill.name == "test-skill"
        assert "testing" in skill.description.lower()

    def test_skill_content_lazy(self, skills_dir):
        loader = SkillLoader(skills_dirs=[skills_dir])
        loader.discover()
        skill = loader.get("test-skill")
        # Content is loaded on demand
        assert skill._content is None
        content = skill.content
        assert "run_command" in content
        assert skill._content is not None  # Now cached

    def test_all_skills(self, skills_dir):
        loader = SkillLoader(skills_dirs=[skills_dir])
        loader.discover()
        all_skills = loader.all()
        assert len(all_skills) == 2
        names = {s.name for s in all_skills}
        assert "test-skill" in names
        assert "web-research" in names

    def test_match(self, skills_dir):
        loader = SkillLoader(skills_dirs=[skills_dir])
        loader.discover()
        matches = loader.match("search web information")
        assert len(matches) > 0
        assert matches[0].name == "web-research"

    def test_system_prompt_section(self, skills_dir):
        loader = SkillLoader(skills_dirs=[skills_dir])
        loader.discover()
        section = loader.get_system_prompt_section()
        assert "test-skill" in section
        assert "web-research" in section

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            loader = SkillLoader(skills_dirs=[d])
            count = loader.discover()
            assert count == 0

    def test_nonexistent_dir(self):
        loader = SkillLoader(skills_dirs=["/nonexistent/path"])
        count = loader.discover()
        assert count == 0
