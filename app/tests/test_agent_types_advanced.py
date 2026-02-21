"""Tests for Agent Type tool filtering logic."""

import pytest
from aether.agents.agent_types import (
    AgentTypeDefinition,
    get_filtered_tools,
    register_agent_type,
)


def test_whitelist_filtering():
    """Agent with allowed_tools only gets those tools."""
    custom_agent = AgentTypeDefinition(
        name="custom_whitelist",
        description="Test",
        system_prompt="Test",
        allowed_tools=["read_file", "list_directory"],
    )
    register_agent_type(custom_agent)

    all_tools = ["read_file", "write_file", "list_directory", "run_command"]
    filtered = get_filtered_tools("custom_whitelist", all_tools)

    assert set(filtered) == {"read_file", "list_directory"}


def test_blacklist_filtering():
    """Agent with denied_tools gets all tools EXCEPT those denied."""
    custom_agent = AgentTypeDefinition(
        name="custom_blacklist",
        description="Test",
        system_prompt="Test",
        denied_tools=["write_file", "run_command"],
    )
    register_agent_type(custom_agent)

    all_tools = ["read_file", "write_file", "list_directory", "run_command"]
    filtered = get_filtered_tools("custom_blacklist", all_tools)

    assert set(filtered) == {"read_file", "list_directory"}


def test_unknown_agent_type_fallback():
    """Unknown agent type falls back to general agent (which denies spawn_task)."""
    all_tools = ["read_file", "write_file", "spawn_task", "check_task"]
    filtered = get_filtered_tools("unknown_agent_type", all_tools)

    # General agent denies spawn_task and check_task
    assert "spawn_task" not in filtered
    assert "check_task" not in filtered
    assert "read_file" in filtered
    assert "write_file" in filtered
