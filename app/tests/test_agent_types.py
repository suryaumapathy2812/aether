"""Tests for Agent Type System — specialized agent definitions."""

import pytest

from aether.agents.agent_types import (
    AgentTypeDefinition,
    get_agent_type,
    get_filtered_tools,
    list_agent_types,
    register_agent_type,
)


# ─── get_agent_type Tests ─────────────────────────────────────


def test_get_general_agent():
    defn = get_agent_type("general")
    assert defn.name == "general"
    assert defn.max_iterations == 25
    assert "spawn_task" in defn.denied_tools


def test_get_explore_agent():
    defn = get_agent_type("explore")
    assert defn.name == "explore"
    assert defn.allowed_tools is not None
    assert "read_file" in defn.allowed_tools
    assert "run_command" not in (defn.allowed_tools or [])


def test_get_planner_agent():
    defn = get_agent_type("planner")
    assert defn.name == "planner"
    assert defn.allowed_tools is not None
    assert "read_file" in defn.allowed_tools


def test_get_default_agent():
    defn = get_agent_type("default")
    assert defn.name == "general"  # default maps to general


def test_get_unknown_falls_back_to_general():
    defn = get_agent_type("nonexistent_type")
    assert defn.name == "general"


# ─── get_filtered_tools Tests ─────────────────────────────────


def test_filtered_tools_general():
    all_tools = ["read_file", "write_file", "run_command", "spawn_task", "check_task"]
    filtered = get_filtered_tools("general", all_tools)

    assert "read_file" in filtered
    assert "write_file" in filtered
    assert "run_command" in filtered
    assert "spawn_task" not in filtered  # Denied
    assert "check_task" not in filtered  # Denied


def test_filtered_tools_explore():
    all_tools = [
        "read_file",
        "write_file",
        "run_command",
        "list_directory",
        "web_search",
    ]
    filtered = get_filtered_tools("explore", all_tools)

    assert "read_file" in filtered
    assert "list_directory" in filtered
    assert "web_search" in filtered
    assert "write_file" not in filtered  # Not in allowed list
    assert "run_command" not in filtered  # Not in allowed list


def test_filtered_tools_planner():
    all_tools = ["read_file", "write_file", "run_command", "list_directory"]
    filtered = get_filtered_tools("planner", all_tools)

    assert "read_file" in filtered
    assert "list_directory" in filtered
    assert "write_file" not in filtered
    assert "run_command" not in filtered


# ─── register_agent_type Tests ────────────────────────────────


def test_register_custom_agent_type():
    custom = AgentTypeDefinition(
        name="test_custom",
        description="Custom test agent",
        system_prompt="You are a test agent.",
        allowed_tools=["read_file"],
        max_iterations=5,
    )
    register_agent_type(custom)

    retrieved = get_agent_type("test_custom")
    assert retrieved.name == "test_custom"
    assert retrieved.max_iterations == 5
    assert retrieved.allowed_tools == ["read_file"]


# ─── list_agent_types Tests ──────────────────────────────────


def test_list_agent_types():
    types = list_agent_types()
    names = [t.name for t in types]

    assert "general" in names
    assert "explore" in names
    assert "planner" in names


def test_list_agent_types_no_duplicates():
    types = list_agent_types()
    names = [t.name for t in types]
    assert len(names) == len(set(names))


# ─── AgentTypeDefinition Tests ────────────────────────────────


def test_agent_type_is_frozen():
    defn = get_agent_type("general")
    with pytest.raises(AttributeError):
        defn.name = "modified"  # type: ignore


def test_agent_type_defaults():
    defn = AgentTypeDefinition(
        name="minimal",
        description="Minimal agent",
        system_prompt="You are minimal.",
    )
    assert defn.allowed_tools is None
    assert defn.denied_tools == []
    assert defn.model_override is None
    assert defn.can_spawn_sub_agents is False


def test_explore_cannot_spawn():
    defn = get_agent_type("explore")
    assert defn.can_spawn_sub_agents is False


def test_general_cannot_spawn():
    defn = get_agent_type("general")
    assert defn.can_spawn_sub_agents is False
