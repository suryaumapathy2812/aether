"""Tests for the tool system — base, registry, and core tools."""

import os
import asyncio
import tempfile
import pytest
import pytest_asyncio

from aether.tools.base import AetherTool, ToolParam, ToolResult
from aether.tools.registry import ToolRegistry
from aether.tools.read_file import ReadFileTool
from aether.tools.write_file import WriteFileTool
from aether.tools.list_directory import ListDirectoryTool
from aether.tools.run_command import RunCommandTool


# --- Fixtures ---


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        # Create some test files
        with open(os.path.join(d, "hello.txt"), "w") as f:
            f.write("Hello, World!")
        os.makedirs(os.path.join(d, "subdir"))
        with open(os.path.join(d, "subdir", "nested.txt"), "w") as f:
            f.write("Nested content")
        yield d


@pytest.fixture
def registry(tmp_dir):
    r = ToolRegistry()
    r.register(ReadFileTool(working_dir=tmp_dir))
    r.register(WriteFileTool(working_dir=tmp_dir))
    r.register(ListDirectoryTool(working_dir=tmp_dir))
    r.register(RunCommandTool(working_dir=tmp_dir))
    return r


# --- ToolResult ---


class TestToolResult:
    def test_success(self):
        result = ToolResult.success("output", key="value")
        assert result.output == "output"
        assert result.error is False
        assert result.metadata["key"] == "value"

    def test_fail(self):
        result = ToolResult.fail("boom")
        assert result.output == "boom"
        assert result.error is True


# --- Registry ---


class TestRegistry:
    def test_register_and_list(self, registry):
        names = registry.tool_names()
        assert "read_file" in names
        assert "write_file" in names
        assert "list_directory" in names
        assert "run_command" in names

    def test_get_tool(self, registry):
        tool = registry.get("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_unknown(self, registry):
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_dispatch(self, registry, tmp_dir):
        result = await registry.dispatch("read_file", {"path": "hello.txt"})
        assert result.output == "Hello, World!"
        assert result.error is False

    @pytest.mark.asyncio
    async def test_dispatch_unknown(self, registry):
        result = await registry.dispatch("nope", {})
        assert result.error is True
        assert "Unknown tool" in result.output

    def test_openai_schema(self, registry):
        schemas = registry.to_openai_tools()
        assert len(schemas) >= 4
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

    def test_anthropic_schema(self, registry):
        schemas = registry.to_anthropic_tools()
        assert len(schemas) >= 4
        for schema in schemas:
            assert "name" in schema
            assert "input_schema" in schema

    def test_status_text(self, registry):
        assert registry.get_status_text("read_file") == "Reading file..."
        assert registry.get_status_text("unknown") == "Working..."


# --- ReadFile ---


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing(self, tmp_dir):
        tool = ReadFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="hello.txt")
        assert result.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_nested(self, tmp_dir):
        tool = ReadFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="subdir/nested.txt")
        assert result.output == "Nested content"

    @pytest.mark.asyncio
    async def test_read_not_found(self, tmp_dir):
        tool = ReadFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="nope.txt")
        assert result.error is True
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_escape_jail(self, tmp_dir):
        tool = ReadFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="../../../etc/passwd")
        assert result.error is True


# --- WriteFile ---


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new(self, tmp_dir):
        tool = WriteFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="new.txt", content="new content")
        assert result.error is False
        assert os.path.exists(os.path.join(tmp_dir, "new.txt"))
        with open(os.path.join(tmp_dir, "new.txt")) as f:
            assert f.read() == "new content"

    @pytest.mark.asyncio
    async def test_write_nested(self, tmp_dir):
        tool = WriteFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="deep/nested/file.txt", content="deep")
        assert result.error is False
        assert os.path.exists(os.path.join(tmp_dir, "deep", "nested", "file.txt"))

    @pytest.mark.asyncio
    async def test_write_escape_jail(self, tmp_dir):
        tool = WriteFileTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="../escape.txt", content="bad")
        assert result.error is True


# --- ListDirectory ---


class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_root(self, tmp_dir):
        tool = ListDirectoryTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path=".")
        assert "hello.txt" in result.output
        assert "subdir/" in result.output

    @pytest.mark.asyncio
    async def test_list_subdir(self, tmp_dir):
        tool = ListDirectoryTool(working_dir=tmp_dir)
        result = await tool.safe_execute(path="subdir")
        assert "nested.txt" in result.output


# --- RunCommand ---


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_echo(self, tmp_dir):
        tool = RunCommandTool(working_dir=tmp_dir)
        result = await tool.safe_execute(command="echo hello")
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_blocked_command(self, tmp_dir):
        tool = RunCommandTool(working_dir=tmp_dir)
        result = await tool.safe_execute(command="sudo reboot")
        assert result.error is True
        assert "not allowed" in result.output.lower()

    @pytest.mark.asyncio
    async def test_ls(self, tmp_dir):
        tool = RunCommandTool(working_dir=tmp_dir)
        result = await tool.safe_execute(command="ls")
        assert "hello.txt" in result.output
