"""Read File tool — read a file's contents."""

from __future__ import annotations

import os

from aether.tools.base import AetherTool, ToolParam, ToolResult

MAX_FILE_SIZE = 100_000  # 100KB limit


class ReadFileTool(AetherTool):
    name = "read_file"
    description = "Read the contents of a file at the given path. Returns the file text."
    status_text = "Reading file..."
    parameters = [
        ToolParam(name="path", type="string", description="Absolute or relative file path to read"),
    ]

    def __init__(self, working_dir: str = "."):
        self.working_dir = os.path.abspath(working_dir)

    def _resolve(self, path: str) -> str:
        resolved = os.path.abspath(os.path.join(self.working_dir, path))
        if not resolved.startswith(self.working_dir):
            raise PermissionError(f"Path escapes working directory: {path}")
        return resolved

    async def execute(self, path: str) -> ToolResult:
        resolved = self._resolve(path)

        if not os.path.exists(resolved):
            return ToolResult.fail(f"File not found: {path}")

        if not os.path.isfile(resolved):
            return ToolResult.fail(f"Not a file: {path}")

        size = os.path.getsize(resolved)
        if size > MAX_FILE_SIZE:
            return ToolResult.fail(f"File too large: {size:,} bytes (max {MAX_FILE_SIZE:,})")

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return ToolResult.success(content, path=path, size=size)
        except Exception as e:
            return ToolResult.fail(f"Error reading file: {e}")
