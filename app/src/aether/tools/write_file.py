"""Write File tool — create or overwrite a file."""

from __future__ import annotations

import os

from aether.tools.base import AetherTool, ToolParam, ToolResult


class WriteFileTool(AetherTool):
    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories as needed."
    status_text = "Writing file..."
    parameters = [
        ToolParam(name="path", type="string", description="File path to write to"),
        ToolParam(name="content", type="string", description="Content to write to the file"),
    ]

    def __init__(self, working_dir: str = "."):
        self.working_dir = os.path.abspath(working_dir)

    def _resolve(self, path: str) -> str:
        resolved = os.path.abspath(os.path.join(self.working_dir, path))
        if not resolved.startswith(self.working_dir):
            raise PermissionError(f"Path escapes working directory: {path}")
        return resolved

    async def execute(self, path: str, content: str) -> ToolResult:
        resolved = self._resolve(path)

        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            size = os.path.getsize(resolved)
            return ToolResult.success(
                f"Written {size:,} bytes to {path}",
                path=path, size=size,
            )
        except PermissionError as e:
            return ToolResult.fail(f"Permission denied: {e}")
        except Exception as e:
            return ToolResult.fail(f"Error writing file: {e}")
