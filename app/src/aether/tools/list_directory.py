"""List Directory tool — show files in a directory."""

from __future__ import annotations

import os

from aether.tools.base import AetherTool, ToolParam, ToolResult

MAX_ENTRIES = 200


class ListDirectoryTool(AetherTool):
    name = "list_directory"
    description = "List files and subdirectories in the given directory path. Shows names, types (file/dir), and sizes."
    status_text = "Listing files..."
    parameters = [
        ToolParam(name="path", type="string", description="Directory path to list", required=False, default="."),
    ]

    def __init__(self, working_dir: str = "."):
        self.working_dir = os.path.abspath(working_dir)

    def _resolve(self, path: str) -> str:
        resolved = os.path.abspath(os.path.join(self.working_dir, path))
        if not resolved.startswith(self.working_dir):
            raise PermissionError(f"Path escapes working directory: {path}")
        return resolved

    async def execute(self, path: str = ".") -> ToolResult:
        resolved = self._resolve(path)

        if not os.path.exists(resolved):
            return ToolResult.fail(f"Directory not found: {path}")

        if not os.path.isdir(resolved):
            return ToolResult.fail(f"Not a directory: {path}")

        try:
            entries = sorted(os.listdir(resolved))
            if len(entries) > MAX_ENTRIES:
                entries = entries[:MAX_ENTRIES]
                truncated = True
            else:
                truncated = False

            lines = []
            for name in entries:
                full = os.path.join(resolved, name)
                if os.path.isdir(full):
                    lines.append(f"  {name}/")
                else:
                    size = os.path.getsize(full)
                    lines.append(f"  {name}  ({size:,} bytes)")

            output = "\n".join(lines) if lines else "(empty directory)"
            if truncated:
                output += f"\n... truncated at {MAX_ENTRIES} entries"

            return ToolResult.success(output, path=path, count=len(entries))
        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error listing directory: {e}")
