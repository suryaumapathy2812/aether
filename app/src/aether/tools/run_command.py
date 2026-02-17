"""Run Command tool — execute a shell command with safety constraints."""

from __future__ import annotations

import asyncio
import os
import shlex

from aether.tools.base import AetherTool, ToolParam, ToolResult

# Safety: only allow these commands
ALLOWED_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "echo",
    "rm",
    "mkdir",
    "cp",
    "mv",
    "touch",
    "date",
    "pwd",
    "whoami",
    "python",
    "python3",
    "pip",
    "node",
    "npm",
    "npx",
    "git",
    "curl",
    "wget",
    "jq",
    "sort",
    "uniq",
    "tr",
    "cut",
    "du",
    "df",
    "tree",
    "file",
    "which",
    "env",
    "uname",
}

MAX_OUTPUT = 50_000  # 50KB output limit
TIMEOUT = 30  # seconds


class RunCommandTool(AetherTool):
    name = "run_command"
    description = "Execute a shell command and return its output. Commands run in the working directory with a 30-second timeout. Only safe commands are allowed."
    status_text = "Running command..."
    parameters = [
        ToolParam(
            name="command", type="string", description="The shell command to execute"
        ),
    ]

    def __init__(self, working_dir: str = "."):
        self.working_dir = os.path.abspath(working_dir)

    def _check_allowed(self, command: str) -> str | None:
        """Check if the base command is allowed. Returns error msg or None."""
        try:
            parts = shlex.split(command)
        except ValueError:
            return "Invalid command syntax"

        if not parts:
            return "Empty command"

        base = os.path.basename(parts[0])
        if base not in ALLOWED_COMMANDS:
            return f"Command not allowed: {base}. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
        return None

    async def execute(self, command: str) -> ToolResult:
        error = self._check_allowed(command)
        if error:
            return ToolResult.fail(error)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult.fail(f"Command timed out after {TIMEOUT}s")

            output = stdout.decode("utf-8", errors="replace")
            err_output = stderr.decode("utf-8", errors="replace")

            if len(output) > MAX_OUTPUT:
                output = (
                    output[:MAX_OUTPUT] + f"\n... truncated at {MAX_OUTPUT:,} chars"
                )

            if proc.returncode != 0:
                combined = output
                if err_output:
                    combined += f"\nSTDERR:\n{err_output[:5000]}"
                return ToolResult.fail(
                    f"Command exited with code {proc.returncode}:\n{combined}",
                    exit_code=proc.returncode,
                )

            if err_output and not output:
                output = err_output

            return ToolResult.success(
                output or "(no output)", exit_code=proc.returncode
            )

        except Exception as e:
            return ToolResult.fail(f"Failed to execute command: {e}")
