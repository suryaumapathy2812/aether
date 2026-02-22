"""Install-package tool — install system packages via apt-get.

Controlled, scoped access to apt-get install. The agent can install packages
the user asks for, but cannot remove, purge, upgrade, or install anything
from a blocked list (e.g. openssh-server, sudo).

Safety model (mirrors openclaw's approach):
  - Only `apt-get install` is exposed — no remove/purge/upgrade/dist-upgrade
  - Package names are validated: alphanumeric + [-_.+] only, no shell chars
  - A blocked list prevents installing packages that open security holes
  - Runs as whatever user the container runs as (root in python:3.12-slim)
  - The LLM can only call this tool when the user explicitly asks for it

Typical flow:
  User: "install ffmpeg"
  Agent: install_package(packages="ffmpeg")
  → apt-get update && apt-get install -y --no-install-recommends ffmpeg
"""

from __future__ import annotations

import asyncio
import logging
import re

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# Only alphanumeric, hyphens, dots, underscores, and plus signs.
# This covers all valid Debian package names and blocks shell metacharacters.
_VALID_PKG_RE = re.compile(r"^[a-z0-9][a-z0-9._+\-]*$")

# Packages that must never be installed — they open security holes or
# fundamentally change the container's security posture.
_BLOCKED_PACKAGES: frozenset[str] = frozenset(
    {
        "sudo",
        "openssh-server",
        "sshd",
        "telnetd",
        "telnet-server",
        "rsh-server",
        "vsftpd",
        "proftpd",
        "pure-ftpd",
        "nfs-kernel-server",
        "samba",
        "x11-apps",
        "xorg",
        "xserver-xorg",
    }
)

# apt-get install can be slow on first run (update + download). 120s is generous.
TIMEOUT = 120


def _parse_packages(packages: str) -> list[str] | str:
    """Parse and validate a package list string.

    Accepts space- or comma-separated names.
    Returns a list of validated names, or an error string.
    """
    # Split on whitespace and/or commas
    raw = re.split(r"[\s,]+", packages.strip())
    names = [p.strip().lower() for p in raw if p.strip()]

    if not names:
        return "No package names provided."

    errors = []
    for name in names:
        if not _VALID_PKG_RE.match(name):
            errors.append(
                f"'{name}' is not a valid package name "
                "(only lowercase letters, digits, hyphens, dots, underscores, and + allowed)."
            )
        elif name in _BLOCKED_PACKAGES:
            errors.append(
                f"'{name}' is blocked for security reasons and cannot be installed."
            )

    if errors:
        return "\n".join(errors)

    return names


class InstallPackageTool(AetherTool):
    """Install one or more system packages via apt-get."""

    name = "install_package"
    description = (
        "Install system packages using apt-get. "
        "Use this when the user asks to install a tool like ffmpeg, imagemagick, "
        "curl, or any other system utility. "
        "Accepts space- or comma-separated package names. "
        "Example: install_package(packages='ffmpeg imagemagick')"
    )
    status_text = "Installing package..."
    parameters = [
        ToolParam(
            name="packages",
            type="string",
            description=(
                "One or more package names to install, space- or comma-separated. "
                "Must be valid Debian/Ubuntu package names. "
                "Examples: 'ffmpeg', 'imagemagick ghostscript', 'curl,wget,jq'"
            ),
            required=True,
        ),
    ]

    async def execute(self, packages: str, **_) -> ToolResult:
        packages = packages.strip()
        if not packages:
            return ToolResult.fail("No packages specified.")

        result = _parse_packages(packages)
        if isinstance(result, str):
            return ToolResult.fail(result)

        pkg_list = result
        pkg_str = " ".join(pkg_list)

        # Run: apt-get update && apt-get install -y --no-install-recommends <pkgs>
        command = (
            f"apt-get update -qq && "
            f"apt-get install -y --no-install-recommends {pkg_str} 2>&1"
        )

        logger.info("Installing packages: %s", pkg_str)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout
            )

            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult.fail(
                    f"Package installation timed out after {TIMEOUT}s. "
                    "The package may still be installing in the background."
                )

            output = stdout.decode("utf-8", errors="replace")

            # Truncate very long apt output (lots of progress lines)
            if len(output) > 10_000:
                # Keep the tail — that's where errors and summaries are
                output = "...(truncated)...\n" + output[-10_000:]

            if proc.returncode != 0:
                logger.error(
                    "Package install failed (exit %d): %s", proc.returncode, pkg_str
                )
                return ToolResult.fail(
                    f"Failed to install {pkg_str} (exit code {proc.returncode}):\n{output}"
                )

            logger.info("Installed packages: %s", pkg_str)

            # Extract the "newly installed" summary line from apt output if present
            summary = ""
            for line in output.splitlines():
                if "newly installed" in line.lower() or "upgraded" in line.lower():
                    summary = line.strip()
                    break

            success_msg = f"Successfully installed: **{pkg_str}**"
            if summary:
                success_msg += f"\n{summary}"

            return ToolResult.success(
                success_msg,
                packages=pkg_list,
                exit_code=proc.returncode,
            )

        except Exception as e:
            logger.error("InstallPackageTool error: %s", e, exc_info=True)
            return ToolResult.fail(f"Installation failed: {e}")
