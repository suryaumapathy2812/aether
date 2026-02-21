"""Shared base for plugin OAuth token refresh tools.

Each OAuth plugin ships a thin subclass that sets ``plugin_name`` and
``name``.  The actual refresh is handled by the orchestrator's
``GET /api/internal/plugins/{name}/config`` endpoint (which auto-refreshes
expired tokens) followed by ``POST /config/reload`` to push the fresh token
into the agent's PluginContextStore.

The LLM calls the plugin-specific tool (e.g. ``refresh_google_drive_token``).
The tool calls the orchestrator.  The orchestrator refreshes the token in DB
and signals the agent to reload.  No OAuth mechanics live here.
"""

from __future__ import annotations

import logging
import os

import httpx

from aether.tools.base import AetherTool, ToolResult

logger = logging.getLogger(__name__)

_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
_AGENT_USER_ID = os.getenv("AETHER_USER_ID", "")
_AGENT_SECRET = os.getenv("AGENT_SECRET", "")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_AGENT_SECRET}"} if _AGENT_SECRET else {}


class RefreshOAuthTokenTool(AetherTool):
    """Base class for plugin-specific OAuth token refresh tools.

    Subclasses must set:
        name        — tool name the LLM calls (e.g. "refresh_google_drive_token")
        plugin_name — plugin identifier matching AVAILABLE_PLUGINS key
        description — human-readable description for the LLM
    """

    plugin_name: str = ""  # override in subclass
    parameters = []  # no LLM arguments needed — fully automatic
    status_text = "Refreshing access token..."

    async def execute(self, **_) -> ToolResult:  # type: ignore[override]
        """Trigger token refresh via the orchestrator, then reload agent config."""
        if not _ORCHESTRATOR_URL:
            return ToolResult.fail(
                f"Cannot refresh {self.plugin_name} token: ORCHESTRATOR_URL not configured."
            )
        if not _AGENT_USER_ID:
            return ToolResult.fail(
                f"Cannot refresh {self.plugin_name} token: AETHER_USER_ID not configured."
            )

        # Step 1: Hit the config endpoint — orchestrator auto-refreshes if expiring
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_ORCHESTRATOR_URL}/api/internal/plugins/{self.plugin_name}/config",
                    params={"user_id": _AGENT_USER_ID},
                    headers=_auth_headers(),
                )
                if resp.status_code == 404:
                    return ToolResult.fail(
                        f"{self.plugin_name} is not installed or not enabled. "
                        "Please connect it first."
                    )
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Token refresh config fetch failed for %s: %s", self.plugin_name, e
            )
            return ToolResult.fail(
                f"Failed to refresh {self.plugin_name} token (HTTP {e.response.status_code}). "
                "The orchestrator may be unavailable."
            )
        except Exception as e:
            logger.error("Token refresh failed for %s: %s", self.plugin_name, e)
            return ToolResult.fail(f"Failed to refresh {self.plugin_name} token: {e}")

        # Step 2: Reload plugin configs into PluginContextStore (agent self-call)
        _agent_port = os.getenv("PORT", "8000")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"http://localhost:{_agent_port}/config/reload",
                    json={},
                )
        except Exception as e:
            # Non-fatal — token was refreshed in DB; next tool call will re-fetch
            logger.warning(
                "Config reload signal failed for %s (token still refreshed in DB): %s",
                self.plugin_name,
                e,
            )

        logger.info("Token refreshed successfully for plugin: %s", self.plugin_name)
        return ToolResult.success(
            f"{self.plugin_name} access token refreshed successfully. "
            "All tools for this plugin will use the new token."
        )
