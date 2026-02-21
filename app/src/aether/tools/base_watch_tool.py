"""Base class for plugin watch setup/renewal tools.

Each plugin that supports push notifications ships two thin subclasses:

    SetupXxxWatchTool  — registers the watch with the external service after
                         OAuth connect. Called once by the cron system (one-shot
                         job scheduled 30 s after OAuth callback).

    RenewXxxWatchTool  — renews the watch before it expires. Scheduled by
                         SetupXxxWatchTool itself using CronClient, at
                         (expiry - 1 day). Recurring at renew_interval seconds.

Both subclasses must set:
    name         — tool name the LLM calls
    plugin_name  — plugin identifier matching AVAILABLE_PLUGINS key
    description  — human-readable description for the LLM

The base class provides:
    _get_token()          — access token from plugin context
    _register_watch()     — store watch registration in orchestrator DB
    _schedule_renewal()   — schedule renewal cron job via CronClient
    _auth_headers()       — bearer auth headers for plugin API calls
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import httpx

from aether.kernel.cron import CronClient
from aether.tools.base import AetherTool, ToolResult

logger = logging.getLogger(__name__)

_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
_AGENT_USER_ID = os.getenv("AETHER_USER_ID", "")
_AGENT_SECRET = os.getenv("AGENT_SECRET", "")


def _agent_auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_AGENT_SECRET}"} if _AGENT_SECRET else {}


class BaseWatchTool(AetherTool):
    """Shared helpers for watch setup and renewal tools."""

    plugin_name: str = ""  # override in subclass
    parameters = []  # no LLM arguments — fully automatic
    status_text = "Setting up push notifications..."

    # ── Helpers ──────────────────────────────────────────

    def _get_token(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("access_token") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        token = self._get_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def _register_watch(
        self,
        watch_id: str,
        resource_id: str,
        protocol: str,
        expires_at: int | None,
    ) -> None:
        """
        Persist watch registration to the orchestrator DB.

        The orchestrator exposes POST /api/internal/watches so the agent
        can record the watch without direct DB access.
        """
        if not _ORCHESTRATOR_URL or not _AGENT_USER_ID:
            logger.warning(
                "Cannot register watch: ORCHESTRATOR_URL or AETHER_USER_ID not set"
            )
            return

        payload = {
            "user_id": _AGENT_USER_ID,
            "plugin_name": self.plugin_name,
            "protocol": protocol,
            "watch_id": watch_id,
            "resource_id": resource_id,
        }
        if expires_at:
            payload["expires_at"] = datetime.fromtimestamp(
                expires_at / 1000, tz=timezone.utc
            ).isoformat()

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{_ORCHESTRATOR_URL}/api/internal/watches",
                    json=payload,
                    headers=_agent_auth_headers(),
                )
                resp.raise_for_status()
                logger.info("Watch registered: %s/%s", self.plugin_name, watch_id)
        except Exception as e:
            logger.warning("Failed to register watch for %s: %s", self.plugin_name, e)

    async def _schedule_renewal(
        self,
        renew_tool: str,
        expires_at_ms: int | None,
        renew_interval_s: int = 518400,
    ) -> None:
        """
        Schedule a recurring renewal cron job via CronClient.

        Fires at (expires_at - 1 day) or in renew_interval_s seconds,
        whichever is sooner.
        """
        cron = CronClient()

        if expires_at_ms:
            # Schedule 1 day before expiry
            expiry_s = expires_at_ms / 1000
            run_at = datetime.fromtimestamp(expiry_s - 86400, tz=timezone.utc)
        else:
            run_at = datetime.fromtimestamp(
                time.time() + renew_interval_s, tz=timezone.utc
            )

        instruction = (
            f"Watch renewal: the {self.plugin_name} push notification watch is about "
            f"to expire. Call the `{renew_tool}` tool now to renew it and keep "
            f"receiving real-time events."
        )
        job_id = await cron.schedule(
            instruction=instruction,
            run_at=run_at,
            plugin=self.plugin_name,
            interval_s=renew_interval_s,
        )
        if job_id:
            logger.info(
                "Watch renewal job scheduled: %s (run_at=%s, interval=%ds)",
                self.plugin_name,
                run_at.isoformat(),
                renew_interval_s,
            )
