"""CronClient — thin async wrapper around the orchestrator's cron API.

Used by agent-side tools (e.g. ScheduleReminderTool) to schedule and
cancel jobs without knowing the orchestrator's internal details.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
_AGENT_USER_ID = os.getenv("AETHER_USER_ID", "")
_AGENT_SECRET = os.getenv("AGENT_SECRET", "")


def _auth_headers() -> dict[str, str]:
    if _AGENT_SECRET:
        return {"Authorization": f"Bearer {_AGENT_SECRET}"}
    return {}


class CronClient:
    """Async client for the orchestrator's /api/internal/cron endpoints.

    All methods are fire-and-forget friendly — they log warnings on failure
    rather than raising, so a cron scheduling error never crashes a tool call.
    """

    def __init__(
        self,
        orchestrator_url: str = _ORCHESTRATOR_URL,
        user_id: str = _AGENT_USER_ID,
    ) -> None:
        self._base = orchestrator_url.rstrip("/")
        self._user_id = user_id

    async def schedule(
        self,
        instruction: str,
        run_at: datetime,
        plugin: str | None = None,
        interval_s: int | None = None,
    ) -> str | None:
        """Schedule a cron job. Returns the job_id on success, None on failure.

        Args:
            instruction: Plain-language instruction for the LLM (e.g. "Remind
                         the user their meeting starts in 10 minutes.").
            run_at:      When to fire the job (timezone-aware datetime).
            plugin:      Optional plugin hint (e.g. "google-drive").
            interval_s:  Recurrence interval in seconds. None = one-shot.
        """
        if not self._base:
            logger.warning("CronClient: ORCHESTRATOR_URL not set — cannot schedule job")
            return None

        # Ensure UTC ISO-8601
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)

        payload: dict[str, Any] = {
            "instruction": instruction,
            "run_at": run_at.isoformat(),
        }
        if plugin:
            payload["plugin"] = plugin
        if interval_s is not None:
            payload["interval_s"] = interval_s

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self._base}/api/internal/cron/schedule",
                    params={"user_id": self._user_id},
                    json=payload,
                    headers=_auth_headers(),
                )
                resp.raise_for_status()
                job_id: str = resp.json().get("job_id", "")
                logger.info(
                    "Cron job scheduled: %s (run_at=%s, interval_s=%s)",
                    job_id,
                    run_at.isoformat(),
                    interval_s,
                )
                return job_id
        except Exception as e:
            logger.warning("CronClient.schedule failed: %s", e)
            return None

    async def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled job. Returns True on success."""
        if not self._base:
            logger.warning("CronClient: ORCHESTRATOR_URL not set — cannot cancel job")
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.delete(
                    f"{self._base}/api/internal/cron/{job_id}",
                    params={"user_id": self._user_id},
                    headers=_auth_headers(),
                )
                resp.raise_for_status()
                logger.info("Cron job cancelled: %s", job_id)
                return True
        except Exception as e:
            logger.warning("CronClient.cancel(%s) failed: %s", job_id, e)
            return False
