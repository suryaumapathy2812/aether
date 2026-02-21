"""ScheduleReminderTool — let the LLM schedule future user reminders.

The LLM calls this tool when the user asks to be reminded of something at a
future time.  The tool posts a one-shot cron job to the orchestrator; when
the job fires, the orchestrator delivers a /cron_event back to the agent and
the LLM relays the reminder to the user in natural language.

Example LLM call:
    schedule_reminder(
        message="Remind the user their standup meeting starts in 5 minutes.",
        iso_datetime="2026-02-21T09:55:00Z",
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from aether.kernel.cron import CronClient
from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)


class ScheduleReminderTool(AetherTool):
    """Schedule a one-shot reminder to be delivered to the user at a future time."""

    name = "schedule_reminder"
    description = (
        "Schedule a reminder to be delivered to the user at a specific future date and time. "
        "Use this when the user asks to be reminded of something later. "
        "The reminder will be spoken or shown to the user at the requested time."
    )
    parameters = [
        ToolParam(
            name="message",
            type="string",
            description=(
                "The reminder message to deliver to the user. "
                "Write it as a natural spoken sentence, e.g. "
                "'Remind the user their dentist appointment is in 10 minutes.'"
            ),
            required=True,
        ),
        ToolParam(
            name="iso_datetime",
            type="string",
            description=(
                "The exact date and time to deliver the reminder, in ISO-8601 format "
                "(e.g. '2026-02-21T15:30:00Z'). Must be in the future."
            ),
            required=True,
        ),
    ]

    def __init__(self) -> None:
        self._cron = CronClient()

    async def execute(self, message: str, iso_datetime: str) -> ToolResult:  # type: ignore[override]
        """Schedule the reminder via the orchestrator cron API."""
        # Parse and validate the datetime
        try:
            run_at = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
        except ValueError:
            return ToolResult.fail(
                f"Invalid datetime format: {iso_datetime!r}. Use ISO-8601, e.g. '2026-02-21T15:30:00Z'."
            )

        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        if run_at <= now:
            return ToolResult.fail(
                f"The requested time {iso_datetime} is in the past. Please provide a future time."
            )

        # Build the instruction the LLM will receive when the cron fires
        instruction = (
            f"Scheduled reminder: {message} "
            f"Please relay this reminder to the user now in a natural, friendly way."
        )

        job_id = await self._cron.schedule(
            instruction=instruction,
            run_at=run_at,
            plugin=None,  # not plugin-specific
            interval_s=None,  # one-shot
        )

        if not job_id:
            return ToolResult.fail(
                "Failed to schedule the reminder — the orchestrator could not be reached. Please try again."
            )

        friendly_time = run_at.strftime("%B %d, %Y at %H:%M UTC")
        logger.info(
            "Reminder scheduled: job_id=%s, run_at=%s", job_id, run_at.isoformat()
        )

        return ToolResult.success(
            f"Reminder scheduled for {friendly_time}. I'll remind you then.",
            job_id=job_id,
            run_at=run_at.isoformat(),
        )

        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        if run_at <= now:
            return ToolResult(
                success=False,
                output=f"The requested time {iso_datetime} is in the past. Please provide a future time.",
            )

        # Build the instruction the LLM will receive when the cron fires
        instruction = (
            f"Scheduled reminder: {message} "
            f"Please relay this reminder to the user now in a natural, friendly way."
        )

        job_id = await self._cron.schedule(
            instruction=instruction,
            run_at=run_at,
            plugin=None,  # not plugin-specific
            interval_s=None,  # one-shot
        )

        if not job_id:
            return ToolResult(
                success=False,
                output="Failed to schedule the reminder — the orchestrator could not be reached. Please try again.",
            )

        friendly_time = run_at.strftime("%B %d, %Y at %H:%M UTC")
        logger.info(
            "Reminder scheduled: job_id=%s, run_at=%s", job_id, run_at.isoformat()
        )

        return ToolResult(
            success=True,
            output=f"Reminder scheduled for {friendly_time}. I'll remind you then.",
            metadata={"job_id": job_id, "run_at": run_at.isoformat()},
        )
