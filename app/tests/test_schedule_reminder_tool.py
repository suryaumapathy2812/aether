"""Tests for ScheduleReminderTool — validates input, delegates to CronClient."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

from aether.tools.schedule_reminder import ScheduleReminderTool


# ── Helpers ────────────────────────────────────────────────


def _future_iso(minutes: int = 30) -> str:
    dt = datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _past_iso(minutes: int = 5) -> str:
    dt = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Tests ──────────────────────────────────────────────────


class TestScheduleReminderTool:
    def setup_method(self):
        self.tool = ScheduleReminderTool()

    # ── Metadata ──

    def test_tool_name(self):
        assert self.tool.name == "schedule_reminder"

    def test_tool_has_two_params(self):
        assert len(self.tool.parameters) == 2
        names = [p.name for p in self.tool.parameters]
        assert "message" in names
        assert "iso_datetime" in names

    def test_both_params_required(self):
        for p in self.tool.parameters:
            assert p.required is True

    # ── Happy path ──

    @pytest.mark.asyncio
    async def test_schedules_successfully(self):
        self.tool._cron = AsyncMock()
        self.tool._cron.schedule = AsyncMock(return_value="job-abc")

        result = await self.tool.execute(
            message="Remind the user about their dentist appointment.",
            iso_datetime=_future_iso(60),
        )

        assert result.error is False
        assert "job-abc" in str(result.metadata)
        assert "Reminder scheduled" in result.output
        self.tool._cron.schedule.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_instruction_wraps_message(self):
        """The instruction sent to the cron must include the original message."""
        self.tool._cron = AsyncMock()
        self.tool._cron.schedule = AsyncMock(return_value="job-xyz")
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            return "job-xyz"

        self.tool._cron.schedule = AsyncMock(side_effect=capture)

        await self.tool.execute(
            message="Meeting starts in 5 minutes.",
            iso_datetime=_future_iso(5),
        )

        assert "Meeting starts in 5 minutes." in captured["instruction"]
        assert captured["interval_s"] is None  # one-shot
        assert captured["plugin"] is None

    @pytest.mark.asyncio
    async def test_z_suffix_iso_parsed_correctly(self):
        """'Z' suffix (UTC shorthand) must be accepted."""
        self.tool._cron = AsyncMock()
        self.tool._cron.schedule = AsyncMock(return_value="job-z")

        result = await self.tool.execute(
            message="Test Z suffix.",
            iso_datetime=_future_iso(10),
        )

        assert result.error is False

    # ── Validation errors ──

    @pytest.mark.asyncio
    async def test_rejects_past_datetime(self):
        result = await self.tool.execute(
            message="Too late.",
            iso_datetime=_past_iso(10),
        )

        assert result.error is True
        assert "past" in result.output.lower()

    @pytest.mark.asyncio
    async def test_rejects_invalid_datetime_format(self):
        result = await self.tool.execute(
            message="Bad format.",
            iso_datetime="not-a-date",
        )

        assert result.error is True
        assert "invalid" in result.output.lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_datetime(self):
        result = await self.tool.execute(
            message="Empty dt.",
            iso_datetime="",
        )

        assert result.error is True

    # ── CronClient failure ──

    @pytest.mark.asyncio
    async def test_returns_error_when_cron_client_fails(self):
        self.tool._cron = AsyncMock()
        self.tool._cron.schedule = AsyncMock(return_value=None)  # None = failure

        result = await self.tool.execute(
            message="Remind me.",
            iso_datetime=_future_iso(20),
        )

        assert result.error is True
        assert "failed" in result.output.lower()

    @pytest.mark.asyncio
    async def test_cron_not_called_for_past_datetime(self):
        """CronClient.schedule must NOT be called when validation fails."""
        self.tool._cron = AsyncMock()
        self.tool._cron.schedule = AsyncMock(return_value="should-not-be-called")

        await self.tool.execute(
            message="Past reminder.",
            iso_datetime=_past_iso(1),
        )

        self.tool._cron.schedule.assert_not_awaited()
