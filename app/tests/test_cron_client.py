"""Tests for CronClient — the thin async wrapper around the orchestrator cron API."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aether.kernel.cron import CronClient


# ── Helpers ────────────────────────────────────────────────


def _future(minutes: int = 10) -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)


def _mock_response(status_code: int, body: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception(f"HTTP {status_code}")
    )
    return resp


# ── CronClient.schedule ────────────────────────────────────


class TestCronClientSchedule:
    @pytest.mark.asyncio
    async def test_schedule_returns_job_id_on_success(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )
        mock_resp = _mock_response(200, {"job_id": "abc123", "status": "scheduled"})

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_resp))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            job_id = await client.schedule(
                instruction="Remind the user about their meeting.",
                run_at=_future(30),
            )

        assert job_id == "abc123"

    @pytest.mark.asyncio
    async def test_schedule_sends_correct_payload(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )
        run_at = _future(60)
        mock_resp = _mock_response(200, {"job_id": "xyz", "status": "scheduled"})
        captured = {}

        async def fake_post(url, params=None, json=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["json"] = json
            return mock_resp

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(side_effect=fake_post))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.schedule(
                instruction="Token refresh needed.",
                run_at=run_at,
                plugin="google-drive",
                interval_s=3000,
            )

        assert captured["params"] == {"user_id": "user-1"}
        assert captured["json"]["instruction"] == "Token refresh needed."
        assert captured["json"]["plugin"] == "google-drive"
        assert captured["json"]["interval_s"] == 3000
        assert "run_at" in captured["json"]

    @pytest.mark.asyncio
    async def test_schedule_returns_none_when_no_orchestrator_url(self):
        client = CronClient(orchestrator_url="", user_id="user-1")
        job_id = await client.schedule(
            instruction="anything",
            run_at=_future(),
        )
        assert job_id is None

    @pytest.mark.asyncio
    async def test_schedule_returns_none_on_http_error(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )
        mock_resp = _mock_response(500, {})

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_resp))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            job_id = await client.schedule(
                instruction="anything",
                run_at=_future(),
            )

        assert job_id is None

    @pytest.mark.asyncio
    async def test_schedule_returns_none_on_network_error(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    post=AsyncMock(side_effect=Exception("connection refused"))
                )
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            job_id = await client.schedule(
                instruction="anything",
                run_at=_future(),
            )

        assert job_id is None

    @pytest.mark.asyncio
    async def test_schedule_adds_utc_if_naive_datetime(self):
        """Naive datetimes should be treated as UTC without raising."""
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )
        naive_dt = datetime.utcnow() + timedelta(hours=1)  # no tzinfo
        mock_resp = _mock_response(200, {"job_id": "naive-ok"})
        captured = {}

        async def fake_post(url, params=None, json=None, headers=None):
            captured["json"] = json
            return mock_resp

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(side_effect=fake_post))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            job_id = await client.schedule(instruction="test", run_at=naive_dt)

        assert job_id == "naive-ok"
        # ISO string should contain timezone info
        assert "+00:00" in captured["json"]["run_at"]


# ── CronClient.cancel ──────────────────────────────────────


class TestCronClientCancel:
    @pytest.mark.asyncio
    async def test_cancel_returns_true_on_success(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )
        mock_resp = _mock_response(200, {"job_id": "abc123", "status": "cancelled"})

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(delete=AsyncMock(return_value=mock_resp))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.cancel("abc123")

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_returns_false_when_no_orchestrator_url(self):
        client = CronClient(orchestrator_url="", user_id="user-1")
        result = await client.cancel("abc123")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_returns_false_on_404(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-1",
        )
        mock_resp = _mock_response(404, {"detail": "not found"})

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(delete=AsyncMock(return_value=mock_resp))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.cancel("missing-job")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_sends_correct_url_and_params(self):
        client = CronClient(
            orchestrator_url="http://orchestrator:8080",
            user_id="user-99",
        )
        mock_resp = _mock_response(200, {"status": "cancelled"})
        captured = {}

        async def fake_delete(url, params=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            return mock_resp

        with patch("aether.kernel.cron.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(delete=AsyncMock(side_effect=fake_delete))
            )
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.cancel("job-xyz")

        assert "job-xyz" in captured["url"]
        assert captured["params"] == {"user_id": "user-99"}
