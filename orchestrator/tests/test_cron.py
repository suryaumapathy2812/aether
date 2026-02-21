"""Tests for the orchestrator cron system.

Covers:
- POST /api/internal/cron/schedule
- DELETE /api/internal/cron/{job_id}
- _ensure_agent_running() helper
- _cron_runner() background loop
- _upsert_token_refresh_job() auto-schedule on OAuth

All DB operations are mocked — no real Postgres needed.
Agent HTTP calls are mocked — no real agent container needed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from tests.conftest import make_record, MockPool


# ── Shared fixtures ────────────────────────────────────────


@pytest.fixture
def mock_pool():
    return MockPool()


@pytest.fixture
def app_client(mock_pool):
    """TestClient with mocked DB, auth bypassed, startup events disabled."""

    async def _get_pool():
        return mock_pool

    async def _noop(*a, **kw):
        pass

    with (
        patch("src.db.get_pool", _get_pool),
        patch("src.auth.get_pool", _get_pool),
        patch("src.main.get_pool", _get_pool),
        patch("src.main.bootstrap_schema", _noop),
        patch("src.main.close_pool", _noop),
        patch("src.main.reconcile_containers", AsyncMock()),
        patch("src.main.ensure_shared_models", AsyncMock()),
        patch("src.main._idle_reaper", AsyncMock()),
        patch("src.main._deferred_flusher", AsyncMock()),
        patch("src.main._cron_runner", AsyncMock()),
    ):
        from src.main import app
        from src.auth import get_user_id

        async def _mock_user():
            return "user-1"

        app.dependency_overrides[get_user_id] = _mock_user

        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool

        app.dependency_overrides.clear()


def _agent_auth_header():
    return {"Authorization": "Bearer test-agent-secret"}


def _future_iso(minutes: int = 30) -> str:
    dt = datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)
    return dt.isoformat()


# ── POST /api/internal/cron/schedule ──────────────────────


class TestCronScheduleRoute:
    def test_schedule_creates_job(self, app_client, mock_pool):
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/cron/schedule",
                params={"user_id": "user-1"},
                json={
                    "instruction": "Refresh the Google Drive token now.",
                    "run_at": _future_iso(50),
                    "plugin": "google-drive",
                    "interval_s": 3000,
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "scheduled"
        pool.execute.assert_awaited()

    def test_schedule_one_shot_no_interval(self, app_client, mock_pool):
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/cron/schedule",
                params={"user_id": "user-1"},
                json={
                    "instruction": "Remind the user about their meeting.",
                    "run_at": _future_iso(10),
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "scheduled"

    def test_schedule_rejects_invalid_datetime(self, app_client):
        client, _ = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/cron/schedule",
                params={"user_id": "user-1"},
                json={
                    "instruction": "Bad datetime.",
                    "run_at": "not-a-date",
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 400

    def test_schedule_rejects_zero_interval(self, app_client):
        client, _ = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/cron/schedule",
                params={"user_id": "user-1"},
                json={
                    "instruction": "Bad interval.",
                    "run_at": _future_iso(10),
                    "interval_s": 0,
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 400

    def test_schedule_rejects_negative_interval(self, app_client):
        client, _ = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/cron/schedule",
                params={"user_id": "user-1"},
                json={
                    "instruction": "Negative interval.",
                    "run_at": _future_iso(10),
                    "interval_s": -60,
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 400

    def test_schedule_requires_agent_secret(self, app_client):
        client, _ = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/cron/schedule",
                params={"user_id": "user-1"},
                json={"instruction": "No auth.", "run_at": _future_iso()},
                # No Authorization header
            )

        assert resp.status_code in (401, 403)


# ── DELETE /api/internal/cron/{job_id} ────────────────────


class TestCronCancelRoute:
    def test_cancel_existing_job(self, app_client, mock_pool):
        client, pool = app_client
        pool.execute = AsyncMock(return_value="UPDATE 1")

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.delete(
                "/api/internal/cron/job-abc",
                params={"user_id": "user-1"},
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-abc"
        assert data["status"] == "cancelled"

    def test_cancel_nonexistent_job_returns_404(self, app_client, mock_pool):
        client, pool = app_client
        pool.execute = AsyncMock(return_value="UPDATE 0")

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.delete(
                "/api/internal/cron/missing-job",
                params={"user_id": "user-1"},
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 404

    def test_cancel_requires_agent_secret(self, app_client):
        client, _ = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.delete(
                "/api/internal/cron/job-abc",
                params={"user_id": "user-1"},
                # No Authorization header
            )

        assert resp.status_code in (401, 403)


# ── _ensure_agent_running ──────────────────────────────────


class TestEnsureAgentRunning:
    @pytest.mark.asyncio
    async def test_returns_agent_if_already_running(self, mock_pool):
        mock_pool.fetchrow = AsyncMock(
            return_value=make_record(host="agent", port=8000, status="running")
        )

        with patch("src.main.get_pool", AsyncMock(return_value=mock_pool)):
            from src.main import _ensure_agent_running

            result = await _ensure_agent_running("user-1")

        assert result is not None
        assert result["host"] == "agent"
        assert result["port"] == 8000

    @pytest.mark.asyncio
    async def test_returns_none_if_agent_never_comes_up(self, mock_pool):
        """Agent stays in 'starting' status — should time out and return None."""
        mock_pool.fetchrow = AsyncMock(
            return_value=make_record(host="agent", port=8000, status="starting")
        )

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch("src.main._local_agent_target", return_value=None),
            patch("src.main._ensure_agent", AsyncMock()),
            patch("asyncio.sleep", AsyncMock()),  # skip real sleeps
        ):
            from src.main import _ensure_agent_running

            result = await _ensure_agent_running("user-1", timeout_s=2)

        assert result is None

    @pytest.mark.asyncio
    async def test_starts_agent_when_stopped(self, mock_pool):
        """When agent is stopped, _ensure_agent should be called."""
        # First call: stopped; subsequent calls: running
        mock_pool.fetchrow = AsyncMock(
            side_effect=[
                make_record(host="agent", port=8000, status="stopped"),
                make_record(host="agent", port=8000, status="running"),
            ]
        )

        mock_ensure = AsyncMock()

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch("src.main._local_agent_target", return_value=None),
            patch("src.main._ensure_agent", mock_ensure),
            patch("asyncio.sleep", AsyncMock()),
        ):
            from src.main import _ensure_agent_running

            result = await _ensure_agent_running("user-1", timeout_s=5)

        mock_ensure.assert_awaited_once_with("user-1")
        assert result is not None
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_local_agent_mode_skips_provisioning(self, mock_pool):
        """In local-agent mode, no container management — return whatever is registered."""
        mock_pool.fetchrow = AsyncMock(
            return_value=make_record(host="localhost", port=8000, status="stopped")
        )

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch(
                "src.main._local_agent_target",
                return_value={"host": "localhost", "port": 8000},
            ),
        ):
            from src.main import _ensure_agent_running

            result = await _ensure_agent_running("user-1")

        # Should return the registered agent without trying to start it
        assert result is not None


# ── _cron_runner ───────────────────────────────────────────


class TestCronRunner:
    @pytest.mark.asyncio
    async def test_delivers_due_job_to_agent(self, mock_pool):
        """A due job should be POSTed to /cron_event on the agent."""
        due_job = make_record(
            id="job-1",
            user_id="user-1",
            plugin="google-drive",
            instruction="Refresh the token.",
            interval_s=3000,
        )
        mock_pool.fetch = AsyncMock(
            side_effect=[
                [due_job],  # first poll: one due job
                [],  # second poll: nothing (loop exits via cancel)
            ]
        )

        agent_row = {"host": "agent", "port": 8000}
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        sleep_count = 0

        async def fake_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch("src.main._ensure_agent_running", AsyncMock(return_value=agent_row)),
            patch("src.main._httpx") as mock_httpx,
            patch("asyncio.sleep", fake_sleep),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            from src.main import _cron_runner

            with pytest.raises(asyncio.CancelledError):
                await _cron_runner()

        # Verify /cron_event was called with the right payload
        mock_client.post.assert_awaited()
        call_args = mock_client.post.call_args
        assert "/cron_event" in call_args[0][0]
        assert call_args[1]["json"]["plugin"] == "google-drive"
        assert call_args[1]["json"]["instruction"] == "Refresh the token."

    @pytest.mark.asyncio
    async def test_advances_recurring_job_after_delivery(self, mock_pool):
        """After delivering a recurring job, run_at should be advanced."""
        due_job = make_record(
            id="job-recurring",
            user_id="user-1",
            plugin="google-drive",
            instruction="Refresh.",
            interval_s=3000,
        )
        mock_pool.fetch = AsyncMock(side_effect=[[due_job], []])
        mock_pool.execute = AsyncMock(return_value=None)

        agent_row = {"host": "agent", "port": 8000}
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        sleep_count = 0

        async def fake_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch("src.main._ensure_agent_running", AsyncMock(return_value=agent_row)),
            patch("src.main._httpx") as mock_httpx,
            patch("asyncio.sleep", fake_sleep),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            from src.main import _cron_runner

            with pytest.raises(asyncio.CancelledError):
                await _cron_runner()

        # execute should have been called to advance run_at (not disable)
        execute_calls = mock_pool.execute.call_args_list
        sql_calls = [str(c) for c in execute_calls]
        assert any("run_at" in s and "interval" in s.lower() for s in sql_calls)
        assert not any("enabled = false" in s for s in sql_calls)

    @pytest.mark.asyncio
    async def test_disables_one_shot_job_after_delivery(self, mock_pool):
        """After delivering a one-shot job (interval_s=None), it should be disabled."""
        due_job = make_record(
            id="job-oneshot",
            user_id="user-1",
            plugin=None,
            instruction="Remind the user.",
            interval_s=None,
        )
        mock_pool.fetch = AsyncMock(side_effect=[[due_job], []])
        mock_pool.execute = AsyncMock(return_value=None)

        agent_row = {"host": "agent", "port": 8000}
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        sleep_count = 0

        async def fake_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch("src.main._ensure_agent_running", AsyncMock(return_value=agent_row)),
            patch("src.main._httpx") as mock_httpx,
            patch("asyncio.sleep", fake_sleep),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            from src.main import _cron_runner

            with pytest.raises(asyncio.CancelledError):
                await _cron_runner()

        execute_calls = mock_pool.execute.call_args_list
        sql_calls = [str(c) for c in execute_calls]
        assert any("enabled = false" in s for s in sql_calls)

    @pytest.mark.asyncio
    async def test_skips_job_when_agent_unavailable(self, mock_pool):
        """If the agent can't be started, the job should be skipped (not crash the runner)."""
        due_job = make_record(
            id="job-no-agent",
            user_id="user-dead",
            plugin="google-drive",
            instruction="Refresh.",
            interval_s=3000,
        )
        mock_pool.fetch = AsyncMock(side_effect=[[due_job], []])
        mock_pool.execute = AsyncMock(return_value=None)

        sleep_count = 0

        async def fake_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch(
                "src.main._ensure_agent_running", AsyncMock(return_value=None)
            ),  # agent down
            patch("asyncio.sleep", fake_sleep),
        ):
            from src.main import _cron_runner

            with pytest.raises(asyncio.CancelledError):
                await _cron_runner()

        # Job should still be advanced (not stuck) even when agent is down
        mock_pool.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_skips_poll_when_no_due_jobs(self, mock_pool):
        """When no jobs are due, no HTTP calls should be made."""
        mock_pool.fetch = AsyncMock(side_effect=[[], []])

        sleep_count = 0

        async def fake_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("src.main.get_pool", AsyncMock(return_value=mock_pool)),
            patch("src.main._ensure_agent_running", AsyncMock()) as mock_ensure,
            patch("asyncio.sleep", fake_sleep),
        ):
            from src.main import _cron_runner

            with pytest.raises(asyncio.CancelledError):
                await _cron_runner()

        mock_ensure.assert_not_awaited()


# ── _upsert_token_refresh_job ──────────────────────────────


class TestUpsertTokenRefreshJob:
    @pytest.mark.asyncio
    async def test_inserts_recurring_job(self, mock_pool):
        mock_pool.execute = AsyncMock(return_value=None)

        with patch("src.main.get_pool", AsyncMock(return_value=mock_pool)):
            from src.main import _upsert_token_refresh_job

            await _upsert_token_refresh_job(
                pool=mock_pool,
                user_id="user-1",
                plugin_name="google-drive",
                provider_name="google",
            )

        # Should have called execute twice: DELETE old + INSERT new
        assert mock_pool.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_instruction_mentions_plugin_and_provider(self, mock_pool):
        mock_pool.execute = AsyncMock(return_value=None)
        captured_sql = []

        async def capture_execute(sql, *args):
            captured_sql.append((sql, args))

        mock_pool.execute = AsyncMock(side_effect=capture_execute)

        with patch("src.main.get_pool", AsyncMock(return_value=mock_pool)):
            from src.main import _upsert_token_refresh_job

            await _upsert_token_refresh_job(
                pool=mock_pool,
                user_id="user-1",
                plugin_name="gmail",
                provider_name="google",
            )

        # The INSERT call (second execute) should have the instruction as an arg
        insert_args = captured_sql[1][1]  # (sql, args) → args tuple
        instruction = insert_args[3]  # 4th positional arg = instruction
        assert "gmail" in instruction
        assert "google" in instruction

    @pytest.mark.asyncio
    async def test_deletes_existing_job_before_inserting(self, mock_pool):
        """Re-connecting a plugin should reset the schedule cleanly."""
        mock_pool.execute = AsyncMock(return_value=None)
        call_sqls = []

        async def capture(sql, *args):
            call_sqls.append(sql)

        mock_pool.execute = AsyncMock(side_effect=capture)

        with patch("src.main.get_pool", AsyncMock(return_value=mock_pool)):
            from src.main import _upsert_token_refresh_job

            await _upsert_token_refresh_job(
                pool=mock_pool,
                user_id="user-1",
                plugin_name="google-drive",
                provider_name="google",
            )

        assert "DELETE" in call_sqls[0].upper()
        assert "INSERT" in call_sqls[1].upper()
