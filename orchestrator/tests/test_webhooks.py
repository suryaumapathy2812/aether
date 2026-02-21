"""Tests for the orchestrator webhook system.

Covers:
- POST /api/hooks/http/{plugin}/{user_id}  — HTTP push adapter
- POST /api/hooks/pubsub/{plugin}/{user_id} — Pub/Sub adapter
- POST /api/internal/watches — watch registration endpoint
- _upsert_watch_setup_job() — one-shot cron after OAuth
- _reconcile_watch_setup_jobs() — startup sweep

All DB operations are mocked — no real Postgres needed.
Agent HTTP calls are mocked — no real agent container needed.
"""

from __future__ import annotations

import base64
import json
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


def _installed_plugin_row(plugin_name: str = "gmail", enabled: bool = True) -> dict:
    """Simulate a fetchrow result for a plugin."""
    return make_record(
        id="plugin-id-1",
        enabled=enabled,
    )


# ── POST /api/hooks/http/{plugin}/{user_id} ────────────────


class TestHttpHookAdapter:
    def test_http_hook_stores_event_for_installed_plugin(self, app_client, mock_pool):
        """A valid HTTP push event is stored and returns 'received'."""
        client, pool = app_client

        # fetchrow(plugins) → installed; fetchrow(agents) → no running agent
        pool.fetchrow.side_effect = [
            _installed_plugin_row("google-calendar"),  # plugin check
            None,  # agent lookup
        ]

        with patch(
            "src.main.AVAILABLE_PLUGINS",
            {"google-calendar": {"webhook": {"handle_tool": "handle_calendar_event"}}},
        ):
            resp = client.post(
                "/api/hooks/http/google-calendar/user-1",
                json={"some": "payload"},
                headers={"x-goog-resource-state": "exists"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "received"
        assert "event_id" in data

    def test_http_hook_ignores_sync_notification(self, app_client, mock_pool):
        """x-goog-resource-state: sync is acknowledged without storing an event."""
        client, pool = app_client

        resp = client.post(
            "/api/hooks/http/google-calendar/user-1",
            json={},
            headers={"x-goog-resource-state": "sync"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "sync_acknowledged"
        # No DB write should happen for sync notifications
        pool.execute.assert_not_awaited()

    def test_http_hook_returns_200_for_unknown_plugin(self, app_client, mock_pool):
        """Unknown plugin returns 200 with plugin_not_installed (prevents retries)."""
        client, pool = app_client

        pool.fetchrow.return_value = None  # plugin not installed

        resp = client.post(
            "/api/hooks/http/unknown-plugin/user-1",
            json={},
            headers={"x-goog-resource-state": "exists"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "plugin_not_installed"

    def test_http_hook_returns_200_for_disabled_plugin(self, app_client, mock_pool):
        """Disabled plugin returns 200 with plugin_disabled (prevents retries)."""
        client, pool = app_client

        pool.fetchrow.return_value = _installed_plugin_row(
            "google-calendar", enabled=False
        )

        resp = client.post(
            "/api/hooks/http/google-calendar/user-1",
            json={},
            headers={"x-goog-resource-state": "exists"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "plugin_disabled"

    def test_http_hook_forwards_to_running_agent(self, app_client, mock_pool):
        """When agent is running, event is forwarded via HTTP."""
        client, pool = app_client

        pool.fetchrow.side_effect = [
            _installed_plugin_row("google-calendar"),
            make_record(host="agent-host", port=8000),  # running agent
        ]

        mock_agent_resp = MagicMock()
        mock_agent_resp.raise_for_status = MagicMock()
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_agent_resp)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "src.main.AVAILABLE_PLUGINS",
                {
                    "google-calendar": {
                        "webhook": {"handle_tool": "handle_calendar_event"}
                    }
                },
            ),
            patch("src.main.httpx.AsyncClient", return_value=mock_http_client),
        ):
            resp = client.post(
                "/api/hooks/http/google-calendar/user-1",
                json={"some": "payload"},
                headers={"x-goog-resource-state": "exists"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "received"
        mock_http_client.post.assert_awaited_once()
        # Verify handle_tool was included in the forwarded payload
        call_kwargs = mock_http_client.post.call_args.kwargs
        assert call_kwargs["json"]["handle_tool"] == "handle_calendar_event"


# ── POST /api/hooks/pubsub/{plugin}/{user_id} ──────────────


def _pubsub_body(data: dict) -> dict:
    """Build a minimal Pub/Sub push body."""
    encoded = base64.b64encode(json.dumps(data).encode()).decode()
    return {"message": {"data": encoded, "messageId": "msg-1"}}


class TestPubSubHookAdapter:
    def test_pubsub_hook_decodes_and_stores(self, app_client, mock_pool):
        """Valid Pub/Sub message is base64-decoded and stored (no JWT when PUBSUB_AUDIENCE unset)."""
        client, pool = app_client

        pool.fetchrow.side_effect = [
            _installed_plugin_row("gmail"),
            None,  # agent not running
        ]

        payload = {"emailAddress": "user@gmail.com", "historyId": "12345"}

        with patch(
            "src.main.AVAILABLE_PLUGINS",
            {"gmail": {"webhook": {"handle_tool": "handle_gmail_event"}}},
        ):
            # PUBSUB_AUDIENCE not set → JWT verification skipped
            resp = client.post(
                "/api/hooks/pubsub/gmail/user-1",
                json=_pubsub_body(payload),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "received"

    def test_pubsub_hook_rejects_invalid_jwt_when_audience_set(
        self, app_client, mock_pool
    ):
        """Invalid JWT returns 403 when PUBSUB_AUDIENCE is configured."""
        client, pool = app_client

        with (
            patch("src.main.PUBSUB_AUDIENCE", "https://my-orchestrator.example.com"),
            patch(
                "src.main._verify_pubsub_jwt",
                AsyncMock(side_effect=Exception("bad token")),
            ),
        ):
            resp = client.post(
                "/api/hooks/pubsub/gmail/user-1",
                json=_pubsub_body({"historyId": "1"}),
                headers={"Authorization": "Bearer bad-jwt"},
            )

        assert resp.status_code == 403

    def test_pubsub_hook_rejects_missing_bearer_when_audience_set(
        self, app_client, mock_pool
    ):
        """Missing Authorization header returns 401 when PUBSUB_AUDIENCE is configured."""
        client, pool = app_client

        with patch("src.main.PUBSUB_AUDIENCE", "https://my-orchestrator.example.com"):
            resp = client.post(
                "/api/hooks/pubsub/gmail/user-1",
                json=_pubsub_body({"historyId": "1"}),
                # No Authorization header
            )

        assert resp.status_code == 401

    def test_pubsub_hook_rejects_missing_message(self, app_client, mock_pool):
        """Malformed Pub/Sub body (no message key) returns 400."""
        client, pool = app_client

        resp = client.post(
            "/api/hooks/pubsub/gmail/user-1",
            json={"not": "a-pubsub-message"},
        )

        assert resp.status_code == 400

    def test_pubsub_hook_attaches_metadata(self, app_client, mock_pool):
        """Pub/Sub metadata (_pubsub_message_id etc.) is attached to the stored payload."""
        client, pool = app_client

        pool.fetchrow.side_effect = [
            _installed_plugin_row("gmail"),
            None,
        ]

        payload = {"emailAddress": "user@gmail.com", "historyId": "999"}

        with patch(
            "src.main.AVAILABLE_PLUGINS",
            {"gmail": {"webhook": {"handle_tool": "handle_gmail_event"}}},
        ):
            resp = client.post(
                "/api/hooks/pubsub/gmail/user-1",
                json={
                    "message": {
                        "data": base64.b64encode(json.dumps(payload).encode()).decode(),
                        "messageId": "msg-42",
                    }
                },
            )

        assert resp.status_code == 200
        # Verify the stored payload includes Pub/Sub metadata
        stored_json = pool.execute.call_args_list[-1].args
        stored_payload = json.loads(stored_json[-1])
        assert stored_payload["_pubsub_message_id"] == "msg-42"
        assert stored_payload["historyId"] == "999"


# ── POST /api/internal/watches ─────────────────────────────


class TestRegisterWatchEndpoint:
    def test_register_watch_upserts_row(self, app_client, mock_pool):
        """Valid watch registration is upserted into watch_registrations."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/watches",
                json={
                    "user_id": "user-1",
                    "plugin_name": "gmail",
                    "protocol": "pubsub",
                    "watch_id": "hist-123",
                    "resource_id": "projects/my-project/topics/gmail-push",
                    "expires_at": "2026-03-01T00:00:00+00:00",
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        pool.execute.assert_awaited()

    def test_register_watch_missing_user_id(self, app_client):
        """Missing user_id returns 400."""
        client, _ = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/watches",
                json={"plugin_name": "gmail"},
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 400

    def test_register_watch_requires_agent_auth(self, app_client):
        """Missing agent secret returns 401 or 422 (unprotected when AGENT_SECRET unset)."""
        client, _ = app_client

        # When AGENT_SECRET is set, missing auth should be rejected
        with patch("src.main.AGENT_SECRET", "real-secret"):
            resp = client.post(
                "/api/internal/watches",
                json={"user_id": "user-1", "plugin_name": "gmail"},
                # No Authorization header
            )

        # 401 when secret is set and header is missing
        assert resp.status_code in (401, 403)

    def test_register_watch_null_expires_at(self, app_client, mock_pool):
        """Watch with no expiry is accepted (expires_at is optional)."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/api/internal/watches",
                json={
                    "user_id": "user-1",
                    "plugin_name": "google-calendar",
                    "protocol": "http",
                    "watch_id": "chan-abc",
                    "resource_id": "res-xyz",
                },
                headers=_agent_auth_header(),
            )

        assert resp.status_code == 200
        pool.execute.assert_awaited()


# ── _upsert_watch_setup_job() ──────────────────────────────


class TestUpsertWatchSetupJob:
    @pytest.mark.asyncio
    async def test_schedules_one_shot_cron_job(self):
        """_upsert_watch_setup_job inserts a cron job row for the plugin."""
        from src.main import _upsert_watch_setup_job

        pool = MockPool()
        with patch(
            "src.main.AVAILABLE_PLUGINS",
            {
                "gmail": {
                    "webhook": {
                        "protocol": "pubsub",
                        "setup_tool": "setup_gmail_watch",
                        "handle_tool": "handle_gmail_event",
                        "renew_interval": 518400,
                    }
                }
            },
        ):
            await _upsert_watch_setup_job(pool, "user-1", "gmail")

        # Should have called execute twice: DELETE old job + INSERT new job
        assert pool.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_skips_plugin_without_webhook(self):
        """_upsert_watch_setup_job is a no-op for plugins without webhook config."""
        from src.main import _upsert_watch_setup_job

        pool = MockPool()
        with patch("src.main.AVAILABLE_PLUGINS", {"spotify": {}}):
            await _upsert_watch_setup_job(pool, "user-1", "spotify")

        pool.execute.assert_not_awaited()


# ── _reconcile_watch_setup_jobs() ─────────────────────────


class TestReconcileWatchSetupJobs:
    @pytest.mark.asyncio
    async def test_schedules_missing_setup_jobs(self):
        """Reconcile creates setup jobs for connected plugins with no active watch."""
        from src.main import _reconcile_watch_setup_jobs

        pool = MockPool()

        # fetch → connected plugins
        pool.fetch.return_value = [make_record(user_id="user-1", plugin_name="gmail")]
        # fetchrow → no existing watch registration
        pool.fetchrow.return_value = None
        # fetchval → no pending setup job
        pool.fetchval.return_value = None

        with (
            patch(
                "src.main.AVAILABLE_PLUGINS",
                {
                    "gmail": {
                        "webhook": {
                            "protocol": "pubsub",
                            "setup_tool": "setup_gmail_watch",
                            "handle_tool": "handle_gmail_event",
                            "renew_interval": 518400,
                        }
                    }
                },
            ),
            patch("src.main._upsert_watch_setup_job", AsyncMock()) as upsert,
        ):
            await _reconcile_watch_setup_jobs(pool)

        upsert.assert_awaited_once_with(pool, "user-1", "gmail")

    @pytest.mark.asyncio
    async def test_skips_when_no_connected_plugins(self):
        """Reconcile is a no-op when no plugins are connected."""
        from src.main import _reconcile_watch_setup_jobs

        pool = MockPool()
        pool.fetch.return_value = []

        with patch("src.main._upsert_watch_setup_job", AsyncMock()) as upsert:
            await _reconcile_watch_setup_jobs(pool)

        upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_plugin_with_active_watch(self):
        """Reconcile skips plugins that already have an active watch registration."""
        from src.main import _reconcile_watch_setup_jobs

        pool = MockPool()
        pool.fetch.return_value = [make_record(user_id="user-1", plugin_name="gmail")]
        # fetchrow → active watch exists
        pool.fetchrow.return_value = make_record(id="watch-1")

        with (
            patch(
                "src.main.AVAILABLE_PLUGINS",
                {"gmail": {"webhook": {"setup_tool": "setup_gmail_watch"}}},
            ),
            patch("src.main._upsert_watch_setup_job", AsyncMock()) as upsert,
        ):
            await _reconcile_watch_setup_jobs(pool)

        upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_plugin_with_pending_job(self):
        """Reconcile skips plugins that already have a pending setup job."""
        from src.main import _reconcile_watch_setup_jobs

        pool = MockPool()
        pool.fetch.return_value = [make_record(user_id="user-1", plugin_name="gmail")]
        pool.fetchrow.return_value = None  # no active watch
        pool.fetchval.return_value = "job-existing"  # pending job exists

        with (
            patch(
                "src.main.AVAILABLE_PLUGINS",
                {"gmail": {"webhook": {"setup_tool": "setup_gmail_watch"}}},
            ),
            patch("src.main._upsert_watch_setup_job", AsyncMock()) as upsert,
        ):
            await _reconcile_watch_setup_jobs(pool)

        upsert.assert_not_awaited()
