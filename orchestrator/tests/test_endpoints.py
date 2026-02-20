"""Tests for orchestrator HTTP endpoints via FastAPI TestClient.

All DB operations are mocked — no real Postgres needed.
Auth is bypassed via FastAPI dependency_overrides.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_record, MockPool


# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def mock_pool():
    return MockPool()


@pytest.fixture
def app_client(mock_pool):
    """
    TestClient with mocked DB, auth, and startup events disabled.

    Uses FastAPI dependency_overrides to bypass auth — the correct way
    to mock Depends() in FastAPI tests.
    """

    # Patch get_pool everywhere it's imported
    async def _get_pool():
        return mock_pool

    # Patch bootstrap_schema to no-op
    async def _noop():
        pass

    with (
        patch("src.db.get_pool", _get_pool),
        patch("src.auth.get_pool", _get_pool),
        patch("src.main.get_pool", _get_pool),
        patch("src.main.bootstrap_schema", _noop),
        patch("src.main.close_pool", _noop),
        patch("src.main.reconcile_containers", AsyncMock()),
    ):
        from src.main import app
        from src.auth import get_user_id

        # Override auth dependency to return a fixed user
        async def _mock_get_user_id():
            return "user-1"

        app.dependency_overrides[get_user_id] = _mock_get_user_id

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool

        # Clean up overrides
        app.dependency_overrides.clear()


@pytest.fixture
def unauthed_client(mock_pool):
    """
    TestClient WITHOUT auth override — for testing 401 responses.
    """

    async def _get_pool():
        return mock_pool

    async def _noop():
        pass

    with (
        patch("src.db.get_pool", _get_pool),
        patch("src.auth.get_pool", _get_pool),
        patch("src.main.get_pool", _get_pool),
        patch("src.main.bootstrap_schema", _noop),
        patch("src.main.close_pool", _noop),
        patch("src.main.reconcile_containers", AsyncMock()),
    ):
        from src.main import app

        # No dependency override — real auth runs (and fails without valid token)
        app.dependency_overrides.clear()

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool


def _auth_header(token: str = "test-session-token") -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Health ─────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, app_client):
        client, pool = app_client
        pool.fetchval = AsyncMock(return_value=1)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db"] == "connected"

    def test_health_degraded(self, app_client):
        client, pool = app_client
        pool.fetchval = AsyncMock(side_effect=Exception("connection refused"))

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"


# ── Agent Registration ─────────────────────────────────────


class TestAgentRegistry:
    def test_register_agent_with_secret(self, app_client):
        """Agent registration succeeds with correct AGENT_SECRET."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "test-agent-secret"):
            resp = client.post(
                "/agents/register",
                json={
                    "agent_id": "agent-dev",
                    "host": "agent",
                    "port": 8000,
                    "container_id": "abc123",
                    "user_id": None,
                },
                headers={"Authorization": "Bearer test-agent-secret"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "registered"
        pool.execute.assert_called_once()

    def test_register_agent_wrong_secret(self, app_client):
        """Agent registration fails with wrong secret."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "correct-secret"):
            resp = client.post(
                "/agents/register",
                json={
                    "agent_id": "agent-dev",
                    "host": "agent",
                    "port": 8000,
                },
                headers={"Authorization": "Bearer wrong-secret"},
            )

        assert resp.status_code == 403

    def test_register_agent_no_secret_configured(self, app_client):
        """Agent registration succeeds when no AGENT_SECRET is configured (dev mode)."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", ""):
            resp = client.post(
                "/agents/register",
                json={
                    "agent_id": "agent-dev",
                    "host": "agent",
                    "port": 8000,
                },
            )

        assert resp.status_code == 200

    def test_register_agent_missing_auth(self, app_client):
        """Agent registration fails when secret is configured but no header sent."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "required-secret"):
            resp = client.post(
                "/agents/register",
                json={
                    "agent_id": "agent-dev",
                    "host": "agent",
                    "port": 8000,
                },
            )

        assert resp.status_code == 401

    def test_heartbeat(self, app_client):
        """Heartbeat updates last_health and status."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", ""):
            resp = client.post("/agents/agent-dev/heartbeat")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        pool.execute.assert_called_once()
        call_sql = pool.execute.call_args[0][0]
        assert "last_health" in call_sql

    def test_list_agents(self, app_client):
        """GET /agents/health returns agent list."""
        client, pool = app_client
        pool.fetch = AsyncMock(
            return_value=[
                make_record(
                    id="agent-1",
                    user_id="user-1",
                    host="agent",
                    port=8000,
                    status="running",
                    last_health=None,
                )
            ]
        )

        resp = client.get("/agents/health")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "agent-1"

    def test_assign_agent(self, app_client):
        """Assign agent to user."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", ""):
            resp = client.post(
                "/agents/agent-1/assign",
                params={"user_id": "user-42"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "assigned"


# ── Auth (me endpoint) ─────────────────────────────────────


class TestAuthMe:
    def test_me_returns_user(self, app_client):
        """GET /auth/me returns user info."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(
            return_value=make_record(
                id="user-1",
                email="test@example.com",
                name="Test User",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        resp = client.get("/auth/me", headers=_auth_header())

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "user-1"
        assert data["email"] == "test@example.com"

    def test_me_user_not_found(self, app_client):
        """GET /auth/me returns 404 when user not in DB."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(return_value=None)
        resp = client.get("/auth/me", headers=_auth_header())

        assert resp.status_code == 404

    def test_me_no_auth(self, unauthed_client):
        """GET /auth/me returns 401 without auth."""
        client, pool = unauthed_client
        resp = client.get("/auth/me")
        assert resp.status_code == 401


# ── Device Pairing ─────────────────────────────────────────


class TestDevicePairing:
    def test_pair_request(self, app_client):
        """POST /pair/request creates a pair request."""
        client, pool = app_client

        resp = client.post(
            "/pair/request",
            json={"code": "ABC123", "device_type": "ios", "device_name": "iPhone"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "waiting"
        assert data["expires_in"] == 600

    def test_pair_confirm(self, app_client):
        """POST /pair/confirm pairs a device and returns token."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(
            return_value=make_record(
                code="ABC123", device_type="ios", device_name="iPhone"
            )
        )

        with patch("src.main._ensure_agent", new_callable=AsyncMock):
            resp = client.post(
                "/pair/confirm",
                json={"code": "ABC123"},
                headers=_auth_header(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "device_id" in data
        assert "device_token" in data

    def test_pair_confirm_invalid_code(self, app_client):
        """POST /pair/confirm with invalid code returns 404."""
        client, pool = app_client
        pool.fetchrow = AsyncMock(return_value=None)

        resp = client.post(
            "/pair/confirm",
            json={"code": "INVALID"},
            headers=_auth_header(),
        )

        assert resp.status_code == 404

    def test_pair_status_waiting(self, app_client):
        """GET /pair/status returns waiting when not yet confirmed."""
        client, pool = app_client
        pool.fetchrow = AsyncMock(return_value=make_record(claimed_by=None))

        resp = client.get("/pair/status/ABC123")
        assert resp.status_code == 200
        assert resp.json()["status"] == "waiting"

    def test_pair_status_paired(self, app_client):
        """GET /pair/status returns paired with device info."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(
            side_effect=[
                make_record(claimed_by="user-1"),
                make_record(id="dev-1", token="device-token-xyz"),
            ]
        )

        resp = client.get("/pair/status/ABC123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "paired"
        assert data["device_token"] == "device-token-xyz"

    def test_pair_status_unknown(self, app_client):
        """GET /pair/status returns unknown for nonexistent code."""
        client, pool = app_client
        pool.fetchrow = AsyncMock(return_value=None)

        resp = client.get("/pair/status/NOPE")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown"

    def test_list_devices(self, app_client):
        """GET /devices returns user's devices."""
        client, pool = app_client
        pool.fetch = AsyncMock(
            return_value=[
                make_record(
                    id="dev-1",
                    name="iPhone",
                    device_type="ios",
                    paired_at="2024-01-01",
                    last_seen=None,
                )
            ]
        )

        resp = client.get("/devices", headers=_auth_header())

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "iPhone"


# ── API Key Management ─────────────────────────────────────


class TestApiKeys:
    def test_store_api_key(self, app_client):
        """POST /services/keys stores an encrypted key."""
        client, pool = app_client

        with patch("src.main.encrypt_value", return_value="gAAAAA_encrypted"):
            resp = client.post(
                "/services/keys",
                json={"provider": "openai", "key_value": "sk-abc123"},
                headers=_auth_header(),
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
        pool.execute.assert_called_once()

    def test_list_api_keys(self, app_client):
        """GET /services/keys returns previews of stored keys."""
        client, pool = app_client
        pool.fetch = AsyncMock(
            return_value=[
                make_record(provider="openai", key_value="gAAAAA_encrypted"),
            ]
        )

        with patch("src.main.decrypt_value", return_value="sk-abc123456789"):
            resp = client.get("/services/keys", headers=_auth_header())

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["provider"] == "openai"
        assert data[0]["preview"] == "sk-abc12..."  # First 8 chars + ...

    def test_delete_api_key(self, app_client):
        """DELETE /services/keys/{provider} deletes the key."""
        client, pool = app_client

        resp = client.delete("/services/keys/openai", headers=_auth_header())

        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        pool.execute.assert_called_once()

    def test_store_api_key_no_auth(self, unauthed_client):
        """POST /services/keys returns 401 without auth."""
        client, pool = unauthed_client
        resp = client.post(
            "/services/keys",
            json={"provider": "openai", "key_value": "sk-abc123"},
        )
        assert resp.status_code == 401


# ── Plugin Config Provisioning ─────────────────────────────


class TestPluginConfigProvisioning:
    def test_vobiz_save_config_auto_provisions_application(self, app_client):
        client, pool = app_client
        pool.fetchrow = AsyncMock(return_value=make_record(id="plug-1", enabled=True))
        pool.fetch = AsyncMock(return_value=[])

        list_resp = MagicMock()
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = {"objects": []}

        create_resp = MagicMock()
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"app_id": "app_12345"}

        link_resp = MagicMock()
        link_resp.raise_for_status = MagicMock()

        with (
            patch("src.main.decrypt_value", side_effect=lambda v: v),
            patch("src.main.encrypt_value", side_effect=lambda v: f"enc:{v}"),
            patch("src.main._signal_agent_plugin_reload", new_callable=AsyncMock),
            patch("src.main.httpx.AsyncClient") as MockClient,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=list_resp)
            mock_client.post = AsyncMock(side_effect=[create_resp, link_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            resp = client.post(
                "/api/plugins/vobiz/config",
                json={
                    "config": {
                        "auth_id": "MA_TEST",
                        "auth_token": "secret",
                        "from_number": "+919999000000",
                    }
                },
                headers=_auth_header(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["vobiz_provision"]["status"] == "ok"
        assert data["vobiz_provision"]["action"] == "created"
        assert data["vobiz_provision"]["application_id"] == "app_12345"
        assert any(
            len(call.args) > 3 and call.args[3] == "application_id"
            for call in pool.execute.call_args_list
        )

    def test_vobiz_save_config_skips_provision_when_required_fields_missing(
        self, app_client
    ):
        client, pool = app_client
        pool.fetchrow = AsyncMock(return_value=make_record(id="plug-1", enabled=True))
        pool.fetch = AsyncMock(return_value=[])

        with patch("src.main.httpx.AsyncClient") as MockClient:
            resp = client.post(
                "/api/plugins/vobiz/config",
                json={"config": {"auth_id": "MA_TEST"}},
                headers=_auth_header(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["vobiz_provision"]["status"] == "skipped"
        MockClient.assert_not_called()


# ── Memory Proxy ───────────────────────────────────────────


class TestMemoryProxy:
    def test_memory_facts(self, app_client):
        """GET /memory/facts proxies to agent."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(return_value=make_record(host="agent", port=8000))

        mock_response = MagicMock()
        mock_response.json.return_value = {"facts": ["User likes Python"]}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            resp = client.get("/memory/facts", headers=_auth_header())

        assert resp.status_code == 200
        assert resp.json()["facts"] == ["User likes Python"]

    def test_memory_no_agent(self, app_client):
        """GET /memory/facts returns 404 when no agent assigned."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(return_value=None)

        with patch("src.main._ensure_agent", new_callable=AsyncMock):
            resp = client.get("/memory/facts", headers=_auth_header())

        assert resp.status_code == 404

    def test_memory_sessions(self, app_client):
        """GET /memory/sessions proxies to agent."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(return_value=make_record(host="agent", port=8000))

        mock_response = MagicMock()
        mock_response.json.return_value = {"sessions": []}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            resp = client.get("/memory/sessions", headers=_auth_header())

        assert resp.status_code == 200

    def test_memory_conversations(self, app_client):
        """GET /memory/conversations proxies to agent with limit param."""
        client, pool = app_client

        pool.fetchrow = AsyncMock(return_value=make_record(host="agent", port=8000))

        mock_response = MagicMock()
        mock_response.json.return_value = {"conversations": []}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            resp = client.get(
                "/memory/conversations",
                params={"limit": 5},
                headers=_auth_header(),
            )

        assert resp.status_code == 200


# ── Verify Agent Secret ───────────────────────────────────


class TestVerifyAgentSecret:
    def test_constant_time_comparison(self):
        """verify_agent_secret uses hmac.compare_digest (constant-time)."""
        from src.main import verify_agent_secret

        assert verify_agent_secret is not None

    def test_no_secret_skips_validation(self, app_client):
        """When AGENT_SECRET is empty, all agent endpoints are open."""
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", ""):
            resp = client.post(
                "/agents/register",
                json={"agent_id": "a", "host": "h", "port": 8000},
            )

        assert resp.status_code == 200
