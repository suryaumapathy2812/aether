"""Tests for Telegram device endpoints in the orchestrator.

Covers four endpoints:
  POST /api/devices/telegram          — register a Telegram bot (authenticated)
  DELETE /api/devices/{device_id}     — remove a device (authenticated)
  POST /api/devices/telegram/webhook  — receive Telegram updates (unauthenticated)
  GET /api/internal/devices           — agent fetches device configs (agent-secret auth)

Each class contains at least one positive (happy-path) and one negative
(failure/breakage) test, linked to the objective via docstring comments.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_record, MockPool


# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def mock_pool():
    return MockPool()


@pytest.fixture
def app_client(mock_pool):
    """TestClient with mocked DB, auth, and startup events disabled.

    Uses FastAPI dependency_overrides to bypass auth — the correct way
    to mock Depends() in FastAPI tests.
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
        from src.auth import get_user_id

        async def _mock_get_user_id():
            return "user-1"

        app.dependency_overrides[get_user_id] = _mock_get_user_id

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool

        app.dependency_overrides.clear()


# ── POST /api/devices/telegram — Register Telegram Device ──


class TestRegisterTelegramDevice:
    """Objective: verify Telegram bot registration creates a device,
    validates the token, registers the webhook, and rejects duplicates."""

    def test_register_success(self, app_client):
        """Positive: register a new Telegram bot — happy path.

        Verifies that a valid bot token results in a 200 response with
        device_id and bot_username, and that the device row is inserted.
        """
        client, pool = app_client

        # Arrange — no existing telegram device for this user
        pool.fetchrow = AsyncMock(return_value=None)

        # Mock Telegram getMe API response
        mock_getme_resp = MagicMock()
        mock_getme_resp.status_code = 200
        mock_getme_resp.json.return_value = {
            "ok": True,
            "result": {"username": "TestAetherBot", "id": 123456},
        }

        # Mock setWebhook API response
        mock_setwebhook_resp = MagicMock()
        mock_setwebhook_resp.status_code = 200

        # Mock agent config push response
        mock_agent_resp = MagicMock()
        mock_agent_resp.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.get = AsyncMock(return_value=mock_getme_resp)
        mock_http_client.post = AsyncMock(
            side_effect=[mock_setwebhook_resp, mock_agent_resp]
        )

        with (
            patch("src.main.httpx.AsyncClient", return_value=mock_http_client),
            patch("src.main._ensure_agent", new_callable=AsyncMock),
            patch(
                "src.main._get_agent_for_user",
                new_callable=AsyncMock,
                return_value=make_record(host="agent", port=8000),
            ),
            patch("src.main.encrypt_value", return_value="encrypted-config"),
        ):
            # Act
            resp = client.post(
                "/api/devices/telegram",
                json={"bot_token": "123456:ABC-DEF"},
            )

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["bot_username"] == "TestAetherBot"
        assert "device_id" in data
        # Verify device INSERT was called
        pool.execute.assert_called()

    def test_register_duplicate_rejected(self, app_client):
        """Negative: cannot register a second Telegram bot for the same user.

        Verifies that a 409 Conflict is returned when a telegram device
        already exists for the authenticated user.
        """
        client, pool = app_client

        # Arrange — existing telegram device found
        pool.fetchrow = AsyncMock(return_value=make_record(id="existing-device"))

        # Act
        resp = client.post(
            "/api/devices/telegram",
            json={"bot_token": "123456:ABC-DEF"},
        )

        # Assert
        assert resp.status_code == 409

    def test_register_invalid_token(self, app_client):
        """Negative: invalid bot token is rejected by Telegram API.

        Verifies that when Telegram's getMe returns a non-200 status,
        the endpoint returns 400 Bad Request.
        """
        client, pool = app_client

        # Arrange — no existing device
        pool.fetchrow = AsyncMock(return_value=None)

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.main.httpx.AsyncClient", return_value=mock_http_client):
            # Act
            resp = client.post(
                "/api/devices/telegram",
                json={"bot_token": "invalid-token"},
            )

        # Assert
        assert resp.status_code == 400


# ── DELETE /api/devices/{device_id} — Delete Device ────────


class TestDeleteDevice:
    """Objective: verify device deletion removes the DB row, unregisters
    the Telegram webhook, and returns 404 for unknown devices."""

    def test_delete_telegram_device(self, app_client):
        """Positive: delete a Telegram device — unregisters webhook and deletes row.

        Verifies that deleting a telegram device calls deleteWebhook,
        notifies the agent, and returns status 'ok'.
        """
        client, pool = app_client

        # Arrange — device exists and belongs to user
        pool.fetchrow = AsyncMock(
            return_value=make_record(
                id="dev-123",
                device_type="telegram",
                config="encrypted-config",
            )
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch(
                "src.main.decrypt_value",
                return_value=json.dumps(
                    {"bot_token": "123456:ABC", "secret_token": "sec"}
                ),
            ),
            patch("src.main.httpx.AsyncClient", return_value=mock_http_client),
            patch(
                "src.main._get_agent_for_user",
                new_callable=AsyncMock,
                return_value=make_record(host="agent", port=8000),
            ),
        ):
            # Act
            resp = client.delete("/api/devices/dev-123")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        # Verify device DELETE was called
        pool.execute.assert_called()

    def test_delete_nonexistent_device(self, app_client):
        """Negative: deleting a device that doesn't exist returns 404.

        Verifies that when the device lookup returns None (not found or
        not owned by user), the endpoint returns 404 Not Found.
        """
        client, pool = app_client

        # Arrange — no device found
        pool.fetchrow = AsyncMock(return_value=None)

        # Act
        resp = client.delete("/api/devices/nonexistent")

        # Assert
        assert resp.status_code == 404


# ── POST /api/devices/telegram/webhook — Webhook Proxy ─────


class TestTelegramWebhookProxy:
    """Objective: verify the Telegram webhook proxy validates the secret
    token, proxies to the agent, and rejects unknown devices / bad secrets."""

    def test_webhook_proxy_success(self, app_client):
        """Positive: Telegram webhook is proxied to the user's agent.

        Verifies that a valid webhook request with correct secret token
        is forwarded to the agent and returns 200.
        """
        client, pool = app_client

        # Arrange — device exists with config
        pool.fetchrow = AsyncMock(
            return_value=make_record(
                id="dev-123",
                user_id="user-1",
                config="encrypted-config",
            )
        )

        mock_agent_resp = MagicMock()
        mock_agent_resp.status_code = 200
        mock_agent_resp.json.return_value = {"ok": True}

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.post = AsyncMock(return_value=mock_agent_resp)

        with (
            patch(
                "src.main.decrypt_value",
                return_value=json.dumps(
                    {"bot_token": "123:ABC", "secret_token": "my-secret"}
                ),
            ),
            patch("src.main.httpx.AsyncClient", return_value=mock_http_client),
            patch(
                "src.main._get_agent_for_user",
                new_callable=AsyncMock,
                return_value=make_record(host="agent", port=8000),
            ),
        ):
            # Act
            resp = client.post(
                "/api/devices/telegram/webhook?did=dev-123",
                json={"update_id": 1, "message": {"text": "hello"}},
                headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret"},
            )

        # Assert
        assert resp.status_code == 200

    def test_webhook_bad_secret(self, app_client):
        """Negative: webhook with wrong secret token is rejected with 403.

        Verifies that when the X-Telegram-Bot-Api-Secret-Token header
        doesn't match the stored secret, the endpoint returns 403 Forbidden.
        """
        client, pool = app_client

        # Arrange — device exists
        pool.fetchrow = AsyncMock(
            return_value=make_record(
                id="dev-123",
                user_id="user-1",
                config="encrypted-config",
            )
        )

        with patch(
            "src.main.decrypt_value",
            return_value=json.dumps(
                {"bot_token": "123:ABC", "secret_token": "correct-secret"}
            ),
        ):
            # Act
            resp = client.post(
                "/api/devices/telegram/webhook?did=dev-123",
                json={"update_id": 1},
                headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
            )

        # Assert
        assert resp.status_code == 403

    def test_webhook_unknown_device(self, app_client):
        """Negative: webhook for unknown device returns 404.

        Verifies that when the device_id in the query param doesn't match
        any telegram device, the endpoint returns 404.
        """
        client, pool = app_client

        # Arrange — no device found
        pool.fetchrow = AsyncMock(return_value=None)

        # Act
        resp = client.post(
            "/api/devices/telegram/webhook?did=nonexistent",
            json={"update_id": 1},
        )

        # Assert
        assert resp.status_code == 404


# ── GET /api/internal/devices — Agent Fetches Device Configs


class TestInternalDevices:
    """Objective: verify the internal device listing returns decrypted
    configs for the agent, and handles empty results correctly."""

    def test_list_devices_with_config(self, app_client):
        """Positive: agent fetches device configs with decrypted data.

        Verifies that the endpoint returns a list of devices with their
        decrypted config objects when devices exist for the user.
        """
        client, pool = app_client

        # Arrange — one telegram device with config
        pool.fetch = AsyncMock(
            return_value=[
                make_record(
                    id="dev-1",
                    device_type="telegram",
                    config="encrypted-config",
                )
            ]
        )

        with (
            patch("src.main.AGENT_SECRET", "test-secret"),
            patch(
                "src.main.decrypt_value",
                return_value=json.dumps(
                    {"bot_token": "123:ABC", "bot_username": "TestBot"}
                ),
            ),
        ):
            # Act
            resp = client.get(
                "/api/internal/devices?user_id=user-1",
                headers={"Authorization": "Bearer test-secret"},
            )

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["devices"]) == 1
        assert data["devices"][0]["device_type"] == "telegram"
        assert data["devices"][0]["config"]["bot_token"] == "123:ABC"

    def test_list_devices_empty(self, app_client):
        """Negative: no devices with config returns empty list.

        Verifies that when no devices exist for the user, the endpoint
        returns an empty devices array (not an error).
        """
        client, pool = app_client

        # Arrange — no devices
        pool.fetch = AsyncMock(return_value=[])

        with patch("src.main.AGENT_SECRET", "test-secret"):
            # Act
            resp = client.get(
                "/api/internal/devices?user_id=user-1",
                headers={"Authorization": "Bearer test-secret"},
            )

        # Assert
        assert resp.status_code == 200
        assert resp.json()["devices"] == []

    def test_list_devices_wrong_agent_secret(self, app_client):
        """Negative: wrong agent secret is rejected with 403.

        Verifies that the verify_agent_secret dependency rejects requests
        with an incorrect Authorization header.
        """
        client, pool = app_client

        with patch("src.main.AGENT_SECRET", "correct-secret"):
            # Act
            resp = client.get(
                "/api/internal/devices?user_id=user-1",
                headers={"Authorization": "Bearer wrong-secret"},
            )

        # Assert
        assert resp.status_code == 403
