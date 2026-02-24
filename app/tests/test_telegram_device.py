"""Tests for the Telegram device config endpoint.

Objective: Verify POST /devices/telegram/config correctly initializes
Telegram when bot_token is provided, and clears state when bot_token
is absent.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient for the agent app.

    Uses raise_server_exceptions=False so we can inspect error responses
    without pytest blowing up.
    """
    from aether.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestTelegramDeviceConfig:
    """Tests for POST /devices/telegram/config."""

    # ── Positive: configure Telegram with a valid bot_token ──────

    def test_configure_telegram(self, client):
        """POST /devices/telegram/config with bot_token initializes Telegram.

        Positive test: verifies that when a valid bot_token is supplied,
        _init_telegram is called with the correct config dict and the
        response indicates success with the bot_username echoed back.
        """
        # Arrange
        payload = {
            "bot_token": "123456:ABC-DEF",
            "secret_token": "my-secret",
            "allowed_chat_ids": "111,222",
            "bot_username": "TestBot",
        }

        # Act — mock _init_telegram so we don't touch real Telegram infra
        with (
            patch("aether.main._init_telegram", new_callable=AsyncMock) as mock_init,
            patch("aether.main.plugin_context_store") as mock_ctx,
        ):
            resp = client.post("/devices/telegram/config", json=payload)

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "configured"
        assert data["bot_username"] == "TestBot"

        # _init_telegram must be called exactly once with the non-None fields
        mock_init.assert_called_once()
        call_args = mock_init.call_args[0][0]
        assert call_args["bot_token"] == "123456:ABC-DEF"
        assert call_args["secret_token"] == "my-secret"
        assert call_args["allowed_chat_ids"] == "111,222"
        assert call_args["bot_username"] == "TestBot"

        # plugin_context_store.set must be called with the config
        mock_ctx.set.assert_called_once_with("telegram", call_args)

    # ── Positive: clear Telegram when routes module is loaded ────

    def test_clear_telegram_with_routes_module(self, client):
        """POST /devices/telegram/config with empty body clears Telegram state.

        Positive test: when bot_token is absent and the Telegram routes
        module IS present in sys.modules, set_config({}) is called to
        tear down the plugin state.
        """
        # Arrange — fake routes module with a set_config callable
        mock_routes = MagicMock()
        mock_routes.set_config = MagicMock()

        with patch.dict(sys.modules, {"aether_plugin_telegram_routes": mock_routes}):
            # Act
            resp = client.post("/devices/telegram/config", json={})

        # Assert
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        mock_routes.set_config.assert_called_once_with({})

    # ── Negative / edge-case: clear when routes module is absent ─

    def test_clear_telegram_no_routes_module(self, client):
        """Clearing Telegram config when routes module is not loaded is a no-op.

        Negative / edge-case test: when bot_token is None and the
        aether_plugin_telegram_routes module has never been imported,
        the endpoint must still return 200 with status 'cleared' and
        not raise an error.
        """
        # Arrange — ensure the module is NOT in sys.modules
        saved = sys.modules.pop("aether_plugin_telegram_routes", None)
        try:
            # Act
            resp = client.post(
                "/devices/telegram/config",
                json={"bot_token": None},
            )
        finally:
            # Restore if it was there before
            if saved is not None:
                sys.modules["aether_plugin_telegram_routes"] = saved

        # Assert
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
