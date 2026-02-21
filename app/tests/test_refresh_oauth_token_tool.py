"""Tests for RefreshOAuthTokenTool and all 5 plugin-specific subclasses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

# Add app/plugins to sys.path so non-hyphenated plugin dirs are importable
# (e.g. plugins.gmail.tools, plugins.spotify.tools)
_PLUGINS_DIR = Path(__file__).parent.parent / "plugins"
if str(_PLUGINS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR.parent))


def _load_plugin_tools(plugin_dir_name: str):
    """Load a plugin's tools module from a potentially hyphenated directory name."""
    tools_path = _PLUGINS_DIR / plugin_dir_name / "tools.py"
    module_name = f"plugins.{plugin_dir_name.replace('-', '_')}.tools"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    assert spec is not None and spec.loader is not None, f"Cannot load {tools_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ── Concrete subclass for testing the base ─────────────────


class _TestRefreshTool(RefreshOAuthTokenTool):
    name = "refresh_test_plugin_token"
    plugin_name = "test-plugin"
    description = "Test refresh tool"


# ── Helpers ────────────────────────────────────────────────


def _mock_response(status_code: int):
    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status = MagicMock()
    return resp


# ── Base tool metadata ─────────────────────────────────────


class TestRefreshOAuthTokenToolMetadata:
    def test_no_parameters(self):
        tool = _TestRefreshTool()
        assert tool.parameters == []

    def test_plugin_name_set(self):
        tool = _TestRefreshTool()
        assert tool.plugin_name == "test-plugin"

    def test_status_text(self):
        tool = _TestRefreshTool()
        assert "refresh" in tool.status_text.lower()


# ── Happy path ─────────────────────────────────────────────


class TestRefreshOAuthTokenToolExecute:
    @pytest.mark.asyncio
    async def test_success_calls_config_then_reload(self):
        tool = _TestRefreshTool()
        config_resp = _mock_response(200)
        reload_resp = _mock_response(200)

        call_order = []

        async def fake_get(url, **kwargs):
            call_order.append("config")
            return config_resp

        async def fake_post(url, **kwargs):
            call_order.append("reload")
            return reload_resp

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.post = AsyncMock(side_effect=fake_post)

        with (
            patch(
                "aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", "http://orch:8080"
            ),
            patch("aether.tools.refresh_oauth_token._AGENT_USER_ID", "user-1"),
            patch("aether.tools.refresh_oauth_token.httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool.execute()

        assert result.error is False
        assert "refreshed" in result.output.lower()
        assert "config" in call_order
        assert "reload" in call_order

    @pytest.mark.asyncio
    async def test_reload_failure_is_non_fatal(self):
        """Config refresh succeeded but reload signal failed — still success."""
        tool = _TestRefreshTool()
        config_resp = _mock_response(200)

        call_count = 0

        async def fake_get(url, **kwargs):
            return config_resp

        async def fake_post(url, **kwargs):
            raise Exception("reload failed")

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.post = AsyncMock(side_effect=fake_post)

        with (
            patch(
                "aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", "http://orch:8080"
            ),
            patch("aether.tools.refresh_oauth_token._AGENT_USER_ID", "user-1"),
            patch("aether.tools.refresh_oauth_token.httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool.execute()

        # Token was refreshed in DB — non-fatal that reload signal failed
        assert result.error is False

    # ── Failure cases ──

    @pytest.mark.asyncio
    async def test_fails_when_no_orchestrator_url(self):
        tool = _TestRefreshTool()
        with patch("aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", ""):
            result = await tool.execute()
        assert result.error is True
        assert "ORCHESTRATOR_URL" in result.output

    @pytest.mark.asyncio
    async def test_fails_when_no_user_id(self):
        tool = _TestRefreshTool()
        with (
            patch(
                "aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", "http://orch:8080"
            ),
            patch("aether.tools.refresh_oauth_token._AGENT_USER_ID", ""),
        ):
            result = await tool.execute()
        assert result.error is True
        assert "AETHER_USER_ID" in result.output

    @pytest.mark.asyncio
    async def test_fails_on_404_plugin_not_installed(self):
        tool = _TestRefreshTool()
        config_resp = _mock_response(404)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=config_resp)

        with (
            patch(
                "aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", "http://orch:8080"
            ),
            patch("aether.tools.refresh_oauth_token._AGENT_USER_ID", "user-1"),
            patch("aether.tools.refresh_oauth_token.httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool.execute()

        assert result.error is True
        assert (
            "not installed" in result.output.lower()
            or "not enabled" in result.output.lower()
        )

    @pytest.mark.asyncio
    async def test_fails_on_network_error(self):
        tool = _TestRefreshTool()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        with (
            patch(
                "aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", "http://orch:8080"
            ),
            patch("aether.tools.refresh_oauth_token._AGENT_USER_ID", "user-1"),
            patch("aether.tools.refresh_oauth_token.httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool.execute()

        assert result.error is True

    @pytest.mark.asyncio
    async def test_config_url_includes_plugin_name(self):
        """The GET request must target the correct plugin's config endpoint."""
        tool = _TestRefreshTool()
        config_resp = _mock_response(200)
        captured = {}

        async def fake_get(url, **kwargs):
            captured["url"] = url
            return config_resp

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.post = AsyncMock(return_value=_mock_response(200))

        with (
            patch(
                "aether.tools.refresh_oauth_token._ORCHESTRATOR_URL", "http://orch:8080"
            ),
            patch("aether.tools.refresh_oauth_token._AGENT_USER_ID", "user-1"),
            patch("aether.tools.refresh_oauth_token.httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            await tool.execute()

        assert "test-plugin" in captured["url"]
        assert "/api/internal/plugins/" in captured["url"]


# ── Plugin subclass smoke tests ────────────────────────────


class TestPluginRefreshToolSubclasses:
    """Verify each plugin subclass has the correct name and plugin_name."""

    def test_google_drive(self):
        m = _load_plugin_tools("google-drive")
        RefreshGoogleDriveTokenTool = m.RefreshGoogleDriveTokenTool

        t = RefreshGoogleDriveTokenTool()
        assert t.name == "refresh_google_drive_token"
        assert t.plugin_name == "google-drive"
        assert t.parameters == []

    def test_gmail(self):
        from plugins.gmail.tools import RefreshGmailTokenTool

        t = RefreshGmailTokenTool()
        assert t.name == "refresh_gmail_token"
        assert t.plugin_name == "gmail"
        assert t.parameters == []

    def test_google_calendar(self):
        m = _load_plugin_tools("google-calendar")
        RefreshGoogleCalendarTokenTool = m.RefreshGoogleCalendarTokenTool

        t = RefreshGoogleCalendarTokenTool()
        assert t.name == "refresh_google_calendar_token"
        assert t.plugin_name == "google-calendar"
        assert t.parameters == []

    def test_google_contacts(self):
        m = _load_plugin_tools("google-contacts")
        RefreshGoogleContactsTokenTool = m.RefreshGoogleContactsTokenTool

        t = RefreshGoogleContactsTokenTool()
        assert t.name == "refresh_google_contacts_token"
        assert t.plugin_name == "google-contacts"
        assert t.parameters == []

    def test_spotify(self):
        from plugins.spotify.tools import RefreshSpotifyTokenTool

        t = RefreshSpotifyTokenTool()
        assert t.name == "refresh_spotify_token"
        assert t.plugin_name == "spotify"
        assert t.parameters == []
