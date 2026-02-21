"""Tests for BaseWatchTool and plugin-specific watch tool subclasses.

Covers:
- BaseWatchTool helpers: _get_token, _auth_headers, _register_watch, _schedule_renewal
- SetupGmailWatchTool.execute() — happy path + error cases
- HandleGmailEventTool.execute() — history fetch + message fetch
- SetupCalendarWatchTool.execute() — happy path + error cases
- HandleCalendarEventTool.execute() — event list fetch

All HTTP calls are mocked — no real Gmail/Calendar API needed.
All orchestrator calls are mocked — no real orchestrator needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aether.tools.base_watch_tool import BaseWatchTool
from aether.tools.base import ToolResult

# ── Plugin loader helper ───────────────────────────────────

_PLUGINS_DIR = Path(__file__).parent.parent / "plugins"
if str(_PLUGINS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR.parent))


def _load_plugin_tools(plugin_dir_name: str):
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


# Pre-load both plugin modules so patch() can find them by module reference
_gmail_mod = _load_plugin_tools("gmail")
_cal_mod = _load_plugin_tools("google-calendar")


# ── Helpers ────────────────────────────────────────────────


def _mock_http_response(status_code: int, json_data: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        from httpx import HTTPStatusError

        resp.raise_for_status.side_effect = HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=MagicMock(status_code=status_code, text="error"),
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _make_async_client(*responses):
    """Build a mock httpx.AsyncClient context manager."""
    client = MagicMock()
    if len(responses) == 1:
        client.get = AsyncMock(return_value=responses[0])
        client.post = AsyncMock(return_value=responses[0])
    else:
        client.get = AsyncMock(side_effect=list(responses))
        client.post = AsyncMock(side_effect=list(responses))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ── BaseWatchTool ──────────────────────────────────────────


class _ConcreteWatchTool(BaseWatchTool):
    name = "test_watch"
    plugin_name = "test-plugin"
    description = "Test watch tool"

    async def execute(self, **_) -> ToolResult:  # type: ignore[override]
        return ToolResult.success("ok")


class TestBaseWatchToolHelpers:
    def test_get_token_from_context(self):
        tool = _ConcreteWatchTool()
        object.__setattr__(tool, "_context", {"access_token": "tok-123"})
        assert tool._get_token() == "tok-123"

    def test_get_token_no_context(self):
        tool = _ConcreteWatchTool()
        assert tool._get_token() is None

    def test_auth_headers_with_token(self):
        tool = _ConcreteWatchTool()
        object.__setattr__(tool, "_context", {"access_token": "tok-abc"})
        headers = tool._auth_headers()
        assert headers["Authorization"] == "Bearer tok-abc"

    def test_auth_headers_no_token(self):
        tool = _ConcreteWatchTool()
        # No token → no Authorization header (or header with None)
        headers = tool._auth_headers()
        # Either no key or value is falsy
        assert not headers.get("Authorization") or "None" in headers.get(
            "Authorization", ""
        )

    @pytest.mark.asyncio
    async def test_register_watch_calls_orchestrator(self):
        tool = _ConcreteWatchTool()
        mock_resp = _mock_http_response(200)
        mock_client = _make_async_client(mock_resp)

        with (
            patch("aether.tools.base_watch_tool._ORCHESTRATOR_URL", "http://orch:8080"),
            patch("aether.tools.base_watch_tool._AGENT_USER_ID", "user-1"),
            patch(
                "aether.tools.base_watch_tool.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            await tool._register_watch(
                watch_id="w-1",
                resource_id="res-1",
                protocol="pubsub",
                expires_at=9999999999000,
            )

        mock_client.post.assert_awaited_once()
        payload = mock_client.post.call_args.kwargs.get("json", {})
        assert payload["plugin_name"] == "test-plugin"
        assert payload["watch_id"] == "w-1"

    @pytest.mark.asyncio
    async def test_register_watch_skips_when_no_orchestrator_url(self):
        """_register_watch is a no-op when ORCHESTRATOR_URL is not set."""
        tool = _ConcreteWatchTool()
        with (
            patch("aether.tools.base_watch_tool._ORCHESTRATOR_URL", ""),
            patch("aether.tools.base_watch_tool.httpx.AsyncClient") as mock_httpx,
        ):
            await tool._register_watch("w-1", "res-1", "pubsub", None)

        mock_httpx.assert_not_called()

    @pytest.mark.asyncio
    async def test_schedule_renewal_calls_cron_client(self):
        """_schedule_renewal schedules a cron job via CronClient."""
        tool = _ConcreteWatchTool()
        mock_cron = MagicMock()
        mock_cron.schedule = AsyncMock(return_value="job-123")

        with patch("aether.tools.base_watch_tool.CronClient", return_value=mock_cron):
            await tool._schedule_renewal(
                renew_tool="renew_test_watch",
                expires_at_ms=int((__import__("time").time() + 86400 * 7) * 1000),
                renew_interval_s=518400,
            )

        mock_cron.schedule.assert_awaited_once()
        call_kwargs = mock_cron.schedule.call_args.kwargs
        assert "renew_test_watch" in call_kwargs["instruction"]


# ── SetupGmailWatchTool ────────────────────────────────────


class TestSetupGmailWatchTool:
    def _make_tool(self, token: str = "tok-gmail"):
        tool = _gmail_mod.SetupGmailWatchTool()
        object.__setattr__(tool, "_context", {"access_token": token})
        return tool

    @pytest.mark.asyncio
    async def test_success_registers_and_schedules(self):
        """Happy path: watch() succeeds → registers + schedules renewal."""
        tool = self._make_tool()

        watch_resp = _mock_http_response(
            200, {"historyId": "555", "expiration": "9999999999000"}
        )
        mock_client = _make_async_client(watch_resp)

        with (
            patch.object(_gmail_mod, "_ORCHESTRATOR_URL_W", "http://orch:8080"),
            patch.object(_gmail_mod, "_AGENT_USER_ID_W", "user-1"),
            patch.object(_gmail_mod, "_GCP_PROJECT_ID", "my-project"),
            patch.object(_gmail_mod, "httpx") as mock_httpx,
            patch.object(tool, "_register_watch", AsyncMock()) as reg,
            patch.object(tool, "_schedule_renewal", AsyncMock()) as sched,
        ):
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute()

        assert result.error is False
        assert "enabled" in result.output.lower()
        reg.assert_awaited_once()
        sched.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fails_without_token(self):
        tool = _gmail_mod.SetupGmailWatchTool()
        result = await tool.execute()
        assert result.error is True
        assert "access token" in result.output.lower()

    @pytest.mark.asyncio
    async def test_fails_without_gcp_project(self):
        tool = self._make_tool()
        with patch.object(_gmail_mod, "_GCP_PROJECT_ID", ""):
            result = await tool.execute()
        assert result.error is True
        assert "GCP_PROJECT_ID" in result.output

    @pytest.mark.asyncio
    async def test_fails_without_orchestrator_url(self):
        tool = self._make_tool()
        with (
            patch.object(_gmail_mod, "_GCP_PROJECT_ID", "my-project"),
            patch.object(_gmail_mod, "_ORCHESTRATOR_URL_W", ""),
        ):
            result = await tool.execute()
        assert result.error is True

    @pytest.mark.asyncio
    async def test_handles_403_permission_denied(self):
        tool = self._make_tool()

        # Return a 403 response object — the code checks status_code before raise_for_status
        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.raise_for_status = MagicMock()
        mock_client = _make_async_client(resp_403)

        with (
            patch.object(_gmail_mod, "_ORCHESTRATOR_URL_W", "http://orch:8080"),
            patch.object(_gmail_mod, "_AGENT_USER_ID_W", "user-1"),
            patch.object(_gmail_mod, "_GCP_PROJECT_ID", "my-project"),
            patch.object(_gmail_mod, "httpx") as mock_httpx,
        ):
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute()

        assert result.error is True
        assert "permission" in result.output.lower()


# ── RenewGmailWatchTool ────────────────────────────────────


class TestRenewGmailWatchTool:
    @pytest.mark.asyncio
    async def test_delegates_to_setup(self):
        """RenewGmailWatchTool delegates to SetupGmailWatchTool."""
        tool = _gmail_mod.RenewGmailWatchTool()
        object.__setattr__(tool, "_context", {"access_token": "tok"})

        with patch.object(
            _gmail_mod.SetupGmailWatchTool,
            "execute",
            AsyncMock(return_value=ToolResult.success("renewed")),
        ) as mock_exec:
            result = await tool.execute()

        assert result.error is False
        mock_exec.assert_awaited_once()


# ── HandleGmailEventTool ───────────────────────────────────


class TestHandleGmailEventTool:
    def _make_tool(self, token: str = "tok-gmail"):
        tool = _gmail_mod.HandleGmailEventTool()
        object.__setattr__(tool, "_context", {"access_token": token})
        return tool

    @pytest.mark.asyncio
    async def test_success_returns_email_summaries(self):
        """Happy path: history + message fetch returns summaries."""
        tool = self._make_tool()

        history_resp = _mock_http_response(
            200, {"history": [{"messagesAdded": [{"message": {"id": "msg-1"}}]}]}
        )
        message_resp = _mock_http_response(
            200,
            {
                "payload": {
                    "headers": [
                        {"name": "From", "value": "alice@example.com"},
                        {"name": "Subject", "value": "Hello"},
                    ]
                }
            },
        )

        call_count = 0

        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "history" in url:
                return history_resp
            return message_resp

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_gmail_mod, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute(payload={"historyId": "555"})

        assert result.error is False
        assert "alice@example.com" in result.output
        assert "Hello" in result.output

    @pytest.mark.asyncio
    async def test_fails_without_token(self):
        tool = _gmail_mod.HandleGmailEventTool()
        result = await tool.execute(payload={"historyId": "1"})
        assert result.error is True

    @pytest.mark.asyncio
    async def test_fails_without_history_id(self):
        tool = self._make_tool()
        result = await tool.execute(payload={})
        assert result.error is True
        assert "historyId" in result.output

    @pytest.mark.asyncio
    async def test_handles_expired_history(self):
        """404 from history API returns graceful message."""
        tool = self._make_tool()

        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.raise_for_status = MagicMock()  # don't raise — code checks status_code

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=resp_404)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_gmail_mod, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute(payload={"historyId": "old-id"})

        assert result.error is False
        assert "expired" in result.output.lower() or "list_unread" in result.output

    @pytest.mark.asyncio
    async def test_no_new_messages(self):
        """Empty history returns informational success."""
        tool = self._make_tool()

        history_resp = _mock_http_response(200, {"history": []})
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=history_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_gmail_mod, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute(payload={"historyId": "555"})

        assert result.error is False
        assert "no new" in result.output.lower()


# ── SetupCalendarWatchTool ─────────────────────────────────


class TestSetupCalendarWatchTool:
    def _make_tool(self, token: str = "tok-cal"):
        tool = _cal_mod.SetupCalendarWatchTool()
        object.__setattr__(tool, "_context", {"access_token": token})
        return tool

    @pytest.mark.asyncio
    async def test_success_registers_and_schedules(self):
        """Happy path: events.watch() succeeds → registers + schedules renewal."""
        tool = self._make_tool()

        watch_resp = _mock_http_response(
            200,
            {
                "id": "chan-abc",
                "resourceId": "res-xyz",
                "expiration": "9999999999000",
            },
        )
        mock_client = _make_async_client(watch_resp)

        with (
            patch.object(_cal_mod, "_ORCHESTRATOR_URL_W", "http://orch:8080"),
            patch.object(_cal_mod, "_AGENT_USER_ID_W", "user-1"),
            patch.object(_cal_mod, "httpx") as mock_httpx,
            patch.object(tool, "_register_watch", AsyncMock()) as reg,
            patch.object(tool, "_schedule_renewal", AsyncMock()) as sched,
        ):
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute()

        assert result.error is False
        assert "enabled" in result.output.lower()
        reg.assert_awaited_once()
        sched.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fails_without_token(self):
        tool = _cal_mod.SetupCalendarWatchTool()
        result = await tool.execute()
        assert result.error is True
        assert "access token" in result.output.lower()

    @pytest.mark.asyncio
    async def test_fails_without_orchestrator_url(self):
        tool = self._make_tool()
        with (
            patch.object(_cal_mod, "_ORCHESTRATOR_URL_W", ""),
            patch.object(_cal_mod, "_AGENT_USER_ID_W", ""),
        ):
            result = await tool.execute()
        assert result.error is True

    @pytest.mark.asyncio
    async def test_handles_403_permission_denied(self):
        tool = self._make_tool()

        # Return a 403 response object — the code checks status_code before raise_for_status
        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.raise_for_status = MagicMock()
        mock_client = _make_async_client(resp_403)

        with (
            patch.object(_cal_mod, "_ORCHESTRATOR_URL_W", "http://orch:8080"),
            patch.object(_cal_mod, "_AGENT_USER_ID_W", "user-1"),
            patch.object(_cal_mod, "httpx") as mock_httpx,
        ):
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute()

        assert result.error is True
        assert "permission" in result.output.lower()


# ── HandleCalendarEventTool ────────────────────────────────


class TestHandleCalendarEventTool:
    def _make_tool(self, token: str = "tok-cal"):
        tool = _cal_mod.HandleCalendarEventTool()
        object.__setattr__(tool, "_context", {"access_token": token})
        return tool

    @pytest.mark.asyncio
    async def test_success_returns_event_summaries(self):
        """Happy path: updated events list returns summaries."""
        tool = self._make_tool()

        events_resp = _mock_http_response(
            200,
            {
                "items": [
                    {
                        "summary": "Team Standup",
                        "start": {"dateTime": "2026-02-22T09:00:00Z"},
                        "status": "confirmed",
                    }
                ]
            },
        )
        mock_client = _make_async_client(events_resp)

        with patch.object(_cal_mod, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute(payload={"x-goog-resource-state": "exists"})

        assert result.error is False
        assert "Team Standup" in result.output

    @pytest.mark.asyncio
    async def test_not_exists_state_returns_deletion_message(self):
        """x-goog-resource-state: not_exists returns deletion notice without API call."""
        tool = self._make_tool()

        with patch.object(_cal_mod, "httpx") as mock_httpx:
            result = await tool.execute(payload={"x-goog-resource-state": "not_exists"})

        assert result.error is False
        assert "deleted" in result.output.lower()
        mock_httpx.AsyncClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_without_token(self):
        tool = _cal_mod.HandleCalendarEventTool()
        result = await tool.execute(payload={"x-goog-resource-state": "exists"})
        assert result.error is True

    @pytest.mark.asyncio
    async def test_no_updated_events(self):
        """Empty items list returns informational success."""
        tool = self._make_tool()

        events_resp = _mock_http_response(200, {"items": []})
        mock_client = _make_async_client(events_resp)

        with patch.object(_cal_mod, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.HTTPStatusError = __import__("httpx").HTTPStatusError
            result = await tool.execute(payload={"x-goog-resource-state": "exists"})

        assert result.error is False
        assert "no recently updated" in result.output.lower()
