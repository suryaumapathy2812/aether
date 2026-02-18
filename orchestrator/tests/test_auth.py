"""Tests for the auth module — session/device token validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_record


# ── Helpers ────────────────────────────────────────────────


def _make_request(auth_header: str | None = None) -> MagicMock:
    """Create a mock FastAPI Request with optional Authorization header."""
    request = MagicMock()
    headers = {}
    if auth_header:
        headers["authorization"] = auth_header
    request.headers = headers
    return request


def _make_websocket(
    token_param: str | None = None, auth_header: str | None = None
) -> MagicMock:
    """Create a mock FastAPI WebSocket with optional query params and headers."""
    ws = MagicMock()
    ws.query_params = {}
    if token_param:
        ws.query_params["token"] = token_param
    headers = {}
    if auth_header:
        headers["authorization"] = auth_header
    ws.headers = headers
    return ws


# ── Token extraction ───────────────────────────────────────


class TestExtractToken:
    def test_extract_bearer_token(self):
        from src.auth import _extract_token_from_request

        request = _make_request("Bearer abc123")
        assert _extract_token_from_request(request) == "abc123"

    def test_extract_no_header(self):
        from src.auth import _extract_token_from_request

        request = _make_request(None)
        assert _extract_token_from_request(request) is None

    def test_extract_non_bearer(self):
        from src.auth import _extract_token_from_request

        request = _make_request("Basic dXNlcjpwYXNz")
        assert _extract_token_from_request(request) is None

    def test_extract_empty_bearer(self):
        from src.auth import _extract_token_from_request

        request = _make_request("Bearer ")
        assert _extract_token_from_request(request) == ""


# ── Token validation ───────────────────────────────────────


class TestValidateToken:
    @pytest.mark.asyncio
    async def test_valid_session_token(self):
        """Session token found in session table returns user_id."""
        from src.auth import _validate_token

        mock_pool = AsyncMock()
        mock_pool.fetchrow = AsyncMock(return_value=make_record(user_id="user-123"))

        with patch("src.auth.get_pool", return_value=mock_pool):
            result = await _validate_token("valid-session-token")

        assert result == "user-123"
        # Should have queried session table
        mock_pool.fetchrow.assert_called_once()
        call_sql = mock_pool.fetchrow.call_args[0][0]
        assert "session" in call_sql

    @pytest.mark.asyncio
    async def test_valid_device_token(self):
        """Device token found in devices table returns user_id."""
        from src.auth import _validate_token

        mock_pool = AsyncMock()
        # First call (session table) returns None, second (devices) returns match
        mock_pool.fetchrow = AsyncMock(
            side_effect=[None, make_record(user_id="user-456")]
        )
        mock_pool.execute = AsyncMock()

        with patch("src.auth.get_pool", return_value=mock_pool):
            result = await _validate_token("device-token-xyz")

        assert result == "user-456"
        # Should have updated last_seen
        mock_pool.execute.assert_called_once()
        call_sql = mock_pool.execute.call_args[0][0]
        assert "last_seen" in call_sql

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        """Token not found anywhere returns None."""
        from src.auth import _validate_token

        mock_pool = AsyncMock()
        mock_pool.fetchrow = AsyncMock(return_value=None)

        with patch("src.auth.get_pool", return_value=mock_pool):
            result = await _validate_token("bad-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_expired_session_falls_through_to_devices(self):
        """Expired session token (not returned by query) checks devices table."""
        from src.auth import _validate_token

        mock_pool = AsyncMock()
        # Session query returns None (expired), device query also None
        mock_pool.fetchrow = AsyncMock(return_value=None)

        with patch("src.auth.get_pool", return_value=mock_pool):
            result = await _validate_token("expired-token")

        assert result is None
        # Should have been called twice (session + devices)
        assert mock_pool.fetchrow.call_count == 2


# ── get_user_id (HTTP) ─────────────────────────────────────


class TestGetUserId:
    @pytest.mark.asyncio
    async def test_valid_request(self):
        """Valid Bearer token returns user_id."""
        from src.auth import get_user_id

        request = _make_request("Bearer good-token")

        with patch(
            "src.auth._validate_token", new_callable=AsyncMock, return_value="user-789"
        ):
            result = await get_user_id(request)

        assert result == "user-789"

    @pytest.mark.asyncio
    async def test_missing_auth_raises_401(self):
        """No Authorization header raises 401."""
        from fastapi import HTTPException
        from src.auth import get_user_id

        request = _make_request(None)

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        """Invalid token raises 401."""
        from fastapi import HTTPException
        from src.auth import get_user_id

        request = _make_request("Bearer bad-token")

        with patch(
            "src.auth._validate_token", new_callable=AsyncMock, return_value=None
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_user_id(request)
            assert exc_info.value.status_code == 401


# ── get_user_id_from_ws (WebSocket) ───────────────────────


class TestGetUserIdFromWs:
    @pytest.mark.asyncio
    async def test_token_from_query_param(self):
        """WebSocket token from ?token= query param."""
        from src.auth import get_user_id_from_ws

        ws = _make_websocket(token_param="ws-token")

        with patch(
            "src.auth._validate_token", new_callable=AsyncMock, return_value="user-ws"
        ):
            result = await get_user_id_from_ws(ws)

        assert result == "user-ws"

    @pytest.mark.asyncio
    async def test_token_from_header(self):
        """WebSocket token from Authorization header."""
        from src.auth import get_user_id_from_ws

        ws = _make_websocket(auth_header="Bearer header-token")

        with patch(
            "src.auth._validate_token", new_callable=AsyncMock, return_value="user-hdr"
        ):
            result = await get_user_id_from_ws(ws)

        assert result == "user-hdr"

    @pytest.mark.asyncio
    async def test_query_param_takes_precedence(self):
        """Query param token is preferred over header."""
        from src.auth import get_user_id_from_ws

        ws = _make_websocket(
            token_param="param-token", auth_header="Bearer header-token"
        )

        with patch(
            "src.auth._validate_token",
            new_callable=AsyncMock,
            return_value="user-param",
        ) as mock_validate:
            result = await get_user_id_from_ws(ws)

        assert result == "user-param"
        mock_validate.assert_called_once_with("param-token")

    @pytest.mark.asyncio
    async def test_no_token_returns_none(self):
        """No token anywhere returns None (no exception)."""
        from src.auth import get_user_id_from_ws

        ws = _make_websocket()
        result = await get_user_id_from_ws(ws)
        assert result is None
