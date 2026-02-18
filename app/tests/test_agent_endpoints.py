"""Tests for agent endpoints and helpers added in the multi-user session.

Covers:
- _agent_auth_headers() — Bearer header construction
- _send() — WebSocket send with client_state check and timeout
- Memory REST endpoints (/memory/facts, /memory/sessions, /memory/conversations)
- _register_with_orchestrator() — registration call
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── _agent_auth_headers ────────────────────────────────────


class TestAgentAuthHeaders:
    def test_with_secret(self):
        """Returns Bearer header when AGENT_SECRET is set."""
        with patch("aether.main.AGENT_SECRET", "my-secret"):
            from aether.main import _agent_auth_headers

            headers = _agent_auth_headers()
            assert headers == {"Authorization": "Bearer my-secret"}

    def test_without_secret(self):
        """Returns empty dict when AGENT_SECRET is empty."""
        with patch("aether.main.AGENT_SECRET", ""):
            from aether.main import _agent_auth_headers

            headers = _agent_auth_headers()
            assert headers == {}


# ── _send ──────────────────────────────────────────────────


class _ClientState(Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"


class TestSend:
    def _make_ws(self, state: str = "CONNECTED"):
        """Create a mock WebSocket with configurable client_state."""
        ws = MagicMock()
        ws.client_state = MagicMock()
        ws.client_state.name = state
        ws.send_text = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_send_connected(self):
        """Sends JSON message when client is connected."""
        ws = self._make_ws("CONNECTED")

        from aether.main import _send

        await _send(ws, "text_chunk", "hello", index=0)

        ws.send_text.assert_called_once()
        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "text_chunk"
        assert sent["data"] == "hello"
        assert sent["index"] == 0

    @pytest.mark.asyncio
    async def test_send_disconnected_drops_silently(self):
        """Drops message silently when client is disconnected."""
        ws = self._make_ws("DISCONNECTED")

        from aether.main import _send

        await _send(ws, "text_chunk", "hello")

        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_timeout_handled(self):
        """Handles send timeout gracefully (no exception raised)."""
        ws = self._make_ws("CONNECTED")

        async def slow_send(text):
            await asyncio.sleep(100)

        ws.send_text = slow_send

        from aether.main import _send

        # Patch config to have a very short timeout
        with patch("aether.main.config") as mock_config:
            mock_config.server.ws_send_timeout = 0.01
            # Should not raise
            await _send(ws, "text_chunk", "hello")

    @pytest.mark.asyncio
    async def test_send_exception_handled(self):
        """Handles send exceptions gracefully (no exception raised)."""
        ws = self._make_ws("CONNECTED")
        ws.send_text = AsyncMock(side_effect=RuntimeError("connection reset"))

        from aether.main import _send

        # Should not raise
        await _send(ws, "text_chunk", "hello")

    @pytest.mark.asyncio
    async def test_send_default_data_empty_string(self):
        """Default data parameter is empty string."""
        ws = self._make_ws("CONNECTED")

        from aether.main import _send

        await _send(ws, "status")

        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["data"] == ""


# ── Memory REST endpoints ──────────────────────────────────


class TestMemoryEndpoints:
    """Test the agent's /memory/* HTTP endpoints via FastAPI TestClient."""

    @pytest.fixture
    def agent_client(self):
        """TestClient for the agent app with mocked memory store and providers."""
        mock_store = MagicMock()
        mock_store.start = AsyncMock()
        mock_store.stop = AsyncMock()
        mock_store.get_facts = AsyncMock(
            return_value=["User likes Python", "User lives in Chennai"]
        )
        mock_store.get_session_summaries = AsyncMock(
            return_value=[
                {"session_id": "s1", "summary": "Built a Flask app", "turns": 10}
            ]
        )
        mock_store.get_recent = AsyncMock(
            return_value=[{"user_message": "Hello", "assistant_message": "Hi there!"}]
        )

        mock_stt = MagicMock()
        mock_stt.start = AsyncMock()
        mock_stt.stop = AsyncMock()
        mock_llm = MagicMock()
        mock_llm.start = AsyncMock()
        mock_llm.stop = AsyncMock()
        mock_tts = MagicMock()
        mock_tts.start = AsyncMock()
        mock_tts.stop = AsyncMock()

        with (
            patch("aether.main.memory_store", mock_store),
            patch("aether.main.stt_provider", mock_stt),
            patch("aether.main.llm_provider", mock_llm),
            patch("aether.main.tts_provider", mock_tts),
            patch("aether.main._register_with_orchestrator", new_callable=AsyncMock),
            patch("aether.main.ORCHESTRATOR_URL", ""),
        ):
            from fastapi.testclient import TestClient
            from aether.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, mock_store

    def test_memory_facts(self, agent_client):
        """GET /memory/facts returns facts from memory store."""
        client, store = agent_client
        resp = client.get("/memory/facts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["facts"] == ["User likes Python", "User lives in Chennai"]
        store.get_facts.assert_called_once()

    def test_memory_sessions(self, agent_client):
        """GET /memory/sessions returns session summaries."""
        client, store = agent_client
        resp = client.get("/memory/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["summary"] == "Built a Flask app"

    def test_memory_conversations_default_limit(self, agent_client):
        """GET /memory/conversations uses default limit of 20."""
        client, store = agent_client
        resp = client.get("/memory/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["conversations"]) == 1
        store.get_recent.assert_called_once_with(limit=20)

    def test_memory_conversations_custom_limit(self, agent_client):
        """GET /memory/conversations respects limit query param."""
        client, store = agent_client
        resp = client.get("/memory/conversations?limit=5")
        assert resp.status_code == 200
        store.get_recent.assert_called_once_with(limit=5)


# ── Registration ───────────────────────────────────────────


class TestRegistration:
    @pytest.mark.asyncio
    async def test_register_skips_without_url(self):
        """Skips registration when ORCHESTRATOR_URL is empty."""
        with patch("aether.main.ORCHESTRATOR_URL", ""):
            from aether.main import _register_with_orchestrator

            # Should not raise or make any HTTP calls
            await _register_with_orchestrator()

    @pytest.mark.asyncio
    async def test_register_sends_correct_payload(self):
        """Sends correct registration payload to orchestrator."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("aether.main.ORCHESTRATOR_URL", "http://orchestrator:9000"),
            patch("aether.main.AGENT_ID", "agent-test"),
            patch("aether.main.AGENT_SECRET", "test-secret"),
            patch("aether.main.AGENT_USER_ID", ""),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            from aether.main import _register_with_orchestrator

            await _register_with_orchestrator()

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "agents/register" in call_kwargs[0][0]
        payload = call_kwargs[1]["json"]
        assert payload["agent_id"] == "agent-test"
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-secret"

    @pytest.mark.asyncio
    async def test_register_handles_failure(self):
        """Handles registration failure gracefully (no exception raised)."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("aether.main.ORCHESTRATOR_URL", "http://orchestrator:9000"),
            patch("aether.main.AGENT_SECRET", ""),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            from aether.main import _register_with_orchestrator

            # Should not raise
            await _register_with_orchestrator()
