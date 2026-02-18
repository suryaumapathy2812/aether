"""Tests for the agent container manager — Docker SDK lifecycle."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from tests.conftest import make_record


# ── Provider-to-env mapping ───────────────────────────────


class TestProviderToEnv:
    def test_known_providers(self):
        from src.agent_manager import _provider_to_env

        assert _provider_to_env("openai") == "OPENAI_API_KEY"
        assert _provider_to_env("deepgram") == "DEEPGRAM_API_KEY"
        assert _provider_to_env("elevenlabs") == "ELEVENLABS_API_KEY"
        assert _provider_to_env("sarvam") == "SARVAM_API_KEY"

    def test_case_insensitive(self):
        from src.agent_manager import _provider_to_env

        assert _provider_to_env("OpenAI") == "OPENAI_API_KEY"
        assert _provider_to_env("DEEPGRAM") == "DEEPGRAM_API_KEY"

    def test_unknown_provider(self):
        from src.agent_manager import _provider_to_env

        assert _provider_to_env("anthropic") is None
        assert _provider_to_env("") is None


# ── Container provisioning ─────────────────────────────────


class TestProvisionAgent:
    def _mock_docker(self):
        """Create a mock Docker client."""
        docker_client = MagicMock()
        docker_client.volumes = MagicMock()
        docker_client.containers = MagicMock()
        return docker_client

    @pytest.mark.asyncio
    async def test_provision_creates_new_container(self):
        """Creates a new container when none exists."""
        docker_client = self._mock_docker()

        # containers.get raises (container doesn't exist)
        docker_client.containers.get.side_effect = Exception("Not found")

        # volumes.get raises (volumes don't exist)
        docker_client.volumes.get.side_effect = Exception("Not found")
        docker_client.volumes.create.return_value = MagicMock()

        # containers.run returns a mock container
        mock_container = MagicMock()
        mock_container.id = "abc123def456"
        docker_client.containers.run.return_value = mock_container

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import provision_agent

            result = await provision_agent("user-42")

        assert result["agent_id"] == "agent-user-42"
        assert result["host"] == "aether-agent-user-42"
        assert result["port"] == 8000
        assert result["container_id"] == "abc123def456"

        # Should have created volumes
        assert docker_client.volumes.create.call_count == 2

        # Should have called containers.run
        docker_client.containers.run.assert_called_once()
        run_kwargs = docker_client.containers.run.call_args
        assert run_kwargs[1]["name"] == "aether-agent-user-42"
        assert run_kwargs[1]["detach"] is True

    @pytest.mark.asyncio
    async def test_provision_restarts_stopped_container(self):
        """Restarts a stopped container instead of creating a new one."""
        docker_client = self._mock_docker()

        mock_container = MagicMock()
        mock_container.id = "existing123456"
        mock_container.status = "exited"
        docker_client.containers.get.return_value = mock_container

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import provision_agent

            result = await provision_agent("user-42")

        assert result["container_id"] == "existing1234"  # Truncated to 12 chars
        mock_container.start.assert_called_once()
        # Should NOT have called containers.run
        docker_client.containers.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_provision_returns_running_container(self):
        """Returns info for an already-running container without restarting."""
        docker_client = self._mock_docker()

        mock_container = MagicMock()
        mock_container.id = "running1234567"
        mock_container.status = "running"
        docker_client.containers.get.return_value = mock_container

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import provision_agent

            result = await provision_agent("user-42")

        assert result["container_id"] == "running12345"  # Truncated to 12
        mock_container.start.assert_not_called()
        docker_client.containers.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_provision_merges_user_api_keys(self):
        """User API keys override system defaults in container env."""
        docker_client = self._mock_docker()
        docker_client.containers.get.side_effect = Exception("Not found")
        docker_client.volumes.get.side_effect = Exception("Not found")

        mock_container = MagicMock()
        mock_container.id = "newcontainer12"
        docker_client.containers.run.return_value = mock_container

        user_keys = {"openai": "sk-user-key", "deepgram": "dg-user-key"}

        with (
            patch("src.agent_manager._get_docker", return_value=docker_client),
            patch("src.agent_manager.SYSTEM_API_KEYS", {"OPENAI_API_KEY": "sk-system"}),
        ):
            from src.agent_manager import provision_agent

            await provision_agent("user-42", user_keys)

        # Check the environment passed to containers.run
        run_kwargs = docker_client.containers.run.call_args[1]
        env = run_kwargs["environment"]
        assert env["OPENAI_API_KEY"] == "sk-user-key"  # User override
        assert env["DEEPGRAM_API_KEY"] == "dg-user-key"


# ── Container stop/destroy ─────────────────────────────────


class TestStopDestroy:
    @pytest.mark.asyncio
    async def test_stop_agent(self):
        """stop_agent calls container.stop()."""
        docker_client = MagicMock()
        mock_container = MagicMock()
        docker_client.containers.get.return_value = mock_container

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import stop_agent

            await stop_agent("user-42")

        docker_client.containers.get.assert_called_once_with("aether-agent-user-42")
        mock_container.stop.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_stop_nonexistent_container(self):
        """stop_agent handles missing container gracefully."""
        docker_client = MagicMock()
        docker_client.containers.get.side_effect = Exception("Not found")

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import stop_agent

            # Should not raise
            await stop_agent("user-99")

    @pytest.mark.asyncio
    async def test_destroy_agent(self):
        """destroy_agent calls container.remove(force=True)."""
        docker_client = MagicMock()
        mock_container = MagicMock()
        docker_client.containers.get.return_value = mock_container

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import destroy_agent

            await destroy_agent("user-42")

        mock_container.remove.assert_called_once_with(force=True)


# ── Container status ───────────────────────────────────────


class TestGetAgentStatus:
    @pytest.mark.asyncio
    async def test_running_container(self):
        docker_client = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        docker_client.containers.get.return_value = mock_container

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import get_agent_status

            assert await get_agent_status("user-42") == "running"

    @pytest.mark.asyncio
    async def test_nonexistent_container(self):
        docker_client = MagicMock()
        docker_client.containers.get.side_effect = Exception("Not found")

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import get_agent_status

            assert await get_agent_status("user-99") is None


# ── Reconciliation ─────────────────────────────────────────


class TestReconcileContainers:
    @pytest.mark.asyncio
    async def test_removes_orphaned_containers(self):
        """Containers not in DB get removed."""
        docker_client = MagicMock()
        orphan = MagicMock()
        orphan.name = "aether-agent-orphan"
        orphan.status = "running"
        docker_client.containers.list.return_value = [orphan]

        mock_pool = AsyncMock()
        mock_pool.fetchrow = AsyncMock(return_value=None)  # Not in DB

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import reconcile_containers

            await reconcile_containers(mock_pool)

        orphan.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_syncs_running_container_to_db(self):
        """Running container with stopped DB status gets synced."""
        docker_client = MagicMock()
        container = MagicMock()
        container.name = "aether-agent-user1"
        container.status = "running"
        docker_client.containers.list.return_value = [container]

        mock_pool = AsyncMock()
        # First fetchrow: agent exists in DB
        # Second fetchrow: DB says stopped
        mock_pool.fetchrow = AsyncMock(
            side_effect=[
                make_record(id="agent-user1"),
                make_record(status="stopped"),
            ]
        )
        mock_pool.execute = AsyncMock()

        with patch("src.agent_manager._get_docker", return_value=docker_client):
            from src.agent_manager import reconcile_containers

            await reconcile_containers(mock_pool)

        # Should have updated DB to running
        mock_pool.execute.assert_called_once()
        call_sql = mock_pool.execute.call_args[0][0]
        assert "running" in call_sql
