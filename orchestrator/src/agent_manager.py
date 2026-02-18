"""
Agent Container Manager
───────────────────────
Spawns and manages per-user agent containers via Docker SDK.

Only active when MULTI_USER_MODE=true. In dev mode (default),
the orchestrator uses the single shared agent from docker-compose.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("orchestrator.agent_manager")

# ── Configuration ──────────────────────────────────────────

MULTI_USER_MODE = os.getenv("MULTI_USER_MODE", "false").lower() == "true"
AGENT_IMAGE = os.getenv("AGENT_IMAGE", "core-ai-agent:latest")
AGENT_NETWORK = os.getenv("AGENT_NETWORK", "core-ai_default")
IDLE_TIMEOUT_MINUTES = int(os.getenv("AGENT_IDLE_TIMEOUT", "30"))
AGENT_SECRET = os.getenv("AGENT_SECRET", "")

# System-wide fallback API keys (from orchestrator's own env)
SYSTEM_API_KEYS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY", ""),
    "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY", ""),
    "SARVAM_API_KEY": os.getenv("SARVAM_API_KEY", ""),
}

# ── Docker client (lazy) ──────────────────────────────────

_docker_client = None


def _get_docker():
    global _docker_client
    if _docker_client is None:
        import docker

        _docker_client = docker.from_env()
    return _docker_client


def _provider_to_env(provider: str) -> str | None:
    """Map provider name (from api_keys table) to env var name."""
    mapping = {
        "openai": "OPENAI_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "sarvam": "SARVAM_API_KEY",
    }
    return mapping.get(provider.lower())


# ── Container lifecycle ────────────────────────────────────


async def provision_agent(
    user_id: str, user_api_keys: dict[str, str] | None = None
) -> dict:
    """
    Create and start an agent container for a user.

    Returns: {"agent_id": str, "host": str, "port": int, "container_id": str}
    """
    if not MULTI_USER_MODE:
        raise RuntimeError("Multi-user mode not enabled")

    agent_id = f"agent-{user_id}"
    container_name = f"aether-agent-{user_id}"

    # Merge API keys: user-specific override system defaults
    env_keys = {k: v for k, v in SYSTEM_API_KEYS.items() if v}
    if user_api_keys:
        for provider, key in user_api_keys.items():
            env_var = _provider_to_env(provider)
            if env_var and key:
                env_keys[env_var] = key

    environment = {
        **env_keys,
        "AETHER_AGENT_ID": agent_id,
        "AETHER_USER_ID": user_id,
        "AETHER_DB_PATH": "/data/aether_memory.db",
        "AETHER_WORKING_DIR": "/workspace",
        "AETHER_HOST": "0.0.0.0",
        "AETHER_PORT": "8000",
        "ORCHESTRATOR_URL": "http://orchestrator:9000",
        **({"AGENT_SECRET": AGENT_SECRET} if AGENT_SECRET else {}),
    }

    docker_client = _get_docker()

    # Check if container already exists (maybe stopped)
    try:
        existing = docker_client.containers.get(container_name)
        if existing.status == "running":
            log.info(f"Agent {agent_id} already running")
            return {
                "agent_id": agent_id,
                "host": container_name,
                "port": 8000,
                "container_id": existing.id[:12],
            }
        # Restart stopped container
        log.info(f"Restarting stopped agent {agent_id}")
        existing.start()
        return {
            "agent_id": agent_id,
            "host": container_name,
            "port": 8000,
            "container_id": existing.id[:12],
        }
    except Exception:
        pass  # Container doesn't exist — create it

    # Create per-user volumes (idempotent)
    memory_vol = f"aether-memory-{user_id}"
    workspace_vol = f"aether-workspace-{user_id}"

    for vol_name in [memory_vol, workspace_vol]:
        try:
            docker_client.volumes.get(vol_name)
        except Exception:
            docker_client.volumes.create(vol_name)
            log.info(f"Created volume {vol_name}")

    # Create and start container
    container = docker_client.containers.run(
        AGENT_IMAGE,
        name=container_name,
        detach=True,
        environment=environment,
        volumes={
            memory_vol: {"bind": "/data", "mode": "rw"},
            workspace_vol: {"bind": "/workspace", "mode": "rw"},
        },
        network=AGENT_NETWORK,
        # Resource limits
        mem_limit="512m",
        cpu_period=100000,
        cpu_quota=100000,  # 1.0 CPU
    )

    log.info(f"Created agent container {container_name} ({container.id[:12]})")

    return {
        "agent_id": agent_id,
        "host": container_name,
        "port": 8000,
        "container_id": container.id[:12],
    }


async def stop_agent(user_id: str) -> None:
    """Stop a user's agent container (preserves state in volumes)."""
    if not MULTI_USER_MODE:
        return

    container_name = f"aether-agent-{user_id}"
    docker_client = _get_docker()

    try:
        container = docker_client.containers.get(container_name)
        container.stop(timeout=10)
        log.info(f"Stopped agent {container_name}")
    except Exception as e:
        log.warning(f"Failed to stop {container_name}: {e}")


async def destroy_agent(user_id: str) -> None:
    """Remove a user's agent container (volumes preserved for data safety)."""
    if not MULTI_USER_MODE:
        return

    container_name = f"aether-agent-{user_id}"
    docker_client = _get_docker()

    try:
        container = docker_client.containers.get(container_name)
        container.remove(force=True)
        log.info(f"Destroyed agent {container_name}")
    except Exception as e:
        log.warning(f"Failed to destroy {container_name}: {e}")


async def get_agent_status(user_id: str) -> str | None:
    """Check if a user's agent container is running. Returns Docker status or None."""
    if not MULTI_USER_MODE:
        return None

    container_name = f"aether-agent-{user_id}"
    docker_client = _get_docker()

    try:
        container = docker_client.containers.get(container_name)
        return container.status  # "running", "exited", "created", etc.
    except Exception:
        return None


async def reconcile_containers(pool) -> None:
    """
    Startup reconciliation: find orphaned agent containers
    (running but not in DB) and clean them up.
    """
    if not MULTI_USER_MODE:
        return

    try:
        docker_client = _get_docker()
        containers = docker_client.containers.list(
            all=True, filters={"name": "aether-agent-"}
        )
        for container in containers:
            # Extract agent_id from container name (aether-agent-{user_id} → agent-{user_id})
            agent_id = container.name.replace("aether-", "")
            row = await pool.fetchrow("SELECT id FROM agents WHERE id = $1", agent_id)
            if not row:
                log.warning(f"Orphaned container {container.name} — removing")
                container.remove(force=True)
            else:
                # Sync status: if container is running but DB says stopped, update DB
                db_row = await pool.fetchrow(
                    "SELECT status FROM agents WHERE id = $1", agent_id
                )
                if (
                    container.status == "running"
                    and db_row
                    and db_row["status"] != "running"
                ):
                    await pool.execute(
                        "UPDATE agents SET status = 'running', last_health = now() WHERE id = $1",
                        agent_id,
                    )
                    log.info(f"Reconciled {agent_id}: DB updated to running")
                elif (
                    container.status != "running"
                    and db_row
                    and db_row["status"] == "running"
                ):
                    await pool.execute(
                        "UPDATE agents SET status = 'stopped' WHERE id = $1",
                        agent_id,
                    )
                    log.info(f"Reconciled {agent_id}: DB updated to stopped")
    except Exception as e:
        log.error(f"Container reconciliation failed: {e}")
