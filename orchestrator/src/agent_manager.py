"""
Agent Container Manager
───────────────────────
Spawns and manages per-user agent containers via Docker SDK.

Every user gets a dedicated agent container with isolated memory,
workspace, and credentials. The orchestrator provisions containers
on demand (first WS connection) and reaps idle ones automatically.
"""

from __future__ import annotations

import logging
import os
import hashlib
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger("orchestrator.agent_manager")


def _strip_wrapping_quotes(value: str) -> str:
    """Normalize env values that may be wrapped in single/double quotes."""
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


# ── Configuration ──────────────────────────────────────────

AGENT_IMAGE = os.getenv("AGENT_IMAGE", "core-ai-agent:latest")
AGENT_NETWORK = os.getenv("AGENT_NETWORK", "core-ai_default")
IDLE_TIMEOUT_MINUTES = int(os.getenv("AGENT_IDLE_TIMEOUT", "30"))
AGENT_SECRET = os.getenv("AGENT_SECRET", "")

# GCP / Pub/Sub — forwarded into agent containers so watch tools can register
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GMAIL_PUBSUB_TOPIC = os.getenv(
    "GMAIL_PUBSUB_TOPIC",
    f"projects/{GCP_PROJECT_ID}/topics/aether-gmail-events" if GCP_PROJECT_ID else "",
)
# PUBLIC_HOOK_URL: public HTTPS base URL of the orchestrator.
# Required for Google Calendar HTTP push (Google rejects non-HTTPS webhook URLs).
# Example: https://api.yourdomain.com
PUBLIC_HOOK_URL = os.getenv("PUBLIC_HOOK_URL", "")

# Dev: host path to app/ directory for hot-reload mounts into agent containers.
# When set, agent containers get source + plugin mounts with --reload.
# Example: /Users/you/code/core-ai/app
AGENT_DEV_ROOT = os.getenv("AGENT_DEV_ROOT", "")

# Shared model cache (host path mounted into orchestrator + agent containers)
AGENT_SHARED_MODELS_HOST_PATH = os.getenv("AGENT_SHARED_MODELS_HOST_PATH", "")
AGENT_SHARED_MODELS_ORCH_PATH = os.getenv("AGENT_SHARED_MODELS_ORCH_PATH", "")
AGENT_SHARED_MODELS_CONTAINER_PATH = os.getenv(
    "AGENT_SHARED_MODELS_CONTAINER_PATH", "/models"
)

# VAD model bootstrap defaults
AETHER_VAD_MODE_DEFAULT = os.getenv("AETHER_VAD_MODE_DEFAULT", "off")
AETHER_VAD_MODEL_RELATIVE_PATH = os.getenv(
    "AETHER_VAD_MODEL_RELATIVE_PATH", "silero/silero_vad.onnx"
)
AETHER_VAD_MODEL_URL = os.getenv("AETHER_VAD_MODEL_URL", "")
AETHER_VAD_MODEL_SHA256 = os.getenv("AETHER_VAD_MODEL_SHA256", "")

# Turn detector defaults (LiveKit model adapter)
AETHER_TURN_DETECTION_MODE_DEFAULT = os.getenv(
    "AETHER_TURN_DETECTION_MODE_DEFAULT", "off"
)
AETHER_TURN_MODEL_TYPE = os.getenv("AETHER_TURN_MODEL_TYPE", "en")
AETHER_TURN_MODEL_REPO = os.getenv("AETHER_TURN_MODEL_REPO", "livekit/turn-detector")
# v1.2.2-en has model.onnx (float32) which works correctly on ARM/aarch64.
# v0.4.1-intl only has model_q8.onnx (INT8 quantized for x86) which produces
# near-zero probabilities on ARM due to quantization incompatibility.
AETHER_TURN_MODEL_REVISION = os.getenv("AETHER_TURN_MODEL_REVISION", "v1.2.2-en")
AETHER_TURN_MODEL_FILENAME = os.getenv("AETHER_TURN_MODEL_FILENAME", "model.onnx")
AETHER_TURN_MODEL_RELATIVE_DIR = os.getenv(
    "AETHER_TURN_MODEL_RELATIVE_DIR", "turn-detector"
)
AETHER_TURN_MODEL_FILES = [
    p.strip()
    for p in os.getenv(
        "AETHER_TURN_MODEL_FILES",
        "onnx/model.onnx,languages.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,added_tokens.json,merges.txt,vocab.json",
    ).split(",")
    if p.strip()
]
AETHER_TURN_MIN_ENDPOINTING_DELAY = os.getenv(
    "AETHER_TURN_MIN_ENDPOINTING_DELAY", "0.5"
)
AETHER_TURN_MAX_ENDPOINTING_DELAY = os.getenv(
    "AETHER_TURN_MAX_ENDPOINTING_DELAY", "3.0"
)
# float32 model.onnx on ARM takes ~5-110ms — 2.0s is more than enough
AETHER_TURN_INFERENCE_TIMEOUT_SECONDS = os.getenv(
    "AETHER_TURN_INFERENCE_TIMEOUT_SECONDS", "2.0"
)


def _resolve_models_host_path() -> str:
    """Return absolute host path for shared model mounts.

    Docker SDK requires an absolute host path for bind mounts.
    In dev, resolve relative paths against the parent of AGENT_DEV_ROOT.
    """
    if not AGENT_SHARED_MODELS_HOST_PATH:
        return ""

    p = Path(AGENT_SHARED_MODELS_HOST_PATH)
    if p.is_absolute():
        return str(p)

    if AGENT_DEV_ROOT:
        try:
            return str((Path(AGENT_DEV_ROOT).resolve().parent / p).resolve())
        except Exception:
            pass

    # Fallback: keep as-is (may fail if Docker daemon cannot resolve it)
    return str(p)


AGENT_SHARED_MODELS_HOST_PATH_ABS = _resolve_models_host_path()

# System-wide fallback API keys (from orchestrator's own env)
SYSTEM_API_KEYS = {
    "OPENAI_API_KEY": _strip_wrapping_quotes(os.getenv("OPENAI_API_KEY", "")),
    "OPENROUTER_API_KEY": _strip_wrapping_quotes(os.getenv("OPENROUTER_API_KEY", "")),
    "DEEPGRAM_API_KEY": _strip_wrapping_quotes(os.getenv("DEEPGRAM_API_KEY", "")),
    "ELEVENLABS_API_KEY": _strip_wrapping_quotes(os.getenv("ELEVENLABS_API_KEY", "")),
    "SARVAM_API_KEY": _strip_wrapping_quotes(os.getenv("SARVAM_API_KEY", "")),
}

# ── Docker client (lazy) ──────────────────────────────────

_docker_client = None


def _get_docker():
    global _docker_client
    if _docker_client is None:
        import docker

        _docker_client = docker.from_env()
    return _docker_client


def _get_container_ip(container) -> str | None:
    """Get a container's IP address by inspecting its network settings.

    For host-network containers on OrbStack/Docker Desktop, the IP is on
    the Linux VM's network — which is reachable from other containers but
    not from the Mac host.  This is the correct address for orchestrator→agent
    communication.
    """
    try:
        container.reload()
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        # Prefer the compose network if attached
        for net_name, net_info in networks.items():
            ip = net_info.get("IPAddress", "")
            if ip and net_name != "host":
                return ip
        # Fallback: exec hostname -I inside the container
        exit_code, output = container.exec_run("hostname -I")
        if exit_code == 0:
            ip = output.decode().strip().split()[0]
            if ip:
                return ip
    except Exception as e:
        log.warning(f"Could not determine container IP: {e}")
    return None


def _provider_to_env(provider: str) -> str | None:
    """Map provider name (from api_keys table) to env var name."""
    mapping = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "sarvam": "SARVAM_API_KEY",
    }
    return mapping.get(provider.lower())


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_lfs_pointer_file(path: Path) -> bool:
    """Detect Git LFS pointer text file masquerading as model binary."""
    try:
        with path.open("rb") as f:
            head = f.read(256)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _resolve_model_paths() -> tuple[Path | None, str | None]:
    """Return (orchestrator-local-path, agent-container-path)."""
    if not AGENT_SHARED_MODELS_ORCH_PATH or not AGENT_SHARED_MODELS_CONTAINER_PATH:
        return None, None

    rel = AETHER_VAD_MODEL_RELATIVE_PATH.strip("/")
    orch_model = Path(AGENT_SHARED_MODELS_ORCH_PATH) / rel
    agent_model = f"{AGENT_SHARED_MODELS_CONTAINER_PATH.rstrip('/')}/{rel}"
    return orch_model, agent_model


def _resolve_turn_model_paths() -> tuple[Path | None, str | None]:
    """Return (orchestrator-local-turn-model-dir, agent-container-dir)."""
    if not AGENT_SHARED_MODELS_ORCH_PATH or not AGENT_SHARED_MODELS_CONTAINER_PATH:
        return None, None

    rel = AETHER_TURN_MODEL_RELATIVE_DIR.strip("/")
    orch_dir = Path(AGENT_SHARED_MODELS_ORCH_PATH) / rel
    agent_dir = f"{AGENT_SHARED_MODELS_CONTAINER_PATH.rstrip('/')}/{rel}"
    return orch_dir, agent_dir


def _turn_model_file_url(path: str) -> str:
    return (
        f"https://huggingface.co/{AETHER_TURN_MODEL_REPO}/resolve/"
        f"{AETHER_TURN_MODEL_REVISION}/{path}"
    )


def _ensure_turn_model_files() -> Path | None:
    """Ensure LiveKit turn-detector assets exist in shared cache."""
    orch_dir, _ = _resolve_turn_model_paths()
    if orch_dir is None:
        return None

    orch_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in AETHER_TURN_MODEL_FILES if not (orch_dir / p).exists()]
    if not missing:
        return orch_dir

    for rel_path in missing:
        dest = orch_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".{dest.name}.tmp"
        with tempfile.NamedTemporaryFile(
            delete=False, dir=str(dest.parent), suffix=suffix
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            url = _turn_model_file_url(rel_path)
            log.info("Downloading turn model file: %s", rel_path)
            urllib.request.urlretrieve(url, str(tmp_path))
            if _is_lfs_pointer_file(tmp_path):
                raise RuntimeError(
                    f"Downloaded turn model file is LFS pointer: {rel_path}"
                )
            tmp_path.replace(dest)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

    log.info("Turn detector assets ready at %s", orch_dir)
    return orch_dir


def _ensure_vad_model_file() -> Path | None:
    """Ensure VAD model exists in shared cache. Returns local model path or None."""
    orch_model, _ = _resolve_model_paths()
    if orch_model is None:
        return None

    orch_model.parent.mkdir(parents=True, exist_ok=True)

    if orch_model.exists():
        if _is_lfs_pointer_file(orch_model):
            log.warning("VAD model is a Git LFS pointer, re-downloading binary")
        elif AETHER_VAD_MODEL_SHA256:
            actual = _compute_sha256(orch_model)
            if actual.lower() == AETHER_VAD_MODEL_SHA256.lower():
                return orch_model
            log.warning("VAD model checksum mismatch, re-downloading")
        else:
            return orch_model

    if not AETHER_VAD_MODEL_URL:
        log.warning(
            "VAD model missing and AETHER_VAD_MODEL_URL not set; skipping download"
        )
        return None

    suffix = f".{orch_model.name}.tmp"
    with tempfile.NamedTemporaryFile(
        delete=False, dir=str(orch_model.parent), suffix=suffix
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        log.info("Downloading VAD model: %s", AETHER_VAD_MODEL_URL)
        urllib.request.urlretrieve(AETHER_VAD_MODEL_URL, str(tmp_path))

        if _is_lfs_pointer_file(tmp_path):
            raise RuntimeError("Downloaded VAD model is a Git LFS pointer, not binary")

        if AETHER_VAD_MODEL_SHA256:
            actual = _compute_sha256(tmp_path)
            if actual.lower() != AETHER_VAD_MODEL_SHA256.lower():
                raise RuntimeError("Downloaded VAD model checksum mismatch")

        tmp_path.replace(orch_model)
        log.info("VAD model ready at %s", orch_model)
        return orch_model
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


async def ensure_shared_models() -> None:
    """Bootstrap shared model artifacts used by agent containers."""
    try:
        _ensure_vad_model_file()
        _ensure_turn_model_files()
    except Exception as e:
        log.warning("Shared model bootstrap failed: %s", e)


# ── Container lifecycle ────────────────────────────────────


def _build_agent_environment(
    user_id: str,
    user_api_keys: dict[str, str] | None = None,
    user_preferences: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the complete environment dict for an agent container.

    This is the SINGLE source of truth for what env vars an agent gets.
    Priority (highest wins):
      1. Identity & infrastructure (always hardcoded, cannot be overridden)
      2. User preferences from DB (override orchestrator defaults)
      3. Orchestrator defaults (API keys, VAD/turn settings)

    The agent's LLM base_url is hardcoded to OpenRouter in the agent code
    itself (config.py), so we don't pass OPENAI_BASE_URL or AETHER_LLM_PROVIDER.
    """
    agent_id = f"agent-{user_id}"

    # ── Layer 1: Orchestrator defaults (API keys, VAD, turn detection) ──
    env: dict[str, str] = {}

    # API keys: system defaults from orchestrator env
    for key, value in SYSTEM_API_KEYS.items():
        cleaned = _strip_wrapping_quotes(value)
        if cleaned:
            env[key] = cleaned

    # User API keys override system defaults
    if user_api_keys:
        for provider, key in user_api_keys.items():
            env_var = _provider_to_env(provider)
            cleaned = _strip_wrapping_quotes(key)
            if env_var and cleaned:
                env[env_var] = cleaned

    # VAD defaults
    env["AETHER_VAD_MODE"] = AETHER_VAD_MODE_DEFAULT
    _, agent_model_path = _resolve_model_paths()
    if agent_model_path:
        env["AETHER_VAD_MODEL_PATH"] = agent_model_path

    # Turn detection defaults
    env["AETHER_TURN_DETECTION_MODE"] = AETHER_TURN_DETECTION_MODE_DEFAULT
    env["AETHER_TURN_MODEL_TYPE"] = AETHER_TURN_MODEL_TYPE
    env["AETHER_TURN_MODEL_REPO"] = AETHER_TURN_MODEL_REPO
    env["AETHER_TURN_MODEL_REVISION"] = AETHER_TURN_MODEL_REVISION
    env["AETHER_TURN_MODEL_FILENAME"] = AETHER_TURN_MODEL_FILENAME
    _, agent_turn_dir = _resolve_turn_model_paths()
    if agent_turn_dir:
        env["AETHER_TURN_MODEL_DIR"] = agent_turn_dir
    env["AETHER_TURN_MIN_ENDPOINTING_DELAY"] = AETHER_TURN_MIN_ENDPOINTING_DELAY
    env["AETHER_TURN_MAX_ENDPOINTING_DELAY"] = AETHER_TURN_MAX_ENDPOINTING_DELAY
    env["AETHER_TURN_INFERENCE_TIMEOUT_SECONDS"] = AETHER_TURN_INFERENCE_TIMEOUT_SECONDS

    # ── Layer 2: User preferences from DB (override defaults) ──
    if user_preferences:
        for key, value in user_preferences.items():
            if value:
                env[key] = value

    # ── Layer 3: Identity & infrastructure (always win) ──
    env["AETHER_AGENT_ID"] = agent_id
    env["AETHER_USER_ID"] = user_id
    env["AETHER_DB_PATH"] = "/data/aether_memory.db"
    env["AETHER_WORKING_DIR"] = "/workspace"
    env["AETHER_HOST"] = "0.0.0.0"
    env["AETHER_PORT"] = "8000"
    env["ORCHESTRATOR_URL"] = (
        "http://localhost:3080" if AGENT_DEV_ROOT else "http://orchestrator:9000"
    )
    if AGENT_DEV_ROOT:
        env["AETHER_AGENT_HOST"] = "host.docker.internal"
    if AGENT_SECRET:
        env["AGENT_SECRET"] = AGENT_SECRET

    # GCP / Pub/Sub — needed by SetupGmailWatchTool in the agent
    if GCP_PROJECT_ID:
        env["GCP_PROJECT_ID"] = GCP_PROJECT_ID
    if GMAIL_PUBSUB_TOPIC:
        env["GMAIL_PUBSUB_TOPIC"] = GMAIL_PUBSUB_TOPIC

    # Public HTTPS URL — needed by SetupCalendarWatchTool (Google rejects HTTP)
    if PUBLIC_HOOK_URL:
        env["PUBLIC_HOOK_URL"] = PUBLIC_HOOK_URL

    return env


async def provision_agent(
    user_id: str,
    user_api_keys: dict[str, str] | None = None,
    user_preferences: dict[str, str] | None = None,
) -> dict:
    """
    Create and start an agent container for a user.

    Returns: {"agent_id": str, "host": str, "port": int, "container_id": str}
    """
    agent_id = f"agent-{user_id}"
    container_name = f"aether-agent-{user_id}"

    environment = _build_agent_environment(user_id, user_api_keys, user_preferences)

    # Download shared models if VAD/turn detection is enabled.
    if AETHER_VAD_MODE_DEFAULT in ("shadow", "active"):
        _ensure_vad_model_file()
    if AETHER_TURN_DETECTION_MODE_DEFAULT in ("shadow", "active"):
        _ensure_turn_model_files()

    docker_client = _get_docker()

    # Check if container already exists (maybe stopped)
    try:
        existing = docker_client.containers.get(container_name)
        if existing.status == "running":
            log.info(f"Agent {agent_id} already running")
            is_host_net = (
                existing.attrs.get("HostConfig", {}).get("NetworkMode") == "host"
            )
            if is_host_net:
                agent_host = _get_container_ip(existing) or "host.docker.internal"
            else:
                agent_host = container_name
            return {
                "agent_id": agent_id,
                "host": agent_host,
                "port": 8000,
                "container_id": existing.id[:12],
            }
        # Stopped container — remove and recreate to pick up latest image,
        # env vars, and volume mounts (especially important for dev hot-reload).
        log.info(f"Removing stopped agent {agent_id} for fresh recreation")
        existing.remove(force=True)
    except Exception:
        pass  # Container doesn't exist — create it

    # Pull latest image from registry before creating the container.
    # Skipped in dev mode (AGENT_DEV_ROOT set) where the image is built locally
    # and there is no registry to pull from.  Pull failure is non-fatal so that
    # environments without registry access still work (cached image is used).
    if not AGENT_DEV_ROOT:
        try:
            log.info(f"Pulling latest image: {AGENT_IMAGE}")
            docker_client.images.pull(AGENT_IMAGE)
            log.info(f"Image pull complete: {AGENT_IMAGE}")
        except Exception as e:
            log.warning(f"Image pull failed — using cached image ({e})")

    # Create per-user volumes (idempotent)
    memory_vol = f"aether-memory-{user_id}"
    workspace_vol = f"aether-workspace-{user_id}"

    for vol_name in [memory_vol, workspace_vol]:
        try:
            docker_client.volumes.get(vol_name)
        except Exception:
            docker_client.volumes.create(vol_name)
            log.info(f"Created volume {vol_name}")

    # Build volume mounts
    vol_mounts = {
        memory_vol: {"bind": "/data", "mode": "rw"},
        workspace_vol: {"bind": "/workspace", "mode": "rw"},
    }

    # Shared models mount (read-only in agent containers)
    if AGENT_SHARED_MODELS_HOST_PATH_ABS and AGENT_SHARED_MODELS_CONTAINER_PATH:
        vol_mounts[AGENT_SHARED_MODELS_HOST_PATH_ABS] = {
            "bind": AGENT_SHARED_MODELS_CONTAINER_PATH,
            "mode": "ro",
        }

    # Dev: mount host source for hot-reload
    command = None
    if AGENT_DEV_ROOT:
        vol_mounts[f"{AGENT_DEV_ROOT}/src"] = {"bind": "/app/src", "mode": "rw"}
        vol_mounts[f"{AGENT_DEV_ROOT}/plugins"] = {"bind": "/app/plugins", "mode": "rw"}
        vol_mounts[f"{AGENT_DEV_ROOT}/skills"] = {"bind": "/app/skills", "mode": "rw"}
        command = (
            "uv run uvicorn aether.main:app "
            "--host 0.0.0.0 --port 8000 "
            "--reload --reload-dir /app/src"
        )

    # Dev mode: use host networking so WebRTC ICE candidates use the
    # host's real IP (reachable from iOS simulator / local clients).
    # Trade-off: only one agent at a time (port 8000 on host).
    use_host_network = bool(AGENT_DEV_ROOT)

    run_kwargs: dict = {
        "name": container_name,
        "detach": True,
        "environment": environment,
        "volumes": vol_mounts,
        "command": command,
        "mem_limit": "512m",
        "cpu_period": 100000,
        "cpu_quota": 100000,  # 1.0 CPU
    }

    if use_host_network:
        run_kwargs["network_mode"] = "host"
        log.info("Dev mode: agent using host networking (WebRTC-friendly)")
    else:
        run_kwargs["network"] = AGENT_NETWORK

    container = docker_client.containers.run(AGENT_IMAGE, **run_kwargs)

    # Determine the host the orchestrator should use to reach the agent.
    if use_host_network:
        # On OrbStack / Docker Desktop for Mac, network_mode=host puts the
        # container on the Linux VM's network — not the Mac's.
        # host.docker.internal points to the Mac, so it can't reach the agent.
        # Instead, inspect the container to get its actual IP on the VM.
        agent_host = _get_container_ip(container) or "host.docker.internal"

        # Also try to attach to the compose network for DNS-based fallback.
        try:
            net = docker_client.networks.get(AGENT_NETWORK)
            net.connect(container, aliases=[container_name])
            log.info(f"Attached {container_name} to {AGENT_NETWORK}")
        except Exception as e:
            log.warning(f"Could not attach to {AGENT_NETWORK}: {e} — using IP only")
    else:
        agent_host = container_name

    log.info(
        f"Created agent container {container_name} ({container.id[:12]}, host={agent_host})"
    )

    return {
        "agent_id": agent_id,
        "host": agent_host,
        "port": 8000,
        "container_id": container.id[:12],
    }


async def stop_agent(user_id: str) -> None:
    """Stop a user's agent container (preserves state in volumes)."""
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
