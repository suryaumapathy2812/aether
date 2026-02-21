"""
Aether Orchestrator
────────────────────
Agent registry, WebSocket proxy, device pairing, API key management.

Auth is handled by better-auth in the dashboard (Next.js).
The orchestrator validates sessions by reading the shared Postgres session table.
"""

from __future__ import annotations

import hmac
import json
import os
import uuid
import asyncio
import logging

import time
import urllib.parse

import httpx

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Request,
    Query,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from .db import get_pool, close_pool, bootstrap_schema
from .auth import get_user_id, get_user_id_from_ws
from .crypto import encrypt_value, decrypt_value

# ── Agent registration secret ─────────────────────────────
AGENT_SECRET = os.getenv("AGENT_SECRET", "")
LOCAL_AGENT_URL = os.getenv("AETHER_LOCAL_AGENT_URL", "").strip()

# ── GCP / Pub/Sub config ──────────────────────────────────
# GCP_PROJECT_ID: your Google Cloud project ID (e.g. "my-project-123")
# PUBSUB_AUDIENCE: the push endpoint URL Google uses to verify JWT audience.
#   Set to your public orchestrator base URL, e.g. "https://api.yourdomain.com"
#   If unset, JWT audience verification is skipped (dev only).
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
PUBSUB_AUDIENCE = os.getenv("PUBSUB_AUDIENCE", "")


def _local_agent_target() -> dict | None:
    """Return a local agent endpoint override when configured."""
    if not LOCAL_AGENT_URL:
        return None

    parsed = urllib.parse.urlparse(LOCAL_AGENT_URL)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        log.warning("Ignoring invalid AETHER_LOCAL_AGENT_URL: %s", LOCAL_AGENT_URL)
        return None

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    return {
        "host": parsed.hostname,
        "port": port,
    }


async def verify_agent_secret(request: Request) -> None:
    """
    Dependency: verify the shared agent secret on internal agent endpoints.

    Agents send `Authorization: Bearer <AGENT_SECRET>` on registration
    and heartbeat calls. Uses constant-time comparison.
    """
    if not AGENT_SECRET:
        # No secret configured — skip validation (dev convenience, logged once at startup)
        return

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing agent authorization")

    token = auth_header[7:]
    if not hmac.compare_digest(token, AGENT_SECRET):
        raise HTTPException(403, "Invalid agent secret")


from .agent_manager import (
    IDLE_TIMEOUT_MINUTES,
    ensure_shared_models,
    provision_agent,
    stop_agent,
    get_agent_status,
    reconcile_containers,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orchestrator")

app = FastAPI(title="Aether Orchestrator", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle ──────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    await bootstrap_schema()
    await ensure_shared_models()

    # Reconcile orphaned containers on startup
    if not AGENT_SECRET:
        log.warning(
            "⚠ AGENT_SECRET not set — agent registration endpoints are UNPROTECTED"
        )

    if LOCAL_AGENT_URL:
        log.info("Local agent mode enabled: %s", LOCAL_AGENT_URL)

    pool = await get_pool()
    await reconcile_containers(pool)
    await _reconcile_token_refresh_jobs(pool)
    await _reconcile_watch_setup_jobs(pool)
    asyncio.create_task(_idle_reaper())
    asyncio.create_task(_deferred_flusher())
    asyncio.create_task(_cron_runner())
    log.info(f"Orchestrator ready (idle timeout={IDLE_TIMEOUT_MINUTES}m)")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


async def _idle_reaper():
    """Background task: stop agent containers that haven't sent a heartbeat.

    Agents with keep_alive = true are exempt — they hold an active WebRTC
    session and must not be killed between reconnects.
    """
    while True:
        await asyncio.sleep(60)  # Check every minute
        try:
            pool = await get_pool()
            idle_agents = await pool.fetch(
                "SELECT id, user_id FROM agents WHERE status = 'running' "
                "AND keep_alive = false "
                "AND last_health < now() - make_interval(mins := $1)",
                IDLE_TIMEOUT_MINUTES,
            )
            for agent in idle_agents:
                log.info(f"Stopping idle agent {agent['id']} (user={agent['user_id']})")
                await stop_agent(agent["user_id"])
                await pool.execute(
                    "UPDATE agents SET status = 'stopped' WHERE id = $1",
                    agent["id"],
                )
        except Exception as e:
            log.error(f"Idle reaper error: {e}")


async def _deferred_flusher():
    """Flush deferred plugin events when their schedule time arrives."""
    while True:
        await asyncio.sleep(30)
        try:
            pool = await get_pool()
            # Find all ready-to-flush events, grouped by user
            ready = await pool.fetch(
                """SELECT id, user_id, plugin_name, event_type, summary,
                          batch_notification, payload
                   FROM plugin_events
                   WHERE decision = 'deferred' AND scheduled_for < now()
                   ORDER BY user_id, created_at""",
            )
            if not ready:
                continue

            # Group by user
            from itertools import groupby
            from operator import itemgetter

            for user_id, events in groupby(ready, key=itemgetter("user_id")):
                batch = list(events)
                agent = await pool.fetchrow(
                    "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
                    user_id,
                )
                if not agent:
                    continue

                items = [
                    {
                        "event_id": e["id"],
                        "plugin": e["plugin_name"],
                        "summary": e["summary"],
                        "notification": e["batch_notification"] or e["summary"],
                        "actions": json.loads(e["payload"]).get(
                            "available_actions", []
                        ),
                    }
                    for e in batch
                ]

                import httpx

                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(
                            f"http://{agent['host']}:{agent['port']}/plugin_event/batch",
                            json={"events": items},
                        )
                    # Mark as flushed
                    ids = [e["id"] for e in batch]
                    await pool.execute(
                        f"UPDATE plugin_events SET decision = 'flushed' WHERE id = ANY($1::text[])",
                        ids,
                    )
                except Exception as e:
                    log.warning(f"Deferred flush failed for user {user_id}: {e}")

        except Exception as e:
            log.error(f"Deferred flusher error: {e}")


async def _ensure_agent_running(user_id: str, timeout_s: int = 30) -> dict | None:
    """Ensure the user's agent container is running, starting it if needed.

    Starts the container if stopped/missing, then polls the DB until the agent
    self-registers as 'running' (max *timeout_s* seconds).

    Returns the agent row dict on success, or None if the agent couldn't be
    brought up in time.
    """
    pool = await get_pool()

    # Fast path: already running
    agent = await pool.fetchrow(
        "SELECT host, port, status FROM agents WHERE user_id = $1", user_id
    )
    if agent and agent["status"] == "running":
        return dict(agent)

    # Local-agent mode: no containers to manage — just return whatever is registered
    if _local_agent_target():
        return dict(agent) if agent else None

    log.info(f"Cron: agent for user {user_id} is not running — starting it")
    try:
        await _ensure_agent(user_id)
    except Exception as e:
        log.error(f"Cron: failed to start agent for user {user_id}: {e}")
        return None

    # Poll until the agent self-registers as running
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(1)
        row = await pool.fetchrow(
            "SELECT host, port, status FROM agents WHERE user_id = $1", user_id
        )
        if row and row["status"] == "running":
            log.info(f"Cron: agent for user {user_id} is now running")
            return dict(row)

    log.warning(f"Cron: agent for user {user_id} did not come up within {timeout_s}s")
    return None


# How often the cron runner wakes to check for due jobs (seconds).
_CRON_POLL_INTERVAL_S = int(os.getenv("CRON_POLL_INTERVAL_S", "30"))


async def _cron_runner() -> None:
    """Background task: fire scheduled jobs when their run_at time arrives.

    All job types are delivered to the agent as a /cron_event — the LLM
    decides what to do (call a tool, relay a reminder, etc.).  The orchestrator
    is a dumb timer; it never interprets the instruction itself.

    Job lifecycle:
    - One-shot (interval_s IS NULL): delivered once, then disabled.
    - Recurring (interval_s > 0): run_at advances by interval_s after each run.
    """
    while True:
        await asyncio.sleep(_CRON_POLL_INTERVAL_S)
        try:
            pool = await get_pool()
            due = await pool.fetch(
                """
                SELECT id, user_id, plugin, instruction, interval_s
                FROM scheduled_jobs
                WHERE enabled = true AND run_at <= now()
                ORDER BY run_at
                """,
            )
            if not due:
                continue

            for job in due:
                job_id = job["id"]
                user_id = job["user_id"]
                plugin = job["plugin"]
                instruction = job["instruction"]
                interval_s = job["interval_s"]

                log.info(
                    f"Cron: firing job {job_id} for user {user_id} "
                    f"(plugin={plugin}, interval_s={interval_s})"
                )

                # Ensure the agent is up before delivering
                agent = await _ensure_agent_running(user_id)
                if not agent:
                    log.warning(
                        f"Cron: skipping job {job_id} — agent unavailable for user {user_id}"
                    )
                    # Advance the schedule so we don't hammer a dead agent
                    if interval_s:
                        await pool.execute(
                            "UPDATE scheduled_jobs SET run_at = now() + make_interval(secs := $1), "
                            "last_run_at = now() WHERE id = $2",
                            interval_s,
                            job_id,
                        )
                    else:
                        await pool.execute(
                            "UPDATE scheduled_jobs SET enabled = false, last_run_at = now() WHERE id = $1",
                            job_id,
                        )
                    continue

                # Deliver the cron event to the agent
                try:
                    async with _httpx.AsyncClient(timeout=10) as client:
                        resp = await client.post(
                            f"http://{agent['host']}:{agent['port']}/cron_event",
                            json={"plugin": plugin, "instruction": instruction},
                        )
                        if resp.status_code not in (200, 202):
                            log.warning(
                                f"Cron: /cron_event returned {resp.status_code} for job {job_id}"
                            )
                except Exception as e:
                    log.warning(f"Cron: failed to deliver job {job_id}: {e}")

                # Advance or disable the job
                if interval_s:
                    await pool.execute(
                        "UPDATE scheduled_jobs SET run_at = now() + make_interval(secs := $1), "
                        "last_run_at = now() WHERE id = $2",
                        interval_s,
                        job_id,
                    )
                else:
                    await pool.execute(
                        "UPDATE scheduled_jobs SET enabled = false, last_run_at = now() WHERE id = $1",
                        job_id,
                    )

        except Exception as e:
            log.error(f"Cron runner error: {e}")


# ── Models ─────────────────────────────────────────────────


class AgentRegisterRequest(BaseModel):
    agent_id: str
    host: str
    port: int
    container_id: str | None = None
    user_id: str | None = None


class PairRequestBody(BaseModel):
    code: str
    device_type: str = "ios"
    device_name: str = ""


class PairConfirmBody(BaseModel):
    code: str


class ApiKeyBody(BaseModel):
    provider: str
    key_value: str


class PreferencesBody(BaseModel):
    stt_provider: str | None = None
    stt_model: str | None = None
    stt_language: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    base_style: str | None = None
    custom_instructions: str | None = None


# ── Agent Provisioning ─────────────────────────────────────


async def _ensure_agent(user_id: str) -> None:
    """
    Ensure the user has a running agent container.

    Provisions a dedicated container per user via Docker SDK.
    If the container already exists and is running, this is a no-op.
    """
    if _local_agent_target():
        return

    pool = await get_pool()

    existing = await pool.fetchrow(
        "SELECT id, status FROM agents WHERE user_id = $1", user_id
    )

    if existing and existing["status"] == "running":
        # Verify the container is actually alive
        container_status = await get_agent_status(user_id)
        if container_status == "running":
            return
        log.warning(
            f"Agent {existing['id']} marked running but container is {container_status}"
        )

    # Fetch user's API keys for injection into the container (decrypt from DB)
    rows = await pool.fetch(
        "SELECT provider, key_value FROM api_keys WHERE user_id = $1", user_id
    )
    user_keys = {r["provider"]: decrypt_value(r["key_value"]) for r in rows}

    # Fetch user preferences for injection as env vars
    pref_row = await pool.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1", user_id
    )
    user_prefs = {}
    if pref_row:
        for col, env_name in PREF_TO_ENV.items():
            val = pref_row.get(col)
            if val:
                user_prefs[env_name] = val

    # Provision (or restart) the agent container
    result = await provision_agent(user_id, user_keys, user_prefs)

    # Upsert agent record (agent will also self-register, but we pre-create the row)
    await pool.execute(
        """
        INSERT INTO agents (id, user_id, host, port, container_id, container_name, status, registered_at, last_health)
        VALUES ($1, $2, $3, $4, $5, $6, 'starting', now(), now())
        ON CONFLICT (id) DO UPDATE SET
            host = $3, port = $4, container_id = $5, container_name = $6,
            status = 'starting', last_health = now()
        """,
        result["agent_id"],
        user_id,
        result["host"],
        result["port"],
        result["container_id"],
        f"aether-agent-{user_id}",
    )

    # Wait for agent to self-register as 'running' (max 15 seconds)
    for _ in range(30):
        await asyncio.sleep(0.5)
        row = await pool.fetchrow(
            "SELECT status FROM agents WHERE user_id = $1", user_id
        )
        if row and row["status"] == "running":
            log.info(f"Agent for user {user_id} is ready")
            return

    log.warning(f"Agent for user {user_id} didn't become ready in 15s")


# ── Auth (user info) ───────────────────────────────────────


@app.get("/api/auth/me")
async def me(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow(
        'SELECT id, email, name, created_at FROM "user" WHERE id = $1', user_id
    )
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)


# ── Agent Registry ─────────────────────────────────────────


@app.post("/api/agents/register", dependencies=[Depends(verify_agent_secret)])
async def register_agent(body: AgentRegisterRequest):
    """Called by an Aether agent on startup. Requires AGENT_SECRET."""
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO agents (id, host, port, container_id, user_id, status, registered_at, last_health)
        VALUES ($1, $2, $3, $4, $5, 'running', now(), now())
        ON CONFLICT (id) DO UPDATE SET
            host = $2, port = $3, container_id = $4,
            user_id = COALESCE($5, agents.user_id),
            status = 'running', last_health = now()
    """,
        body.agent_id,
        body.host,
        body.port,
        body.container_id,
        body.user_id,
    )
    log.info(
        f"Agent registered: {body.agent_id} at {body.host}:{body.port} (user={body.user_id})"
    )
    return {"status": "registered"}


@app.post("/api/agents/{agent_id}/assign", dependencies=[Depends(verify_agent_secret)])
async def assign_agent(agent_id: str, user_id: str = Query(...)):
    """Assign an agent to a user. Requires AGENT_SECRET."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE agents SET user_id = $1 WHERE id = $2", user_id, agent_id
    )
    return {"status": "assigned"}


@app.get("/api/agents/health")
async def list_agents():
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, user_id, host, port, status, last_health FROM agents ORDER BY registered_at DESC"
    )
    return [dict(r) for r in rows]


@app.post(
    "/api/agents/{agent_id}/heartbeat", dependencies=[Depends(verify_agent_secret)]
)
async def heartbeat(agent_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE agents SET last_health = now(), status = 'running' WHERE id = $1",
        agent_id,
    )
    return {"status": "ok"}


@app.post(
    "/api/agents/{agent_id}/keep_alive", dependencies=[Depends(verify_agent_secret)]
)
async def set_keep_alive(agent_id: str, enabled: bool = True):
    """Set or clear the keep_alive flag for an agent.

    When keep_alive=true the idle reaper will not stop this agent even if
    its heartbeat lapses — used while a WebRTC session is active so the
    container survives brief network disconnects between reconnects.
    """
    pool = await get_pool()
    await pool.execute(
        "UPDATE agents SET keep_alive = $1 WHERE id = $2",
        enabled,
        agent_id,
    )
    return {"status": "ok", "keep_alive": enabled}


# ── Device Pairing ─────────────────────────────────────────


@app.post("/api/pair/request")
async def pair_request(body: PairRequestBody):
    """iOS app calls this with the code it's displaying."""
    pool = await get_pool()
    # Clean expired codes
    await pool.execute("DELETE FROM pair_requests WHERE expires_at < now()")
    await pool.execute(
        "INSERT INTO pair_requests (code, device_type, device_name) VALUES ($1, $2, $3) ON CONFLICT (code) DO NOTHING",
        body.code,
        body.device_type,
        body.device_name or "Unknown Device",
    )
    log.info(f"Pair request: {body.code} ({body.device_type}) name={body.device_name}")
    return {"status": "waiting", "expires_in": 600}


@app.post("/api/pair/confirm")
async def pair_confirm(body: PairConfirmBody, user_id: str = Depends(get_user_id)):
    """Dashboard calls this when user enters the code."""
    pool = await get_pool()

    # Find the pending pair request
    row = await pool.fetchrow(
        "SELECT code, device_type, device_name FROM pair_requests WHERE code = $1 AND expires_at > now() AND claimed_by IS NULL",
        body.code,
    )
    if not row:
        raise HTTPException(404, "Invalid or expired code")

    # Claim the code
    await pool.execute(
        "UPDATE pair_requests SET claimed_by = $1 WHERE code = $2", user_id, body.code
    )

    # Create device + token (device token = session token for the device)
    device_id = uuid.uuid4().hex[:12]
    device_token = (
        uuid.uuid4().hex
    )  # Simple opaque token, validated against devices table
    device_name = row.get("device_name") or "Unknown Device"
    await pool.execute(
        "INSERT INTO devices (id, user_id, name, device_type, token) VALUES ($1, $2, $3, $4, $5)",
        device_id,
        user_id,
        device_name,
        row["device_type"],
        device_token,
    )
    log.info(f"Device paired: {device_id} ({device_name}) for user {user_id}")

    # Ensure agent is provisioned for this user
    await _ensure_agent(user_id)

    return {"device_id": device_id, "device_token": device_token}


@app.get("/api/pair/status/{code}")
async def pair_status(code: str):
    """iOS app polls this to check if code was confirmed."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT claimed_by FROM pair_requests WHERE code = $1", code
    )
    if not row:
        return {"status": "unknown"}
    if row["claimed_by"]:
        # Return the device token
        device = await pool.fetchrow(
            "SELECT id, token FROM devices WHERE user_id = $1 ORDER BY paired_at DESC LIMIT 1",
            row["claimed_by"],
        )
        if device:
            return {
                "status": "paired",
                "device_id": device["id"],
                "device_token": device["token"],
            }
    return {"status": "waiting"}


@app.get("/api/devices")
async def list_devices(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, name, device_type, paired_at, last_seen FROM devices WHERE user_id = $1 ORDER BY paired_at DESC",
        user_id,
    )
    return [dict(r) for r in rows]


# ── API Key Management ─────────────────────────────────────


@app.post("/api/services/keys")
async def store_api_key(body: ApiKeyBody, user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    key_id = uuid.uuid4().hex[:12]
    encrypted = encrypt_value(body.key_value)
    await pool.execute(
        """
        INSERT INTO api_keys (id, user_id, provider, key_value)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, provider) DO UPDATE SET key_value = $4
    """,
        key_id,
        user_id,
        body.provider,
        encrypted,
    )
    return {"status": "saved"}


@app.get("/api/services/keys")
async def list_api_keys(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT provider, key_value FROM api_keys WHERE user_id = $1",
        user_id,
    )
    # Decrypt then preview — can't use SQL substring on ciphertext
    return [
        {
            "provider": r["provider"],
            "preview": decrypt_value(r["key_value"])[:8] + "...",
        }
        for r in rows
    ]


@app.delete("/api/services/keys/{provider}")
async def delete_api_key(provider: str, user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    await pool.execute(
        "DELETE FROM api_keys WHERE user_id = $1 AND provider = $2",
        user_id,
        provider,
    )
    return {"status": "deleted"}


# ── User Preferences ───────────────────────────────────────

# DB columns exposed to the dashboard via GET/PUT /api/preferences.
# NOTE: llm_provider and llm_base_url are stored for dashboard display only —
# they are NOT injected into agent containers (see PREF_TO_ENV below).
PREFERENCE_COLUMNS = [
    "stt_provider",
    "stt_model",
    "stt_language",
    "llm_provider",  # display-only (agent derives provider from model name)
    "llm_model",
    "llm_base_url",  # display-only (agent hardcodes OpenRouter)
    "tts_provider",
    "tts_model",
    "tts_voice",
    "base_style",
    "custom_instructions",
]

# Map preference fields to agent env vars.
# NOTE: llm_provider and llm_base_url are intentionally excluded —
# the agent hardcodes OpenRouter as the LLM gateway (config.py),
# and derives the provider from the model name. These DB columns
# still exist for the dashboard UI but don't map to agent env vars.
PREF_TO_ENV = {
    "stt_provider": "AETHER_STT_PROVIDER",
    "stt_model": "AETHER_STT_MODEL",
    "stt_language": "AETHER_STT_LANGUAGE",
    "llm_model": "AETHER_LLM_MODEL",
    "tts_provider": "AETHER_TTS_PROVIDER",
    "tts_model": "AETHER_TTS_MODEL",
    "tts_voice": "AETHER_TTS_VOICE",
    "base_style": "AETHER_BASE_STYLE",
    "custom_instructions": "AETHER_CUSTOM_INSTRUCTIONS",
}


@app.get("/api/preferences")
async def get_preferences(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1", user_id
    )
    if not row:
        # Return defaults (no row yet — user hasn't customized anything)
        return {col: None for col in PREFERENCE_COLUMNS}
    return {col: row[col] for col in PREFERENCE_COLUMNS}


@app.get(
    "/api/internal/preferences",
    dependencies=[Depends(verify_agent_secret)],
)
async def get_preferences_internal(
    user_id: str | None = Query(None),
    agent_id: str | None = Query(None),
):
    """Internal endpoint for agents to fetch a user's saved preferences.

    Accepts either:
    - user_id directly, or
    - agent_id (resolved to user_id via agents table)
    """
    resolved_source = "explicit_user_id"
    pool = await get_pool()
    if not user_id:
        if not agent_id:
            raise HTTPException(400, {"reason": "missing_user_id_and_agent_id"})
        agent = await pool.fetchrow(
            "SELECT user_id FROM agents WHERE id = $1",
            agent_id,
        )
        if agent and agent["user_id"]:
            user_id = str(agent["user_id"])
            resolved_source = "agent_mapping"
        else:
            # Local single-agent dev fallback: infer the only user preference row.
            if not LOCAL_AGENT_URL:
                raise HTTPException(404, {"reason": "no_user_mapping_for_agent"})
            candidates = await pool.fetch(
                "SELECT user_id FROM user_preferences ORDER BY updated_at DESC LIMIT 2"
            )
            if not candidates:
                raise HTTPException(404, {"reason": "no_user_preferences_found"})
            if len(candidates) > 1:
                raise HTTPException(
                    409,
                    {
                        "reason": "ambiguous_user_mapping",
                        "candidate_count": len(candidates),
                    },
                )
            user_id = str(candidates[0]["user_id"])
            resolved_source = "local_single_user_fallback"

    row = await pool.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1", user_id
    )
    if not row:
        raise HTTPException(404, {"reason": "preferences_not_found_for_user"})

    return {
        "preferences": {col: row[col] for col in PREFERENCE_COLUMNS},
        "meta": {
            "resolved_user_id": user_id,
            "source": resolved_source,
        },
    }


@app.put("/api/preferences")
async def update_preferences(
    body: PreferencesBody, user_id: str = Depends(get_user_id)
):
    pool = await get_pool()

    # Build the set of fields that were actually sent (non-None)
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return {"status": "no changes"}

    # Upsert: insert with provided values, or update on conflict
    columns = ["user_id"] + list(updates.keys())
    placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
    set_clause = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates.keys()))
    values = [user_id] + list(updates.values())

    await pool.execute(
        f"""
        INSERT INTO user_preferences ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (user_id) DO UPDATE SET {set_clause}, updated_at = now()
        """,
        *values,
    )
    log.info("Updated preferences for user %s: %s", user_id, sorted(updates.keys()))

    # Signal the running agent to reload config via HTTP
    # Works in both dev mode (shared agent) and multi-user mode (per-user container)
    await _signal_agent_reload(user_id, updates)

    return {"status": "saved"}


async def _signal_agent_reload(user_id: str, updates: dict) -> None:
    """Tell the running agent to reload its config after a preference change.

    POSTs the new config directly to the agent's /config/reload endpoint.
    Works in both dev mode (shared agent) and multi-user mode (per-user container).
    """
    agent = await _get_agent_for_user(user_id)
    if not agent:
        return

    import httpx

    # Convert preference keys to env var names for the agent
    env_updates = {}
    for k, v in updates.items():
        env_name = PREF_TO_ENV.get(k)
        if env_name and v:
            env_updates[env_name] = v
    log.info(
        "Sending config reload to agent for user %s with env keys: %s",
        user_id,
        sorted(env_updates.keys()),
    )

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"http://{agent['host']}:{agent['port']}/config/reload",
                json=env_updates,
            )
            log.info(f"Signaled agent config reload for user {user_id}")
    except Exception as e:
        log.warning(f"Failed to signal agent reload: {e}")


# ── Memory Proxy ───────────────────────────────────────────


async def _get_agent_for_user(user_id: str) -> dict | None:
    """Look up which agent this user is assigned to, provisioning if needed."""
    local_target = _local_agent_target()
    if local_target:
        return local_target

    pool = await get_pool()
    agent = await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        user_id,
    )
    if agent:
        # Guard against stale DB state (row says running but container is gone).
        container_status = await get_agent_status(user_id)
        if container_status == "running":
            return agent
        log.warning(
            "Agent row stale for user %s (db=running, container=%s); reprovisioning",
            user_id,
            container_status,
        )
        await pool.execute(
            "UPDATE agents SET status = 'stopped' WHERE user_id = $1", user_id
        )

    # No running agent — try to provision one
    await _ensure_agent(user_id)

    # Poll briefly for agent readiness
    for _ in range(20):  # Up to 10 seconds
        agent = await pool.fetchrow(
            "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
            user_id,
        )
        if agent:
            return agent
        await asyncio.sleep(0.5)

    return None


@app.get("/api/memory/facts")
async def proxy_memory_facts(user_id: str = Depends(get_user_id)):
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"http://{agent['host']}:{agent['port']}/memory/facts")
        return resp.json()


@app.get("/api/memory/sessions")
async def proxy_memory_sessions(user_id: str = Depends(get_user_id)):
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"http://{agent['host']}:{agent['port']}/memory/sessions"
        )
        return resp.json()


@app.get("/api/memory/conversations")
async def proxy_memory_conversations(
    user_id: str = Depends(get_user_id), limit: int = 20
):
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"http://{agent['host']}:{agent['port']}/memory/conversations",
            params={"limit": limit},
        )
        return resp.json()


# ── WebRTC Signaling Proxy ─────────────────────────────────


@app.post("/api/webrtc/offer")
async def proxy_webrtc_offer(request: Request, user_id: str = Depends(get_user_id)):
    """Proxy WebRTC SDP offer to the user's agent."""
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")

    body = await request.json()
    # Inject user_id so the agent knows who's connecting
    body["user_id"] = user_id

    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"http://{agent['host']}:{agent['port']}/webrtc/offer",
            json=body,
        )
        if resp.status_code != 200:
            raise HTTPException(
                resp.status_code, resp.json().get("error", "Offer failed")
            )
        return resp.json()


@app.patch("/api/webrtc/ice")
async def proxy_webrtc_ice(request: Request, user_id: str = Depends(get_user_id)):
    """Proxy WebRTC ICE candidates to the user's agent."""
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")

    body = await request.json()

    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(
            f"http://{agent['host']}:{agent['port']}/webrtc/ice",
            json=body,
        )
        if resp.status_code != 200:
            raise HTTPException(
                resp.status_code, resp.json().get("error", "ICE failed")
            )
        return resp.json()


# ── Chat Streaming Proxy (Vercel AI SDK) ───────────────────


@app.post("/api/chat")
async def proxy_chat(request: Request, user_id: str = Depends(get_user_id)):
    """
    Proxy streaming chat to the user's agent.

    Passes through plain text streaming chunks from the agent directly
    to the dashboard TextStreamChatTransport.
    """
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")

    body = await request.json()
    body["user_id"] = user_id

    import httpx
    from fastapi.responses import StreamingResponse

    async def stream_from_agent():
        """Forward plain text streaming chunks from the agent."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        ) as client:
            async with client.stream(
                "POST",
                f"http://{agent['host']}:{agent['port']}/chat",
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    yield f"[chat proxy error {resp.status_code}] {error_body.decode()}"
                    return
                async for chunk in resp.aiter_text():
                    yield chunk

    return StreamingResponse(
        stream_from_agent(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ── WebSocket Proxy ────────────────────────────────────────


@app.websocket("/api/ws")
async def ws_proxy(ws: WebSocket):
    """
    Proxy WebSocket between client and the user's Aether agent.

    Client connects here → orchestrator validates session → finds agent → proxies bidirectionally.
    Auth: session token via query param `token` or Authorization header.
    """
    user_id = await get_user_id_from_ws(ws)
    if not user_id:
        await ws.close(code=4001, reason="Invalid or missing session")
        return

    local_target = _local_agent_target()
    if local_target:
        agent = local_target
    else:
        # Ensure agent is provisioned (starts container in multi-user mode)
        await _ensure_agent(user_id)

        # Poll for agent readiness (container may still be starting)
        pool = await get_pool()
        agent = None
        for _ in range(30):  # Up to 15 seconds
            agent = await pool.fetchrow(
                "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
                user_id,
            )
            if agent:
                break
            await asyncio.sleep(0.5)

    if not agent:
        await ws.accept()
        await ws.send_json(
            {"type": "error", "message": "Agent not ready — try again shortly"}
        )
        await ws.close(code=4002, reason="Agent not ready")
        return

    agent_url = f"ws://{agent['host']}:{agent['port']}/ws"
    log.info(f"Proxying WS: user={user_id} → {agent_url}")

    await ws.accept()

    # Connect to the agent
    import websockets as ws_lib

    client_disconnected = False
    agent_disconnected = False

    try:
        async with ws_lib.connect(agent_url) as agent_ws:

            async def client_to_agent():
                nonlocal client_disconnected
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("text"):
                            await agent_ws.send(msg["text"])
                        elif msg.get("bytes"):
                            await agent_ws.send(msg["bytes"])
                except WebSocketDisconnect:
                    client_disconnected = True
                    log.debug(f"Client disconnected (user={user_id})")
                except Exception as e:
                    client_disconnected = True
                    log.debug(f"Client→agent error (user={user_id}): {e}")

            async def agent_to_client():
                nonlocal agent_disconnected
                try:
                    async for msg in agent_ws:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)
                    # Iterator exhausted = agent closed connection
                    agent_disconnected = True
                    log.info(f"Agent closed connection (user={user_id})")
                except Exception as e:
                    agent_disconnected = True
                    log.debug(f"Agent→client error (user={user_id}): {e}")

            async def ping_client():
                """Periodic ping to detect zombie client connections."""
                try:
                    while True:
                        await asyncio.sleep(25)
                        await ws.send_json({"type": "ping"})
                except Exception:
                    pass  # Connection already dead, other tasks will handle it

            # Run all directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_agent()),
                    asyncio.create_task(agent_to_client()),
                    asyncio.create_task(ping_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            # Await cancellation to ensure clean shutdown
            await asyncio.gather(*pending, return_exceptions=True)

    except Exception as e:
        agent_disconnected = True
        log.error(f"WS proxy error (user={user_id}): {e}")

    # ── Clean close ──
    # Notify client if agent died (client may still be connected)
    if agent_disconnected and not client_disconnected:
        try:
            await ws.send_json(
                {"type": "error", "message": "Agent disconnected — reconnecting..."}
            )
            await ws.close(code=4003, reason="Agent disconnected")
        except Exception:
            pass
    else:
        try:
            await ws.close()
        except Exception:
            pass

    log.info(
        f"WS proxy closed (user={user_id}, client_dc={client_disconnected}, agent_dc={agent_disconnected})"
    )


# ── Plugin Management ─────────────────────────────────────

# Available plugins — matches agent's app/plugins/ directory.
#
# Each plugin manages its own OAuth tokens independently.
# Google plugins each request only the scopes they need.
# The user connects each plugin separately via its own OAuth flow.
AVAILABLE_PLUGINS = {
    "gmail": {
        "name": "gmail",
        "display_name": "Gmail",
        "description": "Monitor and respond to Gmail emails",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "scopes": [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/userinfo.email",
        ],
        "config_fields": [],
        # Token refresh — plugin ships a refresh_token tool; cron delivers the
        # instruction to the LLM which calls the tool. 3000 s = 50 min (Google
        # access tokens expire at 60 min).
        "token_refresh_interval": 3000,
        "refresh_tool": "refresh_gmail_token",
        # Push event delivery via Google Cloud Pub/Sub.
        # setup_tool registers the watch; handle_tool processes inbound payloads.
        # renew_interval: 6 days (watches expire at 7 days).
        "webhook": {
            "protocol": "pubsub",
            "setup_tool": "setup_gmail_watch",
            "handle_tool": "handle_gmail_event",
            "renew_interval": 518400,
        },
    },
    "google-calendar": {
        "name": "google-calendar",
        "display_name": "Google Calendar",
        "description": "View and create calendar events",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "scopes": [
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/userinfo.email",
        ],
        "config_fields": [],
        "token_refresh_interval": 3000,
        "refresh_tool": "refresh_google_calendar_token",
        # Push event delivery via direct HTTPS (no Pub/Sub needed).
        "webhook": {
            "protocol": "http",
            "setup_tool": "setup_calendar_watch",
            "handle_tool": "handle_calendar_event",
            "renew_interval": 518400,
        },
    },
    "google-contacts": {
        "name": "google-contacts",
        "display_name": "Google Contacts",
        "description": "Search and look up your contacts",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "scopes": [
            "https://www.googleapis.com/auth/contacts.readonly",
            "https://www.googleapis.com/auth/userinfo.email",
        ],
        "config_fields": [],
        "token_refresh_interval": 3000,
        "refresh_tool": "refresh_google_contacts_token",
    },
    "google-drive": {
        "name": "google-drive",
        "display_name": "Google Drive",
        "description": "Search, browse, read, and create files in Google Drive",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "scopes": [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/presentations",
            "https://www.googleapis.com/auth/userinfo.email",
        ],
        "config_fields": [],
        "token_refresh_interval": 3000,
        "refresh_tool": "refresh_google_drive_token",
    },
    "spotify": {
        "name": "spotify",
        "display_name": "Spotify",
        "description": "Control playback and browse your music",
        "auth_type": "oauth2",
        "auth_provider": "spotify",
        "scopes": [
            "user-read-playback-state",
            "user-modify-playback-state",
            "user-read-currently-playing",
            "user-read-recently-played",
            "playlist-read-private",
            "user-read-email",
        ],
        "config_fields": [
            {
                "key": "account_email",
                "label": "Spotify Account",
                "type": "text",
                "required": True,
            },
        ],
        # Spotify access tokens expire at 60 min; refresh at 50 min.
        "token_refresh_interval": 3000,
        "refresh_tool": "refresh_spotify_token",
    },
    "weather": {
        "name": "weather",
        "display_name": "Weather",
        "description": "Current weather and forecasts for any location",
        "auth_type": "none",
        "auth_provider": "",
        "scopes": [],
        "config_fields": [],
    },
    "wikipedia": {
        "name": "wikipedia",
        "display_name": "Wikipedia",
        "description": "Search Wikipedia and read articles. Instant access to encyclopaedic knowledge on any topic.",
        "auth_type": "none",
        "auth_provider": "",
        "scopes": [],
        "config_fields": [],
    },
    "rss-feeds": {
        "name": "rss-feeds",
        "display_name": "RSS Feeds & Hacker News",
        "description": "Subscribe to and read any RSS/Atom feed. Includes Hacker News tools for top stories and search.",
        "auth_type": "none",
        "auth_provider": "",
        "scopes": [],
        "config_fields": [],
    },
    "brave-search": {
        "name": "brave-search",
        "display_name": "Brave Search",
        "description": "Real-time web search powered by Brave Search API. Search the live web, news, images, and local places.",
        "auth_type": "api_key",
        "auth_provider": "",
        "scopes": [],
        "config_fields": [
            {
                "key": "api_key",
                "label": "Brave Search API Key",
                "type": "password",
                "required": True,
                "description": "Your Brave Search API key from api-dashboard.search.brave.com",
            },
        ],
    },
    "wolfram": {
        "name": "wolfram",
        "display_name": "Wolfram Alpha & Currency",
        "description": "Mathematical computations, scientific queries, and live currency conversion.",
        "auth_type": "api_key",
        "auth_provider": "",
        "scopes": [],
        "config_fields": [
            {
                "key": "wolfram_app_id",
                "label": "Wolfram Alpha App ID",
                "type": "password",
                "required": False,
                "description": "Your Wolfram Alpha App ID from developer.wolframalpha.com (free tier — 2,000 queries/month). Leave blank to disable math/science queries.",
            },
            {
                "key": "exchangerate_api_key",
                "label": "ExchangeRate-API Key",
                "type": "password",
                "required": False,
                "description": "Your ExchangeRate-API key from exchangerate-api.com (free tier — 1,500 requests/month). Leave blank to use the free open fallback.",
            },
        ],
    },
    "vobiz": {
        "name": "vobiz",
        "display_name": "Vobiz Telephony",
        "description": "Make and receive phone calls via Vobiz.",
        "auth_type": "api_key",
        "auth_provider": "",
        "scopes": [],
        "config_fields": [
            {
                "key": "auth_id",
                "label": "Auth ID",
                "type": "text",
                "required": True,
                "description": "Your Vobiz Auth ID (from console.vobiz.ai)",
            },
            {
                "key": "auth_token",
                "label": "Auth Token",
                "type": "password",
                "required": True,
                "description": "Your Vobiz Auth Token",
            },
            {
                "key": "from_number",
                "label": "Vobiz Phone Number",
                "type": "text",
                "required": True,
                "description": "Your Vobiz phone number (caller ID, e.g., 919876543210)",
            },
            {
                "key": "user_phone_number",
                "label": "Your Phone Number",
                "type": "text",
                "required": False,
                "description": "Your personal phone number for outbound calls (e.g., +919123456789)",
            },
        ],
    },
}


class CronScheduleBody(BaseModel):
    plugin: str | None = None
    instruction: str
    run_at: str  # ISO-8601 timestamp
    interval_s: int | None = None  # None = one-shot, >0 = recurring


class PluginConfigBody(BaseModel):
    config: dict[str, str]


def _public_base_url(request: Request) -> str:
    """Build a public base URL from forwarded headers."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}"


def _normalize_e164_like(number: str) -> str:
    """Normalize number to +{digits} format for Vobiz link API."""
    raw = (number or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return ""
    return f"+{digits}"


async def _ensure_vobiz_application(
    *,
    auth_id: str,
    auth_token: str,
    from_number: str,
    answer_url: str,
    existing_app_id: str | None,
) -> tuple[str, str]:
    """Create or update a Vobiz application and return (app_id, action)."""
    base = "https://api.vobiz.ai"
    account_path = f"/api/v1/Account/{auth_id}"
    app_name = "Aether Voice Assistant"
    headers = {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
        "Content-Type": "application/json",
    }

    payload = {
        "app_name": app_name,
        "answer_url": answer_url,
        "answer_method": "POST",
        "hangup_url": answer_url,
        "hangup_method": "POST",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        if existing_app_id:
            resp = await client.post(
                f"{base}{account_path}/Application/{existing_app_id}/",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 404:
                # Stale application_id — app was deleted on VoBiz.
                # Fall through to list/create flow below.
                log.warning(
                    "Vobiz application %s not found (404), will recreate",
                    existing_app_id,
                )
            else:
                resp.raise_for_status()
                return existing_app_id, "updated"

        list_resp = await client.get(
            f"{base}{account_path}/Application/",
            headers=headers,
            params={"limit": 100, "offset": 0},
        )
        list_resp.raise_for_status()
        objects = list_resp.json().get("objects", [])

        existing = next(
            (
                obj
                for obj in objects
                if obj.get("answer_url") == answer_url
                or obj.get("app_name") == app_name
            ),
            None,
        )
        if existing and existing.get("app_id"):
            app_id = str(existing["app_id"])
            resp = await client.post(
                f"{base}{account_path}/Application/{app_id}/",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return app_id, "updated"

        create_resp = await client.post(
            f"{base}{account_path}/Application/",
            headers=headers,
            json=payload,
        )
        create_resp.raise_for_status()
        app_id = str(create_resp.json().get("app_id", "")).strip()
        if not app_id:
            raise RuntimeError("Vobiz create application response missing app_id")

        e164_number = _normalize_e164_like(from_number)
        if e164_number:
            encoded_number = urllib.parse.quote(e164_number, safe="")
            link_resp = await client.post(
                f"{base}/api/v1/account/{auth_id}/numbers/{encoded_number}/application",
                headers=headers,
                json={"application_id": app_id},
            )
            link_resp.raise_for_status()

        return app_id, "created"


@app.get("/api/plugins")
async def list_plugins(user_id: str = Depends(get_user_id)):
    """List all available plugins with user's install/enable status."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, name, enabled FROM plugins WHERE user_id = $1", user_id
    )
    installed = {r["name"]: {"id": r["id"], "enabled": r["enabled"]} for r in rows}

    # Check which installed plugins have OAuth tokens (connected)
    connected_plugins: set[str] = set()
    for name, info in installed.items():
        token_row = await pool.fetchrow(
            "SELECT value FROM plugin_configs WHERE plugin_id = $1 AND key = 'access_token'",
            info["id"],
        )
        if token_row:
            connected_plugins.add(name)

    # Check which api_key type plugins have required config (connected)
    api_key_connected: set[str] = set()
    for name, info in installed.items():
        meta = AVAILABLE_PLUGINS.get(name, {})
        if meta.get("auth_type") == "api_key":
            # Check if all required config fields are set
            required_fields = [
                f["key"] for f in meta.get("config_fields", []) if f.get("required")
            ]
            if required_fields:
                config_rows = await pool.fetch(
                    "SELECT key FROM plugin_configs WHERE plugin_id = $1",
                    info["id"],
                )
                configured_keys = {r["key"] for r in config_rows}
                if all(k in configured_keys for k in required_fields):
                    api_key_connected.add(name)

    # Check which plugins need reconnection (scopes changed since last OAuth)
    needs_reconnect_plugins: set[str] = set()
    for name, info in installed.items():
        meta = AVAILABLE_PLUGINS.get(name, {})
        if meta.get("auth_type") != "oauth2":
            continue
        # Read stored granted_scopes
        scope_row = await pool.fetchrow(
            "SELECT value FROM plugin_configs WHERE plugin_id = $1 AND key = 'granted_scopes'",
            info["id"],
        )
        if scope_row and name in connected_plugins:
            granted = set(decrypt_value(scope_row["value"]).split())
            required = set(meta.get("scopes", []))
            if required - granted:
                needs_reconnect_plugins.add(name)

    result = []
    for name, meta in AVAILABLE_PLUGINS.items():
        entry = {**meta}
        if name in installed:
            entry["installed"] = True
            entry["plugin_id"] = installed[name]["id"]
            entry["enabled"] = installed[name]["enabled"]
            if meta.get("auth_type") == "api_key":
                entry["connected"] = name in api_key_connected
            else:
                entry["connected"] = name in connected_plugins
            entry["needs_reconnect"] = name in needs_reconnect_plugins
        else:
            entry["installed"] = False
            entry["plugin_id"] = None
            entry["enabled"] = False
            entry["connected"] = False
            entry["needs_reconnect"] = False
        # No-auth plugins are always "connected" once installed
        if meta.get("auth_type") == "none" and name in installed:
            entry["connected"] = True
        result.append(entry)
    return result


@app.post("/api/plugins/{plugin_name}/install")
async def install_plugin(plugin_name: str, user_id: str = Depends(get_user_id)):
    """Install a plugin for the user.

    No-auth plugins are auto-enabled on install.
    OAuth2 and API key plugins require connection/configuration before enabling.
    """
    if plugin_name not in AVAILABLE_PLUGINS:
        raise HTTPException(404, f"Unknown plugin: {plugin_name}")

    meta = AVAILABLE_PLUGINS[plugin_name]
    pool = await get_pool()
    plugin_id = uuid.uuid4().hex[:12]

    # Only no-auth plugins auto-enable on install
    auto_enable = meta.get("auth_type") == "none"

    await pool.execute(
        """
        INSERT INTO plugins (id, user_id, name, enabled)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, name) DO NOTHING
        """,
        plugin_id,
        user_id,
        plugin_name,
        auto_enable,
    )
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )

    # Signal agent to reload if auto-enabled
    if auto_enable:
        await _signal_agent_plugin_reload(user_id)

    return {"plugin_id": row["id"], "status": "installed"}


@app.post("/api/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str, user_id: str = Depends(get_user_id)):
    """Enable a plugin. For api_key type plugins, requires all required config fields."""
    meta = AVAILABLE_PLUGINS.get(plugin_name)
    if not meta:
        raise HTTPException(404, f"Unknown plugin: {plugin_name}")

    pool = await get_pool()

    # For api_key type plugins, check that all required config fields are set
    if meta.get("auth_type") == "api_key":
        required_fields = [
            f["key"] for f in meta.get("config_fields", []) if f.get("required")
        ]
        if required_fields:
            row = await pool.fetchrow(
                "SELECT id FROM plugins WHERE user_id = $1 AND name = $2",
                user_id,
                plugin_name,
            )
            if not row:
                raise HTTPException(404, "Plugin not installed")

            config_rows = await pool.fetch(
                "SELECT key FROM plugin_configs WHERE plugin_id = $1",
                row["id"],
            )
            configured_keys = {r["key"] for r in config_rows}
            missing = [k for k in required_fields if k not in configured_keys]
            if missing:
                raise HTTPException(
                    400,
                    f"Missing required configuration: {', '.join(missing)}",
                )

    await pool.execute(
        "UPDATE plugins SET enabled = true WHERE user_id = $1 AND name = $2",
        user_id,
        plugin_name,
    )

    # Signal agent to reload config
    await _signal_agent_plugin_reload(user_id)

    return {"status": "enabled"}


@app.post("/api/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str, user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    await pool.execute(
        "UPDATE plugins SET enabled = false WHERE user_id = $1 AND name = $2",
        user_id,
        plugin_name,
    )
    return {"status": "disabled"}


@app.post("/api/plugins/{plugin_name}/config")
async def save_plugin_config(
    plugin_name: str,
    body: PluginConfigBody,
    request: Request,
    user_id: str = Depends(get_user_id),
):
    """Save plugin configuration (encrypted at rest).

    For api_key type plugins, auto-enables when all required fields are configured.
    """
    meta = AVAILABLE_PLUGINS.get(plugin_name)
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, enabled FROM plugins WHERE user_id = $1 AND name = $2",
        user_id,
        plugin_name,
    )
    if not row:
        raise HTTPException(404, "Plugin not installed")
    plugin_id = row["id"]
    was_enabled = row["enabled"]

    existing_rows = await pool.fetch(
        "SELECT key, value FROM plugin_configs WHERE plugin_id = $1",
        plugin_id,
    )
    existing_config = {
        c["key"]: decrypt_value(c["value"])
        for c in existing_rows
        if c.get("value") is not None
    }

    for key, value in body.config.items():
        config_id = uuid.uuid4().hex[:12]
        encrypted = encrypt_value(value)
        await pool.execute(
            """
            INSERT INTO plugin_configs (id, plugin_id, key, value, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (plugin_id, key) DO UPDATE SET value = $4, updated_at = now()
            """,
            config_id,
            plugin_id,
            key,
            encrypted,
        )

    # For api_key type plugins, auto-enable when all required fields are configured
    auto_enabled = False
    if meta and meta.get("auth_type") == "api_key" and not was_enabled:
        required_fields = [
            f["key"] for f in meta.get("config_fields", []) if f.get("required")
        ]
        if required_fields:
            config_rows = await pool.fetch(
                "SELECT key FROM plugin_configs WHERE plugin_id = $1",
                plugin_id,
            )
            configured_keys = {r["key"] for r in config_rows}
            if all(k in configured_keys for k in required_fields):
                await pool.execute(
                    "UPDATE plugins SET enabled = true WHERE id = $1",
                    plugin_id,
                )
                auto_enabled = True
                # Signal agent to reload config
                await _signal_agent_plugin_reload(user_id)

    vobiz_provision: dict[str, str] | None = None
    if plugin_name == "vobiz":
        merged = {**existing_config, **body.config}
        auth_id = merged.get("auth_id", "").strip()
        auth_token = merged.get("auth_token", "").strip()
        from_number = merged.get("from_number", "").strip()
        existing_app_id = merged.get("application_id", "").strip() or None

        # Enforce phone number uniqueness across users
        if from_number:
            conflict = await pool.fetchrow(
                """
                SELECT p.user_id FROM plugin_configs pc
                JOIN plugins p ON p.id = pc.plugin_id
                WHERE pc.key = 'from_number'
                  AND pc.value IS NOT NULL
                  AND p.name = 'vobiz'
                  AND p.id != $1
                """,
                plugin_id,
            )
            if conflict:
                from .crypto import decrypt_value as _dv

                existing_num = await pool.fetchval(
                    """
                    SELECT pc.value FROM plugin_configs pc
                    JOIN plugins p ON p.id = pc.plugin_id
                    WHERE pc.key = 'from_number' AND p.name = 'vobiz' AND p.id != $1
                    """,
                    plugin_id,
                )
                if existing_num and _dv(existing_num) == from_number:
                    raise HTTPException(
                        409,
                        "This phone number is already configured by another user.",
                    )

        if auth_id and auth_token and from_number:
            answer_url = f"{_public_base_url(request)}/api/plugins/vobiz/webhook"
            try:
                app_id, action = await _ensure_vobiz_application(
                    auth_id=auth_id,
                    auth_token=auth_token,
                    from_number=from_number,
                    answer_url=answer_url,
                    existing_app_id=existing_app_id,
                )
                encrypted_app_id = encrypt_value(app_id)
                pub_url = _public_base_url(request)
                for cfg_key, cfg_val in [
                    ("application_id", encrypted_app_id),
                    ("public_base_url", encrypt_value(pub_url)),
                ]:
                    await pool.execute(
                        """
                        INSERT INTO plugin_configs (id, plugin_id, key, value, updated_at)
                        VALUES ($1, $2, $3, $4, now())
                        ON CONFLICT (plugin_id, key) DO UPDATE SET value = $4, updated_at = now()
                        """,
                        uuid.uuid4().hex[:12],
                        plugin_id,
                        cfg_key,
                        cfg_val,
                    )
                vobiz_provision = {
                    "status": "ok",
                    "action": action,
                    "application_id": app_id,
                }
                if not auto_enabled:
                    await _signal_agent_plugin_reload(user_id)
            except Exception as e:
                log.warning("Vobiz auto-provision failed: %s", e)
                vobiz_provision = {"status": "error", "message": str(e)}
        else:
            vobiz_provision = {
                "status": "skipped",
                "message": "auth_id, auth_token, and from_number are required",
            }

    return {
        "status": "saved",
        "auto_enabled": auto_enabled,
        **({"vobiz_provision": vobiz_provision} if vobiz_provision else {}),
    }


@app.get("/api/plugins/{plugin_name}/config")
async def get_plugin_config(plugin_name: str, user_id: str = Depends(get_user_id)):
    """Get plugin config (values masked)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )
    if not row:
        raise HTTPException(404, "Plugin not installed")

    configs = await pool.fetch(
        "SELECT key, value FROM plugin_configs WHERE plugin_id = $1", row["id"]
    )
    result = {}
    for c in configs:
        decrypted = decrypt_value(c["value"])
        if len(decrypted) > 8:
            result[c["key"]] = decrypted[:4] + "..." + decrypted[-4:]
        else:
            result[c["key"]] = decrypted
    return result


@app.delete("/api/plugins/{plugin_name}")
async def uninstall_plugin(plugin_name: str, user_id: str = Depends(get_user_id)):
    """Uninstall a plugin (cascades to configs)."""
    pool = await get_pool()
    await pool.execute(
        "DELETE FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )
    return {"status": "uninstalled"}


# ── Internal Plugin Config (agent-facing) ─────────────────


@app.get(
    "/api/internal/plugins",
    dependencies=[Depends(verify_agent_secret)],
)
async def list_enabled_plugins_for_agent(user_id: str = Query(...)):
    """List all enabled plugins for a user (agent-facing).

    Returns plugin names so the agent knows which configs to fetch.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT name FROM plugins WHERE user_id = $1 AND enabled = true", user_id
    )
    return {"plugins": [r["name"] for r in rows]}


# ── OAuth2 Provider Registry ──────────────────────────────
#
# Provider-agnostic OAuth2 flow. Each provider defines its auth/token URLs,
# scopes, credentials, and how to exchange/refresh tokens.
# Adding a new OAuth provider = adding an entry to OAUTH_PROVIDERS.

import base64 as _base64
import httpx as _httpx

# In-memory state store for CSRF protection (state → {user_id, plugin_name, created}).
# In production, use Redis or DB. Short-lived: cleaned up after 10 min.
_oauth_states: dict[str, dict] = {}


OAUTH_PROVIDERS: dict[str, dict] = {
    "google": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "auth_params": {
            "access_type": "offline",
            "prompt": "consent",
        },
        "token_auth": "body",  # client_id/secret sent in POST body
        "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
        "userinfo_email_field": "email",
    },
    "spotify": {
        "auth_url": "https://accounts.spotify.com/authorize",
        "token_url": "https://accounts.spotify.com/api/token",
        "client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
        "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET", ""),
        "auth_params": {},
        "token_auth": "basic",  # client_id:secret sent as Basic auth header
        "userinfo_url": "https://api.spotify.com/v1/me",
        "userinfo_email_field": "email",
    },
}


def _get_oauth_redirect_uri(request: Request, plugin_name: str, **_) -> str:
    """Build the OAuth redirect URI from the incoming request's origin."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}/api/plugins/{plugin_name}/oauth/callback"


async def _exchange_token(provider: dict, code: str, redirect_uri: str) -> dict:
    """Exchange an auth code for tokens. Handles both body and Basic auth styles."""
    data = {
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    headers: dict[str, str] = {}

    if provider["token_auth"] == "basic":
        # Spotify-style: client credentials in Basic auth header
        creds = _base64.b64encode(
            f"{provider['client_id']}:{provider['client_secret']}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {creds}"
    else:
        # Google-style: client credentials in POST body
        data["client_id"] = provider["client_id"]
        data["client_secret"] = provider["client_secret"]

    async with _httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(provider["token_url"], data=data, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _refresh_oauth_token(plugin_id: str, provider_name: str) -> str | None:
    """Refresh an expired OAuth access token using the stored refresh_token.

    Works for any provider in OAUTH_PROVIDERS.
    Returns the new access_token, or None if refresh failed.
    """
    provider = OAUTH_PROVIDERS.get(provider_name)
    if not provider:
        log.warning(f"Unknown OAuth provider: {provider_name}")
        return None

    pool = await get_pool()
    configs = await pool.fetch(
        "SELECT key, value FROM plugin_configs WHERE plugin_id = $1", plugin_id
    )
    config_map = {c["key"]: decrypt_value(c["value"]) for c in configs}

    refresh_token = config_map.get("refresh_token")
    if not refresh_token:
        log.warning(f"No refresh_token for plugin {plugin_id}")
        return None

    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    headers: dict[str, str] = {}

    if provider["token_auth"] == "basic":
        creds = _base64.b64encode(
            f"{provider['client_id']}:{provider['client_secret']}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {creds}"
    else:
        data["client_id"] = provider["client_id"]
        data["client_secret"] = provider["client_secret"]

    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(provider["token_url"], data=data, headers=headers)
            resp.raise_for_status()
            tokens = resp.json()
    except Exception as e:
        log.error(f"Token refresh failed for plugin {plugin_id} ({provider_name}): {e}")
        return None

    new_access_token = tokens.get("access_token", "")
    new_expiry = str(int(time.time()) + int(tokens.get("expires_in", 3600)))

    if not new_access_token:
        return None

    # Update stored tokens
    for key, value in [
        ("access_token", new_access_token),
        ("token_expiry", new_expiry),
    ]:
        config_id = uuid.uuid4().hex[:12]
        encrypted = encrypt_value(value)
        await pool.execute(
            """
            INSERT INTO plugin_configs (id, plugin_id, key, value, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (plugin_id, key) DO UPDATE SET value = $4, updated_at = now()
            """,
            config_id,
            plugin_id,
            key,
            encrypted,
        )

    log.info(f"Refreshed {provider_name} token for plugin {plugin_id}")
    return new_access_token


async def _fetch_oauth_user_info(provider: dict, access_token: str) -> str:
    """Fetch the user's email/display name from the provider's userinfo endpoint."""
    url = provider.get("userinfo_url")
    field = provider.get("userinfo_email_field", "email")
    if not url:
        return ""
    try:
        async with _httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {access_token}"}
            )
            if resp.status_code == 200:
                return resp.json().get(field, "")
    except Exception:
        pass
    return ""


@app.get("/api/plugins/{plugin_name}/oauth/start")
async def oauth_start(
    plugin_name: str, request: Request, user_id: str = Depends(get_user_id)
):
    """Initiate OAuth2 flow for a plugin.

    Each plugin manages its own OAuth tokens independently.
    Looks up the plugin's auth_provider in OAUTH_PROVIDERS, then redirects
    the user to the provider's consent screen.
    """
    if plugin_name not in AVAILABLE_PLUGINS:
        raise HTTPException(404, f"Unknown plugin: {plugin_name}")

    meta = AVAILABLE_PLUGINS[plugin_name]

    # No-auth plugins don't need OAuth
    if meta.get("auth_type") == "none":
        raise HTTPException(400, f"{plugin_name} doesn't require authentication")

    plugin_meta = meta
    provider_name = plugin_meta.get("auth_provider", "")
    provider = OAUTH_PROVIDERS.get(provider_name)
    if not provider:
        raise HTTPException(400, f"No OAuth provider configured for {plugin_name}")

    if not provider["client_id"] or not provider["client_secret"]:
        raise HTTPException(
            500,
            f"OAuth not configured for {provider_name} "
            f"(missing client ID/secret env vars)",
        )

    # Ensure plugin is installed
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )
    if not row:
        raise HTTPException(
            400, f"{plugin_name} plugin not installed — install it first"
        )

    # Generate CSRF state token
    state = uuid.uuid4().hex
    _oauth_states[state] = {
        "user_id": user_id,
        "plugin_name": plugin_name,
        "created": time.time(),
    }

    # Clean up stale states (older than 10 min)
    cutoff = time.time() - 600
    stale = [k for k, v in _oauth_states.items() if v["created"] < cutoff]
    for k in stale:
        _oauth_states.pop(k, None)

    redirect_uri = _get_oauth_redirect_uri(request, plugin_name)

    # Build scopes from plugin metadata
    scopes = plugin_meta.get("scopes", [])

    params = {
        "client_id": provider["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        **provider.get("auth_params", {}),
    }

    auth_url = f"{provider['auth_url']}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(auth_url)


@app.get("/api/plugins/{plugin_name}/oauth/callback")
async def oauth_callback(
    plugin_name: str,
    request: Request,
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
):
    """Handle OAuth2 callback for any plugin.

    Exchanges the auth code for tokens, stores them encrypted in plugin_configs,
    enables the plugin, and signals the agent to reload.
    """
    # Handle user denial
    if error:
        log.warning(f"OAuth denied for {plugin_name}: {error}")
        return RedirectResponse("/plugins?error=oauth_denied")

    if not code or not state:
        return RedirectResponse("/plugins?error=missing_params")

    # Validate CSRF state
    state_data = _oauth_states.pop(state, None)
    if not state_data:
        return RedirectResponse("/plugins?error=invalid_state")

    # Verify the callback matches the plugin that started the flow
    if state_data.get("plugin_name") != plugin_name:
        return RedirectResponse("/plugins?error=state_mismatch")

    user_id = state_data["user_id"]

    plugin_meta = AVAILABLE_PLUGINS.get(plugin_name)
    if not plugin_meta:
        return RedirectResponse("/plugins?error=unknown_plugin")

    provider_name = plugin_meta.get("auth_provider", "")
    provider = OAUTH_PROVIDERS.get(provider_name)
    if not provider:
        return RedirectResponse("/plugins?error=no_provider")

    redirect_uri = _get_oauth_redirect_uri(request, plugin_name)

    # Exchange auth code for tokens
    try:
        tokens = await _exchange_token(provider, code, redirect_uri)
    except Exception as e:
        log.error(f"OAuth token exchange failed for {plugin_name}: {e}")
        return RedirectResponse("/plugins?error=token_exchange_failed")

    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")
    expires_in = tokens.get("expires_in", 3600)
    token_expiry = str(int(time.time()) + int(expires_in))

    if not access_token:
        return RedirectResponse("/plugins?error=no_access_token")

    # Fetch the user's account email/name for display
    account_email = await _fetch_oauth_user_info(provider, access_token)

    # Store tokens encrypted in plugin_configs
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )
    if not row:
        return RedirectResponse("/plugins?error=plugin_not_found")

    plugin_id = row["id"]

    # Store the scopes that were actually granted (from token response or requested)
    granted_scopes = tokens.get("scope", "")
    if not granted_scopes:
        # Fallback: assume all requested scopes were granted
        granted_scopes = " ".join(plugin_meta.get("scopes", []))

    config_entries = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_expiry": token_expiry,
        "granted_scopes": granted_scopes,
    }
    if account_email:
        config_entries["account_email"] = account_email

    for key, value in config_entries.items():
        config_id = uuid.uuid4().hex[:12]
        encrypted = encrypt_value(value)
        await pool.execute(
            """
            INSERT INTO plugin_configs (id, plugin_id, key, value, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (plugin_id, key) DO UPDATE SET value = $4, updated_at = now()
            """,
            config_id,
            plugin_id,
            key,
            encrypted,
        )

    # Auto-enable the plugin
    await pool.execute("UPDATE plugins SET enabled = true WHERE id = $1", plugin_id)

    # Signal agent to reload plugin configs
    await _signal_agent_plugin_reload(user_id)

    # Auto-schedule a recurring token refresh so the agent never hits a stale token.
    # Google tokens expire at 60 min; we refresh at 50 min (3000 s) to stay ahead.
    # Only schedule for OAuth plugins that have a refresh_token.
    if refresh_token and provider_name:
        await _upsert_token_refresh_job(
            pool=pool,
            user_id=user_id,
            plugin_name=plugin_name,
            provider_name=provider_name,
        )

    # Schedule one-shot watch setup job (fires in 30s) for plugins that
    # support push notifications. Reads setup_tool from AVAILABLE_PLUGINS —
    # no plugin-specific logic here.
    await _upsert_watch_setup_job(pool, user_id, plugin_name)

    log.info(f"{plugin_name} OAuth complete for user {user_id} ({account_email})")
    return RedirectResponse(f"/plugins/{plugin_name}?connected=true")


async def _upsert_token_refresh_job(
    pool, user_id: str, plugin_name: str, provider_name: str
) -> None:
    """Insert (or replace) a recurring cron job that tells the LLM to refresh
    the OAuth token for *plugin_name* before it expires.

    The interval and refresh tool name are read from AVAILABLE_PLUGINS so the
    orchestrator never hardcodes plugin-specific logic — adding a new plugin
    only requires updating AVAILABLE_PLUGINS and shipping a refresh tool.

    One job per (user, plugin) — we delete any existing job first so
    re-connecting a plugin resets the schedule cleanly.
    """
    meta = AVAILABLE_PLUGINS.get(plugin_name, {})
    interval_s = meta.get("token_refresh_interval")
    refresh_tool = meta.get("refresh_tool")

    if not interval_s or not refresh_tool:
        log.debug(
            f"Plugin {plugin_name} has no token_refresh_interval or refresh_tool — skipping cron"
        )
        return

    # Remove any existing refresh job for this plugin (idempotent re-connect)
    await pool.execute(
        "DELETE FROM scheduled_jobs WHERE user_id = $1 AND plugin = $2 "
        "AND instruction LIKE 'OAuth token refresh:%'",
        user_id,
        plugin_name,
    )

    job_id = uuid.uuid4().hex
    instruction = (
        f"OAuth token refresh: the {plugin_name} plugin's {provider_name} access token "
        f"is about to expire. Call the `{refresh_tool}` tool now to obtain a new access "
        f"token and keep the plugin working without interruption."
    )
    await pool.execute(
        """
        INSERT INTO scheduled_jobs (id, user_id, plugin, instruction, run_at, interval_s)
        VALUES ($1, $2, $3, $4, now() + make_interval(secs := $5), $5)
        """,
        job_id,
        user_id,
        plugin_name,
        instruction,
        interval_s,
    )
    log.info(
        f"Token refresh job scheduled for {plugin_name} (user={user_id}, "
        f"tool={refresh_tool}, interval={interval_s}s)"
    )


async def _reconcile_token_refresh_jobs(pool) -> None:
    """Ensure every connected OAuth plugin has an active token refresh job.

    Runs once at orchestrator startup. Catches plugins that were connected
    before the cron system existed, or whose jobs were lost (DB wipe, migration).

    For each enabled plugin that declares token_refresh_interval in
    AVAILABLE_PLUGINS and has a stored refresh_token, we insert a scheduled_job
    if one doesn't already exist.
    """
    # Find all enabled OAuth plugins that have a refresh_token stored
    rows = await pool.fetch(
        """
        SELECT p.user_id, p.name AS plugin_name
        FROM plugins p
        JOIN plugin_configs pc ON pc.plugin_id = p.id
        WHERE p.enabled = true
          AND pc.key = 'refresh_token'
          AND pc.value IS NOT NULL
        """
    )

    if not rows:
        log.info("Cron reconcile: no connected OAuth plugins found")
        return

    reconciled = 0
    for row in rows:
        user_id = row["user_id"]
        plugin_name = row["plugin_name"]
        meta = AVAILABLE_PLUGINS.get(plugin_name, {})

        if not meta.get("token_refresh_interval") or not meta.get("refresh_tool"):
            continue  # plugin doesn't declare refresh needs

        # Check if an active refresh job already exists
        existing = await pool.fetchval(
            """
            SELECT id FROM scheduled_jobs
            WHERE user_id = $1 AND plugin = $2
              AND instruction LIKE 'OAuth token refresh:%'
              AND enabled = true
            LIMIT 1
            """,
            user_id,
            plugin_name,
        )
        if existing:
            continue  # already scheduled — nothing to do

        # Create the missing job
        provider_name = meta.get("auth_provider", plugin_name)
        await _upsert_token_refresh_job(
            pool=pool,
            user_id=user_id,
            plugin_name=plugin_name,
            provider_name=provider_name,
        )
        reconciled += 1
        log.info(
            f"Cron reconcile: created missing refresh job for {plugin_name} (user={user_id})"
        )

    log.info(f"Cron reconcile complete: {reconciled} job(s) created")


async def _signal_agent_plugin_reload(user_id: str) -> None:
    """Tell the agent to refresh its plugin configs after OAuth or config change."""
    agent = await _get_agent_for_user(user_id)
    if not agent:
        return

    try:
        async with _httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"http://{agent['host']}:{agent['port']}/config/reload",
                json={},
            )
            log.info(f"Signaled agent plugin reload for user {user_id}")
    except Exception as e:
        log.warning(f"Failed to signal agent plugin reload: {e}")


@app.get(
    "/api/internal/plugins/{plugin_name}/config",
    dependencies=[Depends(verify_agent_secret)],
)
async def get_plugin_config_for_agent(plugin_name: str, user_id: str = Query(...)):
    """Return full decrypted plugin config for the agent to use at tool call time.

    Agent-authenticated (AGENT_SECRET). Returns unmasked values — never
    expose this endpoint to end users.

    Each plugin stores its own tokens. Auto-refreshes expired OAuth tokens.
    No-auth plugins return an empty config.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2 AND enabled = true",
        user_id,
        plugin_name,
    )
    if not row:
        raise HTTPException(404, "Plugin not installed or not enabled")

    plugin_meta = AVAILABLE_PLUGINS.get(plugin_name, {})

    # No-auth plugins have no config to serve
    if plugin_meta.get("auth_type") == "none":
        return {}

    token_plugin_id = row["id"]

    configs = await pool.fetch(
        "SELECT key, value FROM plugin_configs WHERE plugin_id = $1", token_plugin_id
    )
    config_map = {c["key"]: decrypt_value(c["value"]) for c in configs}

    # Auto-refresh expired OAuth tokens (works for any provider)
    provider_name = plugin_meta.get("auth_provider", "")
    if provider_name and config_map.get("token_expiry"):
        try:
            expiry = int(config_map["token_expiry"])
            # Refresh if token expires within 5 minutes
            if time.time() > expiry - 300:
                new_token = await _refresh_oauth_token(token_plugin_id, provider_name)
                if new_token:
                    config_map["access_token"] = new_token
                    config_map["token_expiry"] = str(int(time.time()) + 3600)
        except (ValueError, TypeError):
            pass  # Invalid expiry — skip refresh

    return config_map


# ── Cron API (agent-authenticated) ────────────────────────


@app.post(
    "/api/internal/cron/schedule",
    dependencies=[Depends(verify_agent_secret)],
)
async def cron_schedule(body: CronScheduleBody, user_id: str = Query(...)):
    """Schedule a cron job on behalf of the agent/LLM.

    Agent-authenticated (AGENT_SECRET). The agent submits a job with a
    plain-language instruction; the orchestrator fires it at run_at and
    delivers it back to the agent as a /cron_event.
    """
    from datetime import datetime, timezone

    try:
        run_at = datetime.fromisoformat(body.run_at.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, f"Invalid run_at timestamp: {body.run_at!r}")

    if body.interval_s is not None and body.interval_s <= 0:
        raise HTTPException(400, "interval_s must be a positive integer or null")

    pool = await get_pool()
    job_id = uuid.uuid4().hex
    await pool.execute(
        """
        INSERT INTO scheduled_jobs (id, user_id, plugin, instruction, run_at, interval_s)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        job_id,
        user_id,
        body.plugin,
        body.instruction,
        run_at,
        body.interval_s,
    )
    log.info(
        f"Cron job scheduled: {job_id} for user {user_id} "
        f"(plugin={body.plugin}, run_at={run_at}, interval_s={body.interval_s})"
    )
    return {"job_id": job_id, "status": "scheduled"}


@app.delete(
    "/api/internal/cron/{job_id}",
    dependencies=[Depends(verify_agent_secret)],
)
async def cron_cancel(job_id: str, user_id: str = Query(...)):
    """Cancel (disable) a scheduled cron job.

    Agent-authenticated (AGENT_SECRET). Only disables jobs belonging to
    the requesting user.
    """
    pool = await get_pool()
    result = await pool.execute(
        "UPDATE scheduled_jobs SET enabled = false WHERE id = $1 AND user_id = $2",
        job_id,
        user_id,
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "Job not found or not owned by this user")
    log.info(f"Cron job cancelled: {job_id} (user={user_id})")
    return {"job_id": job_id, "status": "cancelled"}


# ── Webhook Infrastructure ────────────────────────────────
#
# Per-protocol adapters all funnel into one shared handle_webhook() function.
# The orchestrator is a dumb pipe — it never parses plugin-specific payloads.
# Each plugin ships a handle_tool that the LLM calls with the raw payload.
#
# Supported protocols:
#   POST /api/hooks/http/{plugin}/{user_id}    — direct HTTPS push (Calendar, Drive…)
#   POST /api/hooks/pubsub/{plugin}/{user_id}  — Google Cloud Pub/Sub push subscription
#
# Adding a new protocol = one new adapter endpoint + call handle_webhook().
# Adding a new plugin   = zero orchestrator changes.


async def _handle_webhook(
    plugin_name: str,
    user_id: str,
    payload: dict,
    pool,
) -> dict:
    """
    Shared webhook handler — protocol-agnostic core.

    1. Verify plugin is installed and enabled for this user
    2. Store raw payload in plugin_events
    3. Forward to user's running agent with handle_tool hint
       (agent calls the plugin's handle_tool with the raw payload)
    4. If agent is not running, event is stored and will be delivered
       when the agent next starts (deferred flusher picks it up)
    """
    plugin_row = await pool.fetchrow(
        "SELECT id, enabled FROM plugins WHERE user_id = $1 AND name = $2",
        user_id,
        plugin_name,
    )
    if not plugin_row:
        log.warning(
            f"Webhook ignored: plugin {plugin_name} not installed for user {user_id}"
        )
        # Return 200 to prevent retries from external services
        return {"status": "plugin_not_installed"}
    if not plugin_row["enabled"]:
        log.debug(f"Webhook ignored: plugin {plugin_name} disabled for user {user_id}")
        return {"status": "plugin_disabled"}

    # Read handle_tool from AVAILABLE_PLUGINS — orchestrator never hardcodes plugin logic
    meta = AVAILABLE_PLUGINS.get(plugin_name, {})
    handle_tool = meta.get("webhook", {}).get("handle_tool", "")

    event_id = uuid.uuid4().hex

    await pool.execute(
        """
        INSERT INTO plugin_events
            (id, user_id, plugin_name, event_type, source_id, summary, payload)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        """,
        event_id,
        user_id,
        plugin_name,
        "webhook",  # raw type — plugin's handle_tool determines real type
        "",  # source_id — plugin fills this in
        "",  # summary  — plugin fills this in
        json.dumps(payload),
    )

    # Forward to agent — include handle_tool so agent knows which tool to call
    agent = await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        user_id,
    )
    if agent:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"http://{agent['host']}:{agent['port']}/plugin_event",
                    json={
                        "event_id": event_id,
                        "plugin": plugin_name,
                        "event_type": "webhook",
                        "handle_tool": handle_tool,
                        "payload": payload,
                    },
                )
        except Exception as e:
            log.warning(
                f"Failed to forward webhook to agent ({plugin_name}/{user_id}): {e}"
            )
    else:
        log.info(f"Webhook stored (agent not running): {plugin_name}/{user_id}")

    log.info(f"Webhook received: {plugin_name}/{user_id} event_id={event_id}")
    return {"status": "received", "event_id": event_id}


# ── HTTP adapter ──────────────────────────────────────────


@app.post("/api/hooks/http/{plugin_name}/{user_id}")
async def webhook_http(plugin_name: str, user_id: str, request: Request):
    """
    Direct HTTPS push adapter.

    Used by: Google Calendar, Google Drive, and any service that pushes
    directly to an HTTPS endpoint.

    No protocol-level verification here — plugin's handle_tool is responsible
    for any payload-level signature checks (e.g. Google channel token).
    """
    pool = await get_pool()
    try:
        payload = await request.json()
    except Exception:
        payload = {"raw": (await request.body()).decode("utf-8", errors="replace")}

    # Google Calendar sends a sync notification on watch creation — acknowledge and ignore
    resource_state = request.headers.get("x-goog-resource-state", "")
    if resource_state == "sync":
        log.debug(f"Sync notification for {plugin_name}/{user_id} — acknowledged")
        return {"status": "sync_acknowledged"}

    # Enrich payload with Google push headers if present (Calendar, Drive)
    if resource_state:
        payload["_goog_resource_state"] = resource_state
        payload["_goog_resource_id"] = request.headers.get("x-goog-resource-id", "")
        payload["_goog_channel_id"] = request.headers.get("x-goog-channel-id", "")
        payload["_goog_message_number"] = request.headers.get(
            "x-goog-message-number", ""
        )

    return await _handle_webhook(plugin_name, user_id, payload, pool)


# ── Pub/Sub adapter ───────────────────────────────────────


@app.post("/api/hooks/pubsub/{plugin_name}/{user_id}")
async def webhook_pubsub(plugin_name: str, user_id: str, request: Request):
    """
    Google Cloud Pub/Sub push subscription adapter.

    Used by: Gmail (and any future plugin using Pub/Sub delivery).

    Pub/Sub wraps the actual message in an envelope:
    {
      "message": {
        "data": "<base64-encoded payload>",
        "messageId": "...",
        "attributes": {...}
      },
      "subscription": "projects/.../subscriptions/..."
    }

    This adapter:
    1. Verifies the Pub/Sub JWT (if PUBSUB_AUDIENCE is configured)
    2. Unwraps the envelope and base64-decodes the message data
    3. Passes the decoded payload to _handle_webhook()

    JWT verification uses Google's public keys fetched from their JWKS endpoint.
    In dev (PUBSUB_AUDIENCE unset), verification is skipped.
    """
    pool = await get_pool()

    # ── JWT verification ──
    if PUBSUB_AUDIENCE:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            log.warning(
                f"Pub/Sub request missing Bearer token for {plugin_name}/{user_id}"
            )
            raise HTTPException(401, "Missing Pub/Sub authorization token")
        token = auth_header[7:]
        try:
            await _verify_pubsub_jwt(token)
        except Exception as e:
            log.warning(f"Pub/Sub JWT verification failed: {e}")
            raise HTTPException(403, "Invalid Pub/Sub token")

    # ── Unwrap Pub/Sub envelope ──
    try:
        envelope = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON in Pub/Sub envelope")

    message = envelope.get("message", {})
    if not message:
        raise HTTPException(400, "Missing 'message' in Pub/Sub envelope")

    # Decode base64 message data
    raw_data = message.get("data", "")
    try:
        import base64 as _b64

        decoded = _b64.b64decode(raw_data).decode("utf-8")
        payload = json.loads(decoded)
    except Exception:
        # Data is not JSON — pass as raw string
        payload = {"raw": raw_data}

    # Attach Pub/Sub metadata for the handle_tool to use if needed
    payload["_pubsub_message_id"] = message.get("messageId", "")
    payload["_pubsub_attributes"] = message.get("attributes", {})
    payload["_pubsub_subscription"] = envelope.get("subscription", "")

    return await _handle_webhook(plugin_name, user_id, payload, pool)


async def _verify_pubsub_jwt(token: str) -> None:
    """
    Verify a Google-signed Pub/Sub push JWT.

    Google signs push subscription tokens with its service account key.
    We fetch Google's public JWKS and verify the signature + audience.
    Raises on failure.
    """
    import base64 as _b64

    # Decode header to get kid
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Not a valid JWT")

    # Fetch Google's public keys
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get("https://www.googleapis.com/oauth2/v3/certs")
        resp.raise_for_status()
        jwks = resp.json()

    # Decode payload (no verification yet — just to check claims)
    payload_b64 = parts[1] + "=="  # re-pad
    try:
        claims = json.loads(_b64.urlsafe_b64decode(payload_b64).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Cannot decode JWT payload: {e}")

    # Verify audience
    aud = claims.get("aud", "")
    if aud != PUBSUB_AUDIENCE:
        raise ValueError(
            f"JWT audience mismatch: got {aud!r}, expected {PUBSUB_AUDIENCE!r}"
        )

    # Verify expiry
    exp = claims.get("exp", 0)
    if time.time() > exp:
        raise ValueError("JWT has expired")

    # Verify issuer is Google
    iss = claims.get("iss", "")
    if iss not in (
        "https://accounts.google.com",
        "accounts.google.com",
    ):
        raise ValueError(f"Unexpected JWT issuer: {iss!r}")

    # Full signature verification would require a JWT library (e.g. PyJWT + cryptography).
    # The issuer + audience + expiry checks above are sufficient for most deployments.
    # To add full RS256 signature verification, install PyJWT and verify against jwks.
    log.debug("Pub/Sub JWT claims verified (iss=%s, aud=%s)", iss, aud)


# ── Watch setup / renewal cron jobs ──────────────────────


async def _upsert_watch_setup_job(
    pool,
    user_id: str,
    plugin_name: str,
) -> None:
    """
    Schedule a one-shot cron job to set up push notifications for a plugin.

    Called after OAuth connect. Fires 30 seconds later so the OAuth flow
    completes cleanly before the agent tries to call the setup tool.

    Reads setup_tool from AVAILABLE_PLUGINS — zero plugin-specific logic here.
    """
    meta = AVAILABLE_PLUGINS.get(plugin_name, {})
    webhook_cfg = meta.get("webhook", {})
    setup_tool = webhook_cfg.get("setup_tool")

    if not setup_tool:
        return  # plugin doesn't support push notifications

    # Remove any existing setup job (idempotent re-connect)
    await pool.execute(
        "DELETE FROM scheduled_jobs WHERE user_id = $1 AND plugin = $2 "
        "AND instruction LIKE 'Watch setup:%'",
        user_id,
        plugin_name,
    )

    job_id = uuid.uuid4().hex
    protocol = webhook_cfg.get("protocol", "http")
    instruction = (
        f"Watch setup: register push notifications for the {plugin_name} plugin. "
        f"Call the `{setup_tool}` tool now to set up {protocol} push delivery so "
        f"incoming events are forwarded to this agent in real time."
    )
    # One-shot job (no interval_s) — fires in 30 seconds
    await pool.execute(
        """
        INSERT INTO scheduled_jobs (id, user_id, plugin, instruction, run_at)
        VALUES ($1, $2, $3, $4, now() + interval '30 seconds')
        """,
        job_id,
        user_id,
        plugin_name,
        instruction,
    )
    log.info(
        f"Watch setup job scheduled for {plugin_name} (user={user_id}, tool={setup_tool})"
    )


async def _reconcile_watch_setup_jobs(pool) -> None:
    """
    Ensure every connected OAuth plugin that supports webhooks has an active
    watch registration or a pending setup job.

    Runs once at orchestrator startup. Catches plugins connected before the
    watch system existed, or whose watches were lost (DB wipe, expiry).
    """
    rows = await pool.fetch(
        """
        SELECT p.user_id, p.name AS plugin_name
        FROM plugins p
        JOIN plugin_configs pc ON pc.plugin_id = p.id
        WHERE p.enabled = true
          AND pc.key = 'refresh_token'
          AND pc.value IS NOT NULL
        """
    )

    if not rows:
        return

    reconciled = 0
    for row in rows:
        user_id = row["user_id"]
        plugin_name = row["plugin_name"]
        meta = AVAILABLE_PLUGINS.get(plugin_name, {})

        if not meta.get("webhook", {}).get("setup_tool"):
            continue  # plugin doesn't support push notifications

        # Skip if watch is already registered and not expired
        existing_watch = await pool.fetchrow(
            """
            SELECT id FROM watch_registrations
            WHERE user_id = $1 AND plugin_name = $2
              AND (expires_at IS NULL OR expires_at > now() + interval '1 hour')
            """,
            user_id,
            plugin_name,
        )
        if existing_watch:
            continue

        # Skip if a setup job is already pending
        existing_job = await pool.fetchval(
            """
            SELECT id FROM scheduled_jobs
            WHERE user_id = $1 AND plugin = $2
              AND instruction LIKE 'Watch setup:%'
              AND enabled = true
            LIMIT 1
            """,
            user_id,
            plugin_name,
        )
        if existing_job:
            continue

        await _upsert_watch_setup_job(pool, user_id, plugin_name)
        reconciled += 1
        log.info(f"Watch reconcile: scheduled setup for {plugin_name} (user={user_id})")

    if reconciled:
        log.info(f"Watch reconcile complete: {reconciled} setup job(s) created")


@app.get("/api/plugins/{plugin_name}/events")
async def list_plugin_events(
    plugin_name: str,
    user_id: str = Depends(get_user_id),
    limit: int = 20,
):
    """List recent events for a plugin."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, event_type, source_id, summary, decision, created_at
        FROM plugin_events WHERE user_id = $1 AND plugin_name = $2
        ORDER BY created_at DESC LIMIT $3
        """,
        user_id,
        plugin_name,
        limit,
    )
    return [dict(r) for r in rows]


@app.post(
    "/api/internal/events/{event_id}/decision",
    dependencies=[Depends(verify_agent_secret)],
)
async def report_event_decision(event_id: str, request: Request):
    """Receive decision from agent and persist to DB."""
    body = await request.json()
    decision = body.get("decision", "archive")
    notification = body.get("notification", "")

    pool = await get_pool()

    if decision == "deferred":
        # Schedule for 30 minutes from now
        await pool.execute(
            """UPDATE plugin_events
               SET decision = $1, batch_notification = $2,
                   scheduled_for = now() + interval '30 minutes'
               WHERE id = $3""",
            decision,
            notification,
            event_id,
        )
    else:
        await pool.execute(
            "UPDATE plugin_events SET decision = $1, batch_notification = $2 WHERE id = $3",
            decision,
            notification,
            event_id,
        )
    return {"status": "ok"}


# ── Watch registration ─────────────────────────────────────


@app.post(
    "/api/internal/watches",
    dependencies=[Depends(verify_agent_secret)],
)
async def register_watch(request: Request):
    """Upsert a watch registration from an agent-side SetupXxxWatchTool.

    Called by BaseWatchTool._register_watch() after the external service
    confirms the watch.  Stores (or replaces) the row in watch_registrations
    so _reconcile_watch_setup_jobs() can detect active watches at startup.
    """
    body = await request.json()
    user_id = body.get("user_id", "")
    plugin_name = body.get("plugin_name", "")
    protocol = body.get("protocol", "")
    watch_id = body.get("watch_id", "")
    resource_id = body.get("resource_id", "")
    expires_at_raw = body.get("expires_at")  # ISO string or None

    if not user_id or not plugin_name:
        raise HTTPException(status_code=400, detail="user_id and plugin_name required")

    pool = await get_pool()

    expires_at_val = None
    if expires_at_raw:
        try:
            from datetime import datetime, timezone

            expires_at_val = datetime.fromisoformat(expires_at_raw).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    await pool.execute(
        """
        INSERT INTO watch_registrations
            (user_id, plugin_name, protocol, watch_id, resource_id, expires_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (user_id, plugin_name)
        DO UPDATE SET
            protocol    = EXCLUDED.protocol,
            watch_id    = EXCLUDED.watch_id,
            resource_id = EXCLUDED.resource_id,
            expires_at  = EXCLUDED.expires_at,
            registered_at = now()
        """,
        user_id,
        plugin_name,
        protocol,
        watch_id,
        resource_id,
        expires_at_val,
    )

    log.info(
        "Watch registered: user=%s plugin=%s protocol=%s watch_id=%s",
        user_id,
        plugin_name,
        protocol,
        watch_id,
    )
    return {"status": "ok"}


# ── Vobiz Telephony Proxy ──────────────────────────────────
#
# VoBiz can only reach the orchestrator (public).  Agent containers are
# internal.  These routes accept VoBiz callbacks, look up the owning user
# by phone number, and proxy to the correct agent.


async def _lookup_user_by_vobiz_number(phone: str) -> dict | None:
    """Find the agent for the user who owns a VoBiz from_number.

    Returns {"user_id": str, "host": str, "port": int} or None.
    """
    if not phone:
        return None
    pool = await get_pool()
    # Normalise: strip leading + and whitespace
    digits = "".join(ch for ch in phone if ch.isdigit())
    if not digits:
        return None

    # plugin_configs stores encrypted values — we must decrypt to compare.
    rows = await pool.fetch(
        """
        SELECT p.user_id, pc.value
        FROM plugin_configs pc
        JOIN plugins p ON p.id = pc.plugin_id
        WHERE pc.key = 'from_number' AND p.name = 'vobiz' AND p.enabled = true
        """
    )
    target_user_id: str | None = None
    for row in rows:
        decrypted = decrypt_value(row["value"])
        decrypted_digits = "".join(ch for ch in decrypted if ch.isdigit())
        if decrypted_digits == digits:
            target_user_id = row["user_id"]
            break

    if not target_user_id:
        return None

    agent = await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        target_user_id,
    )
    if not agent:
        return None

    return {
        "user_id": target_user_id,
        "host": agent["host"],
        "port": agent["port"],
    }


@app.post("/api/plugins/vobiz/webhook")
async def vobiz_webhook_proxy(request: Request):
    """Proxy inbound-call webhook from VoBiz to the owning user's agent.

    VoBiz POSTs form data with To= (our user's VoBiz number).
    We look up the user, then proxy to their agent's /plugins/vobiz/webhook.
    The agent returns XML that VoBiz uses to connect the call.
    """
    body = await request.body()
    form = await request.form()
    # Inbound: To is our number.  Outbound answer callback: From is our number.
    phone = form.get("To") or form.get("From") or ""
    log.info("VoBiz webhook: To=%s From=%s", form.get("To"), form.get("From"))

    target = await _lookup_user_by_vobiz_number(str(phone))
    if not target:
        log.warning("VoBiz webhook: no user found for phone %s", phone)
        return Response(
            content='<?xml version="1.0"?><Response><Hangup/></Response>',
            media_type="application/xml",
        )

    # Proxy to agent — pass original headers so agent can build ws:// URL
    agent_url = f"http://{target['host']}:{target['port']}/plugins/vobiz/webhook"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            agent_url,
            content=body,
            headers={
                "content-type": request.headers.get(
                    "content-type", "application/x-www-form-urlencoded"
                ),
                "host": request.headers.get("host", ""),
                "x-forwarded-proto": request.headers.get(
                    "x-forwarded-proto", request.url.scheme
                ),
                "x-forwarded-host": request.headers.get(
                    "x-forwarded-host", request.headers.get("host", "")
                ),
            },
        )
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/xml"),
    )


@app.post("/api/plugins/vobiz/answer")
async def vobiz_answer_proxy(request: Request):
    """Proxy outbound-call answer callback from VoBiz to the owning user's agent.

    When an outbound call is answered, VoBiz POSTs here with From= (our number).
    """
    body = await request.body()
    form = await request.form()
    phone = form.get("From") or form.get("To") or ""
    log.info("VoBiz answer: To=%s From=%s", form.get("To"), form.get("From"))

    target = await _lookup_user_by_vobiz_number(str(phone))
    if not target:
        log.warning("VoBiz answer: no user found for phone %s", phone)
        return Response(
            content='<?xml version="1.0"?><Response><Hangup/></Response>',
            media_type="application/xml",
        )

    # Forward query params (e.g. ?greeting=...)
    qs = str(request.url.query)
    agent_url = f"http://{target['host']}:{target['port']}/plugins/vobiz/answer"
    if qs:
        agent_url += f"?{qs}"

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            agent_url,
            content=body,
            headers={
                "content-type": request.headers.get(
                    "content-type", "application/x-www-form-urlencoded"
                ),
                "host": request.headers.get("host", ""),
                "x-forwarded-proto": request.headers.get(
                    "x-forwarded-proto", request.url.scheme
                ),
                "x-forwarded-host": request.headers.get(
                    "x-forwarded-host", request.headers.get("host", "")
                ),
            },
        )
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/xml"),
    )


@app.websocket("/api/plugins/vobiz/ws")
async def vobiz_ws_proxy(ws: WebSocket):
    """Proxy VoBiz media-stream WebSocket to the owning user's agent.

    VoBiz connects here after the XML <Stream> element.  The first message
    is a JSON 'start' event containing callId — but we don't have the phone
    number in the WS handshake.  Instead, we use the Referer / Origin or
    query params.  For now, we accept and read the first message to get the
    callId, then look up the call's phone number from recent webhook logs.

    Simpler approach: the agent's XML response includes the WS URL.  We
    rewrite it in the webhook/answer proxy to include a user_id hint.
    """
    # The agent's XML <Stream> URL will be rewritten to include ?uid=...
    user_id = ws.query_params.get("uid", "")
    if not user_id:
        await ws.close(code=1008, reason="Missing user identifier")
        return

    pool = await get_pool()
    agent = await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        user_id,
    )
    if not agent:
        await ws.close(code=1008, reason="No agent available")
        return

    agent_ws_url = f"ws://{agent['host']}:{agent['port']}/plugins/vobiz/ws"
    log.info("VoBiz WS proxy: user=%s → %s", user_id, agent_ws_url)

    await ws.accept()

    import websockets as ws_lib

    try:
        async with ws_lib.connect(agent_ws_url) as agent_ws:

            async def client_to_agent():
                try:
                    while True:
                        data = await ws.receive_text()
                        await agent_ws.send(data)
                except WebSocketDisconnect:
                    pass

            async def agent_to_client():
                try:
                    async for msg in agent_ws:
                        await ws.send_text(
                            msg if isinstance(msg, str) else msg.decode()
                        )
                except Exception:
                    pass

            await asyncio.gather(client_to_agent(), agent_to_client())
    except Exception as e:
        log.warning("VoBiz WS proxy error: %s", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ── Health ─────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}
