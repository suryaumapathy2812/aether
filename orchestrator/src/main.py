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
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .db import get_pool, close_pool, bootstrap_schema
from .auth import get_user_id, get_user_id_from_ws
from .crypto import encrypt_value, decrypt_value

# ── Agent registration secret ─────────────────────────────
AGENT_SECRET = os.getenv("AGENT_SECRET", "")


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

    pool = await get_pool()
    await reconcile_containers(pool)
    asyncio.create_task(_idle_reaper())
    asyncio.create_task(_deferred_flusher())
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

PREFERENCE_COLUMNS = [
    "stt_provider",
    "stt_model",
    "stt_language",
    "llm_provider",
    "llm_model",
    "tts_provider",
    "tts_model",
    "tts_voice",
    "base_style",
    "custom_instructions",
]

# Map preference fields to agent env vars
PREF_TO_ENV = {
    "stt_provider": "AETHER_STT_PROVIDER",
    "stt_model": "AETHER_STT_MODEL",
    "stt_language": "AETHER_STT_LANGUAGE",
    "llm_provider": "AETHER_LLM_PROVIDER",
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
# Plugins with "token_source" share OAuth tokens from another plugin.
# e.g. google-calendar uses Gmail's Google OAuth tokens — no separate connect.
# When the user connects Gmail, Calendar & Contacts get tokens automatically.
#
# The "scopes" on Gmail include Calendar + Contacts scopes so a single
# Google consent covers all three plugins.
AVAILABLE_PLUGINS = {
    "gmail": {
        "name": "gmail",
        "display_name": "Gmail",
        "description": "Monitor and respond to Gmail emails",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "scopes": [
            # Gmail
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
            # Calendar
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",
            # Contacts
            "https://www.googleapis.com/auth/contacts.readonly",
            # User info
            "https://www.googleapis.com/auth/userinfo.email",
        ],
        "config_fields": [
            {
                "key": "account_email",
                "label": "Gmail Address",
                "type": "text",
                "required": True,
            },
        ],
    },
    "google-calendar": {
        "name": "google-calendar",
        "display_name": "Google Calendar",
        "description": "View and create calendar events",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "token_source": "gmail",  # shares Gmail's OAuth tokens
        "scopes": [],  # scopes are on gmail's entry
        "config_fields": [],
    },
    "google-contacts": {
        "name": "google-contacts",
        "display_name": "Google Contacts",
        "description": "Search and look up your contacts",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "token_source": "gmail",  # shares Gmail's OAuth tokens
        "scopes": [],  # scopes are on gmail's entry
        "config_fields": [],
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
}


class PluginConfigBody(BaseModel):
    config: dict[str, str]


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

    result = []
    for name, meta in AVAILABLE_PLUGINS.items():
        entry = {**meta}
        if name in installed:
            entry["installed"] = True
            entry["plugin_id"] = installed[name]["id"]
            entry["enabled"] = installed[name]["enabled"]
            # Plugins with token_source are connected if their source is connected
            token_source = meta.get("token_source")
            if token_source:
                entry["connected"] = token_source in connected_plugins
            else:
                entry["connected"] = name in connected_plugins
        else:
            entry["installed"] = False
            entry["plugin_id"] = None
            entry["enabled"] = False
            entry["connected"] = False
        # No-auth plugins are always "connected" once installed
        if meta.get("auth_type") == "none" and name in installed:
            entry["connected"] = True
        result.append(entry)
    return result


@app.post("/api/plugins/{plugin_name}/install")
async def install_plugin(plugin_name: str, user_id: str = Depends(get_user_id)):
    """Install a plugin for the user.

    Plugins with token_source (e.g. google-calendar → gmail) are auto-enabled
    if the source plugin is already connected. No-auth plugins are auto-enabled.
    """
    if plugin_name not in AVAILABLE_PLUGINS:
        raise HTTPException(404, f"Unknown plugin: {plugin_name}")

    meta = AVAILABLE_PLUGINS[plugin_name]
    pool = await get_pool()
    plugin_id = uuid.uuid4().hex[:12]

    # Check if we should auto-enable
    auto_enable = False
    if meta.get("auth_type") == "none":
        auto_enable = True
    elif meta.get("token_source"):
        # Auto-enable if the source plugin has tokens
        source_row = await pool.fetchrow(
            "SELECT p.id FROM plugins p JOIN plugin_configs pc ON pc.plugin_id = p.id "
            "WHERE p.user_id = $1 AND p.name = $2 AND pc.key = 'access_token'",
            user_id,
            meta["token_source"],
        )
        if source_row:
            auto_enable = True

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
    pool = await get_pool()
    await pool.execute(
        "UPDATE plugins SET enabled = true WHERE user_id = $1 AND name = $2",
        user_id,
        plugin_name,
    )
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
    plugin_name: str, body: PluginConfigBody, user_id: str = Depends(get_user_id)
):
    """Save plugin configuration (encrypted at rest)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )
    if not row:
        raise HTTPException(404, "Plugin not installed")
    plugin_id = row["id"]

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
    return {"status": "saved"}


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
    """Initiate OAuth2 flow for any plugin.

    Plugins with token_source redirect to the source plugin's OAuth flow.
    Looks up the plugin's auth_provider in OAUTH_PROVIDERS, then redirects
    the user to the provider's consent screen.
    """
    if plugin_name not in AVAILABLE_PLUGINS:
        raise HTTPException(404, f"Unknown plugin: {plugin_name}")

    meta = AVAILABLE_PLUGINS[plugin_name]

    # No-auth plugins don't need OAuth
    if meta.get("auth_type") == "none":
        raise HTTPException(400, f"{plugin_name} doesn't require authentication")

    # Plugins with token_source use the source plugin's OAuth
    token_source = meta.get("token_source")
    if token_source:
        plugin_name = token_source
        if plugin_name not in AVAILABLE_PLUGINS:
            raise HTTPException(500, f"Token source plugin '{plugin_name}' not found")

    plugin_meta = AVAILABLE_PLUGINS[plugin_name]
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

    config_entries = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_expiry": token_expiry,
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

    # Auto-enable any installed plugins that share this plugin's tokens
    for dep_name, dep_meta in AVAILABLE_PLUGINS.items():
        if dep_meta.get("token_source") == plugin_name:
            await pool.execute(
                "UPDATE plugins SET enabled = true WHERE user_id = $1 AND name = $2",
                user_id,
                dep_name,
            )

    # Signal agent to reload plugin configs
    await _signal_agent_plugin_reload(user_id)

    log.info(f"{plugin_name} OAuth complete for user {user_id} ({account_email})")
    return RedirectResponse(f"/plugins/{plugin_name}?connected=true")


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

    Plugins with token_source (e.g. google-calendar) read tokens from
    the source plugin (gmail). Auto-refreshes expired OAuth tokens.
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

    # Determine which plugin holds the tokens
    token_source = plugin_meta.get("token_source")
    if token_source:
        # Read tokens from the source plugin (e.g. gmail)
        source_row = await pool.fetchrow(
            "SELECT id FROM plugins WHERE user_id = $1 AND name = $2 AND enabled = true",
            user_id,
            token_source,
        )
        if not source_row:
            raise HTTPException(404, f"Source plugin '{token_source}' not enabled")
        token_plugin_id = source_row["id"]
    else:
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


# ── Webhook Receiver ──────────────────────────────────────


@app.post("/api/hooks/{plugin_name}/{user_id}")
async def webhook_receiver(plugin_name: str, user_id: str, request: Request):
    """
    Receive webhook events from third-party services.

    1. Verify plugin is installed and enabled
    2. Store raw event in plugin_events
    3. Forward to user's agent as a plugin_event
    """
    pool = await get_pool()

    plugin = await pool.fetchrow(
        "SELECT id, enabled FROM plugins WHERE user_id = $1 AND name = $2",
        user_id,
        plugin_name,
    )
    if not plugin:
        raise HTTPException(404, "Plugin not found for user")
    if not plugin["enabled"]:
        return {"status": "plugin_disabled"}

    try:
        payload = await request.json()
    except Exception:
        payload = {"raw": (await request.body()).decode("utf-8", errors="replace")}

    event_id = uuid.uuid4().hex
    event_type = payload.get("event_type", payload.get("type", "unknown"))
    summary = payload.get("summary", "")
    source_id = payload.get("source_id", payload.get("message_id", ""))

    await pool.execute(
        """
        INSERT INTO plugin_events (id, user_id, plugin_name, event_type, source_id, summary, payload)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        """,
        event_id,
        user_id,
        plugin_name,
        event_type,
        source_id,
        summary,
        json.dumps(payload),
    )

    # Forward to agent via HTTP
    agent = await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        user_id,
    )
    if agent:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"http://{agent['host']}:{agent['port']}/plugin_event",
                    json={
                        "event_id": event_id,
                        "plugin": plugin_name,
                        "event_type": event_type,
                        "source_id": source_id,
                        "summary": summary,
                        "payload": payload,
                    },
                )
        except Exception as e:
            log.warning(f"Failed to forward plugin event to agent: {e}")

    log.info(f"Webhook: {plugin_name}/{event_type} for user {user_id}")
    return {"status": "received", "event_id": event_id}


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


# ── Health ─────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}
