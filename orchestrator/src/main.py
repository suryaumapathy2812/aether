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
    MULTI_USER_MODE,
    IDLE_TIMEOUT_MINUTES,
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

    # Reconcile orphaned containers on startup
    if not AGENT_SECRET:
        log.warning(
            "⚠ AGENT_SECRET not set — agent registration endpoints are UNPROTECTED"
        )

    if MULTI_USER_MODE:
        pool = await get_pool()
        await reconcile_containers(pool)
        asyncio.create_task(_idle_reaper())
        log.info(
            f"Orchestrator ready (multi-user mode, idle timeout={IDLE_TIMEOUT_MINUTES}m)"
        )
    else:
        log.info("Orchestrator ready (dev mode — single agent)")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


async def _idle_reaper():
    """Background task: stop agent containers that haven't sent a heartbeat."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        try:
            pool = await get_pool()
            idle_agents = await pool.fetch(
                "SELECT id, user_id FROM agents WHERE status = 'running' "
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
    Ensure the user has a running agent.

    Dev mode: grab any running agent (single shared agent from docker-compose).
    Multi-user mode: provision a dedicated container per user via Docker SDK.
    """
    pool = await get_pool()

    if not MULTI_USER_MODE:
        # ── Dev mode: single shared agent ──────────────────────
        # Always prefer the most recently healthy agent (handles restarts/stale records)
        freshest = await pool.fetchrow(
            "SELECT id FROM agents WHERE status = 'running' ORDER BY last_health DESC NULLS LAST LIMIT 1"
        )
        if not freshest:
            return  # No agents at all — agent hasn't registered yet

        # Assign (or reassign) the user to the freshest agent
        existing = await pool.fetchrow(
            "SELECT id FROM agents WHERE user_id = $1 AND status = 'running'", user_id
        )
        if existing and existing["id"] == freshest["id"]:
            return  # Already on the right agent

        # Clear old assignment if any
        if existing:
            await pool.execute(
                "UPDATE agents SET user_id = NULL WHERE id = $1", existing["id"]
            )

        await pool.execute(
            "UPDATE agents SET user_id = $1 WHERE id = $2", user_id, freshest["id"]
        )
        log.info(f"Dev mode: assigned agent {freshest['id']} to user {user_id}")
        return

    # ── Multi-user mode: dedicated container per user ──────
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
        return agent

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
AVAILABLE_PLUGINS = {
    "gmail": {
        "name": "gmail",
        "display_name": "Gmail",
        "description": "Monitor and respond to Gmail emails",
        "auth_type": "oauth2",
        "auth_provider": "google",
        "config_fields": [
            {
                "key": "account_email",
                "label": "Gmail Address",
                "type": "text",
                "required": True,
            },
        ],
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

    result = []
    for name, meta in AVAILABLE_PLUGINS.items():
        entry = {**meta}
        if name in installed:
            entry["installed"] = True
            entry["plugin_id"] = installed[name]["id"]
            entry["enabled"] = installed[name]["enabled"]
        else:
            entry["installed"] = False
            entry["plugin_id"] = None
            entry["enabled"] = False
        result.append(entry)
    return result


@app.post("/api/plugins/{plugin_name}/install")
async def install_plugin(plugin_name: str, user_id: str = Depends(get_user_id)):
    """Install a plugin for the user."""
    if plugin_name not in AVAILABLE_PLUGINS:
        raise HTTPException(404, f"Unknown plugin: {plugin_name}")

    pool = await get_pool()
    plugin_id = uuid.uuid4().hex[:12]
    await pool.execute(
        """
        INSERT INTO plugins (id, user_id, name, enabled)
        VALUES ($1, $2, $3, false)
        ON CONFLICT (user_id, name) DO NOTHING
        """,
        plugin_id,
        user_id,
        plugin_name,
    )
    row = await pool.fetchrow(
        "SELECT id FROM plugins WHERE user_id = $1 AND name = $2", user_id, plugin_name
    )
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


# ── Health ─────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}
