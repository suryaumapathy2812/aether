"""
Aether Orchestrator
────────────────────
Agent registry, WebSocket proxy, device pairing, API key management.

Auth is handled by better-auth in the dashboard (Next.js).
The orchestrator validates sessions by reading the shared Postgres session table.
"""

from __future__ import annotations

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

    # Fetch user's API keys for injection into the container
    rows = await pool.fetch(
        "SELECT provider, key_value FROM api_keys WHERE user_id = $1", user_id
    )
    user_keys = {r["provider"]: r["key_value"] for r in rows}

    # Provision (or restart) the agent container
    result = await provision_agent(user_id, user_keys)

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


@app.get("/auth/me")
async def me(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    row = await pool.fetchrow(
        'SELECT id, email, name, created_at FROM "user" WHERE id = $1', user_id
    )
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)


# ── Agent Registry ─────────────────────────────────────────


@app.post("/agents/register")
async def register_agent(body: AgentRegisterRequest):
    """Called by an Aether agent on startup."""
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


@app.post("/agents/{agent_id}/assign")
async def assign_agent(agent_id: str, user_id: str = Query(...)):
    """Assign an agent to a user."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE agents SET user_id = $1 WHERE id = $2", user_id, agent_id
    )
    return {"status": "assigned"}


@app.get("/agents/health")
async def list_agents():
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, user_id, host, port, status, last_health FROM agents ORDER BY registered_at DESC"
    )
    return [dict(r) for r in rows]


@app.post("/agents/{agent_id}/heartbeat")
async def heartbeat(agent_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE agents SET last_health = now(), status = 'running' WHERE id = $1",
        agent_id,
    )
    return {"status": "ok"}


# ── Device Pairing ─────────────────────────────────────────


@app.post("/pair/request")
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


@app.post("/pair/confirm")
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


@app.get("/pair/status/{code}")
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


@app.get("/devices")
async def list_devices(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, name, device_type, paired_at, last_seen FROM devices WHERE user_id = $1 ORDER BY paired_at DESC",
        user_id,
    )
    return [dict(r) for r in rows]


# ── API Key Management ─────────────────────────────────────


@app.post("/services/keys")
async def store_api_key(body: ApiKeyBody, user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    key_id = uuid.uuid4().hex[:12]
    await pool.execute(
        """
        INSERT INTO api_keys (id, user_id, provider, key_value)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, provider) DO UPDATE SET key_value = $4
    """,
        key_id,
        user_id,
        body.provider,
        body.key_value,
    )
    return {"status": "saved"}


@app.get("/services/keys")
async def list_api_keys(user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT provider, substring(key_value, 1, 8) || '...' as preview FROM api_keys WHERE user_id = $1",
        user_id,
    )
    return [dict(r) for r in rows]


@app.delete("/services/keys/{provider}")
async def delete_api_key(provider: str, user_id: str = Depends(get_user_id)):
    pool = await get_pool()
    await pool.execute(
        "DELETE FROM api_keys WHERE user_id = $1 AND provider = $2",
        user_id,
        provider,
    )
    return {"status": "deleted"}


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


@app.get("/memory/facts")
async def proxy_memory_facts(user_id: str = Depends(get_user_id)):
    agent = await _get_agent_for_user(user_id)
    if not agent:
        raise HTTPException(404, "No agent assigned")
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"http://{agent['host']}:{agent['port']}/memory/facts")
        return resp.json()


@app.get("/memory/sessions")
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


@app.get("/memory/conversations")
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


@app.websocket("/ws")
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


# ── Health ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}
