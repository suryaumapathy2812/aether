"""
Aether Orchestrator
────────────────────
Agent registry, WebSocket proxy, device pairing, user auth.
"""

from __future__ import annotations

import os
import uuid
import asyncio
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import get_pool, close_pool, bootstrap_schema
from .auth import hash_password, verify_password, create_token, decode_token

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orchestrator")

app = FastAPI(title="Aether Orchestrator", version="0.1.0")

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
    log.info("Orchestrator ready")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


# ── Models ─────────────────────────────────────────────────


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class AgentRegisterRequest(BaseModel):
    agent_id: str
    host: str
    port: int
    container_id: str | None = None


class PairRequestBody(BaseModel):
    code: str
    device_type: str = "ios"
    device_name: str = ""


class PairConfirmBody(BaseModel):
    code: str


class ApiKeyBody(BaseModel):
    provider: str
    key_value: str


# ── Auth ───────────────────────────────────────────────────


async def _auto_assign_agent(user_id: str):
    """Dev convenience: if user has no agent, assign an available one."""
    pool = await get_pool()
    # Already has an agent?
    existing = await pool.fetchrow(
        "SELECT id FROM agents WHERE user_id = $1 AND status = 'running'", user_id
    )
    if existing:
        return
    # Find an unassigned running agent
    agent = await pool.fetchrow(
        "SELECT id FROM agents WHERE user_id IS NULL AND status = 'running' LIMIT 1"
    )
    if agent:
        await pool.execute(
            "UPDATE agents SET user_id = $1 WHERE id = $2", user_id, agent["id"]
        )
        log.info(f"Auto-assigned agent {agent['id']} to user {user_id}")
    else:
        # All agents are assigned — reassign any running agent (single-agent dev mode)
        agent = await pool.fetchrow(
            "SELECT id FROM agents WHERE status = 'running' LIMIT 1"
        )
        if agent:
            await pool.execute(
                "UPDATE agents SET user_id = $1 WHERE id = $2", user_id, agent["id"]
            )
            log.info(f"Reassigned agent {agent['id']} to user {user_id} (dev mode)")


@app.post("/auth/signup")
async def signup(body: SignupRequest):
    pool = await get_pool()
    user_id = uuid.uuid4().hex[:12]
    hashed = hash_password(body.password)
    try:
        await pool.execute(
            "INSERT INTO users (id, email, name, password) VALUES ($1, $2, $3, $4)",
            user_id,
            body.email.lower(),
            body.name,
            hashed,
        )
    except Exception:
        raise HTTPException(400, "Email already registered")
    token = create_token(user_id)
    await _auto_assign_agent(user_id)
    return {"user_id": user_id, "token": token}


@app.post("/auth/login")
async def login(body: LoginRequest):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, password FROM users WHERE email = $1", body.email.lower()
    )
    if not row or not verify_password(body.password, row["password"]):
        raise HTTPException(401, "Invalid credentials")
    token = create_token(row["id"])
    await _auto_assign_agent(row["id"])
    return {"user_id": row["id"], "token": token}


@app.get("/auth/me")
async def me(token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, email, name, created_at FROM users WHERE id = $1", payload["sub"]
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
        INSERT INTO agents (id, host, port, container_id, status, registered_at, last_health)
        VALUES ($1, $2, $3, $4, 'running', now(), now())
        ON CONFLICT (id) DO UPDATE SET
            host = $2, port = $3, container_id = $4,
            status = 'running', last_health = now()
    """,
        body.agent_id,
        body.host,
        body.port,
        body.container_id,
    )
    log.info(f"Agent registered: {body.agent_id} at {body.host}:{body.port}")
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
async def pair_confirm(body: PairConfirmBody, token: str = Query(...)):
    """Dashboard calls this when user enters the code."""
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    user_id = payload["sub"]
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

    # Create device + token
    device_id = uuid.uuid4().hex[:12]
    device_token = create_token(user_id, device_id=device_id)
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
async def list_devices(token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, name, device_type, paired_at, last_seen FROM devices WHERE user_id = $1 ORDER BY paired_at DESC",
        payload["sub"],
    )
    return [dict(r) for r in rows]


# ── API Key Management ─────────────────────────────────────


@app.post("/services/keys")
async def store_api_key(body: ApiKeyBody, token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    pool = await get_pool()
    key_id = uuid.uuid4().hex[:12]
    await pool.execute(
        """
        INSERT INTO api_keys (id, user_id, provider, key_value)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, provider) DO UPDATE SET key_value = $4
    """,
        key_id,
        payload["sub"],
        body.provider,
        body.key_value,
    )
    return {"status": "saved"}


@app.get("/services/keys")
async def list_api_keys(token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT provider, substring(key_value, 1, 8) || '...' as preview FROM api_keys WHERE user_id = $1",
        payload["sub"],
    )
    return [dict(r) for r in rows]


@app.delete("/services/keys/{provider}")
async def delete_api_key(provider: str, token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    pool = await get_pool()
    await pool.execute(
        "DELETE FROM api_keys WHERE user_id = $1 AND provider = $2",
        payload["sub"],
        provider,
    )
    return {"status": "deleted"}


# ── Memory Proxy ───────────────────────────────────────────


async def _get_agent_for_user(user_id: str) -> dict | None:
    """Look up which agent this user is assigned to."""
    pool = await get_pool()
    return await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        user_id,
    )


@app.get("/memory/facts")
async def proxy_memory_facts(token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    agent = await _get_agent_for_user(payload["sub"])
    if not agent:
        raise HTTPException(404, "No agent assigned")
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"http://{agent['host']}:{agent['port']}/memory/facts")
        return resp.json()


@app.get("/memory/sessions")
async def proxy_memory_sessions(token: str = Query(...)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    agent = await _get_agent_for_user(payload["sub"])
    if not agent:
        raise HTTPException(404, "No agent assigned")
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"http://{agent['host']}:{agent['port']}/memory/sessions"
        )
        return resp.json()


@app.get("/memory/conversations")
async def proxy_memory_conversations(token: str = Query(...), limit: int = 20):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    agent = await _get_agent_for_user(payload["sub"])
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
async def ws_proxy(ws: WebSocket, token: str = Query(...)):
    """
    Proxy WebSocket between client and the user's Aether agent.

    Client connects here → orchestrator validates token → finds agent → proxies bidirectionally.
    """
    payload = decode_token(token)
    if not payload:
        await ws.close(code=4001, reason="Invalid token")
        return

    user_id = payload["sub"]

    # Find user's agent
    pool = await get_pool()
    agent = await pool.fetchrow(
        "SELECT host, port FROM agents WHERE user_id = $1 AND status = 'running'",
        user_id,
    )
    if not agent:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "No agent available"})
        await ws.close(code=4002, reason="No agent")
        return

    agent_url = f"ws://{agent['host']}:{agent['port']}/ws"
    log.info(f"Proxying WS: user={user_id} → {agent_url}")

    await ws.accept()

    # Connect to the agent
    import websockets as ws_lib

    try:
        async with ws_lib.connect(agent_url) as agent_ws:
            # Bidirectional proxy
            async def client_to_agent():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("text"):
                            await agent_ws.send(msg["text"])
                        elif msg.get("bytes"):
                            await agent_ws.send(msg["bytes"])
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            async def agent_to_client():
                try:
                    async for msg in agent_ws:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)
                except Exception:
                    pass

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_agent()),
                    asyncio.create_task(agent_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as e:
        log.error(f"WS proxy error: {e}")
        try:
            await ws.send_json({"type": "error", "message": "Agent connection failed"})
            await ws.close()
        except Exception:
            pass


# ── Health ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}
