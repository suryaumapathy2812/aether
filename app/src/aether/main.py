"""
Aether v0.08 — Transport Layer.

Changes from v0.07:
- Transport layer: modular connection handling (WebSocket, future WebRTC/Push)
- KernelCore: adapter between transport (CoreMsg) and core (Frames)
- TransportManager: facade routing messages between transports and core
- Old monolithic websocket_endpoint removed — /ws delegates to WebSocketTransport
- Plugin events routed through transport_manager.broadcast()
- Providers started once via KernelCore.start() (no double startup)

Run: uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import aether.core.config as _config_mod
from aether.core.config import config, reload_config

# FrameType removed — /chat now routes through scheduler
from aether.core.logging import setup_logging
from aether.core.metrics import metrics
from aether.memory.store import MemoryStore
from aether.providers import get_stt_provider, get_llm_provider, get_tts_provider
from aether.tools.registry import ToolRegistry
from aether.tools.read_file import ReadFileTool
from aether.tools.write_file import WriteFileTool
from aether.tools.list_directory import ListDirectoryTool
from aether.tools.run_command import RunCommandTool
from aether.tools.web_search import WebSearchTool
from aether.tools.run_task import RunTaskTool
from aether.skills.loader import Skill, SkillLoader
from aether.agents.task_runner import TaskRunner
from aether.plugins.loader import PluginLoader
from aether.plugins.context import PluginContextStore
from aether.plugins.event import PluginEvent
from aether.processors.event import EventProcessor

# --- Transport Layer ---
from aether.kernel.core import KernelCore
from aether.transport import CoreMsg, TransportManager, WebSocketTransport

# Optional: WebRTC transport (requires aiortc)
SmallWebRTCTransport = None
try:
    from aether.transport.webrtc import SmallWebRTCTransport as _WebRTCTransport
    from aether.transport.webrtc import AIORTC_AVAILABLE

    SmallWebRTCTransport = _WebRTCTransport
except ImportError:
    AIORTC_AVAILABLE = False

# --- Setup ---
setup_logging()
logger = logging.getLogger("aether")

# --- Orchestrator registration ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
AGENT_ID = os.getenv("AETHER_AGENT_ID", f"agent-{socket.gethostname()}")
AGENT_USER_ID = os.getenv("AETHER_USER_ID", "")
AGENT_SECRET = os.getenv("AGENT_SECRET", "")

# --- App ---
app = FastAPI(title="Aether", version="0.0.8")

# Static files: serve from client/web/ (new structure) or fallback to legacy static/
CLIENT_WEB_DIR = Path(__file__).parent.parent.parent.parent.parent / "client" / "web"
STATIC_DIR = (
    CLIENT_WEB_DIR if CLIENT_WEB_DIR.exists() else Path(__file__).parent / "static"
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Shared state (created once, shared across connections) ---
memory_store = MemoryStore()
stt_provider = get_stt_provider()
llm_provider = get_llm_provider()
tts_provider = get_tts_provider()

# --- Tool Registry ---
WORKING_DIR = config.server.working_dir

tool_registry = ToolRegistry()
tool_registry.register(ReadFileTool(working_dir=WORKING_DIR))
tool_registry.register(WriteFileTool(working_dir=WORKING_DIR))
tool_registry.register(ListDirectoryTool(working_dir=WORKING_DIR))
tool_registry.register(RunCommandTool(working_dir=WORKING_DIR))
tool_registry.register(WebSearchTool())

# --- Background Task Runner ---
task_runner = TaskRunner(provider=llm_provider, tool_registry=tool_registry)
tool_registry.register(RunTaskTool(task_runner))

# --- Skill Loader ---
APP_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIRS = [
    str(APP_ROOT / "skills"),
    str(APP_ROOT / ".agents" / "skills"),
]
skill_loader = SkillLoader(skills_dirs=SKILLS_DIRS)
skill_loader.discover()

# --- Plugin Loader ---
PLUGINS_DIR = str(APP_ROOT / "plugins")
plugin_loader = PluginLoader(PLUGINS_DIR)
loaded_plugins = plugin_loader.discover()
for plugin in loaded_plugins:
    for tool in plugin.tools:
        tool_registry.register(tool, plugin_name=plugin.manifest.name)

# --- Plugin Context Store ---
plugin_context_store = PluginContextStore()

# --- Inject plugin SKILL.md into skill_loader ---
for plugin in loaded_plugins:
    if plugin.skill_content:
        skill = Skill(
            name=plugin.manifest.name,
            description=plugin.manifest.description or plugin.manifest.display_name,
            location=f"{plugin.manifest.location}/SKILL.md",
            _content=plugin.skill_content,
        )
        skill_loader.register(skill)

# --- Event Processor (decision engine for plugin events) ---
event_processor = EventProcessor(llm_provider, memory_store)

# --- Transport Manager (initialized at startup) ---
transport_manager: TransportManager | None = None


# ── Orchestrator helpers ───────────────────────────────────────


def _agent_auth_headers() -> dict[str, str]:
    if AGENT_SECRET:
        return {"Authorization": f"Bearer {AGENT_SECRET}"}
    return {}


async def _register_with_orchestrator():
    if not ORCHESTRATOR_URL:
        logger.info("No ORCHESTRATOR_URL — skipping registration")
        return
    if not AGENT_SECRET:
        logger.warning("AGENT_SECRET not set — registration calls are unauthenticated")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ORCHESTRATOR_URL}/api/agents/register",
                json={
                    "agent_id": AGENT_ID,
                    "host": os.getenv("AETHER_AGENT_HOST", socket.gethostname()),
                    "port": config.server.port,
                    "container_id": os.getenv("HOSTNAME", ""),
                    "user_id": AGENT_USER_ID or None,
                },
                headers=_agent_auth_headers(),
            )
            logger.info(f"Registered with orchestrator: {resp.status_code}")
    except Exception as e:
        logger.error(f"Failed to register with orchestrator: {e}")


async def _heartbeat_loop():
    import httpx

    while True:
        await asyncio.sleep(30)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"{ORCHESTRATOR_URL}/api/agents/{AGENT_ID}/heartbeat",
                    headers=_agent_auth_headers(),
                )
        except Exception:
            logger.warning("Heartbeat to orchestrator failed")


async def _fetch_plugin_configs():
    if not ORCHESTRATOR_URL or not AGENT_USER_ID:
        logger.debug(
            "No ORCHESTRATOR_URL or AGENT_USER_ID — skipping plugin config fetch"
        )
        return

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORCHESTRATOR_URL}/api/internal/plugins",
                params={"user_id": AGENT_USER_ID},
                headers=_agent_auth_headers(),
            )
            if resp.status_code != 200:
                logger.debug(f"No enabled plugins (status={resp.status_code})")
                return

            enabled = resp.json().get("plugins", [])
            for plugin_name in enabled:
                cfg_resp = await client.get(
                    f"{ORCHESTRATOR_URL}/api/internal/plugins/{plugin_name}/config",
                    params={"user_id": AGENT_USER_ID},
                    headers=_agent_auth_headers(),
                )
                if cfg_resp.status_code == 200:
                    plugin_context_store.set(plugin_name, cfg_resp.json())
                else:
                    logger.debug(
                        f"No config for plugin {plugin_name} (status={cfg_resp.status_code})"
                    )
    except Exception as e:
        logger.warning(f"Failed to fetch plugin configs: {e}")


# ── Startup / Shutdown ─────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global transport_manager

    # Register with orchestrator and start heartbeat
    await _register_with_orchestrator()
    if ORCHESTRATOR_URL:
        asyncio.create_task(_heartbeat_loop())

    # Fetch plugin configs
    await _fetch_plugin_configs()

    # Build the transport layer
    #   KernelCore starts providers (STT, LLM, TTS, Memory) via start_all()
    #   No separate provider.start() calls needed.
    core = KernelCore(
        llm_provider=llm_provider,
        memory_store=memory_store,
        tool_registry=tool_registry,
        skill_loader=skill_loader,
        plugin_context=plugin_context_store,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
    )

    transport_manager = TransportManager(core=core)

    # Register transports
    ws_transport = WebSocketTransport()
    await transport_manager.register_transport(ws_transport)

    # Optional: WebRTC transport (self-hosted via aiortc)
    if AIORTC_AVAILABLE and SmallWebRTCTransport is not None:
        try:
            ice_servers = []
            stun_url = os.getenv("WEBRTC_STUN_URL", "stun:stun.l.google.com:19302")
            if stun_url:
                ice_servers.append({"urls": stun_url})

            turn_url = os.getenv("WEBRTC_TURN_URL", "")
            turn_username = os.getenv("WEBRTC_TURN_USERNAME", "")
            turn_credential = os.getenv("WEBRTC_TURN_CREDENTIAL", "")
            if turn_url:
                ice_servers.append(
                    {
                        "urls": turn_url,
                        "username": turn_username,
                        "credential": turn_credential,
                    }
                )

            webrtc_transport = SmallWebRTCTransport(
                ice_servers=ice_servers or None,
            )
            await transport_manager.register_transport(webrtc_transport)
            logger.info("WebRTC transport registered (aiortc)")
        except Exception as e:
            logger.warning(f"Failed to register WebRTC transport: {e}")
    else:
        logger.info("WebRTC transport not available (aiortc not installed)")

    # Start everything (core + transports)
    await transport_manager.start_all()

    # Inject scheduler reference into WebSocket transport for disconnect cancellation
    scheduler_ref = getattr(core, "_scheduler", None)
    if scheduler_ref is not None and ws_transport is not None:
        ws_transport._scheduler = scheduler_ref
        logger.info(
            "Scheduler wired into WebSocket transport for disconnect cancellation"
        )

    logger.info(
        "Aether v0.09 ready (id=%s, providers: STT=%s, LLM=%s, TTS=%s, tools=%s, skills=%s, plugin_ctx=%s)",
        AGENT_ID,
        config.stt.provider,
        config.llm.provider,
        config.tts.provider,
        tool_registry.tool_names(),
        [s.name for s in skill_loader.all()],
        plugin_context_store.loaded_plugins(),
    )


@app.on_event("shutdown")
async def shutdown():
    global transport_manager
    if transport_manager:
        await transport_manager.stop_all()
        transport_manager = None


# ── WebSocket endpoint — delegates to transport layer ──────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, user_id: str = ""):
    """
    WebSocket endpoint — thin delegate to the transport layer.

    The WebSocketTransport handles:
    - Connection accept/close
    - Protocol parsing (JSON messages)
    - Normalizing to CoreMsg
    - Routing through TransportManager → KernelCore
    - Serializing responses back to the client
    """
    if not transport_manager:
        await ws.close(code=1013, reason="Server not ready")
        return

    ws_transport: WebSocketTransport | None = transport_manager.get_transport(
        "websocket"
    )  # type: ignore[assignment]
    if not ws_transport:
        await ws.close(code=1013, reason="WebSocket transport not available")
        return

    await ws_transport.handle_connection(ws, user_id=user_id)


# ── WebRTC signaling endpoints ─────────────────────────────────


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request):
    """
    WebRTC signaling: receive SDP offer, return SDP answer.

    Client sends:
        {"sdp": "...", "type": "offer", "user_id": "...", "pc_id": "..."}

    Server returns:
        {"sdp": "...", "type": "answer", "pc_id": "pc-xxxx"}
    """
    if not transport_manager:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    webrtc = transport_manager.get_transport("webrtc")
    if not webrtc:
        return JSONResponse(
            {"error": "WebRTC transport not available"}, status_code=404
        )

    body = await request.json()
    sdp = body.get("sdp", "")
    sdp_type = body.get("type", "offer")
    user_id = body.get("user_id", "")
    pc_id = body.get("pc_id")

    if not sdp:
        return JSONResponse({"error": "Missing SDP"}, status_code=400)

    try:
        answer = await webrtc.handle_offer(  # type: ignore[union-attr]
            sdp=sdp,
            sdp_type=sdp_type,
            user_id=user_id,
            pc_id=pc_id,
        )
        return JSONResponse(answer)
    except Exception as e:
        logger.error(f"WebRTC offer error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.patch("/webrtc/ice")
async def webrtc_ice(request: Request):
    """
    WebRTC signaling: add ICE candidates.

    Client sends:
        {
            "pc_id": "pc-xxxx",
            "candidates": [
                {"candidate": "...", "sdpMid": "0", "sdpMLineIndex": 0}
            ]
        }
    """
    if not transport_manager:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    webrtc = transport_manager.get_transport("webrtc")
    if not webrtc:
        return JSONResponse(
            {"error": "WebRTC transport not available"}, status_code=404
        )

    body = await request.json()
    pc_id = body.get("pc_id", "")
    candidates = body.get("candidates", [])

    if not pc_id:
        return JSONResponse({"error": "Missing pc_id"}, status_code=400)

    try:
        for c in candidates:
            await webrtc.handle_ice_candidate(  # type: ignore[union-attr]
                pc_id=pc_id,
                candidate=c.get("candidate", ""),
                sdp_mid=c.get("sdpMid"),
                sdp_mline_index=c.get("sdpMLineIndex"),
            )
        return JSONResponse({"status": "ok"})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        logger.error(f"WebRTC ICE error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ── HTTP Chat Streaming (Vercel AI SDK compatible) ─────────────


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    HTTP streaming chat endpoint for dashboard text chat.

    Accepts AI SDK UI messages ("parts") and legacy messages ("content").
    Streams plain text chunks via the kernel scheduler.
    """
    if not transport_manager:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    body = await request.json()
    incoming_messages = body.get("messages", [])
    user_id = body.get("user_id", "")

    if not incoming_messages:
        return JSONResponse({"error": "No messages"}, status_code=400)

    # Extract the latest user message.
    # AI SDK format: {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
    # Legacy format: {"role": "user", "content": "hello"}
    last_user_msg = None
    for m in reversed(incoming_messages):
        if m.get("role") == "user":
            # Try AI SDK format first (parts array)
            parts = m.get("parts", [])
            if parts:
                last_user_msg = " ".join(
                    p.get("text", "") for p in parts if p.get("type") == "text"
                )
            # Fall back to legacy format (content string)
            else:
                last_user_msg = m.get("content", "")
            break

    if not last_user_msg:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    # Get core and session
    core: KernelCore = transport_manager.core  # type: ignore[assignment]
    session_id = f"http-{user_id or 'anon'}"
    session = core._get_session(session_id, mode="text")
    await core._ensure_session_started(session)

    async def generate():
        """Stream plain text via scheduler."""
        from aether.kernel.contracts import JobPriority, KernelRequest

        try:
            session.turn_count += 1
            request = KernelRequest(
                kind="reply_text",
                modality="text",
                user_id=user_id or "anon",
                session_id=session_id,
                payload={
                    "text": last_user_msg,
                    "history": list(session.conversation_history),
                },
                priority=JobPriority.INTERACTIVE.value,
            )

            job_id = await core._scheduler.submit(request)
            collected_text: list[str] = []

            async for event in core._scheduler.stream(job_id):
                if event.stream_type == "text_chunk":
                    chunk = event.payload.get("text", "")
                    collected_text.append(chunk)
                    yield chunk

            # Update session history
            session.conversation_history.append(
                {"role": "user", "content": last_user_msg}
            )
            assistant_text = " ".join(collected_text).strip()
            if assistant_text:
                session.conversation_history.append(
                    {"role": "assistant", "content": assistant_text}
                )

        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield f"\n[error] {e}\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ── REST endpoints (unchanged) ─────────────────────────────────


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Aether v0.08</h1><p>Client files not found.</p>")


@app.get("/workspace")
@app.get("/workspace/{path:path}")
async def browse_workspace(path: str = ""):
    """Browse the workspace directory."""
    workspace = Path(WORKING_DIR)
    target = workspace / path

    try:
        target.resolve().relative_to(workspace.resolve())
    except ValueError:
        return JSONResponse({"error": "Path outside workspace"}, status_code=403)

    if not target.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    if target.is_file():
        if target.stat().st_size > 100_000:
            return JSONResponse({"error": "File too large"}, status_code=413)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            return JSONResponse(
                {
                    "type": "file",
                    "path": str(path),
                    "content": content,
                    "size": target.stat().st_size,
                }
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    entries = []
    for item in sorted(target.iterdir()):
        entries.append(
            {
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            }
        )

    return JSONResponse(
        {"type": "directory", "path": str(path) or ".", "entries": entries}
    )


@app.get("/health")
async def health():
    """Health check — providers, memory, transport, and metrics snapshot."""
    # Transport status includes core health
    transport_status: dict = {"enabled": False}
    core_health: dict = {}
    if transport_manager:
        try:
            status = await transport_manager.get_status()
            transport_status = {
                "enabled": True,
                "transports": list(status.get("transports", {}).keys()),
                "connections": len(status.get("connections", {})),
            }
            core_health = status.get("core", {})
        except Exception as e:
            transport_status = {"enabled": True, "error": str(e)}

    # Fallback: direct provider health if core didn't report
    providers = core_health.get("providers", {})
    if not providers:
        providers = {
            "stt": await stt_provider.health_check(),
            "llm": await llm_provider.health_check(),
            "tts": await tts_provider.health_check(),
        }

    facts = await memory_store.get_facts()
    recent = await memory_store.get_recent(limit=1)

    return JSONResponse(
        {
            "status": "ok",
            "version": "0.0.9",
            "providers": providers,
            "memory": {
                "facts_count": len(facts),
                "has_conversations": len(recent) > 0,
            },
            "tools": tool_registry.tool_names(),
            "skills": [s.name for s in skill_loader.all()],
            "transport": transport_status,
            "metrics": metrics.snapshot(),
        }
    )


@app.get("/metrics")
async def metrics_endpoint():
    """Full in-process metrics snapshot (counters, gauges, histogram percentiles)."""
    return JSONResponse(metrics.snapshot())


@app.get("/metrics/latency")
async def latency_metrics():
    """SLO-focused latency metrics — p50/p95 for all critical paths.

    Compare against SLO thresholds from REFACTOR.md:
        /chat TTFT p95      ≤ 1800ms  (fail > 2000ms)
        Voice TTFT p95      ≤ 2200ms  (fail > 2500ms)
        Voice audio p95     ≤  900ms  (fail > 1100ms)
        Tool execution p95  ≤ +15% baseline
        Notification p95    ≤ 1200ms  (fail > 1500ms)
    """
    return JSONResponse(
        {
            "chat": {
                "ttft_p50_ms": metrics.percentile(
                    "llm.ttft_ms", 50, labels={"kind": "reply_text"}
                ),
                "ttft_p95_ms": metrics.percentile(
                    "llm.ttft_ms", 95, labels={"kind": "reply_text"}
                ),
            },
            "voice": {
                "ttft_p50_ms": metrics.percentile(
                    "llm.ttft_ms", 50, labels={"kind": "reply_voice"}
                ),
                "ttft_p95_ms": metrics.percentile(
                    "llm.ttft_ms", 95, labels={"kind": "reply_voice"}
                ),
                "tts_p50_ms": metrics.percentile("provider.tts.latency_ms", 50),
                "tts_p95_ms": metrics.percentile("provider.tts.latency_ms", 95),
            },
            "kernel": {
                "job_duration_p50_ms": metrics.percentile("kernel.job.duration_ms", 50),
                "job_duration_p95_ms": metrics.percentile("kernel.job.duration_ms", 95),
                "enqueue_delay_p95_ms": metrics.percentile(
                    "kernel.enqueue_delay_ms", 95
                ),
            },
            "services": {
                "notification_decision_p95_ms": metrics.percentile(
                    "service.notification.decision_ms", 95
                ),
                "tool_execution_p95_ms": metrics.percentile(
                    "service.tool.duration_ms", 95
                ),
                "memory_extraction_p95_ms": metrics.percentile(
                    "service.memory.extraction_ms", 95
                ),
            },
        }
    )


@app.get("/memory/facts")
async def memory_facts():
    facts = await memory_store.get_facts()
    return JSONResponse({"facts": facts})


@app.get("/memory/sessions")
async def memory_sessions():
    sessions = await memory_store.get_session_summaries()
    return JSONResponse({"sessions": sessions})


@app.get("/memory/conversations")
async def memory_conversations(limit: int = 20):
    conversations = await memory_store.get_recent(limit=limit)
    return JSONResponse({"conversations": conversations})


# ── Config endpoints ───────────────────────────────────────────


@app.get("/config")
async def get_config():
    cfg = _config_mod.config
    return JSONResponse(
        {
            "stt": {
                "provider": cfg.stt.provider,
                "model": cfg.stt.model,
                "language": cfg.stt.language,
            },
            "llm": {
                "provider": cfg.llm.provider,
                "model": cfg.llm.model,
            },
            "tts": {
                "provider": cfg.tts.provider,
                "model": cfg.tts.model,
                "voice": cfg.tts.voice,
            },
            "personality": {
                "base_style": cfg.personality.base_style,
                "custom_instructions": cfg.personality.custom_instructions,
            },
        }
    )


@app.post("/config/reload")
async def config_reload(request_body: dict | None = None):
    if request_body:
        for key, value in request_body.items():
            os.environ[key] = str(value)

    new_config = reload_config()
    await _fetch_plugin_configs()

    logger.info(
        "Config reloaded (STT=%s/%s, LLM=%s/%s, TTS=%s/%s/%s, style=%s, plugin_ctx=%s)",
        new_config.stt.provider,
        new_config.stt.model,
        new_config.llm.provider,
        new_config.llm.model,
        new_config.tts.provider,
        new_config.tts.model,
        new_config.tts.voice,
        new_config.personality.base_style,
        plugin_context_store.loaded_plugins(),
    )
    return JSONResponse({"status": "reloaded"})


@app.get("/plugins")
async def list_plugins_endpoint():
    return JSONResponse(
        {
            "plugins": [
                {
                    "name": p.manifest.name,
                    "display_name": p.manifest.display_name,
                    "description": p.manifest.description,
                    "tools": [t.name for t in p.tools],
                    "has_skill": bool(p.skill_content),
                }
                for p in loaded_plugins
            ]
        }
    )


# ── Plugin event endpoints — routed through transport layer ────


@app.post("/plugin_event")
async def receive_plugin_event(request: Request):
    """
    Receive a plugin event from the orchestrator webhook receiver.
    Runs through EventProcessor → broadcasts to connected clients via transport layer.
    """
    body = await request.json()

    event = PluginEvent(
        id=body.get("event_id", str(uuid.uuid4())),
        plugin=body.get("plugin", "unknown"),
        event_type=body.get("event_type", "unknown"),
        source_id=body.get("source_id", ""),
        timestamp=time.time(),
        summary=body.get("summary", ""),
        content=body.get("payload", {}).get("content", ""),
        sender=body.get("payload", {}).get("sender", {}),
        urgency=body.get("payload", {}).get("urgency", "medium"),
        category=body.get("payload", {}).get("category", ""),
        requires_action=body.get("payload", {}).get("requires_action", False),
        available_actions=body.get("payload", {}).get("available_actions", []),
        metadata=body.get("payload", {}).get("metadata", {}),
    )

    decision = await event_processor.process(event)

    # Broadcast notification to connected clients via transport layer
    if decision.action in ("surface", "action_required") and transport_manager:
        level = "speak" if decision.action == "action_required" else "nudge"
        notification = CoreMsg.event(
            event_type="notification",
            user_id="",  # broadcast — no specific user
            session_id="",
            payload={
                "event_id": event.id,
                "plugin": event.plugin,
                "level": level,
                "text": decision.notification,
                "actions": event.available_actions,
            },
            transport="notification",
        )
        await transport_manager.broadcast(notification)

    # Report decision back to orchestrator
    if ORCHESTRATOR_URL and body.get("event_id"):
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"{ORCHESTRATOR_URL}/api/internal/events/{body['event_id']}/decision",
                    json={
                        "decision": decision.action,
                        "notification": decision.notification,
                    },
                    headers=_agent_auth_headers(),
                )
        except Exception as e:
            logger.warning(f"Decision callback failed: {e}")

    return JSONResponse(
        {"action": decision.action, "notification": decision.notification}
    )


@app.post("/plugin_event/batch")
async def receive_batch_notification(request: Request):
    """Receive a batch of deferred plugin events for flushing."""
    body = await request.json()
    items = body.get("events", [])
    if not items:
        return JSONResponse({"status": "empty"})

    count = len(items)
    summaries = [e.get("notification", e.get("summary", "")) for e in items[:5]]
    batch_text = (
        f"While you were away, {count} thing{'s' if count > 1 else ''} happened. "
    )
    batch_text += " ".join(summaries)

    if transport_manager:
        notification = CoreMsg.event(
            event_type="notification",
            user_id="",
            session_id="",
            payload={
                "level": "batch",
                "text": batch_text,
                "items": items,
            },
            transport="notification",
        )
        await transport_manager.broadcast(notification)

    return JSONResponse({"status": "delivered", "count": count})
