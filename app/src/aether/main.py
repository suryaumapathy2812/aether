"""
Aether v0.10 — Clean Transport Architecture.

Three independent transports, one AgentCore facade:
  1. OpenAI-compatible HTTP API (/v1/chat/completions) — text chat
  2. Simplified WebRTC Voice (/webrtc/offer) — per-session STT, direct audio
  3. WS Notification Sidecar (/ws) — push-only for dashboard

All transports call AgentCore. AgentCore wraps the KernelScheduler.
No CoreMsg, no TransportManager, no KernelCore.

Run: uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import aether.core.config as _config_mod
from aether.core.config import reload_config
from aether.core.logging import setup_logging
from aether.core.metrics import metrics
from aether.memory.store import MemoryStore
from aether.providers import get_llm_provider, get_stt_provider, get_tts_provider
from aether.tools.registry import ToolRegistry
from aether.tools.read_file import ReadFileTool
from aether.tools.write_file import WriteFileTool
from aether.tools.list_directory import ListDirectoryTool
from aether.tools.run_command import RunCommandTool
from aether.tools.web_search import WebSearchTool
from aether.tools.world_time import WorldTimeTool
from aether.tools.schedule_reminder import ScheduleReminderTool
from aether.tools.run_task import RunTaskTool
from aether.tools.save_memory import SaveMemoryTool
from aether.tools.search_memory import SearchMemoryTool
from aether.skills.loader import Skill, SkillLoader
from aether.agents.task_runner import TaskRunner
from aether.plugins.loader import PluginLoader
from aether.plugins.context import PluginContextStore
from aether.plugins.event import PluginEvent
from aether.processors.event import EventProcessor

# --- New architecture ---
from aether.agent import AgentCore
from aether.kernel.scheduler import KernelScheduler, ServiceRouter
from aether.services.reply_service import ReplyService
from aether.services.memory_service import MemoryService
from aether.services.notification_service import NotificationService
from aether.services.tool_service import ToolService
from aether.llm.core import LLMCore
from aether.llm.context_builder import ContextBuilder
from aether.tools.orchestrator import ToolOrchestrator
from aether.http.openai_compat import create_router as create_openai_router
from aether.http.sessions import create_session_router
from aether.kernel.event_bus import EventBus
from aether.session.store import SessionStore
from aether.agents.manager import SubAgentManager
from aether.tools.spawn_task import SpawnTaskTool
from aether.tools.check_task import CheckTaskTool
from aether.ws.sidecar import WSSidecar

# Optional: WebRTC voice transport (requires aiortc)
WebRTCVoiceTransport = None
try:
    from aether.voice.webrtc import WebRTCVoiceTransport as _WebRTCVoiceTransport
    from aether.voice.webrtc import AIORTC_AVAILABLE

    WebRTCVoiceTransport = _WebRTCVoiceTransport
except ImportError:
    AIORTC_AVAILABLE = False

# Optional: Telephony transport
TelephonyTransport = None
try:
    from aether.voice.telephony import TelephonyTransport as _TelephonyTransport

    TelephonyTransport = _TelephonyTransport
except ImportError:
    pass

# --- Setup ---
setup_logging()
logger = logging.getLogger("aether")

# --- Orchestrator registration ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
AGENT_ID = os.getenv("AETHER_AGENT_ID", f"agent-{socket.gethostname()}")
AGENT_USER_ID = os.getenv("AETHER_USER_ID", "")
AGENT_SECRET = os.getenv("AGENT_SECRET", "")

# --- App ---
app = FastAPI(title="Aether", version="0.10.0")

# Static files: serve from client/web/ (new structure) or fallback to legacy static/
# __file__ = app/src/aether/main.py → .parent×4 = core-ai/ (repo root)
CLIENT_WEB_DIR = Path(__file__).parent.parent.parent.parent / "client" / "web"
STATIC_DIR = (
    CLIENT_WEB_DIR if CLIENT_WEB_DIR.exists() else Path(__file__).parent / "static"
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Shared state (created once, shared across connections) ---
memory_store = MemoryStore()
session_store = SessionStore()
event_bus = EventBus()
stt_provider = get_stt_provider()
llm_provider = get_llm_provider()
tts_provider = get_tts_provider()

# --- Tool Registry ---
WORKING_DIR = _config_mod.config.server.working_dir

tool_registry = ToolRegistry()
tool_registry.register(ReadFileTool(working_dir=WORKING_DIR))
tool_registry.register(WriteFileTool(working_dir=WORKING_DIR))
tool_registry.register(ListDirectoryTool(working_dir=WORKING_DIR))
tool_registry.register(RunCommandTool(working_dir=WORKING_DIR))
tool_registry.register(WebSearchTool())
tool_registry.register(WorldTimeTool())
tool_registry.register(ScheduleReminderTool())
tool_registry.register(SaveMemoryTool(memory_store=memory_store))
tool_registry.register(SearchMemoryTool(memory_store=memory_store))

# --- Background Task Runner ---
# TaskRunner is created after SubAgentManager (below) since it now delegates to it.
# The RunTaskTool registration is also deferred.

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
            plugin_name=plugin.manifest.name,
        )
        skill_loader.register(skill)

# --- Register plugin routes (for telephony plugins) ---
for plugin in loaded_plugins:
    if plugin.router_factory and plugin.manifest.plugin_type == "telephony":
        try:
            router = plugin.router_factory()
            app.include_router(router)
            logger.info(
                f"Registered routes for telephony plugin: {plugin.manifest.name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to register routes for plugin {plugin.manifest.name}: {e}"
            )

# --- Event Processor (decision engine for plugin events) ---
event_processor = EventProcessor(llm_provider, memory_store)

# --- Build the core stack ---
# LLMCore + ContextBuilder + ToolOrchestrator
tool_orchestrator = ToolOrchestrator(tool_registry)
llm_core = LLMCore(llm_provider, tool_orchestrator)
context_builder = ContextBuilder(
    memory_store=memory_store,
    tool_registry=tool_registry,
    skill_loader=skill_loader,
    plugin_context_store=plugin_context_store,
)

# Services
reply_service = ReplyService(llm_core, context_builder, memory_store)
memory_service = MemoryService(llm_core, memory_store)
notification_service = NotificationService(llm_core, memory_store)
tool_service = ToolService(tool_orchestrator)

# ServiceRouter → KernelScheduler
service_router = ServiceRouter(
    reply_service=reply_service,
    memory_service=memory_service,
    notification_service=notification_service,
    tool_service=tool_service,
)

kernel_cfg = _config_mod.config.kernel
scheduler = KernelScheduler(
    service_router=service_router,
    max_interactive_workers=kernel_cfg.workers_interactive,
    max_background_workers=kernel_cfg.workers_background,
    interactive_queue_limit=kernel_cfg.interactive_queue_limit,
    background_queue_limit=kernel_cfg.background_queue_limit,
)

# AgentCore — the single facade for all transports
agent_core = AgentCore(
    scheduler=scheduler,
    memory_store=memory_store,
    llm_provider=llm_provider,
    tool_registry=tool_registry,
    skill_loader=skill_loader,
    plugin_context=plugin_context_store,
    session_store=session_store,
    event_bus=event_bus,
    llm_core=llm_core,
    context_builder=context_builder,
)

# --- Sub-Agent Manager ---
sub_agent_manager = SubAgentManager(
    session_store=session_store,
    llm_core=llm_core,
    context_builder=context_builder,
    event_bus=event_bus,
)

# --- Background Task Runner (delegates to SubAgentManager) ---
task_runner = TaskRunner(sub_agent_manager=sub_agent_manager)
tool_registry.register(RunTaskTool(task_runner))

# Register spawn_task and check_task tools
tool_registry.register(SpawnTaskTool(sub_agent_manager))
tool_registry.register(CheckTaskTool(sub_agent_manager))

# --- Mount OpenAI-compatible HTTP API ---
openai_router = create_openai_router(agent_core)
app.include_router(openai_router)

# --- Mount Session API ---
session_router = create_session_router(
    agent=agent_core,
    session_store=session_store,
    event_bus=event_bus,
    sub_agent_manager=sub_agent_manager,
)
app.include_router(session_router)

# --- WS Notification Sidecar ---
ws_sidecar = WSSidecar(agent=agent_core, event_bus=event_bus)

# --- WebRTC Voice Transport (optional) ---
webrtc_transport = None
if AIORTC_AVAILABLE and WebRTCVoiceTransport is not None:
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

        webrtc_transport = WebRTCVoiceTransport(
            agent=agent_core,
            tts_provider=tts_provider,
            ice_servers=ice_servers or None,
        )

        async def _on_sessions_empty() -> None:
            await _set_keep_alive(False)

        webrtc_transport.on_sessions_empty = _on_sessions_empty
        # Wire transport reference for spoken notification delivery
        agent_core._voice_transport = webrtc_transport
        logger.info("WebRTC voice transport initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize WebRTC transport: {e}")
else:
    logger.info("WebRTC transport not available (aiortc not installed)")

# --- Telephony Transport (plugin-based) ---
telephony_transport = None
# Telephony is now handled by plugins (e.g., Vobiz)
# The plugin loader will initialize telephony transport when configured


# ── Orchestrator helpers ───────────────────────────────────────


def _agent_auth_headers() -> dict[str, str]:
    if AGENT_SECRET:
        return {"Authorization": f"Bearer {AGENT_SECRET}"}
    return {}


def _detect_agent_host() -> str:
    """Determine the host address the orchestrator should use to reach this agent."""
    explicit = os.getenv("AETHER_AGENT_HOST", "")
    if explicit and explicit != "host.docker.internal":
        return explicit

    try:
        import subprocess

        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                if not ip.startswith("127.") and ":" not in ip:
                    logger.info(f"Auto-detected agent host: {ip}")
                    return ip
    except Exception:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            detected = s.getsockname()[0]
        if detected and not detected.startswith("127."):
            logger.info(f"Auto-detected agent host (UDP): {detected}")
            return detected
    except Exception:
        pass

    return socket.gethostname()


async def _register_with_orchestrator() -> None:
    if not ORCHESTRATOR_URL:
        logger.info("No ORCHESTRATOR_URL — skipping registration")
        return
    if not AGENT_SECRET:
        logger.warning("AGENT_SECRET not set — registration calls are unauthenticated")

    import httpx

    agent_host = _detect_agent_host()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ORCHESTRATOR_URL}/api/agents/register",
                json={
                    "agent_id": AGENT_ID,
                    "host": agent_host,
                    "port": _config_mod.config.server.port,
                    "container_id": os.getenv("HOSTNAME", ""),
                    "user_id": AGENT_USER_ID or None,
                },
                headers=_agent_auth_headers(),
            )
            logger.info(
                f"Registered with orchestrator: {resp.status_code} (host={agent_host})"
            )
    except Exception as e:
        logger.error(f"Failed to register with orchestrator: {e}")


async def _heartbeat_loop() -> None:
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


async def _set_keep_alive(enabled: bool) -> None:
    """Tell the orchestrator to exempt this agent from idle-kill.

    Called when a WebRTC session becomes active (enabled=True) or when
    all WebRTC sessions are torn down (enabled=False).
    """
    if not ORCHESTRATOR_URL:
        return
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{ORCHESTRATOR_URL}/api/agents/{AGENT_ID}/keep_alive",
                params={"enabled": str(enabled).lower()},
                headers=_agent_auth_headers(),
            )
        logger.info("keep_alive=%s signalled to orchestrator", enabled)
    except Exception as e:
        logger.warning("Failed to set keep_alive=%s: %s", enabled, e)


async def _fetch_plugin_configs() -> None:
    if not ORCHESTRATOR_URL or not AGENT_USER_ID:
        logger.warning(
            "Plugin config fetch skipped: ORCHESTRATOR_URL=%s, AGENT_USER_ID=%s",
            bool(ORCHESTRATOR_URL),
            bool(AGENT_USER_ID),
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
                logger.warning(
                    "Failed to list enabled plugins (status=%d, body=%s)",
                    resp.status_code,
                    resp.text[:200],
                )
                return

            enabled = resp.json().get("plugins", [])
            logger.info("Enabled plugins from orchestrator: %s", enabled)

            for plugin_name in enabled:
                cfg_resp = await client.get(
                    f"{ORCHESTRATOR_URL}/api/internal/plugins/{plugin_name}/config",
                    params={"user_id": AGENT_USER_ID},
                    headers=_agent_auth_headers(),
                )
                if cfg_resp.status_code == 200:
                    config_data = cfg_resp.json()
                    logger.info(
                        "Plugin %s config loaded (keys=%s)",
                        plugin_name,
                        sorted(config_data.keys()),
                    )
                    plugin_context_store.set(plugin_name, config_data)

                    # Initialize telephony transport for telephony plugins
                    if plugin_name == "vobiz":
                        await _init_vobiz_telephony(config_data)
                else:
                    logger.warning(
                        "Failed to fetch config for plugin %s (status=%d, body=%s)",
                        plugin_name,
                        cfg_resp.status_code,
                        cfg_resp.text[:200],
                    )
    except Exception as e:
        logger.warning("Failed to fetch plugin configs: %s", e)


async def _init_vobiz_telephony(config_data: dict) -> None:
    """Initialize telephony transport for Vobiz plugin."""
    global telephony_transport

    if not config_data.get("auth_id") or not config_data.get("auth_token"):
        logger.debug("Vobiz plugin not fully configured — skipping telephony init")
        return

    if TelephonyTransport is None:
        logger.warning("TelephonyTransport not available — cannot init Vobiz")
        return

    try:
        telephony_transport = TelephonyTransport(
            agent=agent_core,
            tts_provider=tts_provider,
        )

        # Use the SAME module instance that PluginLoader registered with FastAPI.
        # Previously this re-imported routes.py, creating a separate module whose
        # globals (set_transport/set_config) were invisible to the live router.
        import sys

        routes_module = sys.modules.get("aether_plugin_vobiz_routes")
        if routes_module:
            set_transport = getattr(routes_module, "set_transport", None)
            set_config = getattr(routes_module, "set_config", None)

            if set_transport and set_config:
                # Resolve the public base URL for VoBiz callbacks.
                # Priority: public_base_url from orchestrator (stored at
                # provision time) → AETHER_BASE_URL env → localhost fallback.
                base_url = (
                    config_data.get("public_base_url")
                    or os.getenv("AETHER_BASE_URL")
                    or f"http://localhost:{_config_mod.config.server.port}"
                )
                config_with_base = {
                    **config_data,
                    "base_url": base_url,
                    "user_id": os.getenv("AETHER_USER_ID", ""),
                }

                set_transport(telephony_transport)
                set_config(config_with_base)

                # Also update plugin context store with base_url
                plugin_context_store.set("vobiz", config_with_base)
        else:
            logger.warning(
                "Vobiz routes module not found in sys.modules — "
                "routes will not receive transport/config"
            )

        logger.info("Vobiz telephony transport initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Vobiz telephony: {e}")


# ── Startup / Shutdown ─────────────────────────────────────────


async def _fetch_user_preferences() -> str:
    """Fetch user preferences from orchestrator and apply to env."""
    if not ORCHESTRATOR_URL:
        logger.warning(
            "No ORCHESTRATOR_URL — skipping preferences fetch; "
            "startup will use env/defaults instead of DB preferences"
        )
        return "env/defaults:no_orchestrator"

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORCHESTRATOR_URL}/api/internal/preferences",
                params={
                    "agent_id": AGENT_ID,
                    "user_id": AGENT_USER_ID or None,
                },
                headers=_agent_auth_headers(),
            )
            if resp.status_code != 200:
                details = resp.text[:300]
                logger.warning(
                    "Failed to fetch preferences (status=%s, user_id=%s, agent_id=%s, details=%s)",
                    resp.status_code,
                    AGENT_USER_ID or "<unset>",
                    AGENT_ID,
                    details,
                )
                return f"env/defaults:http_{resp.status_code}"

            payload = resp.json()
            if isinstance(payload, dict) and isinstance(
                payload.get("preferences"), dict
            ):
                prefs = payload["preferences"]
                meta = (
                    payload.get("meta", {})
                    if isinstance(payload.get("meta"), dict)
                    else {}
                )
                pref_source = str(meta.get("source", "db"))
                resolved_user = str(
                    meta.get("resolved_user_id", AGENT_USER_ID or "<unset>")
                )
            else:
                prefs = payload if isinstance(payload, dict) else {}
                pref_source = "db_legacy"
                resolved_user = AGENT_USER_ID or "<unset>"
            # Map preference keys to env vars.
            # llm_provider and llm_base_url are excluded — OpenRouter is
            # hardcoded in config.py; provider is derived from model name.
            pref_to_env = {
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

            for key, env_name in pref_to_env.items():
                value = prefs.get(key)
                if value:
                    os.environ[env_name] = str(value)
                    logger.debug(f"Applied preference: {env_name}={value}")

            logger.info(
                "User preferences loaded from orchestrator (source=%s, resolved_user_id=%s)",
                pref_source,
                resolved_user,
            )
            return f"db:{pref_source}"

    except Exception as e:
        logger.warning(f"Failed to fetch preferences: {e}")
        return "env/defaults:error"


@app.on_event("startup")
async def startup() -> None:
    # Register with orchestrator and start heartbeat
    await _register_with_orchestrator()
    if ORCHESTRATOR_URL:
        asyncio.create_task(_heartbeat_loop())

    # Fetch user preferences (applies to env vars before providers start)
    pref_source = await _fetch_user_preferences()

    # Reload config with user preferences
    reload_config()
    startup_effective_model = _effective_llm_model(_config_mod.config.llm.model)
    logger.info(
        "LLM startup: provider=%s, model=%s, effective_model=%s, base_url=%s, source=%s",
        _config_mod.config.llm.provider,
        _config_mod.config.llm.model,
        startup_effective_model,
        _config_mod.config.llm.base_url,
        pref_source,
    )

    # Fetch plugin configs
    await _fetch_plugin_configs()

    # Start providers and stores
    await llm_provider.start()
    await tts_provider.start()
    await memory_store.start()
    await session_store.start()

    # Start AgentCore (starts the scheduler worker pools)
    await agent_core.start()

    # Start WebRTC session TTL sweep
    if webrtc_transport:
        await webrtc_transport.start_session_ttl_sweep()

    logger.info(
        "Aether v0.10 ready (id=%s, providers: STT=%s, LLM=%s, TTS=%s, "
        "tools=%s, skills=%s, plugin_ctx=%s, webrtc=%s)",
        AGENT_ID,
        _config_mod.config.stt.provider,
        _config_mod.config.llm.provider,
        _config_mod.config.tts.provider,
        tool_registry.tool_names(),
        [s.name for s in skill_loader.all()],
        plugin_context_store.loaded_plugins(),
        "available" if webrtc_transport else "unavailable",
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    # Stop AgentCore (stops scheduler)
    await agent_core.stop()

    # Close WebRTC connections
    if webrtc_transport:
        await webrtc_transport.close_all()

    # Stop providers and stores
    await llm_provider.stop()
    await tts_provider.stop()
    await session_store.stop()


# ── WebSocket endpoint — notification sidecar ──────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint — push-only notification sidecar for dashboard."""
    await ws_sidecar.handle_connection(ws)


# ── WebRTC signaling endpoints ─────────────────────────────────


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request) -> JSONResponse:
    """WebRTC signaling: receive SDP offer, return SDP answer."""
    if not webrtc_transport:
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
        answer = await webrtc_transport.handle_offer(
            sdp=sdp,
            sdp_type=sdp_type,
            user_id=user_id,
            pc_id=pc_id,
        )
        # Exempt this agent from idle-kill while a WebRTC session is active.
        # Fire-and-forget — don't block the offer response on this.
        asyncio.create_task(_set_keep_alive(True))
        return JSONResponse(answer)
    except Exception as e:
        logger.error(f"WebRTC offer error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.patch("/webrtc/ice")
async def webrtc_ice(request: Request) -> JSONResponse:
    """WebRTC signaling: add ICE candidates."""
    if not webrtc_transport:
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
            await webrtc_transport.handle_ice_candidate(
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


# ── Legacy /chat endpoint (backward compat) ───────────────────


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Legacy HTTP streaming chat endpoint for dashboard text chat.

    Streams plain text chunks via AgentCore. For OpenAI-compatible
    format, use /v1/chat/completions instead.
    """
    body = await request.json()
    incoming_messages = body.get("messages", [])
    user_id = body.get("user_id", "")

    if not incoming_messages:
        return JSONResponse({"error": "No messages"}, status_code=400)

    # Extract the latest user message (AI SDK + legacy format)
    last_user_msg = None
    for m in reversed(incoming_messages):
        if m.get("role") == "user":
            parts = m.get("parts", [])
            if parts:
                last_user_msg = " ".join(
                    p.get("text", "") for p in parts if p.get("type") == "text"
                )
            else:
                last_user_msg = m.get("content", "")
            break

    if not last_user_msg:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    session_id = f"http-{user_id or 'anon'}"

    async def generate():
        try:
            async for event in agent_core.generate_reply(last_user_msg, session_id):
                if event.stream_type == "text_chunk":
                    chunk = event.payload.get("text", "")
                    yield chunk
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


# ── REST endpoints ─────────────────────────────────────────────


@app.get("/")
async def root() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Aether v0.10</h1><p>Client files not found.</p>")


@app.get("/webrtc")
async def webrtc_test_page() -> HTMLResponse:
    """Serve the WebRTC voice test page."""
    webrtc_path = STATIC_DIR / "webrtc.html"
    if webrtc_path.exists():
        return HTMLResponse(webrtc_path.read_text())
    return HTMLResponse("<h1>WebRTC test page not found</h1>", status_code=404)


@app.get("/workspace")
@app.get("/workspace/{path:path}")
async def browse_workspace(path: str = "") -> JSONResponse:
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
async def health() -> JSONResponse:
    """Health check — AgentCore, providers, memory, and metrics snapshot."""
    # AgentCore health includes scheduler status
    agent_health = await agent_core.health_check()

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
            "version": "0.10.0",
            "providers": providers,
            "memory": {
                "facts_count": len(facts),
                "has_conversations": len(recent) > 0,
            },
            "tools": tool_registry.tool_names(),
            "skills": [s.name for s in skill_loader.all()],
            "agent_core": agent_health,
            "transports": {
                "http": True,
                "webrtc": webrtc_transport is not None,
                "telephony": telephony_transport is not None,
                "ws_sidecar": True,
                "ws_connections": ws_sidecar.connection_count,
            },
            "metrics": metrics.snapshot(),
        }
    )


@app.get("/metrics")
async def metrics_endpoint() -> JSONResponse:
    """Full in-process metrics snapshot (counters, gauges, histogram percentiles)."""
    return JSONResponse(metrics.snapshot())


@app.get("/metrics/latency")
async def latency_metrics() -> JSONResponse:
    """SLO-focused latency metrics — p50/p95 for all critical paths."""
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
async def memory_facts() -> JSONResponse:
    facts = await memory_store.get_facts()
    return JSONResponse({"facts": facts})


@app.get("/memory/sessions")
async def memory_sessions() -> JSONResponse:
    sessions = await memory_store.get_session_summaries()
    return JSONResponse({"sessions": sessions})


@app.get("/memory/conversations")
async def memory_conversations(limit: int = 20) -> JSONResponse:
    conversations = await memory_store.get_recent(limit=limit)
    return JSONResponse({"conversations": conversations})


# ── Config endpoints ───────────────────────────────────────────


def _effective_llm_model(model: str) -> str:
    """Return the runtime model id that will be sent to the OpenAI SDK.

    All LLM traffic routes through OpenRouter, so models always get
    a provider prefix (e.g. 'gpt-4o' → 'openai/gpt-4o').
    """
    if "/" not in model:
        from aether.core.config import _infer_provider_from_model

        return f"{_infer_provider_from_model(model)}/{model}"
    return model


@app.get("/config")
async def get_config() -> JSONResponse:
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
async def config_reload(request_body: dict | None = None) -> JSONResponse:
    if request_body:
        for key, value in request_body.items():
            os.environ[key] = str(value)

    new_config = reload_config()
    await _fetch_plugin_configs()

    # Restart providers to pick up new config
    await llm_provider.stop()
    await llm_provider.start()

    # TTS provider may also need restart
    await tts_provider.stop()
    await tts_provider.start()

    effective_model = _effective_llm_model(new_config.llm.model)
    logger.info(
        "Config reloaded (STT=%s/%s, LLM=%s/%s, effective_llm=%s, TTS=%s/%s/%s, style=%s, plugin_ctx=%s)",
        new_config.stt.provider,
        new_config.stt.model,
        new_config.llm.provider,
        new_config.llm.model,
        effective_model,
        new_config.tts.provider,
        new_config.tts.model,
        new_config.tts.voice,
        new_config.personality.base_style,
        plugin_context_store.loaded_plugins(),
    )
    return JSONResponse({"status": "reloaded"})


@app.get("/plugins")
async def list_plugins_endpoint() -> JSONResponse:
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


# ── Plugin event endpoints — routed through AgentCore ──────────


@app.post("/plugin_event")
async def receive_plugin_event(request: Request) -> JSONResponse:
    """
    Receive a plugin event from the orchestrator webhook receiver.
    Runs through EventProcessor → broadcasts to WS sidecar clients.
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

    # Broadcast notification to WS sidecar clients via AgentCore
    if decision.action in ("surface", "action_required"):
        level = "speak" if decision.action == "action_required" else "nudge"

        # For "speak" level, deliver through active voice sessions
        if level == "speak" and decision.notification:
            await agent_core.speak_notification(decision.notification)
        else:
            await agent_core.broadcast_notification(
                {
                    "event_id": event.id,
                    "plugin": event.plugin,
                    "level": level,
                    "text": decision.notification,
                    "actions": event.available_actions,
                }
            )

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
async def receive_batch_notification(request: Request) -> JSONResponse:
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

    await agent_core.broadcast_notification(
        {
            "level": "batch",
            "text": batch_text,
            "items": items,
        }
    )

    return JSONResponse({"status": "delivered", "count": count})


@app.post("/cron_event")
async def receive_cron_event(request: Request) -> JSONResponse:
    """Receive a scheduled cron event from the orchestrator.

    The orchestrator delivers a plain-language instruction (e.g. "refresh the
    Google Drive token") and an optional plugin hint.  We inject it as a system
    message into a dedicated cron session so the LLM can act on it — calling
    whatever tool is appropriate — without any user interaction.

    The session ID is stable per plugin so the LLM has context across runs.
    """
    body = await request.json()
    plugin: str = body.get("plugin") or "system"
    instruction: str = body.get("instruction", "").strip()

    if not instruction:
        return JSONResponse(
            {"status": "ignored", "reason": "empty instruction"}, status_code=400
        )

    # Use a stable, background session per plugin so history accumulates
    session_id = f"cron-{plugin}"

    logger.info(
        "Cron event received (plugin=%s, session=%s): %s",
        plugin,
        session_id,
        instruction[:120],
    )

    # Run the LLM in the background — don't block the orchestrator's HTTP call.
    # run_session with background=True returns immediately; the LLM loop runs
    # autonomously, calls tools, and logs results to the session store.
    asyncio.create_task(
        agent_core.run_session(
            session_id=session_id,
            user_message=instruction,
            background=False,  # run fully before task completes; errors surface in logs
        )
    )

    return JSONResponse({"status": "accepted", "session_id": session_id})
