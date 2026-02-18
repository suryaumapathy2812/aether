"""
Aether v0.07 — Memory That Matters.

Changes from v0.06:
- Four-tier memory: conversations, facts, actions, sessions
- Action memory: tool calls stored with embeddings for semantic search
- Session summaries: generated on disconnect, loaded into greeting on reconnect
- Action compaction: old actions summarized into facts on startup
- Enhanced fact extraction: expanded prompt for preferences and tool patterns
- Cross-session continuity: returning users get context from previous sessions

Run: uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import aether.core.config as _config_mod
from aether.core.config import config, reload_config
from aether.core.frames import (
    Frame,
    FrameType,
    audio_frame,
    text_frame,
    vision_frame,
)
from aether.core.logging import PipelineTimer, setup_logging
from aether.memory.store import MemoryStore
from aether.processors.llm import LLMProcessor
from aether.processors.memory import MemoryRetrieverProcessor
from aether.processors.stt import STTProcessor
from aether.processors.tts import TTSProcessor
from aether.processors.vision import VisionProcessor
from aether.greeting import generate_greeting
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

# --- Setup ---
setup_logging()
logger = logging.getLogger("aether")

# --- Orchestrator registration ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
AGENT_ID = os.getenv("AETHER_AGENT_ID", f"agent-{socket.gethostname()}")
AGENT_USER_ID = os.getenv("AETHER_USER_ID", "")
AGENT_SECRET = os.getenv("AGENT_SECRET", "")

# --- App ---
app = FastAPI(title="Aether", version="0.0.7")

# Static files: serve from client/web/ (new structure) or fallback to legacy static/
CLIENT_WEB_DIR = Path(__file__).parent.parent.parent.parent.parent / "client" / "web"
STATIC_DIR = (
    CLIENT_WEB_DIR if CLIENT_WEB_DIR.exists() else Path(__file__).parent / "static"
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Shared state ---
memory_store = MemoryStore()

# Providers (created once, shared across connections)
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
    str(APP_ROOT / "skills"),  # Custom skills
    str(APP_ROOT / ".agents" / "skills"),  # skills.sh marketplace
]
skill_loader = SkillLoader(skills_dirs=SKILLS_DIRS)
skill_loader.discover()

# --- Plugin Loader ---
PLUGINS_DIR = str(APP_ROOT / "plugins")  # /app/plugins/
plugin_loader = PluginLoader(PLUGINS_DIR)
loaded_plugins = plugin_loader.discover()
for plugin in loaded_plugins:
    for tool in plugin.tools:
        tool_registry.register(tool, plugin_name=plugin.manifest.name)

# --- Plugin Context Store (runtime credentials for plugin tools) ---
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

# Track connected WebSocket clients for plugin event forwarding
_connected_clients: list[WebSocket] = []


def _agent_auth_headers() -> dict[str, str]:
    """Build auth headers for orchestrator calls (agent secret)."""
    if AGENT_SECRET:
        return {"Authorization": f"Bearer {AGENT_SECRET}"}
    return {}


async def _register_with_orchestrator():
    """Register this agent with the orchestrator on startup."""
    if not ORCHESTRATOR_URL:
        logger.info("No ORCHESTRATOR_URL — skipping registration")
        return

    if not AGENT_SECRET:
        logger.warning(
            "⚠ AGENT_SECRET not set — registration calls are unauthenticated"
        )

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
    """Send periodic heartbeats to the orchestrator."""
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
    """Fetch decrypted plugin configs from the orchestrator.

    Called at startup (after registration) and on /config/reload.
    Populates the plugin_context_store so tools get their credentials.
    """
    if not ORCHESTRATOR_URL or not AGENT_USER_ID:
        logger.debug(
            "No ORCHESTRATOR_URL or AGENT_USER_ID — skipping plugin config fetch"
        )
        return

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # 1. Get list of enabled plugins for this user
            resp = await client.get(
                f"{ORCHESTRATOR_URL}/api/internal/plugins",
                params={"user_id": AGENT_USER_ID},
                headers=_agent_auth_headers(),
            )
            if resp.status_code != 200:
                logger.debug(f"No enabled plugins (status={resp.status_code})")
                return

            enabled = resp.json().get("plugins", [])

            # 2. Fetch config for each enabled plugin
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


@app.on_event("startup")
async def startup():
    await memory_store.start()
    await stt_provider.start()
    await llm_provider.start()
    await tts_provider.start()

    # Register with orchestrator and start heartbeat
    await _register_with_orchestrator()
    if ORCHESTRATOR_URL:
        asyncio.create_task(_heartbeat_loop())

    # Fetch plugin configs (needs orchestrator registration first)
    await _fetch_plugin_configs()

    logger.info(
        "Aether v0.07 ready (id=%s, providers: STT=%s, LLM=%s, TTS=%s, tools=%s, skills=%s, plugin_ctx=%s)",
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
    await stt_provider.stop()
    await llm_provider.stop()
    await tts_provider.stop()
    await memory_store.stop()


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Aether v0.07</h1><p>Client files not found.</p>")


@app.get("/workspace")
@app.get("/workspace/{path:path}")
async def browse_workspace(path: str = ""):
    """Browse the workspace directory. Returns file listing as JSON."""
    import os

    workspace = Path(WORKING_DIR)
    target = workspace / path

    # Safety: don't escape workspace
    try:
        target.resolve().relative_to(workspace.resolve())
    except ValueError:
        return JSONResponse({"error": "Path outside workspace"}, status_code=403)

    if not target.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    if target.is_file():
        # Return file contents (text only, max 100KB)
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

    # Directory listing
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
        {
            "type": "directory",
            "path": str(path) or ".",
            "entries": entries,
        }
    )


@app.get("/health")
async def health():
    """Health check — reports provider status and memory stats."""
    stt_health = await stt_provider.health_check()
    llm_health = await llm_provider.health_check()
    tts_health = await tts_provider.health_check()

    # Memory stats
    facts = await memory_store.get_facts()
    recent = await memory_store.get_recent(limit=1)

    return JSONResponse(
        {
            "status": "ok",
            "version": "0.0.7",
            "providers": {
                "stt": stt_health,
                "llm": llm_health,
                "tts": tts_health,
            },
            "memory": {
                "facts_count": len(facts),
                "has_conversations": len(recent) > 0,
            },
            "tools": tool_registry.tool_names(),
            "skills": [s.name for s in skill_loader.all()],
        }
    )


@app.get("/memory/facts")
async def memory_facts():
    """Return all extracted facts from memory."""
    facts = await memory_store.get_facts()
    return JSONResponse({"facts": facts})


@app.get("/memory/sessions")
async def memory_sessions():
    """Return session summaries."""
    sessions = await memory_store.get_session_summaries()
    return JSONResponse({"sessions": sessions})


@app.get("/memory/conversations")
async def memory_conversations(limit: int = 20):
    """Return recent conversations."""
    conversations = await memory_store.get_recent(limit=limit)
    return JSONResponse({"conversations": conversations})


# ── Config endpoints ───────────────────────────────────────


@app.get("/config")
async def get_config():
    """Return the agent's current configuration (for dashboard display)."""
    cfg = _config_mod.config  # Always read the latest (survives reload)
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
    """Hot-reload config from env vars (or from provided overrides).

    Called by the orchestrator when a user changes preferences in dev mode.
    In multi-user mode, the container is restarted instead.
    """
    if request_body:
        # Apply overrides to os.environ so reload_config() picks them up
        for key, value in request_body.items():
            os.environ[key] = str(value)

    new_config = reload_config()

    # Also refresh plugin configs (user may have changed plugin settings)
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
    """List loaded plugins and their tools."""
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


@app.post("/plugin_event")
async def receive_plugin_event(request: Request):
    """
    Receive a plugin event forwarded from orchestrator webhook receiver.
    Runs through EventProcessor → surfaces to connected clients if needed.
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

    # Send structured notification to connected clients
    if decision.action in ("surface", "action_required"):
        level = "speak" if decision.action == "action_required" else "nudge"
        for client_ws in _connected_clients:
            try:
                await _send(
                    client_ws,
                    "notification",
                    json.dumps(
                        {
                            "event_id": event.id,
                            "plugin": event.plugin,
                            "level": level,
                            "text": decision.notification,
                            "actions": event.available_actions,
                        }
                    ),
                )
            except Exception as e:
                logger.debug(f"Plugin notification send failed: {e}")

    # Report decision back to orchestrator for persistence
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

    # Compose batch text
    count = len(items)
    summaries = [e.get("notification", e.get("summary", "")) for e in items[:5]]
    batch_text = (
        f"While you were away, {count} thing{'s' if count > 1 else ''} happened. "
    )
    batch_text += " ".join(summaries)

    for client_ws in _connected_clients:
        try:
            await _send(
                client_ws,
                "notification",
                json.dumps(
                    {
                        "level": "batch",
                        "text": batch_text,
                        "items": items,
                    }
                ),
            )
        except Exception:
            pass

    return JSONResponse({"status": "delivered", "count": count})


async def _send(ws: WebSocket, msg_type: str, data: str = "", **extra):
    """Send a JSON message over WebSocket with timeout protection.

    Silently drops messages if the connection is already dead,
    preventing wasted work in the LLM/TTS pipeline.
    """
    try:
        if ws.client_state.name != "CONNECTED":
            return
        msg = json.dumps({"type": msg_type, "data": data, **extra})
        logger.info(f"→ WS OUT: {msg[:200]}...")  # Log outgoing message
        await asyncio.wait_for(
            ws.send_text(msg),
            timeout=config.server.ws_send_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"WebSocket send timeout ({msg_type})")
    except Exception as e:
        logger.debug(f"WebSocket send skipped ({msg_type}): {e}")


async def _speak_status(ws: WebSocket, tts: TTSProcessor, text: str):
    """Fire-and-forget: synthesize status text as a short TTS clip.

    Runs concurrently with tool execution so the user hears
    "Reading that file..." while the tool is already working.
    Uses index=-1 to tag as status audio — client inserts a pause after.
    """
    try:
        status_frame = text_frame(text, role="assistant")
        async for tts_out in tts.process(status_frame):
            if tts_out.type == FrameType.AUDIO:
                await _send(
                    ws,
                    "audio_chunk",
                    base64.b64encode(tts_out.data).decode("utf-8"),
                    index=-1,
                    status_audio=True,
                )
    except Exception as e:
        logger.debug(f"Status TTS skipped: {e}")


SESSION_SUMMARY_PROMPT = """Summarize this conversation session in 2-3 sentences.
Focus on what was accomplished (files created, questions answered, tasks completed), not what was said.
If tools were used, mention what they did.

Conversation:
{conversation}

Summary:"""


async def _summarize_session(
    session_id: str,
    started_at: float,
    conversation_history: list[dict],
    turn_count: int,
) -> None:
    """Summarize a session and store it for cross-session continuity."""
    import time as _time

    try:
        # Build conversation text for the summary prompt
        conv_lines = []
        tools_used = set()
        for msg in conversation_history[-20:]:  # Last 20 messages max
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conv_lines.append(f"User: {content}")
            elif role == "assistant" and content:
                conv_lines.append(f"Aether: {content}")
            elif role == "tool":
                tools_used.add("tool")  # We don't have the name here

        if not conv_lines:
            return

        conv_text = "\n".join(conv_lines)
        prompt = SESSION_SUMMARY_PROMPT.format(conversation=conv_text)

        # Generate summary
        summary = ""
        async for token in llm_provider.generate_stream(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
        ):
            summary += token

        summary = summary.strip()
        if not summary:
            return

        await memory_store.add_session_summary(
            session_id=session_id,
            summary=summary,
            started_at=started_at,
            ended_at=_time.time(),
            turns=turn_count,
            tools_used=list(tools_used),
        )

    except Exception as e:
        logger.error(f"Session summary failed: {e}")


async def _run_llm_tts_streaming(
    ws: WebSocket,
    user_text: str,
    memory_retriever: MemoryRetrieverProcessor,
    llm: LLMProcessor,
    tts: TTSProcessor,
    pending_vision: Frame | None = None,
):
    """
    Run Memory → LLM → TTS streaming pipeline for a complete utterance.
    """
    timer = PipelineTimer()

    # Phase 1: Memory retrieval
    pre_frames: list[Frame] = []
    user_frame = text_frame(user_text, role="user")

    if pending_vision:
        async for f in VisionProcessor().process(pending_vision):
            pre_frames.append(f)

    async for f in memory_retriever.process(user_frame):
        pre_frames.append(f)

    timer.mark("Memory")

    # Phase 2: Stream LLM → TTS
    sentence_index = 0
    total_chars = 0
    total_audio_bytes = 0
    llm_marked = False

    for pf in pre_frames:
        async for llm_frame in llm.process(pf):
            if (
                llm_frame.type == FrameType.TEXT
                and llm_frame.metadata.get("role") == "assistant"
            ):
                if not llm_marked:
                    timer.mark("LLM")
                    llm_marked = True

                sentence_text = llm_frame.data
                total_chars += len(sentence_text)
                await _send(ws, "text_chunk", sentence_text, index=sentence_index)

                try:
                    async for tts_frame in tts.process(llm_frame):
                        if tts_frame.type == FrameType.AUDIO:
                            total_audio_bytes += len(tts_frame.data)
                            await _send(
                                ws,
                                "audio_chunk",
                                base64.b64encode(tts_frame.data).decode("utf-8"),
                                index=sentence_index,
                            )
                except Exception as e:
                    logger.error(f"TTS error for sentence {sentence_index}: {e}")

                sentence_index += 1

            elif llm_frame.type == FrameType.STATUS:
                status = llm_frame.data.get("text", "Working...")
                await _send(ws, "status", status)
                # Voice acknowledgment — fire-and-forget TTS while tool executes
                asyncio.create_task(_speak_status(ws, tts, status))

            elif llm_frame.type == FrameType.TOOL_RESULT:
                await _send(
                    ws,
                    "tool_result",
                    json.dumps(
                        {
                            "name": llm_frame.data["tool_name"],
                            "output": llm_frame.data["output"][
                                :500
                            ],  # Truncate for client
                            "error": llm_frame.data.get("error", False),
                        }
                    ),
                )

            elif llm_frame.type == FrameType.CONTROL:
                if llm_frame.data.get("action") == "llm_done":
                    await _send(ws, "stream_end")

    timer.mark("TTS")

    # --- Clean pipeline summary ---
    audio_kb = total_audio_bytes / 1024
    llm_time = timer.elapsed("LLM")
    tts_time = timer.elapsed("TTS")
    llm_str = f"{llm_time:.1f}s" if llm_time else "?"
    tts_str = f"{tts_time:.1f}s" if tts_time else "?"
    logger.info(
        f"LLM: {sentence_index} sentences, {total_chars} chars ({llm_str}) | "
        f"TTS: {sentence_index} chunks, {audio_kb:.0f}KB ({tts_str}) | "
        f"{timer.summary()}"
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint supporting both batch and streaming STT.

    Protocol:
      Client sends:
        {"type": "audio", "data": "<base64 blob>"}              — batch mode
        {"type": "audio_chunk", "data": "<base64 chunk>"}       — streaming mode
        {"type": "stream_start"}                                  — open STT connection
        {"type": "stream_stop"}                                   — close STT connection
        {"type": "mute"}                                          — pause mic (keep connection)
        {"type": "unmute"}                                        — resume mic
        {"type": "text", "data": "message"}                      — text fallback
        {"type": "image", "data": "<base64>", "mime": "..."}    — vision input

      Server sends:
        {"type": "transcript", "data": "text", "interim": true}  — live transcription
        {"type": "text_chunk", "data": "sentence", "index": 0}   — LLM response
        {"type": "audio_chunk", "data": "<base64>", "index": 0}  — TTS audio
        {"type": "stream_end"}                                     — response complete
        {"type": "status", "data": "..."}                         — status updates
    """
    await ws.accept()
    _connected_clients.append(ws)
    logger.info("Client connected")

    # Client liveness — set to False on disconnect to short-circuit sends
    client_alive = True

    # Session tracking
    session_id = str(uuid.uuid4())[:8]
    session_started_at = __import__("time").time()
    turn_count = 0
    tools_used_in_session: list[str] = []

    # Create per-connection processor instances (they hold conversation state)
    batch_stt = STTProcessor(stt_provider)
    memory_retriever = MemoryRetrieverProcessor(memory_store)
    llm = LLMProcessor(
        llm_provider,
        memory_store,
        tool_registry=tool_registry,
        skill_loader=skill_loader,
        plugin_context=plugin_context_store,
    )
    llm.session_id = session_id  # For action memory
    tts = TTSProcessor(tts_provider)

    await batch_stt.start()
    await llm.start()
    await tts.start()

    pending_vision_frame: Frame | None = None
    stt_event_task: asyncio.Task | None = None

    # --- Turn state ---
    is_responding = False
    is_muted = False
    debounce_task: asyncio.Task | None = None
    accumulated_transcript = ""

    async def _trigger_response(transcript: str):
        """Run the LLM/TTS pipeline for a complete utterance."""
        nonlocal is_responding, pending_vision_frame, turn_count

        is_responding = True
        await _send(ws, "status", "thinking...")

        try:
            turn_count += 1
            logger.info(f'STT: "{transcript}"')
            await _send(ws, "transcript", transcript, interim=False)

            vision = pending_vision_frame
            pending_vision_frame = None
            await _run_llm_tts_streaming(
                ws,
                transcript,
                memory_retriever,
                llm,
                tts,
                pending_vision=vision,
            )
        except Exception as e:
            logger.error(f"Response pipeline error: {e}", exc_info=True)
        finally:
            is_responding = False
            await _send(ws, "status", "listening...")

    async def _debounce_and_trigger():
        """Wait for silence, then trigger the response."""
        nonlocal accumulated_transcript, debounce_task

        try:
            await asyncio.sleep(config.server.debounce_delay)

            transcript = accumulated_transcript.strip()
            accumulated_transcript = ""

            if not transcript:
                return

            await _trigger_response(transcript)

        except asyncio.CancelledError:
            logger.debug("Debounce reset (user still speaking)")
        except Exception as e:
            logger.error(f"Debounce/trigger error: {e}", exc_info=True)
        finally:
            debounce_task = None

    async def _handle_stt_events():
        """Background task: listen to streaming STT events.

        Half-duplex: drops all STT events while assistant is responding or muted.
        """
        nonlocal accumulated_transcript, debounce_task

        try:
            async for event in stt_provider.stream_events():
                # Drop everything while assistant is speaking or muted
                if is_responding or is_muted:
                    continue

                if event.type == FrameType.TEXT and event.metadata.get("interim"):
                    await _send(ws, "transcript", event.data, interim=True)

                elif event.type == FrameType.CONTROL:
                    action = event.data.get("action")

                    if action == "utterance_end":
                        transcript = event.data.get("transcript", "")
                        if transcript:
                            accumulated_transcript += (
                                " " + transcript
                                if accumulated_transcript
                                else transcript
                            )

                            if debounce_task and not debounce_task.done():
                                debounce_task.cancel()
                            debounce_task = asyncio.create_task(_debounce_and_trigger())

                    elif action == "reconnected":
                        await _send(ws, "status", "listening...")
                        logger.info("STT reconnected — resuming")

                    elif action == "connection_lost":
                        await _send(ws, "status", "Connection lost. Please refresh.")
                        logger.error("STT connection permanently lost")

        except asyncio.CancelledError:
            logger.info("STT event handler cancelled")
        except Exception as e:
            logger.error(f"STT event handler error: {e}", exc_info=True)

    async def _session_greeting(
        ws_conn: WebSocket, tts_proc: TTSProcessor, is_reconnect: bool = False
    ):
        """Generate and speak a personalized greeting."""
        nonlocal is_responding
        try:
            if is_reconnect:
                # Silent reconnect — don't greet again
                logger.info("Silent reconnect — skipping greeting")
                return

            is_responding = True

            # Generate personalized greeting text
            greeting_text = await generate_greeting(memory_store, llm_provider)

            # Send as text
            await _send(ws_conn, "text_chunk", greeting_text, index=0)

            # Speak it
            greeting_frame = text_frame(greeting_text, role="assistant")
            try:
                async for tts_out in tts_proc.process(greeting_frame):
                    if tts_out.type == FrameType.AUDIO:
                        await _send(
                            ws_conn,
                            "audio_chunk",
                            base64.b64encode(tts_out.data).decode("utf-8"),
                            index=0,
                        )
            except Exception as e:
                logger.error(f"Greeting TTS error: {e}")

            await _send(ws_conn, "stream_end")

        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
        finally:
            is_responding = False
            await _send(ws_conn, "status", "listening...")

    logger.debug("Session pipeline initialized")

    try:
        while True:
            raw = await ws.receive_text()
            logger.info(f"← WS IN: {raw[:200]}...")  # Log incoming message
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "stream_start":
                await stt_provider.connect_stream()
                stt_event_task = asyncio.create_task(_handle_stt_events())
                await _send(ws, "status", "listening...")

                # Session greeting — Aether says hello when you "pick up"
                asyncio.create_task(
                    _session_greeting(ws, tts, is_reconnect=msg.get("reconnect", False))
                )

            elif msg_type == "stream_stop":
                if stt_event_task:
                    stt_event_task.cancel()
                    try:
                        await stt_event_task
                    except asyncio.CancelledError:
                        pass
                    stt_event_task = None
                await stt_provider.disconnect_stream()

            elif msg_type == "mute":
                is_muted = True
                # Cancel any pending debounce — don't trigger response from pre-mute speech
                if debounce_task and not debounce_task.done():
                    debounce_task.cancel()
                accumulated_transcript = ""
                await _send(ws, "status", "muted")
                logger.info("Session muted")

            elif msg_type == "unmute":
                is_muted = False
                await _send(ws, "status", "listening...")
                logger.info("Session unmuted")

            elif msg_type == "audio_chunk":
                if not is_muted:
                    audio_data = base64.b64decode(msg["data"])
                    await stt_provider.send_audio(audio_data)

            elif msg_type == "audio":
                # Batch mode fallback
                audio_data = base64.b64decode(msg["data"])
                user_text = await stt_provider.transcribe(audio_data)

                if user_text:
                    await _send(ws, "transcript", user_text, interim=False)
                    vision = pending_vision_frame
                    pending_vision_frame = None
                    await _run_llm_tts_streaming(
                        ws,
                        user_text,
                        memory_retriever,
                        llm,
                        tts,
                        pending_vision=vision,
                    )
                else:
                    await _send(ws, "status", "Didn't catch that, try again")

            elif msg_type == "text":
                user_text = msg["data"]
                vision = pending_vision_frame
                pending_vision_frame = None
                await _run_llm_tts_streaming(
                    ws,
                    user_text,
                    memory_retriever,
                    llm,
                    tts,
                    pending_vision=vision,
                )

            elif msg_type == "image":
                image_data = base64.b64decode(msg["data"])
                mime = msg.get("mime", "image/jpeg")
                pending_vision_frame = vision_frame(image_data, mime_type=mime)
                await _send(ws, "status", "Image received, listening...")

            elif msg_type == "notification_feedback":
                # Handle user feedback on notifications (engaged, dismissed, muted)
                data = msg.get("data", {})
                event_id = data.get("event_id", "")
                feedback = data.get("action", "")  # "engaged" | "dismissed" | "muted"
                plugin = data.get("plugin", "unknown")
                sender = data.get("sender", "")

                fact = ""
                if feedback == "engaged":
                    fact = (
                        f"User immediately reads {plugin} notifications from {sender}"
                    )
                elif feedback == "dismissed":
                    fact = f"User dismisses {plugin} notifications from {sender}"
                elif feedback == "muted":
                    fact = (
                        f"User wants to mute all {plugin} notifications from {sender}"
                    )

                if fact:
                    await memory_store.store_preference(fact)
                    logger.info(f"Preference stored: {fact}")

            elif msg_type == "pong":
                # Keep-alive response from client — no action needed
                pass

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        client_alive = False
        logger.info("Client disconnected")
    except Exception as e:
        client_alive = False
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        client_alive = False
        if ws in _connected_clients:
            _connected_clients.remove(ws)

        # Cancel pending debounce (could fire after cleanup otherwise)
        if debounce_task and not debounce_task.done():
            debounce_task.cancel()

        if stt_event_task:
            stt_event_task.cancel()
            try:
                await stt_event_task
            except asyncio.CancelledError:
                pass
        await stt_provider.disconnect_stream()

        # Session summary BEFORE stopping providers — needs llm_provider.client
        if turn_count > 0:
            await _summarize_session(
                session_id=session_id,
                started_at=session_started_at,
                conversation_history=llm.conversation_history,
                turn_count=turn_count,
            )

        await batch_stt.stop()
        await llm.stop()
        await tts.stop()
