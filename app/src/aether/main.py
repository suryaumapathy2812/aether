"""
Aether v0.05 — Hands.

Changes from v0.04:
- Tool system: 5 core tools (read_file, write_file, list_directory, run_command, web_search)
- Agentic loop: LLM can call tools, get results, continue
- Status/acknowledge: each tool has status_text for UI spinners / voice acknowledgment
- Skill system: SKILL.md discovery + description-matching trigger
- Background tasks: simple sub-agents (max 3 concurrent, 60s timeout)

Run: uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aether.core.config import config
from aether.core.frames import (
    Frame,
    FrameType,
    audio_frame,
    text_frame,
    vision_frame,
)
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
from aether.skills.loader import SkillLoader
from aether.agents.task_runner import TaskRunner

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aether")

# --- App ---
app = FastAPI(title="Aether", version="0.0.5")

STATIC_DIR = Path(__file__).parent / "static"
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


@app.on_event("startup")
async def startup():
    await memory_store.start()
    await stt_provider.start()
    await llm_provider.start()
    await tts_provider.start()
    logger.info(
        "Aether v0.05 ready (providers: STT=%s, LLM=%s, TTS=%s, tools=%s, skills=%s)",
        config.stt.provider,
        config.llm.provider,
        config.tts.provider,
        tool_registry.tool_names(),
        [s.name for s in skill_loader.all()],
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
    return HTMLResponse("<h1>Aether v0.03</h1><p>Static files not found.</p>")


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
            "version": "0.0.5",
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


async def _send(ws: WebSocket, msg_type: str, data: str = "", **extra):
    """Send a JSON message over WebSocket with timeout protection."""
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": msg_type, "data": data, **extra})),
            timeout=config.server.ws_send_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"WebSocket send timeout ({msg_type})")
    except Exception as e:
        logger.error(f"WebSocket send error ({msg_type}): {e}")


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
    # Phase 1: Memory retrieval
    pre_frames: list[Frame] = []
    user_frame = text_frame(user_text, role="user")

    if pending_vision:
        async for f in VisionProcessor().process(pending_vision):
            pre_frames.append(f)

    async for f in memory_retriever.process(user_frame):
        pre_frames.append(f)

    # Phase 2: Stream LLM → TTS
    sentence_index = 0

    for pf in pre_frames:
        async for llm_frame in llm.process(pf):
            if (
                llm_frame.type == FrameType.TEXT
                and llm_frame.metadata.get("role") == "assistant"
            ):
                sentence_text = llm_frame.data
                await _send(ws, "text_chunk", sentence_text, index=sentence_index)

                try:
                    async for tts_frame in tts.process(llm_frame):
                        if tts_frame.type == FrameType.AUDIO:
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
                await _send(ws, "status", llm_frame.data.get("text", "Working..."))

            elif llm_frame.type == FrameType.TOOL_CALL:
                await _send(
                    ws,
                    "tool_call",
                    json.dumps(
                        {
                            "name": llm_frame.data["tool_name"],
                            "call_id": llm_frame.data.get("call_id", ""),
                        }
                    ),
                )

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
    logger.info("Client connected")

    # Create per-connection processor instances (they hold conversation state)
    batch_stt = STTProcessor(stt_provider)
    memory_retriever = MemoryRetrieverProcessor(memory_store)
    llm = LLMProcessor(
        llm_provider,
        memory_store,
        tool_registry=tool_registry,
        skill_loader=skill_loader,
    )
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
        nonlocal is_responding, pending_vision_frame

        is_responding = True
        await _send(ws, "status", "thinking...")

        try:
            logger.info(f"Triggering LLM: '{transcript[:60]}'")
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
            logger.info("Response cycle complete, ready for next utterance")

    async def _debounce_and_trigger():
        """Wait for silence, then trigger the response."""
        nonlocal accumulated_transcript, debounce_task

        try:
            logger.info(
                f"Debounce started (accumulated: '{accumulated_transcript[:60]}')"
            )
            await asyncio.sleep(config.server.debounce_delay)

            transcript = accumulated_transcript.strip()
            accumulated_transcript = ""

            if not transcript:
                logger.info("Debounce fired but transcript empty, ignoring")
                return

            logger.info("Debounce fired -> triggering response")
            await _trigger_response(transcript)

        except asyncio.CancelledError:
            logger.info("Debounce cancelled (user still speaking)")
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
                                logger.info(
                                    "Cancelling previous debounce (new utterance)"
                                )
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

    logger.info("Pipeline ready: StreamingSTT -> Memory -> LLM -> TTS")

    try:
        while True:
            raw = await ws.receive_text()
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

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if stt_event_task:
            stt_event_task.cancel()
            try:
                await stt_event_task
            except asyncio.CancelledError:
                pass
        await stt_provider.disconnect_stream()
        await batch_stt.stop()
        await llm.stop()
        await tts.stop()
