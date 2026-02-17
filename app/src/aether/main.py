"""
Aether v0.02 — Streaming Pipeline with Live STT.

Two modes of operation:
1. Batch mode: Client records full audio → sends blob → STT → LLM → TTS (v0.01 compat)
2. Stream mode: Client streams audio chunks → live transcription → on utterance end → LLM → TTS

Stream mode is the primary path. Batch mode is fallback for browsers that
don't support streaming audio capture.

Run: uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from aether.core.frames import (
    Frame,
    FrameType,
    audio_frame,
    text_frame,
    vision_frame,
    memory_frame,
)
from aether.core.pipeline import Pipeline
from aether.memory.store import MemoryStore
from aether.processors.llm import LLMProcessor
from aether.processors.memory import MemoryRetrieverProcessor
from aether.processors.stt import STTProcessor, StreamingSTTProcessor
from aether.processors.tts import TTSProcessor
from aether.processors.vision import VisionProcessor

# --- Setup ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aether")

# --- App ---
app = FastAPI(title="Aether", version="0.0.2")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Shared state ---
memory_store = MemoryStore()


@app.on_event("startup")
async def startup():
    await memory_store.start()
    logger.info("Aether v0.02 ready (streaming STT + streaming LLM)")


@app.on_event("shutdown")
async def shutdown():
    await memory_store.stop()


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Aether v0.02</h1><p>Static files not found.</p>")


async def _send(ws: WebSocket, msg_type: str, data: str = "", **extra):
    """Send a JSON message over WebSocket with timeout protection."""
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": msg_type, "data": data, **extra})),
            timeout=5.0,
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
    Sends text_chunk and audio_chunk messages to client as they're produced.
    """
    # Phase 1: Memory retrieval
    pre_frames: list[Frame] = []
    user_frame = text_frame(user_text, role="user")

    # Include vision if present
    if pending_vision:
        async for f in VisionProcessor().process(pending_vision):
            pre_frames.append(f)

    # Search memory
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

                # Text arrives immediately
                await _send(ws, "text_chunk", sentence_text, index=sentence_index)

                # TTS this sentence
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

            elif llm_frame.type == FrameType.CONTROL:
                if llm_frame.data.get("action") == "llm_done":
                    await _send(ws, "stream_end")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint supporting both batch and streaming STT.

    Protocol:
      Client sends:
        {"type": "audio", "data": "<base64 blob>"}              — batch mode (full recording)
        {"type": "audio_chunk", "data": "<base64 chunk>"}       — streaming mode (live audio)
        {"type": "stream_start"}                                  — open Deepgram live connection
        {"type": "stream_stop"}                                   — close Deepgram live connection
        {"type": "text", "data": "message"}                      — text fallback
        {"type": "image", "data": "<base64>", "mime": "..."}    — vision input

      Server sends:
        {"type": "transcript", "data": "text", "interim": true}  — live transcription
        {"type": "text_chunk", "data": "sentence", "index": 0}   — LLM response sentence
        {"type": "audio_chunk", "data": "<base64>", "index": 0}  — TTS audio for sentence
        {"type": "stream_end"}                                     — response complete
    """
    await ws.accept()
    logger.info("Client connected")

    # Processors
    batch_stt = STTProcessor()
    streaming_stt = StreamingSTTProcessor()
    memory_retriever = MemoryRetrieverProcessor(memory_store)
    llm = LLMProcessor(memory_store)
    tts = TTSProcessor()

    await batch_stt.start()
    await streaming_stt.start()
    await llm.start()
    await tts.start()

    pending_vision_frame: Frame | None = None
    stt_event_task: asyncio.Task | None = None

    # --- Turn state ---
    is_responding = False
    debounce_task: asyncio.Task | None = None
    accumulated_transcript = ""  # Accumulates across multiple utterance_end events

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
                f"⏱ Debounce started (accumulated: '{accumulated_transcript[:60]}')"
            )
            await asyncio.sleep(1.5)  # 1.5-second debounce window

            # Timer expired — user is done speaking
            transcript = accumulated_transcript.strip()
            accumulated_transcript = ""

            if not transcript:
                logger.info("⏱ Debounce fired but transcript empty, ignoring")
                return

            logger.info("⏱ Debounce fired → triggering response")
            await _trigger_response(transcript)

        except asyncio.CancelledError:
            logger.info("⏱ Debounce cancelled (user still speaking)")
        except Exception as e:
            logger.error(f"Debounce/trigger error: {e}", exc_info=True)
        finally:
            debounce_task = None

    async def _handle_stt_events():
        """Background task: listen to streaming STT events and react.

        Turn-taking model: strict half-duplex. While the assistant is
        responding (is_responding=True), all STT events are dropped.
        The user's turn starts only after the response cycle completes.
        """
        nonlocal accumulated_transcript, debounce_task

        try:
            async for event in streaming_stt.events():
                # --- Drop everything while assistant is speaking ---
                if is_responding:
                    continue

                if event.type == FrameType.TEXT and event.metadata.get("interim"):
                    # Send interim transcript to client for live display
                    await _send(ws, "transcript", event.data, interim=True)

                elif event.type == FrameType.CONTROL:
                    action = event.data.get("action")

                    if action == "utterance_end":
                        # User paused — accumulate transcript and start/reset debounce timer
                        transcript = event.data.get("transcript", "")
                        if transcript:
                            accumulated_transcript += (
                                " " + transcript
                                if accumulated_transcript
                                else transcript
                            )

                            # Cancel existing debounce timer and start a new one
                            if debounce_task and not debounce_task.done():
                                logger.info(
                                    "⏱ Cancelling previous debounce (new utterance)"
                                )
                                debounce_task.cancel()
                            debounce_task = asyncio.create_task(_debounce_and_trigger())

        except asyncio.CancelledError:
            logger.info("STT event handler cancelled")
        except Exception as e:
            logger.error(f"STT event handler error: {e}", exc_info=True)

    logger.info("Pipeline ready: StreamingSTT → Memory → LLM ⇒ TTS")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "stream_start":
                # Client wants to start streaming audio
                await streaming_stt.connect()
                stt_event_task = asyncio.create_task(_handle_stt_events())
                await _send(ws, "status", "listening...")

            elif msg_type == "stream_stop":
                # Client stopped streaming
                if stt_event_task:
                    stt_event_task.cancel()
                    try:
                        await stt_event_task
                    except asyncio.CancelledError:
                        pass
                    stt_event_task = None
                await streaming_stt.disconnect()

            elif msg_type == "audio_chunk":
                # Streaming audio chunk — forward to Deepgram
                audio_data = base64.b64decode(msg["data"])
                await streaming_stt.send_audio(audio_data)

            elif msg_type == "audio":
                # Batch mode fallback — full audio blob
                audio_data = base64.b64decode(msg["data"])

                # Run through batch STT
                user_text = None
                async for f in batch_stt.process(audio_frame(audio_data)):
                    if f.type == FrameType.TEXT:
                        user_text = f.data
                        break

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
                # Text fallback
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
        await streaming_stt.disconnect()
        await batch_stt.stop()
        await llm.stop()
        await tts.stop()
