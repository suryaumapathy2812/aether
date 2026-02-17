"""
Aether v0.02 — Streaming Pipeline.

FastAPI server with WebSocket endpoint for real-time voice conversation.
Key change from v0.01: LLM streams sentence-by-sentence, TTS processes each
sentence immediately, client receives and plays audio chunks as they arrive.

Run: uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

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
)
from aether.core.pipeline import Pipeline
from aether.memory.store import MemoryStore
from aether.processors.llm import LLMProcessor
from aether.processors.memory import MemoryRetrieverProcessor
from aether.processors.stt import STTProcessor
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
    logger.info("Aether v0.02 ready (streaming)")


@app.on_event("shutdown")
async def shutdown():
    await memory_store.stop()


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Aether v0.02</h1><p>Static files not found.</p>")


async def _send_ws(ws: WebSocket, msg_type: str, data: str, **extra):
    """Helper to send a JSON message over WebSocket."""
    payload = {"type": msg_type, "data": data, **extra}
    await ws.send_text(json.dumps(payload))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint with streaming response.

    v0.02 protocol additions:
      Server sends: {"type": "text_chunk", "data": "sentence", "index": 0}
                    {"type": "audio_chunk", "data": "<base64 mp3>", "index": 0}
                    {"type": "stream_end"}
    """
    await ws.accept()
    logger.info("Client connected")

    # Create processors individually (not as a rigid pipeline for the hot path)
    stt = STTProcessor()
    vision = VisionProcessor()
    memory_retriever = MemoryRetrieverProcessor(memory_store)
    llm = LLMProcessor(memory_store)
    tts = TTSProcessor()

    # Pre-processing pipeline: STT → Vision → Memory (still sequential, fast)
    pre_pipeline = Pipeline([stt, vision, memory_retriever])
    await pre_pipeline.start()
    await llm.start()
    await tts.start()
    logger.info("Streaming pipeline ready: [STT → Vision → Memory] → LLM ⇒ TTS")

    pending_vision_frame: Frame | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            input_frames: list[Frame] = []

            if msg_type == "audio":
                audio_data = base64.b64decode(msg["data"])
                input_frames.append(audio_frame(audio_data))
                if pending_vision_frame:
                    input_frames.insert(0, pending_vision_frame)
                    pending_vision_frame = None

            elif msg_type == "image":
                image_data = base64.b64decode(msg["data"])
                mime = msg.get("mime", "image/jpeg")
                pending_vision_frame = vision_frame(image_data, mime_type=mime)
                await _send_ws(ws, "status", "Image received, listening...")
                continue

            elif msg_type == "text":
                input_frames.append(text_frame(msg["data"], role="user"))
                if pending_vision_frame:
                    input_frames.insert(0, pending_vision_frame)
                    pending_vision_frame = None

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                continue

            # --- Phase 1: Pre-processing (STT → Vision → Memory) ---
            pre_frames: list[Frame] = []
            async for f in pre_pipeline.run(input_frames):
                pre_frames.append(f)

            if not pre_frames:
                await _send_ws(ws, "status", "Didn't catch that, try again")
                continue

            # --- Phase 2: Streaming LLM → TTS ---
            # Feed pre-processed frames into LLM, stream sentences out,
            # TTS each sentence immediately, send to client as chunks arrive.
            sentence_index = 0

            for pf in pre_frames:
                async for llm_frame in llm.process(pf):
                    # LLM yields: text frames (sentences) and control frames
                    if llm_frame.type == FrameType.TEXT and llm_frame.metadata.get("role") == "assistant":
                        sentence_text = llm_frame.data

                        # Send text chunk to client immediately (text appears fast)
                        await _send_ws(ws, "text_chunk", sentence_text, index=sentence_index)

                        # TTS this sentence and send audio chunk
                        async for tts_frame in tts.process(llm_frame):
                            if tts_frame.type == FrameType.AUDIO:
                                await _send_ws(
                                    ws,
                                    "audio_chunk",
                                    base64.b64encode(tts_frame.data).decode("utf-8"),
                                    index=sentence_index,
                                )

                        sentence_index += 1

                    elif llm_frame.type == FrameType.CONTROL:
                        action = llm_frame.data.get("action")
                        if action == "llm_done":
                            # All sentences streamed
                            await _send_ws(ws, "stream_end", "")

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await pre_pipeline.stop()
        await llm.stop()
        await tts.stop()
