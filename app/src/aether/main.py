"""
Aether v0.01 — The Bootloader.

FastAPI server with WebSocket endpoint for real-time voice conversation.
Single pipeline: AudioIn → STT → Memory → LLM → TTS → AudioOut

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
app = FastAPI(title="Aether", version="0.0.1")

# Serve static files (web client)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Shared state ---
memory_store = MemoryStore()


@app.on_event("startup")
async def startup():
    await memory_store.start()
    logger.info("Aether v0.01 ready")


@app.on_event("shutdown")
async def shutdown():
    await memory_store.stop()


@app.get("/")
async def root():
    """Serve the web client."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Aether v0.01</h1><p>Static files not found.</p>")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time conversation.

    Protocol:
      Client sends: {"type": "audio", "data": "<base64>"}
                    {"type": "image", "data": "<base64>", "mime": "image/jpeg"}
                    {"type": "text", "data": "typed message"}
      Server sends: {"type": "audio", "data": "<base64 mp3>"}
                    {"type": "text", "data": "response text"}
    """
    await ws.accept()
    logger.info("Client connected")

    # Build pipeline for this session
    stt = STTProcessor()
    vision = VisionProcessor()
    memory_retriever = MemoryRetrieverProcessor(memory_store)
    llm = LLMProcessor(memory_store)
    tts = TTSProcessor()

    pipeline = Pipeline([stt, vision, memory_retriever, llm, tts])
    await pipeline.start()
    logger.info(f"Pipeline ready: {pipeline}")

    # Track pending vision frame for multimodal queries
    pending_vision_frame: Frame | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # Build input frames based on message type
            input_frames: list[Frame] = []

            if msg_type == "audio":
                audio_data = base64.b64decode(msg["data"])
                input_frames.append(audio_frame(audio_data))

                # If there's a pending vision frame, include it
                if pending_vision_frame:
                    input_frames.insert(0, pending_vision_frame)
                    pending_vision_frame = None

            elif msg_type == "image":
                # Store vision frame — it'll be included with the next audio/text message
                image_data = base64.b64decode(msg["data"])
                mime = msg.get("mime", "image/jpeg")
                pending_vision_frame = vision_frame(image_data, mime_type=mime)
                # Acknowledge receipt
                await ws.send_text(json.dumps({"type": "status", "data": "Image received, listening..."}))
                continue

            elif msg_type == "text":
                # Text input (typed message, fallback mode)
                input_frames.append(text_frame(msg["data"], role="user"))

                if pending_vision_frame:
                    input_frames.insert(0, pending_vision_frame)
                    pending_vision_frame = None

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                continue

            # Run through pipeline
            async for output_frame in pipeline.run(input_frames):
                if output_frame.type == FrameType.AUDIO:
                    await ws.send_text(json.dumps({
                        "type": "audio",
                        "data": base64.b64encode(output_frame.data).decode("utf-8"),
                    }))
                elif output_frame.type == FrameType.TEXT:
                    await ws.send_text(json.dumps({
                        "type": "text",
                        "data": output_frame.data,
                        "role": output_frame.metadata.get("role", "assistant"),
                    }))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await pipeline.stop()
