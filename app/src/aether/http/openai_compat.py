"""
OpenAI-compatible HTTP API — /v1/chat/completions and /v1/models.

Drop-in compatible with the OpenAI Python SDK:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    response = client.chat.completions.create(
        model="aether",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

Supports:
- Streaming (SSE) and non-streaming responses
- Multimodal (vision) via content array with image_url
- Tool call events in OpenAI format
- /v1/models endpoint
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from aether.agent import AgentCore

logger = logging.getLogger(__name__)


def create_router(agent: "AgentCore") -> APIRouter:
    """Create the OpenAI-compatible router bound to an AgentCore instance."""

    router = APIRouter()

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        body = await request.json()

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        user = body.get("user", "")

        # Extract the latest user message
        last_msg = _extract_last_user_message(messages)
        if not last_msg:
            return JSONResponse(
                {
                    "error": {
                        "message": "No user message found in messages array",
                        "type": "invalid_request_error",
                        "code": "missing_message",
                    }
                },
                status_code=400,
            )

        vision = _extract_vision(messages)

        # Session ID from user field or generate
        session_id = f"http-{user or 'anon'}"

        # Generate completion metadata
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model = os.getenv("AETHER_LLM_MODEL", "aether")

        if stream:
            return StreamingResponse(
                _stream_response(
                    agent, completion_id, created, model, last_msg, session_id, vision
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _sync_response(
                agent, completion_id, created, model, last_msg, session_id, vision
            )

    @router.get("/v1/models")
    async def list_models() -> JSONResponse:
        """Return configured model in OpenAI format."""
        model = os.getenv("AETHER_LLM_MODEL", "aether")
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "aether",
                    }
                ],
            }
        )

    return router


# ─── Streaming Response ──────────────────────────────────────────


async def _stream_response(
    agent: "AgentCore",
    completion_id: str,
    created: int,
    model: str,
    text: str,
    session_id: str,
    vision: dict | None,
) -> AsyncGenerator[str, None]:
    """SSE stream in OpenAI chat.completion.chunk format."""

    # Initial role chunk
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    # Stream from AgentCore
    finish_reason = "stop"
    try:
        async for event in agent.generate_reply(text, session_id, vision=vision):
            if event.stream_type == "text_chunk":
                chunk_text = event.payload.get("text", "")
                if chunk_text:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

            elif event.stream_type == "tool_call":
                # Stream tool calls in OpenAI format
                tc = event.payload
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tc.get("call_id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": tc.get("tool_name", ""),
                                                "arguments": json.dumps(
                                                    tc.get("arguments", {})
                                                ),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                finish_reason = "tool_calls"

            elif event.stream_type == "error":
                # Stream error as a content chunk so the client sees it
                err_msg = event.payload.get("message", "Unknown error")
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"\n[error] {err_msg}"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

            # tool_result, status, stream_end — internal, not streamed to client

    except Exception as e:
        logger.error("Stream error: %s", e, exc_info=True)
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n[error] {e}"},
                        "finish_reason": None,
                    }
                ],
            }
        )

    # Final chunk with finish_reason
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )

    yield "data: [DONE]\n\n"


# ─── Non-Streaming Response ─────────────────────────────────────


async def _sync_response(
    agent: "AgentCore",
    completion_id: str,
    created: int,
    model: str,
    text: str,
    session_id: str,
    vision: dict | None,
) -> JSONResponse:
    """Non-streaming response in OpenAI chat.completion format."""
    collected: list[str] = []

    try:
        async for event in agent.generate_reply(text, session_id, vision=vision):
            if event.stream_type == "text_chunk":
                collected.append(event.payload.get("text", ""))
    except Exception as e:
        logger.error("Sync response error: %s", e, exc_info=True)
        return JSONResponse(
            {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
            status_code=500,
        )

    content = "".join(collected)
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


# ─── Helpers ─────────────────────────────────────────────────────


def _extract_last_user_message(messages: list[dict]) -> str:
    """Extract text from last user message.

    Handles both standard and multimodal (vision) formats:
    - Standard: {"role": "user", "content": "hello"}
    - Multimodal: {"role": "user", "content": [{"type": "text", "text": "..."}]}
    - AI SDK: {"role": "user", "parts": [{"type": "text", "text": "..."}]}
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")

        # Standard string content
        if isinstance(content, str):
            return content

        # Multimodal content array (OpenAI format)
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            if texts:
                return " ".join(texts)

        # AI SDK format (parts array)
        parts = msg.get("parts", [])
        if parts:
            texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            if texts:
                return " ".join(texts)

    return ""


def _extract_vision(messages: list[dict]) -> dict | None:
    """Extract base64 image from last user message if present.

    Supports OpenAI vision format:
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # data:image/jpeg;base64,...
                    try:
                        header, data = url.split(",", 1)
                        mime = header.split(";")[0].split(":")[1]
                        return {"mime": mime, "data": data}
                    except (ValueError, IndexError):
                        logger.warning("Malformed data URL in vision content")
                        continue

    return None


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
