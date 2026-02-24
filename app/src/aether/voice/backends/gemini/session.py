"""Gemini live session implementation for the realtime voice abstraction."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from google.genai import types as genai_types

from aether.core.metrics import metrics
from aether.voice.realtime import (
    FunctionCallResult,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSession,
)

logger = logging.getLogger(__name__)

MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_BASE_SECONDS = 0.5
RECONNECT_BACKOFF_MAX_SECONDS = 8.0
OUTBOUND_TEXT_QUEUE_MAX_SIZE = 16


def _is_retryable_session_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_tokens = (
        "deadline expired",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "internal error",
        "1011",
    )
    return any(token in message for token in retryable_tokens)


class GeminiRealtimeSession(RealtimeSession):
    """Live Gemini websocket session implementing the realtime session ABC."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        live_config: Any,
        input_sample_rate: int,
        output_sample_rate: int,
        requested_resumption: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._config = live_config
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._requested_resumption = requested_resumption

        self._session: Any = None
        self._session_cm: Any = None
        self._connected = False
        self._closed = False
        self._session_id = f"gemini-{uuid.uuid4().hex[:12]}"
        self._resumption_token: str | None = None

        self._event_queue: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self._receive_task: asyncio.Task[None] | None = None
        self._pending_text_queue: deque[str] = deque()
        self._pending_text_lock = asyncio.Lock()
        self._session_setup_emitted = False
        self._connection_dropped = False
        self._lifecycle_state = "cold"
        self._last_error: str | None = None
        self._last_error_source: str | None = None
        self._last_error_at: float | None = None
        self._recovery_count = 0
        self._degraded_since: float | None = None

    async def connect(self) -> None:
        if self._connected:
            return
        self._closed = False
        self._set_lifecycle_state("connecting")
        started_at = asyncio.get_event_loop().time()
        connected = await self._connect_with_retries(reason="initial_connect")
        if not connected:
            metrics.inc("voice.realtime.connect.failed", labels={"backend": "gemini"})
            self._set_lifecycle_state("degraded")
            raise RuntimeError("Failed to connect Gemini realtime session")
        metrics.observe(
            "voice.realtime.connect_ms",
            (asyncio.get_event_loop().time() - started_at) * 1000,
            labels={"backend": "gemini"},
        )
        metrics.inc("voice.realtime.connect.ok", labels={"backend": "gemini"})
        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name=f"gemini-live-recv-{self._session_id}",
        )

    async def push_audio(self, data: bytes) -> None:
        if not data or not self._connected or self._session is None:
            return
        try:
            await self._session.send_realtime_input(
                audio=genai_types.Blob(
                    data=data,
                    mime_type=f"audio/pcm;rate={self._input_sample_rate}",
                )
            )
        except Exception:
            logger.exception("Failed to send realtime audio")

    async def send_function_result(self, result: FunctionCallResult) -> None:
        if not self._connected or self._session is None:
            return

        response_payload = _parse_json_or_text(result.result)
        try:
            await self._session.send_tool_response(
                function_responses=[
                    genai_types.FunctionResponse(
                        id=result.call_id,
                        name=result.name,
                        response=response_payload,
                    )
                ]
            )
        except Exception:
            logger.exception("Failed to send function result")

    async def interrupt(self) -> None:
        await self._event_queue.put(RealtimeEvent(type=RealtimeEventType.INTERRUPTED))

    async def generate_reply(self, text: str) -> None:
        if not text:
            return

        if not self._connected or self._session is None:
            await self._enqueue_pending_text(text)
            return

        sent = await self._send_text_content(text)
        if sent:
            return

        await self._enqueue_pending_text(text)
        await self._drop_connection(
            RuntimeError("Text send failed; reconnect scheduled"),
            source="generate_reply",
            retry_attempt=0,
            recoverable=True,
        )

    async def update_instructions(self, instructions: str) -> None:
        if not instructions:
            return
        self._config.system_instruction = genai_types.Content(
            parts=[genai_types.Part(text=instructions)]
        )

    async def update_tools(self, tools: list[dict[str, Any]]) -> None:
        declarations: list[genai_types.FunctionDeclaration] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=name,
                    description=str(tool.get("description", "")),
                    parameters=tool.get("parameters", {"type": "object"}),
                )
            )
        self._config.tools = [genai_types.Tool(function_declarations=declarations)]

    def events(self) -> AsyncIterator[RealtimeEvent]:
        return self._event_stream()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._session_cm is not None:
            await self._close_session_context()

        self._session = None
        self._session_cm = None
        self._set_lifecycle_state("closed")

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def resumption_token(self) -> str | None:
        return self._resumption_token

    def diagnostics(self) -> dict[str, Any]:
        return {
            "state": self._lifecycle_state,
            "connected": self._connected,
            "last_error": self._last_error,
            "last_error_source": self._last_error_source,
            "last_error_at": self._last_error_at,
            "recovery_count": self._recovery_count,
            "session_id": self._session_id,
            "queued_text": len(self._pending_text_queue),
        }

    async def _event_stream(self) -> AsyncIterator[RealtimeEvent]:
        while not self._closed or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            yield event

    async def _receive_loop(self) -> None:
        while not self._closed:
            if not self._connected or self._session is None:
                connected = await self._connect_with_retries(reason="receive_reconnect")
                if not connected:
                    self._set_lifecycle_state("degraded")
                    return

            try:
                async for message in self._session.receive():
                    for event in self._parse_live_message(message):
                        await self._event_queue.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Gemini receive loop failed")
                recoverable = _is_retryable_session_error(exc)
                await self._drop_connection(
                    exc,
                    source="receive_loop",
                    retry_attempt=0,
                    recoverable=recoverable,
                )
                if not recoverable:
                    return
                continue

            await self._drop_connection(
                RuntimeError("Gemini live session closed unexpectedly"),
                source="receive_loop",
                retry_attempt=0,
                recoverable=True,
            )

    def _parse_live_message(self, message: Any) -> list[RealtimeEvent]:
        events: list[RealtimeEvent] = []

        resumption_update = _pick(message, "session_resumption_update")
        new_handle = _pick(resumption_update, "new_handle")
        if isinstance(new_handle, str) and new_handle:
            self._resumption_token = new_handle

        if _pick(message, "setup_complete"):
            if not self._session_setup_emitted:
                event_type = (
                    RealtimeEventType.SESSION_RESUMED
                    if self._requested_resumption
                    else RealtimeEventType.SESSION_CREATED
                )
                events.append(RealtimeEvent(type=event_type))
                self._session_setup_emitted = True

        go_away = _pick(message, "go_away")
        if go_away:
            events.append(
                RealtimeEvent(
                    type=RealtimeEventType.SESSION_ERROR,
                    data={
                        "error": str(go_away),
                        "source": "go_away",
                        "retry_attempt": 0,
                        "recoverable": False,
                    },
                )
            )

        input_transcription = _pick(message, "input_transcription")
        transcript = _pick(input_transcription, "text")
        if isinstance(transcript, str) and transcript.strip():
            events.append(
                RealtimeEvent(
                    type=RealtimeEventType.INPUT_SPEECH_TRANSCRIPTION_COMPLETED,
                    data={"transcript": transcript},
                )
            )

        server_content = _pick(message, "server_content")
        if server_content:
            if _pick(server_content, "interrupted"):
                events.append(RealtimeEvent(type=RealtimeEventType.INTERRUPTED))

            if _pick(server_content, "input_speech_started"):
                events.append(
                    RealtimeEvent(type=RealtimeEventType.INPUT_SPEECH_STARTED)
                )
            if _pick(server_content, "input_speech_stopped"):
                events.append(
                    RealtimeEvent(type=RealtimeEventType.INPUT_SPEECH_STOPPED)
                )
            if _pick(server_content, "input_speech_committed"):
                events.append(
                    RealtimeEvent(type=RealtimeEventType.INPUT_SPEECH_COMMITTED)
                )

            model_turn = _pick(server_content, "model_turn")
            parts = _pick(model_turn, "parts") if model_turn else None
            if isinstance(parts, list) and parts:
                events.append(RealtimeEvent(type=RealtimeEventType.TURN_STARTED))
                for part in parts:
                    text_part = _pick(part, "text")
                    if isinstance(text_part, str) and text_part:
                        events.append(
                            RealtimeEvent(
                                type=RealtimeEventType.TEXT_DELTA,
                                data={"text": text_part},
                            )
                        )
                        continue

                    inline_data = _pick(part, "inline_data")
                    audio_bytes = _decode_inline_audio(inline_data)
                    if audio_bytes is not None:
                        events.append(
                            RealtimeEvent(
                                type=RealtimeEventType.AUDIO_DELTA,
                                data={
                                    "audio": audio_bytes,
                                    "sample_rate": self._output_sample_rate,
                                },
                            )
                        )

            if _pick(server_content, "turn_complete"):
                events.append(RealtimeEvent(type=RealtimeEventType.AUDIO_DONE))
                events.append(RealtimeEvent(type=RealtimeEventType.TEXT_DONE))
                events.append(RealtimeEvent(type=RealtimeEventType.TURN_DONE))

        tool_call = _pick(message, "tool_call")
        function_calls = _pick(tool_call, "function_calls") if tool_call else None
        if isinstance(function_calls, list):
            for call in function_calls:
                call_id = str(_pick(call, "id") or "")
                name = str(_pick(call, "name") or "")
                arguments = _stringify_tool_args(_pick(call, "args"))
                payload = {
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
                events.append(
                    RealtimeEvent(type=RealtimeEventType.FUNCTION_CALL, data=payload)
                )
                events.append(
                    RealtimeEvent(
                        type=RealtimeEventType.FUNCTION_CALL_DONE, data=payload
                    )
                )

        return events

    async def _connect_with_retries(self, *, reason: str) -> bool:
        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            if self._closed:
                return False

            self._set_lifecycle_state(
                "reconnecting" if self._connection_dropped else "connecting"
            )

            try:
                self._session_cm = self._client.aio.live.connect(
                    model=self._model,
                    config=self._config,
                )
                self._session = await self._session_cm.__aenter__()
                self._connected = True
                self._set_lifecycle_state("ready")

                was_reconnect = self._connection_dropped
                self._connection_dropped = False
                if was_reconnect:
                    self._recovery_count += 1
                    metrics.inc(
                        "voice.realtime.reconnect.ok",
                        labels={"backend": "gemini", "reason": reason},
                    )
                    if self._degraded_since is not None:
                        metrics.observe(
                            "voice.realtime.reconnect_ms",
                            max(0.0, (time.time() - self._degraded_since) * 1000.0),
                            labels={"backend": "gemini", "reason": reason},
                        )
                        self._degraded_since = None
                    await self._event_queue.put(
                        RealtimeEvent(type=RealtimeEventType.SESSION_RESUMED)
                    )
                await self._flush_pending_text_queue()
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_error(exc, source=reason)
                recoverable = attempt < MAX_RECONNECT_ATTEMPTS
                metrics.inc(
                    "voice.realtime.reconnect.attempt",
                    labels={
                        "backend": "gemini",
                        "reason": reason,
                        "recoverable": str(recoverable).lower(),
                    },
                )
                await self._emit_session_error(
                    exc,
                    source=reason,
                    retry_attempt=attempt,
                    recoverable=recoverable,
                )
                await self._close_session_context()
                self._connected = False

                if not recoverable:
                    self._set_lifecycle_state("degraded")
                    metrics.inc(
                        "voice.realtime.reconnect.exhausted",
                        labels={"backend": "gemini", "reason": reason},
                    )
                    return False

                await asyncio.sleep(_backoff_delay_seconds(attempt))

        return False

    async def _drop_connection(
        self,
        exc: Exception,
        *,
        source: str,
        retry_attempt: int,
        recoverable: bool,
    ) -> None:
        self._connected = False
        self._connection_dropped = True
        if self._degraded_since is None:
            self._degraded_since = time.time()
        self._set_lifecycle_state("degraded")
        self._record_error(exc, source=source)
        await self._emit_session_error(
            exc,
            source=source,
            retry_attempt=retry_attempt,
            recoverable=recoverable,
        )
        await self._close_session_context()

    async def _close_session_context(self) -> None:
        if self._session_cm is None:
            self._session = None
            return
        try:
            await self._session_cm.__aexit__(None, None, None)
        except Exception:
            logger.exception("Failed closing Gemini live session")
        finally:
            self._session = None
            self._session_cm = None

    async def _emit_session_error(
        self,
        exc: Exception,
        *,
        source: str,
        retry_attempt: int,
        recoverable: bool,
    ) -> None:
        await self._event_queue.put(
            RealtimeEvent(
                type=RealtimeEventType.SESSION_ERROR,
                data={
                    "error": str(exc),
                    "source": source,
                    "retry_attempt": retry_attempt,
                    "recoverable": recoverable,
                },
            )
        )

    def _set_lifecycle_state(self, state: str) -> None:
        if self._lifecycle_state == state:
            return
        self._lifecycle_state = state
        metrics.gauge_set(
            "voice.realtime.state",
            {
                "cold": 0,
                "connecting": 1,
                "ready": 2,
                "reconnecting": 3,
                "degraded": 4,
                "closed": 5,
            }.get(state, 0),
            labels={"backend": "gemini"},
        )

    def _record_error(self, exc: Exception, *, source: str) -> None:
        self._last_error = str(exc)
        self._last_error_source = source
        self._last_error_at = time.time()

    async def _enqueue_pending_text(self, text: str) -> None:
        async with self._pending_text_lock:
            if len(self._pending_text_queue) >= OUTBOUND_TEXT_QUEUE_MAX_SIZE:
                self._pending_text_queue.popleft()
                metrics.inc(
                    "voice.realtime.pending_text.drop_oldest",
                    labels={"backend": "gemini"},
                )
            self._pending_text_queue.append(text)
            metrics.gauge_set(
                "voice.realtime.pending_text.queue_size",
                len(self._pending_text_queue),
                labels={"backend": "gemini"},
            )

    async def _flush_pending_text_queue(self) -> None:
        replayed = 0
        while self._connected and self._session is not None:
            text: str | None = None
            async with self._pending_text_lock:
                if not self._pending_text_queue:
                    if replayed:
                        metrics.inc(
                            "voice.realtime.pending_text.replayed",
                            value=replayed,
                            labels={"backend": "gemini"},
                        )
                    metrics.gauge_set(
                        "voice.realtime.pending_text.queue_size",
                        0,
                        labels={"backend": "gemini"},
                    )
                    return
                text = self._pending_text_queue.popleft()

            if text is None:
                return

            sent = await self._send_text_content(text)
            if sent:
                replayed += 1
                continue

            async with self._pending_text_lock:
                self._pending_text_queue.appendleft(text)
                metrics.gauge_set(
                    "voice.realtime.pending_text.queue_size",
                    len(self._pending_text_queue),
                    labels={"backend": "gemini"},
                )
                if replayed:
                    metrics.inc(
                        "voice.realtime.pending_text.replayed",
                        value=replayed,
                        labels={"backend": "gemini"},
                    )
            return

    async def _send_text_content(self, text: str) -> bool:
        if self._session is None:
            return False
        try:
            await self._session.send_client_content(
                turns=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=text)],
                ),
                turn_complete=True,
            )
            return True
        except Exception:
            logger.exception("Failed to send text content")
            return False


def _pick(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _decode_inline_audio(inline_data: Any) -> bytes | None:
    if inline_data is None:
        return None

    mime = _pick(inline_data, "mime_type")
    if isinstance(mime, str) and "audio" not in mime.lower():
        return None

    raw = _pick(inline_data, "data")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str):
        try:
            return base64.b64decode(raw)
        except Exception:
            logger.debug("Failed to decode inline audio chunk", exc_info=True)
            return None
    return None


def _stringify_tool_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    if args is None:
        return "{}"
    try:
        return json.dumps(args)
    except Exception:
        return "{}"


def _parse_json_or_text(value: str) -> dict[str, Any]:
    if not value:
        return {"result": ""}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"result": value}

    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def _backoff_delay_seconds(attempt: int) -> float:
    power = max(0, attempt - 1)
    delay = RECONNECT_BACKOFF_BASE_SECONDS * (2**power)
    return min(delay, RECONNECT_BACKOFF_MAX_SECONDS)
