"""VoiceSession orchestrates RealtimeSession + AudioIO for one connection."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from aether.core.metrics import metrics
from aether.session.models import TaskType
from aether.voice.io import (
    AudioFrame,
    AudioInput,
    AudioInputEvent,
    AudioOutput,
    NullTextOutput,
    TextOutput,
)
from aether.voice.realtime import (
    FunctionCallResult,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeModel,
    RealtimeSession,
)

if TYPE_CHECKING:
    from aether.session.ledger import TaskLedger
    from aether.session.store import SessionStore

logger = logging.getLogger(__name__)

_MAX_PENDING_AUDIO_CHUNKS = 256
_MAX_PENDING_TEXT_INJECTIONS = 32


class VoiceSession:
    """Thin orchestrator over a realtime model session."""

    def __init__(
        self,
        *,
        session_id: str,
        realtime_model: RealtimeModel,
        on_function_call: Callable[[str, str, str], Awaitable[str]] | None = None,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        session_store: SessionStore | None = None,
        task_ledger: TaskLedger | None = None,
    ) -> None:
        self.session_id = session_id
        self._model = realtime_model
        self._on_function_call = on_function_call
        self._instructions = instructions
        self._tools = tools
        self._session_store = session_store
        self._task_ledger = task_ledger

        self._session: RealtimeSession | None = None
        self._resumption_token: str | None = None

        self._audio_input: AudioInput | None = None
        self._audio_output: AudioOutput | None = None
        self._text_output: TextOutput = NullTextOutput()

        self._input_task: asyncio.Task[None] | None = None
        self._event_task: asyncio.Task[None] | None = None

        self._pending_user_text: str = ""
        self._assistant_parts: list[str] = []
        self._pending_audio_chunks: deque[bytes] = deque()
        self._pending_text_injections: deque[str] = deque()
        self._degraded = False
        self._replay_lock = asyncio.Lock()

        self.is_streaming = False
        self.is_muted = False
        self._stopped = True

    def set_io(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        text_output: TextOutput | None = None,
    ) -> None:
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._text_output = text_output or NullTextOutput()

    async def start(self) -> None:
        if self.is_streaming:
            return
        if self._audio_input is None or self._audio_output is None:
            raise RuntimeError("set_io() must be called before start()")

        self._session = await self._model.create_session(
            instructions=self._instructions,
            tools=self._tools,
            resumption_token=self._resumption_token,
        )
        self.is_streaming = True
        self._stopped = False
        self._input_task = asyncio.create_task(self._audio_input_loop())
        self._event_task = asyncio.create_task(self._event_loop())
        await self._text_output.push_state("listening")

    async def stop(self) -> None:
        self._stopped = True
        self.is_streaming = False
        await self._cancel_tasks()
        await self._flush_transcript_on_pause_or_stop()
        await self._close_session()
        await self._close_io()

    async def pause(self) -> None:
        self.is_streaming = False
        await self._cancel_tasks()
        await self._flush_transcript_on_pause_or_stop()
        await self._close_session()
        await self._close_io()

    async def resume(self) -> None:
        await self.start()

    async def inject_text(self, text: str) -> None:
        clean = text.strip()
        if not clean or self._session is None:
            return
        if not self._is_session_connected():
            self._buffer_pending_text(clean)
            await self._mark_reconnecting_if_needed()
            return
        await self._recover_and_replay_if_connected()
        self._pending_user_text = clean
        await self._text_output.push_state("thinking")
        await self._session.generate_reply(clean)

    async def deliver_notification(self, text: str) -> None:
        await self.inject_text(text)

    def mute(self) -> None:
        self.is_muted = True

    def unmute(self) -> None:
        self.is_muted = False

    async def _audio_input_loop(self) -> None:
        assert self._audio_input is not None
        while self.is_streaming:
            try:
                async for frame in self._audio_input:
                    if not self.is_streaming:
                        return
                    if frame.event == AudioInputEvent.START_OF_SPEECH:
                        await self._text_output.push_state("listening")
                    if self.is_muted:
                        continue
                    if frame.data and self._session is not None:
                        if self._is_session_connected():
                            await self._recover_and_replay_if_connected()
                            await self._session.push_audio(frame.data)
                        else:
                            self._buffer_pending_audio(frame.data)
                            await self._mark_reconnecting_if_needed()
                return
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Audio input loop error (%s)", self.session_id)
                await asyncio.sleep(0.1)

    async def _event_loop(self) -> None:
        if self._session is None:
            return
        try:
            async for event in self._session.events():
                if not self.is_streaming:
                    break
                await self._handle_event(event)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Realtime event loop error (%s)", self.session_id)

    async def _handle_event(self, event: RealtimeEvent) -> None:
        if self._audio_output is None:
            return

        if event.type == RealtimeEventType.AUDIO_DELTA:
            audio = event.data.get("audio", b"")
            if isinstance(audio, bytes) and audio:
                await self._audio_output.push_frame(
                    AudioFrame(
                        data=audio,
                        sample_rate=int(event.data.get("sample_rate", 24000)),
                        samples_per_channel=len(audio) // 2,
                    )
                )
            return

        if event.type == RealtimeEventType.TEXT_DELTA:
            text = str(event.data.get("text", ""))
            if text:
                self._assistant_parts.append(text)
                await self._text_output.push_text(text)
            return

        if event.type in {
            RealtimeEventType.SESSION_CREATED,
            RealtimeEventType.SESSION_RESUMED,
        }:
            await self._mark_recovered_if_needed()
            await self._replay_pending_if_connected()
            return

        if event.type == RealtimeEventType.TEXT_DONE:
            await self._text_output.push_text("", final=True)
            await self._sync_transcript_best_effort(event)
            self._assistant_parts.clear()
            self._pending_user_text = ""
            return

        if event.type == RealtimeEventType.INPUT_SPEECH_STARTED:
            await self._audio_output.clear()
            await self._text_output.push_state("listening")
            return

        if event.type == RealtimeEventType.INPUT_SPEECH_COMMITTED:
            await self._text_output.push_state("thinking")
            return

        if event.type == RealtimeEventType.INPUT_SPEECH_TRANSCRIPTION_COMPLETED:
            transcript = str(event.data.get("transcript", "")).strip()
            if transcript:
                self._pending_user_text = transcript
            return

        if event.type in {
            RealtimeEventType.FUNCTION_CALL,
            RealtimeEventType.FUNCTION_CALL_DONE,
        }:
            await self._handle_function_call(event)
            return

        if event.type == RealtimeEventType.TURN_STARTED:
            await self._text_output.push_state("speaking")
            return

        if event.type == RealtimeEventType.INTERRUPTED:
            await self._audio_output.clear()
            return

        if event.type == RealtimeEventType.TURN_DONE:
            await self._text_output.push_state("idle")
            await self._sync_transcript_best_effort(event)
            self._assistant_parts.clear()
            self._pending_user_text = ""
            return

        if event.type == RealtimeEventType.SESSION_ERROR:
            logger.error("Realtime session error (%s): %s", self.session_id, event.data)
            await self._mark_reconnecting_if_needed()

    def _is_session_connected(self) -> bool:
        return self._session is not None and self._session.is_connected

    def _buffer_pending_audio(self, data: bytes) -> None:
        if len(self._pending_audio_chunks) >= _MAX_PENDING_AUDIO_CHUNKS:
            self._pending_audio_chunks.popleft()
            metrics.inc("voice.session.pending_audio.drop_oldest")
            logger.warning(
                "Pending audio buffer overflow; dropped oldest chunk (%s)",
                self.session_id,
            )
        self._pending_audio_chunks.append(data)
        metrics.gauge_set(
            "voice.session.pending_audio.queue_size",
            len(self._pending_audio_chunks),
        )

    def _buffer_pending_text(self, text: str) -> None:
        if len(self._pending_text_injections) >= _MAX_PENDING_TEXT_INJECTIONS:
            self._pending_text_injections.popleft()
            metrics.inc("voice.session.pending_text.drop_oldest")
            logger.warning(
                "Pending text buffer overflow; dropped oldest injection (%s)",
                self.session_id,
            )
        self._pending_text_injections.append(text)
        metrics.gauge_set(
            "voice.session.pending_text.queue_size",
            len(self._pending_text_injections),
        )

    async def _mark_reconnecting_if_needed(self) -> None:
        if self._degraded:
            return
        self._degraded = True
        metrics.inc("voice.session.state.reconnecting")
        await self._text_output.push_state("reconnecting")

    async def _mark_recovered_if_needed(self) -> None:
        if not self._degraded:
            return
        self._degraded = False
        metrics.inc("voice.session.state.recovered")
        await self._text_output.push_state("recovered")

    async def _replay_pending_if_connected(self) -> None:
        session = self._session
        if session is None or not session.is_connected:
            return

        async with self._replay_lock:
            session = self._session
            if session is None or not session.is_connected:
                return

            while self._pending_text_injections and session.is_connected:
                text = self._pending_text_injections.popleft()
                self._pending_user_text = text
                await self._text_output.push_state("thinking")
                await session.generate_reply(text)

            metrics.gauge_set(
                "voice.session.pending_text.queue_size",
                len(self._pending_text_injections),
            )

            while self._pending_audio_chunks and session.is_connected:
                audio = self._pending_audio_chunks.popleft()
                await session.push_audio(audio)

            metrics.gauge_set(
                "voice.session.pending_audio.queue_size",
                len(self._pending_audio_chunks),
            )

    async def _recover_and_replay_if_connected(self) -> None:
        if not self._is_session_connected():
            return
        await self._mark_recovered_if_needed()
        if self._pending_text_injections or self._pending_audio_chunks:
            await self._replay_pending_if_connected()

    async def _handle_function_call(self, event: RealtimeEvent) -> None:
        if self._session is None:
            return

        call_id = str(event.data.get("call_id", ""))
        name = str(event.data.get("name", ""))
        args = event.data.get("arguments", "")
        if not call_id or not name:
            return

        if isinstance(args, str):
            arguments = args
        else:
            arguments = json.dumps(args)

        if self._on_function_call is None:
            result = json.dumps({"error": "no function handler registered"})
        else:
            try:
                result = await self._on_function_call(call_id, name, arguments)
            except Exception as exc:
                logger.exception("Function call failed (%s:%s)", name, call_id)
                result = json.dumps({"error": str(exc)})

        await self._session.send_function_result(
            FunctionCallResult(call_id=call_id, name=name, result=result)
        )

    async def _sync_transcript_best_effort(self, event: RealtimeEvent) -> None:
        user_text = self._pending_user_text.strip()
        assistant_text = "".join(self._assistant_parts).strip()
        if not assistant_text:
            assistant_text = str(event.data.get("assistant_text", "")).strip()
        if not user_text:
            user_text = str(event.data.get("user_text", "")).strip()

        logger.debug(
            "Transcript sync attempt (%s): user=%r, assistant=%r (len=%d)",
            self.session_id,
            user_text[:80] if user_text else "",
            assistant_text[:80] if assistant_text else "",
            len(self._assistant_parts),
        )

        if not user_text and not assistant_text:
            return

        try:
            if self._session_store is not None:
                await self._session_store.ensure_session(self.session_id)
                if user_text:
                    await self._session_store.append_user_message(
                        self.session_id, user_text
                    )
                if assistant_text:
                    await self._session_store.append_assistant_message(
                        self.session_id, assistant_text
                    )

            if self._task_ledger is not None and user_text and assistant_text:
                await self._task_ledger.submit(
                    session_id=self.session_id,
                    task_type=TaskType.MEMORY_EXTRACT.value,
                    payload={
                        "user_message": user_text,
                        "assistant_message": assistant_text,
                    },
                    priority="low",
                )
        except Exception:
            logger.warning(
                "Transcript sync failed (%s)", self.session_id, exc_info=True
            )

    async def _flush_transcript_on_pause_or_stop(self) -> None:
        if (
            not self._pending_user_text.strip()
            and not "".join(self._assistant_parts).strip()
        ):
            return
        await self._sync_transcript_best_effort(
            RealtimeEvent(type=RealtimeEventType.TEXT_DONE)
        )
        self._assistant_parts.clear()
        self._pending_user_text = ""

    async def _close_session(self) -> None:
        if self._session is None:
            return
        try:
            self._resumption_token = self._session.resumption_token
        except Exception:
            self._resumption_token = None
        try:
            await self._session.close()
        finally:
            self._session = None

    async def _cancel_tasks(self) -> None:
        tasks = [self._input_task, self._event_task]
        self._input_task = None
        self._event_task = None
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

    async def _close_io(self) -> None:
        if self._audio_input is not None:
            try:
                await self._audio_input.close()
            except Exception:
                pass
        if self._audio_output is not None:
            try:
                await self._audio_output.close()
            except Exception:
                pass
