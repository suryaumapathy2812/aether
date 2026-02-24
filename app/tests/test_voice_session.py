"""VoiceSession tests for realtime AudioIO orchestration."""

from __future__ import annotations

import asyncio
import json

import pytest

from aether.voice.io import (
    AudioFrame,
    AudioInput,
    AudioInputEvent,
    AudioOutput,
    TextOutput,
)
from aether.voice.realtime import (
    FunctionCallResult,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeModel,
    RealtimeModelConfig,
    RealtimeSession,
)
from aether.voice.session import VoiceSession


class _FakeAudioInput(AudioInput):
    def __init__(self, frames: list[AudioFrame] | None = None) -> None:
        self.frames = frames or []
        self.closed = False

    async def __aiter__(self):
        for frame in self.frames:
            yield frame

    async def close(self) -> None:
        self.closed = True


class _FakeAudioOutput(AudioOutput):
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.clear_calls = 0
        self.closed = False

    async def push_frame(self, frame: AudioFrame) -> None:
        self.frames.append(frame)

    async def clear(self) -> None:
        self.clear_calls += 1

    async def close(self) -> None:
        self.closed = True


class _FakeTextOutput(TextOutput):
    def __init__(self) -> None:
        self.text: list[tuple[str, bool]] = []
        self.states: list[str] = []

    async def push_text(self, text: str, *, final: bool = False) -> None:
        self.text.append((text, final))

    async def push_state(self, state: str) -> None:
        self.states.append(state)


class _FakeRealtimeSession(RealtimeSession):
    def __init__(self, events: list[RealtimeEvent] | None = None) -> None:
        self._events = events or []
        self._closed = False
        self._connected = True
        self._resumption = "resume-token"
        self.pushed_audio: list[bytes] = []
        self.generated_replies: list[str] = []
        self.function_results: list[FunctionCallResult] = []

    async def push_audio(self, data: bytes) -> None:
        self.pushed_audio.append(data)

    def events(self):
        async def _gen():
            for event in self._events:
                yield event

        return _gen()

    async def send_function_result(self, result: FunctionCallResult) -> None:
        self.function_results.append(result)

    async def interrupt(self) -> None:
        return None

    async def generate_reply(self, text: str) -> None:
        self.generated_replies.append(text)

    async def update_instructions(self, instructions: str) -> None:
        return None

    async def update_tools(self, tools: list[dict[str, object]]) -> None:
        return None

    async def close(self) -> None:
        self._closed = True
        self._connected = False

    @property
    def session_id(self) -> str:
        return "fake-session"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def resumption_token(self) -> str | None:
        return self._resumption


class _FakeRealtimeModel(RealtimeModel):
    def __init__(self, session: _FakeRealtimeSession) -> None:
        self._session = session
        self.create_calls: list[
            tuple[str, list[dict[str, object]] | None, str | None]
        ] = []

    async def create_session(
        self,
        *,
        instructions: str = "",
        tools: list[dict[str, object]] | None = None,
        resumption_token: str | None = None,
    ) -> RealtimeSession:
        self.create_calls.append((instructions, tools, resumption_token))
        return self._session

    async def close(self) -> None:
        return None

    @property
    def config(self) -> RealtimeModelConfig:
        return RealtimeModelConfig(model="fake")


class _FakeSessionStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def ensure_session(self, session_id: str) -> None:
        self.calls.append(("ensure", session_id, ""))

    async def append_user_message(self, session_id: str, text: str) -> None:
        self.calls.append(("user", session_id, text))

    async def append_assistant_message(self, session_id: str, text: str) -> None:
        self.calls.append(("assistant", session_id, text))


class _FakeTaskLedger:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    async def submit(
        self,
        *,
        session_id: str,
        task_type: str,
        payload: dict[str, object],
        priority: str,
    ) -> str:
        self.submissions.append(
            {
                "session_id": session_id,
                "task_type": task_type,
                "payload": payload,
                "priority": priority,
            }
        )
        return "task-1"


@pytest.mark.asyncio
async def test_start_routes_audio_and_realtime_events() -> None:
    events = [
        RealtimeEvent(
            type=RealtimeEventType.AUDIO_DELTA,
            data={"audio": b"\x00\x00" * 10, "sample_rate": 24000},
        ),
        RealtimeEvent(type=RealtimeEventType.TEXT_DELTA, data={"text": "hello"}),
        RealtimeEvent(type=RealtimeEventType.TEXT_DONE),
        RealtimeEvent(type=RealtimeEventType.TURN_DONE),
    ]
    rt_session = _FakeRealtimeSession(events)
    model = _FakeRealtimeModel(rt_session)
    audio_in = _FakeAudioInput(
        [AudioFrame(data=b"\x01\x02" * 8, sample_rate=16000, samples_per_channel=8)]
    )
    audio_out = _FakeAudioOutput()
    text_out = _FakeTextOutput()

    session = VoiceSession(session_id="s1", realtime_model=model)
    session.set_io(audio_in, audio_out, text_out)
    await session.start()
    await asyncio.sleep(0.05)
    await session.stop()

    assert model.create_calls
    assert rt_session.pushed_audio == [b"\x01\x02" * 8]
    assert len(audio_out.frames) == 1
    assert text_out.text[0] == ("hello", False)
    assert text_out.text[-1] == ("", True)


@pytest.mark.asyncio
async def test_input_speech_started_clears_output() -> None:
    rt_session = _FakeRealtimeSession(
        [RealtimeEvent(type=RealtimeEventType.INPUT_SPEECH_STARTED)]
    )
    model = _FakeRealtimeModel(rt_session)
    audio_in = _FakeAudioInput()
    audio_out = _FakeAudioOutput()
    text_out = _FakeTextOutput()

    session = VoiceSession(session_id="s2", realtime_model=model)
    session.set_io(audio_in, audio_out, text_out)
    await session.start()
    await asyncio.sleep(0.05)
    await session.stop()

    assert audio_out.clear_calls == 1
    assert "listening" in text_out.states


@pytest.mark.asyncio
async def test_function_call_dispatches_and_returns_result() -> None:
    async def _on_function_call(call_id: str, name: str, arguments: str) -> str:
        parsed = json.loads(arguments)
        return json.dumps(
            {"ok": True, "id": call_id, "name": name, "x": parsed.get("x")}
        )

    rt_session = _FakeRealtimeSession(
        [
            RealtimeEvent(
                type=RealtimeEventType.FUNCTION_CALL_DONE,
                data={"call_id": "c1", "name": "tool_x", "arguments": '{"x":1}'},
            )
        ]
    )
    model = _FakeRealtimeModel(rt_session)

    session = VoiceSession(
        session_id="s3",
        realtime_model=model,
        on_function_call=_on_function_call,
    )
    session.set_io(_FakeAudioInput(), _FakeAudioOutput(), _FakeTextOutput())
    await session.start()
    await asyncio.sleep(0.05)
    await session.stop()

    assert len(rt_session.function_results) == 1
    result = rt_session.function_results[0]
    assert result.call_id == "c1"
    assert result.name == "tool_x"
    assert json.loads(result.result)["ok"] is True


@pytest.mark.asyncio
async def test_inject_text_forwards_to_realtime_session() -> None:
    rt_session = _FakeRealtimeSession([])
    model = _FakeRealtimeModel(rt_session)
    text_out = _FakeTextOutput()

    session = VoiceSession(session_id="s4", realtime_model=model)
    session.set_io(_FakeAudioInput(), _FakeAudioOutput(), text_out)
    await session.start()
    await session.inject_text("hello from text")
    await session.stop()

    assert rt_session.generated_replies == ["hello from text"]
    assert "thinking" in text_out.states


@pytest.mark.asyncio
async def test_turn_done_syncs_transcript_and_memory_task() -> None:
    store = _FakeSessionStore()
    ledger = _FakeTaskLedger()
    rt_session = _FakeRealtimeSession(
        [
            RealtimeEvent(
                type=RealtimeEventType.INPUT_SPEECH_TRANSCRIPTION_COMPLETED,
                data={"transcript": "hello"},
            ),
            RealtimeEvent(type=RealtimeEventType.TEXT_DELTA, data={"text": "world"}),
            RealtimeEvent(type=RealtimeEventType.TURN_DONE),
        ]
    )
    model = _FakeRealtimeModel(rt_session)

    session = VoiceSession(
        session_id="s5",
        realtime_model=model,
        session_store=store,  # type: ignore[arg-type]
        task_ledger=ledger,  # type: ignore[arg-type]
    )
    session.set_io(_FakeAudioInput(), _FakeAudioOutput(), _FakeTextOutput())
    await session.start()
    await asyncio.sleep(0.05)
    await session.stop()

    assert ("user", "s5", "hello") in store.calls
    assert ("assistant", "s5", "world") in store.calls
    assert ledger.submissions
    assert ledger.submissions[0]["task_type"] == "memory_extract"


@pytest.mark.asyncio
async def test_stop_closes_io_and_session_and_keeps_resumption() -> None:
    rt_session = _FakeRealtimeSession([])
    model = _FakeRealtimeModel(rt_session)
    audio_in = _FakeAudioInput()
    audio_out = _FakeAudioOutput()

    session = VoiceSession(session_id="s6", realtime_model=model)
    session.set_io(audio_in, audio_out, _FakeTextOutput())
    await session.start()
    await session.stop()

    assert audio_in.closed is True
    assert audio_out.closed is True
    assert rt_session.is_connected is False
    assert session._resumption_token == "resume-token"
