"""Focused tests for voice-session chunking and STT gating."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from aether.voice.session import (
    _STATE_IDLE_VAD_ONLY,
    _STATE_THINKING,
    _STATE_STT_STREAMING,
    _extract_tts_chunks,
    VoiceSession,
)
import aether.voice.session as voice_session_module


class _DummyAgent:
    def __init__(self) -> None:
        self.cancel_calls = 0

    async def generate_greeting(
        self, session_id: str | None = None, is_resume: bool = False
    ):
        return None

    async def cancel_session(self, _session_id: str) -> None:
        self.cancel_calls += 1
        return None

    async def generate_reply_voice(self, _text: str, _session_id: str):
        if False:
            yield None


class _DummyTTS:
    async def synthesize_stream(self, _text: str):
        if False:
            yield b""


class _ChunkTTS:
    async def synthesize_stream(self, _text: str):
        yield b"\x00\x00" * 160


class _FakeSTT:
    def __init__(self, is_open: bool) -> None:
        self.is_open = is_open
        self.sent: list[bytes] = []

    async def send_audio(self, chunk: bytes) -> None:
        self.sent.append(chunk)


class _LifecycleSTT:
    def __init__(self) -> None:
        self.is_open = False
        self.started = False
        self.start_calls = 0
        self.connect_calls = 0
        self.disconnect_calls = 0

    async def start(self) -> None:
        self.started = True
        self.start_calls += 1

    async def stop(self) -> None:
        self.started = False
        self.is_open = False

    async def connect_stream(self) -> None:
        if not self.started:
            raise RuntimeError("Deepgram STT not started")
        self.connect_calls += 1
        self.is_open = True

    async def disconnect_stream(self) -> None:
        self.disconnect_calls += 1
        self.is_open = False


class _OpenSTT:
    def __init__(self, is_open: bool = True) -> None:
        self.is_open = is_open


def test_extract_tts_chunks_prefers_pause_boundaries() -> None:
    buffer = ("a" * 70) + ", continue speaking without sentence splitting"
    pieces, remaining = _extract_tts_chunks(buffer)

    assert len(pieces) == 1
    assert pieces[0].endswith(",")
    assert remaining.startswith("continue")


def test_extract_tts_chunks_falls_back_to_size_boundaries() -> None:
    buffer = "word " * 45
    pieces, remaining = _extract_tts_chunks(buffer)

    assert pieces
    assert len(pieces[0]) <= 140
    assert remaining


@pytest.mark.asyncio
async def test_on_audio_in_skips_when_stt_not_open() -> None:
    session = VoiceSession(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="test-session",
    )  # type: ignore[arg-type]
    fake_stt = _FakeSTT(is_open=False)
    session.stt = fake_stt  # type: ignore[assignment]
    session.is_streaming = True
    session._stt_connected = True
    session._state = _STATE_STT_STREAMING

    await session.on_audio_in(b"\x00\x00" * 320)

    assert not fake_stt.sent

    fake_stt.is_open = True
    await session.on_audio_in(b"\x00\x00" * 320)

    assert len(fake_stt.sent) == 1


@pytest.mark.asyncio
async def test_barge_in_is_idempotent() -> None:
    agent = _DummyAgent()
    session = VoiceSession(
        agent=agent,  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="test-session",
    )  # type: ignore[arg-type]
    session.is_streaming = True
    session._assistant_speaking_until = time.time() + 10
    on_barge_calls = 0

    async def _on_barge() -> None:
        nonlocal on_barge_calls
        on_barge_calls += 1

    session.on_barge_in = _on_barge

    task = asyncio.create_task(asyncio.sleep(10))
    session._response_task = task
    session.is_responding = True

    await session._handle_barge_in("vad", probability=0.9)
    await session._handle_barge_in("vad", probability=0.9)
    await asyncio.sleep(0)

    assert on_barge_calls == 1
    assert agent.cancel_calls == 1
    assert session._state == _STATE_IDLE_VAD_ONLY
    assert session.is_responding is False


@pytest.mark.asyncio
async def test_stale_generation_skips_tts_output() -> None:
    session = VoiceSession(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        tts_provider=_ChunkTTS(),  # type: ignore[arg-type]
        session_id="test-session",
    )  # type: ignore[arg-type]
    sent_audio: list[bytes] = []

    async def _on_audio(chunk: bytes) -> None:
        sent_audio.append(chunk)

    session.on_audio_out = _on_audio
    session._turn_generation = 5

    await session._synthesize_and_send("hello", generation=4)

    assert sent_audio == []


@pytest.mark.asyncio
async def test_watchdog_recovers_stuck_thinking_state() -> None:
    agent = _DummyAgent()
    session = VoiceSession(
        agent=agent,  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="test-session",
    )  # type: ignore[arg-type]
    session.is_streaming = True
    session.is_responding = True
    session._state = _STATE_THINKING
    session._response_task = None
    session._last_state_progress_at = time.monotonic() - 20

    watchdog = asyncio.create_task(session._state_watchdog_loop())
    await asyncio.sleep(0.7)
    session.is_streaming = False
    watchdog.cancel()
    try:
        await watchdog
    except asyncio.CancelledError:
        pass

    assert session._state == _STATE_IDLE_VAD_ONLY
    assert session.is_responding is False
    assert agent.cancel_calls >= 1


@pytest.mark.asyncio
async def test_resume_restarts_stt_event_loop_after_pause() -> None:
    session = VoiceSession(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="resume-test",
    )  # type: ignore[arg-type]
    fake_stt = _LifecycleSTT()
    session.stt = fake_stt  # type: ignore[assignment]

    # Simulate a previously started and then paused session.
    session.is_streaming = True
    session._stt_connected = True
    session._stt_event_task = asyncio.create_task(asyncio.sleep(10))
    await session.pause()
    assert session._stt_event_task is None

    await session.resume()

    assert fake_stt.start_calls >= 1
    assert session._stt_event_task is not None
    assert not session._stt_event_task.done()

    # Cleanup created task to avoid leakage.
    session.is_streaming = False
    task = session._stt_event_task
    if task and not task.done():
        task.cancel()


@pytest.mark.asyncio
async def test_response_complete_returns_to_stt_streaming_when_stt_open() -> None:
    session = VoiceSession(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="response-state-test",
    )  # type: ignore[arg-type]
    session.is_streaming = True
    session._stt_connected = True
    session.stt = _OpenSTT(is_open=True)  # type: ignore[assignment]

    await session._trigger_response("hello")

    assert session._state == _STATE_STT_STREAMING


@pytest.mark.asyncio
async def test_vad_off_does_not_schedule_idle_stt_disconnect(monkeypatch) -> None:
    session = VoiceSession(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="vad-off-disconnect-test",
    )  # type: ignore[arg-type]
    session.is_streaming = True
    session._stt_connected = True
    session.stt = _OpenSTT(is_open=True)  # type: ignore[assignment]

    called = False

    def _schedule_disconnect() -> None:
        nonlocal called
        called = True

    session._schedule_stt_disconnect = _schedule_disconnect  # type: ignore[assignment]
    monkeypatch.setattr(
        voice_session_module,
        "config",
        SimpleNamespace(vad=SimpleNamespace(mode="off")),
    )

    await session._trigger_response("hello")

    assert called is False


@pytest.mark.asyncio
async def test_vad_active_schedules_idle_stt_disconnect(monkeypatch) -> None:
    session = VoiceSession(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        tts_provider=_DummyTTS(),  # type: ignore[arg-type]
        session_id="vad-active-disconnect-test",
    )  # type: ignore[arg-type]
    session.is_streaming = True
    session._stt_connected = True
    session.stt = _OpenSTT(is_open=True)  # type: ignore[assignment]

    called = False

    def _schedule_disconnect() -> None:
        nonlocal called
        called = True

    session._schedule_stt_disconnect = _schedule_disconnect  # type: ignore[assignment]
    monkeypatch.setattr(
        voice_session_module,
        "config",
        SimpleNamespace(vad=SimpleNamespace(mode="active")),
    )

    await session._trigger_response("hello")

    assert called is True
