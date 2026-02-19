"""Focused tests for voice-session chunking and STT gating."""

import asyncio
import time

import pytest

from aether.voice.session import (
    _STATE_IDLE_VAD_ONLY,
    _STATE_THINKING,
    _STATE_STT_STREAMING,
    _extract_tts_chunks,
    VoiceSession,
)


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
