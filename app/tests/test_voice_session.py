"""Focused tests for voice-session chunking and STT gating."""

import pytest

from aether.voice.session import (
    _STATE_STT_STREAMING,
    _extract_tts_chunks,
    VoiceSession,
)


class _DummyAgent:
    async def generate_greeting(self):
        return None

    async def cancel_session(self, _session_id: str) -> None:
        return None


class _DummyTTS:
    async def synthesize_stream(self, _text: str):
        if False:
            yield b""


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
        agent=_DummyAgent(),
        tts_provider=_DummyTTS(),
        session_id="test-session",
    )
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
