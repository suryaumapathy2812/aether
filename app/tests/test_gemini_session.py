from __future__ import annotations

import pytest

from aether.voice.backends.gemini.session import (
    GeminiRealtimeSession,
    _is_retryable_session_error,
)


def test_retryable_session_error_detects_deadline() -> None:
    exc = RuntimeError("1011 Deadline expired before operation could complete")
    assert _is_retryable_session_error(exc) is True


def test_retryable_session_error_rejects_invalid_argument() -> None:
    exc = RuntimeError("400 INVALID_ARGUMENT bad model")
    assert _is_retryable_session_error(exc) is False


@pytest.mark.asyncio
async def test_drop_connection_records_degraded_diagnostics() -> None:
    session = GeminiRealtimeSession(
        client=object(),
        model="gemini-test",
        live_config=object(),
        input_sample_rate=16000,
        output_sample_rate=24000,
    )

    await session._drop_connection(  # noqa: SLF001
        RuntimeError("deadline expired"),
        source="test",
        retry_attempt=1,
        recoverable=True,
    )

    diagnostics = session.diagnostics()
    assert diagnostics["state"] == "degraded"
    assert diagnostics["last_error_source"] == "test"
    assert "deadline expired" in str(diagnostics["last_error"])
