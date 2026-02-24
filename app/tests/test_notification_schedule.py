from __future__ import annotations

from datetime import datetime

from aether.agent import AgentCore


def _ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute).timestamp()


def test_resolve_morning_hint_targets_next_morning() -> None:
    now_ts = _ts(2026, 2, 24, 21, 30)
    deliver_at = AgentCore._resolve_notification_deliver_at(
        "morning",
        delivery_type="surface",
        now_ts=now_ts,
    )

    assert deliver_at is not None
    dt = datetime.fromtimestamp(deliver_at)
    assert dt.hour == 8
    assert dt.minute == 0


def test_resolve_before_time_hint_parses_clock_time() -> None:
    now_ts = _ts(2026, 2, 24, 10, 0)
    deliver_at = AgentCore._resolve_notification_deliver_at(
        "before 5pm",
        delivery_type="surface",
        now_ts=now_ts,
    )

    assert deliver_at is not None
    dt = datetime.fromtimestamp(deliver_at)
    assert dt.hour == 17
    assert dt.minute == 0


def test_resolve_relative_hint_in_minutes() -> None:
    now_ts = _ts(2026, 2, 24, 10, 0)
    deliver_at = AgentCore._resolve_notification_deliver_at(
        "in 30 minutes",
        delivery_type="nudge",
        now_ts=now_ts,
    )

    assert deliver_at is not None
    assert int(deliver_at - now_ts) == 1800


def test_quiet_hours_defer_non_interrupt_immediate() -> None:
    now_ts = _ts(2026, 2, 24, 23, 0)
    deliver_at = AgentCore._resolve_notification_deliver_at(
        "immediately",
        delivery_type="surface",
        now_ts=now_ts,
    )

    assert deliver_at is not None
    dt = datetime.fromtimestamp(deliver_at)
    assert dt.hour == 8


def test_interrupt_ignores_quiet_hour_deferral() -> None:
    now_ts = _ts(2026, 2, 24, 23, 0)
    deliver_at = AgentCore._resolve_notification_deliver_at(
        "immediately",
        delivery_type="interrupt",
        now_ts=now_ts,
    )

    assert deliver_at is None


def test_delivery_latency_ms_from_created_timestamp() -> None:
    now_ts = _ts(2026, 2, 24, 10, 0)
    created_at = now_ts - 2.5

    latency_ms = AgentCore._delivery_latency_ms(created_at, now_ts=now_ts)
    assert latency_ms == 2500.0


def test_delivery_latency_ms_handles_invalid_created_at() -> None:
    assert AgentCore._delivery_latency_ms("not-a-timestamp", now_ts=0) is None
