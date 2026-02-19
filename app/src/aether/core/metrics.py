"""
Aether Metrics — in-process metrics collector.

No external dependencies. Prometheus export can wrap this later.

Features:
- Counters (monotonically increasing)
- Histograms (rolling window, p50/p95/p99 on read)
- Gauges (current value, inc/dec/set)
- Label support (key=value pairs appended to metric name)
- Thread-safe snapshot for /health and /metrics endpoints

Usage:
    from aether.core.metrics import metrics

    metrics.inc("kernel.jobs.submitted", labels={"kind": "reply_text"})
    metrics.observe("llm.ttft_ms", 342.1, labels={"kind": "reply_text"})
    metrics.gauge_set("kernel.queue.interactive", 3)

    snapshot = metrics.snapshot()  # -> dict for JSON response
"""

from __future__ import annotations

import time
from collections import defaultdict


class MetricsCollector:
    """In-process metrics collector — counters, histograms, gauges."""

    # Rolling window size for histograms — keeps memory bounded
    HISTOGRAM_MAX_SAMPLES = 1000

    _instance: "MetricsCollector | None" = None

    @classmethod
    def get(cls) -> "MetricsCollector":
        """Return the process-wide singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = defaultdict(float)
        self._started_at: float = time.time()

    # ── Counters ──────────────────────────────────────────────────

    def inc(self, name: str, value: int = 1, labels: dict | None = None) -> None:
        """Increment a counter."""
        self._counters[self._key(name, labels)] += value

    # ── Histograms ────────────────────────────────────────────────

    def observe(self, name: str, value: float, labels: dict | None = None) -> None:
        """Record a single observation (e.g. latency in ms).

        Maintains a rolling window of HISTOGRAM_MAX_SAMPLES values.
        Oldest sample is dropped when the window is full.
        """
        key = self._key(name, labels)
        samples = self._histograms[key]
        samples.append(value)
        if len(samples) > self.HISTOGRAM_MAX_SAMPLES:
            samples.pop(0)

    # ── Gauges ────────────────────────────────────────────────────

    def gauge_set(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set a gauge to an absolute value."""
        self._gauges[self._key(name, labels)] = value

    def gauge_inc(
        self, name: str, value: float = 1.0, labels: dict | None = None
    ) -> None:
        """Increment a gauge."""
        self._gauges[self._key(name, labels)] += value

    def gauge_dec(
        self, name: str, value: float = 1.0, labels: dict | None = None
    ) -> None:
        """Decrement a gauge."""
        self._gauges[self._key(name, labels)] -= value

    # ── Percentiles ───────────────────────────────────────────────

    def percentile(
        self, name: str, p: float, labels: dict | None = None
    ) -> float | None:
        """Compute a percentile (0-100) over recorded observations.

        When labels are provided, matches the exact key.
        When labels are None, aggregates across ALL label variants of the metric
        (e.g. "kernel.job.duration_ms" matches "kernel.job.duration_ms{kind=reply_voice}"
        and "kernel.job.duration_ms{kind=reply_text}").

        Returns None if no samples exist yet.
        """
        if labels is not None:
            # Exact key match
            key = self._key(name, labels)
            samples = self._histograms.get(key)
            if not samples:
                return None
            sorted_samples = sorted(samples)
            idx = min(int(len(sorted_samples) * p / 100), len(sorted_samples) - 1)
            return sorted_samples[idx]

        # Aggregate across all label variants
        all_samples: list[float] = []
        prefix = name + "{"
        for key, samples in self._histograms.items():
            if key == name or key.startswith(prefix):
                all_samples.extend(samples)
        if not all_samples:
            return None
        sorted_samples = sorted(all_samples)
        idx = min(int(len(sorted_samples) * p / 100), len(sorted_samples) - 1)
        return sorted_samples[idx]

    # ── Snapshot ──────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full metrics snapshot — suitable for JSON response.

        Returns counters, gauges, and histogram summaries (p50/p95/p99/min/max/count).
        """
        histograms: dict[str, dict] = {}
        for key, samples in self._histograms.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            histograms[key] = {
                "count": n,
                "min": sorted_s[0],
                "max": sorted_s[-1],
                "p50": sorted_s[n // 2],
                "p95": sorted_s[min(int(n * 0.95), n - 1)],
                "p99": sorted_s[min(int(n * 0.99), n - 1)],
            }

        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": histograms,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _key(self, name: str, labels: dict | None) -> str:
        """Build a metric key with optional label suffix.

        Example: "llm.ttft_ms{kind=reply_text,provider=openai}"
        """
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Process-wide singleton — import this directly
metrics = MetricsCollector.get()
