"""
Aether Tracing — lightweight per-job structured tracing.

No external dependencies. Designed to be wrapped by OpenTelemetry later if needed.

Each kernel job gets a JobTrace. Spans are nested via parent_id.
The root span covers the full job lifetime. Child spans cover
individual stages (context build, LLM call, tool execution, etc.).

Usage:
    from aether.core.tracing import JobTrace

    trace = JobTrace()
    root = trace.start_span("kernel.job", kind="reply_text", worker="P-Core-0")

    ctx_span = trace.start_span("context.build")
    # ... build context ...
    ctx_span.finish(tokens=1200)

    llm_span = trace.start_span("llm.generate")
    # ... stream LLM ...
    llm_span.finish(ttft_ms=312.4, tokens=87)

    root.finish()

    result = trace.to_dict()  # attach to job result or log
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Span:
    """A single timed unit of work within a job trace."""

    span_id: str
    name: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict = field(default_factory=dict)

    def finish(self, **attrs) -> "Span":
        """Mark the span complete and attach optional attributes.

        Returns self so callers can chain: span = trace.start_span(...); span.finish(x=1)
        """
        self.end_time = time.time()
        self.attributes.update(attrs)
        return self

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration in milliseconds. None if span is still open."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 2)
            if self.duration_ms is not None
            else None,
            "attributes": self.attributes,
        }


class JobTrace:
    """Collects spans for a single kernel job.

    Spans are linked via parent_id. The most recently started span
    is tracked as the implicit parent for the next start_span() call,
    so callers don't need to pass parent_id explicitly for linear flows.

    For branching (parallel spans), pass parent_id explicitly.
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.spans: list[Span] = []
        self._active_span: Span | None = None

    def start_span(self, name: str, parent_id: str | None = None, **attrs) -> Span:
        """Open a new span.

        If parent_id is not given, the currently active span is used as parent.
        The new span becomes the active span.
        """
        resolved_parent = parent_id or (
            self._active_span.span_id if self._active_span else None
        )
        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            parent_id=resolved_parent,
            start_time=time.time(),
            attributes=attrs,
        )
        self.spans.append(span)
        self._active_span = span
        return span

    def root_spans(self) -> list[Span]:
        """Return top-level spans (no parent)."""
        return [s for s in self.spans if s.parent_id is None]

    def total_ms(self) -> float:
        """Sum of root span durations in milliseconds."""
        return sum(s.duration_ms or 0.0 for s in self.root_spans())

    def to_dict(self) -> dict:
        """Serialise the full trace for logging or attaching to a job result."""
        return {
            "trace_id": self.trace_id,
            "total_ms": round(self.total_ms(), 2),
            "spans": [s.to_dict() for s in self.spans],
        }
