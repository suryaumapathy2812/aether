"""
Tests for KernelScheduler and ServiceRouter.

Tests:
- Job submission and retrieval
- Priority ordering (interactive before background)
- Job cancellation
- Health check
- ServiceRouter routing
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aether.kernel.contracts import (
    JobKind,
    JobModality,
    JobPriority,
    JobStatus,
    KernelEvent,
    KernelRequest,
    KernelResult,
)
from aether.kernel.scheduler import KernelScheduler, ServiceRouter


# ─── Helpers ─────────────────────────────────────────────────────


def _make_request(
    kind: str = "reply_text",
    priority: int = 0,
    text: str = "Hello",
) -> KernelRequest:
    return KernelRequest(
        kind=kind,
        modality=JobModality.TEXT.value,
        user_id="user-1",
        session_id="session-1",
        payload={"text": text},
        priority=priority,
    )


def _make_router_with_events(*events: KernelEvent) -> ServiceRouter:
    """Create a ServiceRouter that yields the given events for any request."""
    router = ServiceRouter()

    async def fake_route(request):
        for event in events:
            yield event

    router.route = fake_route  # type: ignore[assignment]
    return router


# ─── KernelScheduler ─────────────────────────────────────────────


class TestKernelScheduler:
    """Test KernelScheduler job management."""

    @pytest.mark.asyncio
    async def test_submit_returns_job_id(self):
        """submit() returns a unique job_id."""
        scheduler = KernelScheduler()
        job_id = await scheduler.submit(_make_request())
        assert job_id
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_submit_multiple_unique_ids(self):
        """Each submission gets a unique job_id."""
        scheduler = KernelScheduler()
        id1 = await scheduler.submit(_make_request())
        id2 = await scheduler.submit(_make_request())
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self):
        """Can cancel a pending job before it starts."""
        scheduler = KernelScheduler()  # No router, no worker started
        job_id = await scheduler.submit(_make_request())

        result = await scheduler.cancel(job_id)
        assert result is True

        final = await scheduler.await_result(job_id)
        assert final.status == JobStatus.CANCELED.value

    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self):
        """Canceling an unknown job returns False."""
        scheduler = KernelScheduler()
        result = await scheduler.cancel("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_await_result_unknown_job(self):
        """await_result for unknown job returns failed result."""
        scheduler = KernelScheduler()
        result = await scheduler.await_result("nonexistent-id")
        assert result.status == JobStatus.FAILED.value
        assert result.error["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """health_check returns scheduler metrics."""
        scheduler = KernelScheduler()
        await scheduler.submit(_make_request())

        health = await scheduler.health_check()
        assert health["scheduler"] == "kernel"
        assert health["total_jobs"] == 1
        assert health["pending_jobs"] == 1

    @pytest.mark.asyncio
    async def test_queue_depths(self):
        """get_queue_depths returns correct counts."""
        scheduler = KernelScheduler()
        await scheduler.submit(_make_request(priority=0))
        await scheduler.submit(_make_request(priority=1))

        depths = scheduler.get_queue_depths()
        assert depths["interactive"] == 1
        assert depths["background"] == 1

    @pytest.mark.asyncio
    async def test_job_execution_with_router(self):
        """Scheduler executes jobs through the service router."""
        events = [
            KernelEvent.text_chunk("", "Hello!", sequence=1),
            KernelEvent.stream_end("", sequence=2),
        ]
        router = _make_router_with_events(*events)
        scheduler = KernelScheduler(service_router=router)

        await scheduler.start()
        try:
            job_id = await scheduler.submit(_make_request())

            # Wait for completion
            result = await scheduler.await_result(job_id, timeout_ms=5000)
            assert result.status == JobStatus.COMPLETED.value
            assert "Hello!" in result.payload.get("text", "")
        finally:
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        """stream() yields events from a running job."""
        events = [
            KernelEvent.text_chunk("", "Hello!", sequence=1),
            KernelEvent.stream_end("", sequence=2),
        ]
        router = _make_router_with_events(*events)
        scheduler = KernelScheduler(service_router=router)

        await scheduler.start()
        try:
            job_id = await scheduler.submit(_make_request())

            collected = []
            async for event in scheduler.stream(job_id):
                collected.append(event)

            assert len(collected) == 2
            assert collected[0].stream_type == "text_chunk"
            assert collected[1].stream_type == "stream_end"
        finally:
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stream_unknown_job(self):
        """stream() for unknown job yields nothing."""
        scheduler = KernelScheduler()
        collected = []
        async for event in scheduler.stream("nonexistent"):
            collected.append(event)
        assert collected == []

    @pytest.mark.asyncio
    async def test_interactive_before_background(self):
        """Interactive jobs are processed before background jobs."""
        execution_order = []

        async def tracking_route(request):
            execution_order.append(request.kind)
            yield KernelEvent.text_chunk(request.job_id, "ok", sequence=1)

        router = ServiceRouter()
        router.route = tracking_route  # type: ignore[assignment]

        scheduler = KernelScheduler(service_router=router)

        # Submit background first, then interactive
        bg_id = await scheduler.submit(
            _make_request(kind="memory_fact_extract", priority=1)
        )
        int_id = await scheduler.submit(_make_request(kind="reply_text", priority=0))

        await scheduler.start()
        try:
            # Wait for both to complete
            await scheduler.await_result(int_id, timeout_ms=5000)
            await scheduler.await_result(bg_id, timeout_ms=5000)

            # Interactive should have been processed first
            assert execution_order[0] == "reply_text"
            assert execution_order[1] == "memory_fact_extract"
        finally:
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_stop_idempotent(self):
        """Starting/stopping multiple times is safe."""
        scheduler = KernelScheduler()
        await scheduler.start()
        await scheduler.start()  # Should be no-op
        await scheduler.stop()
        await scheduler.stop()  # Should be no-op

    @pytest.mark.asyncio
    async def test_job_failure_produces_error_event(self):
        """A failing job produces an error event and failed result."""

        async def failing_route(request):
            raise RuntimeError("Service exploded")
            yield  # type: ignore[misc]

        router = ServiceRouter()
        router.route = failing_route  # type: ignore[assignment]

        scheduler = KernelScheduler(service_router=router)
        await scheduler.start()
        try:
            job_id = await scheduler.submit(_make_request())
            result = await scheduler.await_result(job_id, timeout_ms=5000)

            assert result.status == JobStatus.FAILED.value
            assert "exploded" in result.error["message"]
        finally:
            await scheduler.stop()


# ─── ServiceRouter ───────────────────────────────────────────────


class TestServiceRouter:
    """Test ServiceRouter job routing."""

    @pytest.mark.asyncio
    async def test_unknown_kind_yields_error(self):
        """Unknown job kind yields an error event."""
        router = ServiceRouter()
        request = _make_request(kind="unknown_kind")

        events = []
        async for event in router.route(request):
            events.append(event)

        assert len(events) == 1
        assert events[0].stream_type == "error"
        assert "Unknown job kind" in events[0].payload["message"]

    @pytest.mark.asyncio
    async def test_reply_without_service_yields_error(self):
        """Reply routing without ReplyService yields error."""
        router = ServiceRouter()  # No services configured
        request = _make_request(kind="reply_text")

        events = []
        async for event in router.route(request):
            events.append(event)

        assert len(events) == 1
        assert events[0].stream_type == "error"
        assert "not configured" in events[0].payload["message"]

    @pytest.mark.asyncio
    async def test_memory_without_service_yields_error(self):
        """Memory routing without MemoryService yields error."""
        router = ServiceRouter()
        request = _make_request(kind="memory_fact_extract")

        events = []
        async for event in router.route(request):
            events.append(event)

        assert len(events) == 1
        assert events[0].stream_type == "error"

    @pytest.mark.asyncio
    async def test_tool_routing(self):
        """Tool execution routes to ToolService."""
        tool_service = AsyncMock()
        tool_service.execute = AsyncMock(
            return_value=MagicMock(
                tool_name="read_file",
                output="contents",
                call_id="c1",
                error=False,
            )
        )

        router = ServiceRouter(tool_service=tool_service)
        request = KernelRequest(
            kind="tool_execute",
            modality="system",
            user_id="u1",
            session_id="s1",
            payload={
                "tool_name": "read_file",
                "arguments": {"path": "/tmp/test"},
                "call_id": "c1",
            },
        )

        events = []
        async for event in router.route(request):
            events.append(event)

        assert len(events) == 1
        assert events[0].stream_type == "tool_result"
        assert events[0].payload["tool_name"] == "read_file"
