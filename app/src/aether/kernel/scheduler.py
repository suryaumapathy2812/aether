"""
Kernel Scheduler — priority queue job scheduler with isolated worker pools.

Architecture:
  Two isolated worker pools (P-Core / E-Core):
  - P-Core workers: interactive jobs only (reply_text, reply_voice)
  - E-Core workers: background jobs only (memory, notifications)

  This ensures background work can never starve user-facing replies.

Bounded queues:
  - Interactive queue: raises QueueFullError when at capacity (caller fails fast)
  - Background queue: sheds oldest job when at capacity (best-effort work)

Cancellation:
  - cancel(job_id): cancel a specific job
  - cancel_by_session(session_id): cancel all jobs for a session (on disconnect)

Metrics:
  All decision points emit to MetricsCollector (jobs submitted, completed,
  canceled, shed, queue depths, worker activity, enqueue delay).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from aether.core.metrics import metrics
from aether.kernel.contracts import (
    JobPriority,
    JobStatus,
    KernelEvent,
    KernelRequest,
    KernelResult,
)
from aether.kernel.interface import KernelInterface

logger = logging.getLogger(__name__)

# Sentinel for stream completion
_STREAM_DONE = object()


class QueueFullError(Exception):
    """Raised when the interactive queue is at capacity."""


@dataclass
class _ScheduledJob:
    """Internal job tracking state."""

    job_id: str
    request: KernelRequest
    queue: asyncio.Queue[KernelEvent | object] = field(
        default_factory=asyncio.Queue,
    )
    status: JobStatus = JobStatus.PENDING
    result: KernelResult | None = None
    task: asyncio.Task[None] | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    sequence: int = 0
    submitted_at: float = field(default_factory=time.time)
    started_at: float = 0.0

    def next_seq(self) -> int:
        seq = self.sequence
        self.sequence += 1
        return seq

    @property
    def lane(self) -> str:
        return (
            "interactive"
            if self.request.priority <= JobPriority.INTERACTIVE.value
            else "background"
        )


class KernelScheduler(KernelInterface):
    """
    Priority queue scheduler with isolated P-Core / E-Core worker pools.

    P-Core workers handle interactive jobs (user-facing replies).
    E-Core workers handle background jobs (memory, notifications).
    The two pools are completely isolated — background work can never
    starve interactive responses.
    """

    def __init__(
        self,
        service_router: "ServiceRouter | None" = None,
        # Legacy single-worker param kept for backward compat
        max_workers: int = 1,
        # Split pool params (preferred)
        max_interactive_workers: int | None = None,
        max_background_workers: int | None = None,
        interactive_queue_limit: int = 20,
        background_queue_limit: int = 50,
    ) -> None:
        self._router = service_router

        # Resolve worker counts — split pool params take precedence
        self._max_interactive = (
            max_interactive_workers
            if max_interactive_workers is not None
            else max_workers
        )
        self._max_background = (
            max_background_workers
            if max_background_workers is not None
            else max_workers
        )

        # Queue limits
        self._interactive_queue_limit = interactive_queue_limit
        self._background_queue_limit = background_queue_limit

        # Job tracking
        self._jobs: dict[str, _ScheduledJob] = {}

        # Session → job_ids index for O(1) cancel_by_session
        self._session_jobs: dict[str, set[str]] = defaultdict(set)

        # Separate queues per pool
        self._interactive_queue: deque[str] = deque()
        self._background_queue: deque[str] = deque()

        # Separate semaphores per pool
        self._interactive_sem = asyncio.Semaphore(self._max_interactive)
        self._background_sem = asyncio.Semaphore(self._max_background)

        # Separate wake events per pool
        self._interactive_event = asyncio.Event()
        self._background_event = asyncio.Event()

        # Worker task lists
        self._interactive_workers: list[asyncio.Task] = []
        self._background_workers: list[asyncio.Task] = []

        self._running = False

    # ─── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start P-Core and E-Core worker pools."""
        if self._running:
            return
        self._running = True

        # Spawn P-Core workers (interactive only)
        for i in range(self._max_interactive):
            task = asyncio.create_task(
                self._worker_loop(
                    name=f"P-Core-{i}",
                    queue=self._interactive_queue,
                    semaphore=self._interactive_sem,
                    event=self._interactive_event,
                ),
                name=f"kernel-p-core-{i}",
            )
            self._interactive_workers.append(task)

        # Spawn E-Core workers (background only)
        for i in range(self._max_background):
            task = asyncio.create_task(
                self._worker_loop(
                    name=f"E-Core-{i}",
                    queue=self._background_queue,
                    semaphore=self._background_sem,
                    event=self._background_event,
                ),
                name=f"kernel-e-core-{i}",
            )
            self._background_workers.append(task)

        logger.info(
            "Kernel started: %d P-Cores (interactive), %d E-Cores (background)",
            self._max_interactive,
            self._max_background,
        )

    async def stop(self) -> None:
        """Stop all workers and cancel pending jobs."""
        self._running = False

        # Wake all workers so they can exit
        self._interactive_event.set()
        self._background_event.set()

        all_workers = self._interactive_workers + self._background_workers
        for task in all_workers:
            if not task.done():
                task.cancel()
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)

        self._interactive_workers.clear()
        self._background_workers.clear()

        # Cancel all pending/running jobs
        for job in self._jobs.values():
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                if job.task and not job.task.done():
                    job.task.cancel()
                job.status = JobStatus.CANCELED
                job.result = KernelResult.canceled(job.job_id)
                job.done_event.set()
                await job.queue.put(_STREAM_DONE)

        logger.info("Kernel scheduler stopped")

    # ─── KernelInterface ──────────────────────────────────────────

    async def submit(self, request: KernelRequest) -> str:
        """Submit a job to the appropriate priority queue.

        Interactive queue: raises QueueFullError when at capacity.
        Background queue: sheds oldest job when at capacity.
        """
        job_id = str(uuid.uuid4())
        tagged = request.with_job_id(job_id)
        job = _ScheduledJob(job_id=job_id, request=tagged)
        self._jobs[job_id] = job
        self._session_jobs[request.session_id].add(job_id)

        is_interactive = request.priority <= JobPriority.INTERACTIVE.value
        lane = "interactive" if is_interactive else "background"

        metrics.inc(
            "kernel.jobs.submitted", labels={"kind": request.kind, "lane": lane}
        )

        if is_interactive:
            if len(self._interactive_queue) >= self._interactive_queue_limit:
                # Remove from tracking and raise — caller fails fast
                self._jobs.pop(job_id, None)
                self._session_jobs[request.session_id].discard(job_id)
                metrics.inc("kernel.jobs.rejected", labels={"lane": "interactive"})
                raise QueueFullError(
                    f"Interactive queue at capacity ({self._interactive_queue_limit})"
                )
            self._interactive_queue.append(job_id)
            self._interactive_event.set()
        else:
            if len(self._background_queue) >= self._background_queue_limit:
                # Shed oldest background job (best-effort work)
                shed_id = self._background_queue.popleft()
                shed = self._jobs.get(shed_id)
                if shed and shed.status == JobStatus.PENDING:
                    shed.status = JobStatus.CANCELED
                    shed.result = KernelResult.failed(
                        job_id=shed_id,
                        error={
                            "code": "queue_shed",
                            "message": "Background queue full",
                        },
                    )
                    shed.done_event.set()
                    await shed.queue.put(_STREAM_DONE)
                    self._session_jobs[shed.request.session_id].discard(shed_id)
                    metrics.inc("kernel.jobs.shed", labels={"kind": shed.request.kind})
                    logger.warning(
                        "Shed background job %s (%s)", shed_id[:8], shed.request.kind
                    )

            self._background_queue.append(job_id)
            self._background_event.set()

        metrics.gauge_set("kernel.queue.interactive", len(self._interactive_queue))
        metrics.gauge_set("kernel.queue.background", len(self._background_queue))

        logger.debug(
            "Job %s submitted (kind=%s, lane=%s)", job_id[:8], request.kind, lane
        )
        return job_id

    async def stream(self, job_id: str) -> AsyncGenerator[KernelEvent, None]:
        """Yield KernelEvents from the job's queue until stream_end."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("stream() called for unknown job %s", job_id[:8])
            return

        while True:
            item = await job.queue.get()
            if item is _STREAM_DONE:
                break
            yield item  # type: ignore[misc]

    async def await_result(
        self, job_id: str, timeout_ms: int | None = None
    ) -> KernelResult:
        """Block until the job finishes and return its terminal result."""
        job = self._jobs.get(job_id)
        if job is None:
            return KernelResult.failed(
                job_id=job_id,
                error={"code": "not_found", "message": f"Job {job_id} not found"},
            )

        timeout_s = timeout_ms / 1000.0 if timeout_ms else None
        try:
            await asyncio.wait_for(job.done_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return KernelResult.failed(
                job_id=job_id,
                error={"code": "timeout", "message": "Job timed out"},
            )

        assert job.result is not None
        return job.result

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return False

        # Remove from queue if still pending
        if job.status == JobStatus.PENDING:
            try:
                self._interactive_queue.remove(job_id)
            except ValueError:
                try:
                    self._background_queue.remove(job_id)
                except ValueError:
                    pass

        if job.task and not job.task.done():
            job.task.cancel()

        job.status = JobStatus.CANCELED
        job.result = KernelResult.canceled(job_id)
        job.done_event.set()
        await job.queue.put(_STREAM_DONE)

        self._session_jobs[job.request.session_id].discard(job_id)
        metrics.inc("kernel.jobs.canceled", labels={"kind": job.request.kind})
        metrics.gauge_set("kernel.queue.interactive", len(self._interactive_queue))
        metrics.gauge_set("kernel.queue.background", len(self._background_queue))

        logger.info("Job %s canceled", job_id[:8])
        return True

    async def cancel_by_session(
        self, session_id: str, kinds: list[str] | None = None
    ) -> int:
        """Cancel all pending/running jobs for a session.

        Uses the session → job_ids index for O(1) lookup.
        Called on WebSocket disconnect or when a new utterance supersedes old.

        Args:
            session_id: Session to cancel jobs for.
            kinds: If given, only cancel jobs of these kinds.

        Returns:
            Number of jobs canceled.
        """
        canceled = 0
        job_ids = list(self._session_jobs.get(session_id, set()))
        for job_id in job_ids:
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if kinds and job.request.kind not in kinds:
                continue
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                if await self.cancel(job_id):
                    canceled += 1
        logger.info("Canceled %d jobs for session %s", canceled, session_id)
        return canceled

    async def health_check(self) -> dict[str, Any]:
        """Return scheduler health metrics."""
        active = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
        pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
        return {
            "scheduler": "kernel",
            "running": self._running,
            "pools": {
                "interactive": {
                    "workers": self._max_interactive,
                    "queue": len(self._interactive_queue),
                },
                "background": {
                    "workers": self._max_background,
                    "queue": len(self._background_queue),
                },
            },
            "active_jobs": active,
            "pending_jobs": pending,
            "total_jobs": len(self._jobs),
        }

    # ─── Worker Loops ─────────────────────────────────────────────

    async def _worker_loop(
        self,
        name: str,
        queue: deque,
        semaphore: asyncio.Semaphore,
        event: asyncio.Event,
    ) -> None:
        """Generic worker loop — works for both P-Core and E-Core pools."""
        logger.info("Worker %s started", name)

        while self._running:
            # Wait for work
            if not queue:
                event.clear()
                try:
                    await event.wait()
                except asyncio.CancelledError:
                    break

            if not self._running:
                break

            try:
                job_id = queue.popleft()
            except IndexError:
                continue

            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.PENDING:
                continue

            async with semaphore:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                metrics.gauge_inc(
                    "kernel.workers.active", labels={"pool": name.split("-")[0]}
                )
                job.task = asyncio.create_task(
                    self._execute_job(job, worker_name=name),
                    name=f"job-{job_id[:8]}",
                )
                await job.task
                metrics.gauge_dec(
                    "kernel.workers.active", labels={"pool": name.split("-")[0]}
                )

        logger.info("Worker %s stopped", name)

    async def _execute_job(self, job: _ScheduledJob, worker_name: str) -> None:
        """Execute a single job through the service router."""
        from aether.core.tracing import JobTrace

        trace = JobTrace()
        root = trace.start_span("kernel.job", kind=job.request.kind, worker=worker_name)

        # Record enqueue delay
        enqueue_delay_ms = (time.time() - job.submitted_at) * 1000
        metrics.observe(
            "kernel.enqueue_delay_ms",
            enqueue_delay_ms,
            labels={"lane": job.lane},
        )

        collected_text: list[str] = []

        try:
            if self._router is None:
                raise RuntimeError("No service router configured")

            svc_span = trace.start_span("service.execute", parent_id=root.span_id)
            async for event in self._router.route(job.request):
                await job.queue.put(event)
                if event.stream_type == "text_chunk":
                    collected_text.append(event.payload.get("text", ""))
            svc_span.finish(status="ok")

            # Success
            elapsed_ms = round((time.time() - job.started_at) * 1000, 1)
            job.status = JobStatus.COMPLETED
            job.result = KernelResult.success(
                job.job_id,
                payload={"text": "".join(collected_text)},
                duration_ms=elapsed_ms,
                trace=trace.to_dict(),
            )
            metrics.observe(
                "kernel.job.duration_ms", elapsed_ms, labels={"kind": job.request.kind}
            )
            metrics.inc(
                "kernel.jobs.completed",
                labels={"kind": job.request.kind, "status": "completed"},
            )

        except asyncio.CancelledError:
            logger.debug("Job %s cancelled", job.job_id[:8])
            if job.status != JobStatus.CANCELED:
                job.status = JobStatus.CANCELED
                job.result = KernelResult.canceled(job.job_id)

        except Exception as exc:
            logger.error("Job %s failed: %s", job.job_id[:8], exc, exc_info=True)
            job.status = JobStatus.FAILED
            job.result = KernelResult.failed(
                job.job_id,
                error={"code": "execution_error", "message": str(exc)},
            )
            metrics.inc(
                "kernel.jobs.completed",
                labels={"kind": job.request.kind, "status": "failed"},
            )
            err_event = KernelEvent.error(job.job_id, str(exc), job.next_seq())
            await job.queue.put(err_event)

        finally:
            root.finish()
            await job.queue.put(_STREAM_DONE)
            job.done_event.set()
            self._session_jobs[job.request.session_id].discard(job.job_id)
            metrics.gauge_set("kernel.queue.interactive", len(self._interactive_queue))
            metrics.gauge_set("kernel.queue.background", len(self._background_queue))

    # ─── Introspection ────────────────────────────────────────────

    def get_queue_depths(self) -> dict[str, int]:
        return {
            "interactive": len(self._interactive_queue),
            "background": len(self._background_queue),
        }

    def get_job_status(self, job_id: str) -> JobStatus | None:
        job = self._jobs.get(job_id)
        return job.status if job else None


class ServiceRouter:
    """
    Routes KernelRequests to the appropriate service.

    Maps job kinds to service methods. Each service yields KernelEvents.
    LLMEventEnvelope events from services are converted to KernelEvents here.
    """

    def __init__(
        self,
        reply_service: Any = None,
        memory_service: Any = None,
        notification_service: Any = None,
        tool_service: Any = None,
    ) -> None:
        self._reply = reply_service
        self._memory = memory_service
        self._notification = notification_service
        self._tool = tool_service

    async def route(self, request: KernelRequest) -> AsyncGenerator[KernelEvent, None]:
        """Route a request to the appropriate service, yielding KernelEvents."""
        kind = request.kind

        if kind.startswith("reply_"):
            async for event in self._handle_reply(request):
                yield event
        elif kind.startswith("memory_"):
            async for event in self._handle_memory(request):
                yield event
        elif kind.startswith("notification_"):
            async for event in self._handle_notification(request):
                yield event
        elif kind == "tool_execute":
            async for event in self._handle_tool(request):
                yield event
        else:
            logger.warning("Unknown job kind: %s", kind)
            yield KernelEvent.error(
                request.job_id,
                f"Unknown job kind: {kind}",
                sequence=0,
                code="unknown_kind",
            )

    def _llm_event_to_kernel_event(
        self, llm_event: Any, job_id: str, sequence: int
    ) -> KernelEvent:
        """Convert LLMEventEnvelope → KernelEvent.

        Mapping:
            text_chunk  → text_chunk
            tool_call   → tool_call
            tool_result → tool_result
            status      → status      (tool acknowledge for voice)
            stream_end  → stream_end
            error       → error
        """
        return KernelEvent(
            job_id=job_id,
            stream_type=llm_event.event_type,  # types align 1:1
            payload=llm_event.payload,
            sequence=sequence,
        )

    async def _handle_reply(
        self, request: KernelRequest
    ) -> AsyncGenerator[KernelEvent, None]:
        """Route reply jobs to ReplyService."""
        if self._reply is None:
            yield KernelEvent.error(
                request.job_id,
                "ReplyService not configured",
                sequence=0,
                code="no_service",
            )
            return

        from aether.llm.context_builder import SessionState

        session = SessionState(
            session_id=request.session_id,
            user_id=request.user_id,
            mode="text" if request.kind == "reply_text" else "voice",
            history=request.payload.get("history", []),
        )

        sequence = 0
        async for llm_event in self._reply.generate_reply(
            user_message=request.payload.get("text", ""),
            session=session,
            enabled_plugins=request.payload.get("enabled_plugins", []),
            pending_memory=request.payload.get("pending_memory"),
            pending_vision=request.payload.get("pending_vision"),
        ):
            sequence += 1
            yield self._llm_event_to_kernel_event(llm_event, request.job_id, sequence)

    async def _handle_memory(
        self, request: KernelRequest
    ) -> AsyncGenerator[KernelEvent, None]:
        """Route memory jobs to MemoryService."""
        if self._memory is None:
            yield KernelEvent.error(
                request.job_id,
                "MemoryService not configured",
                sequence=0,
                code="no_service",
            )
            return

        kind = request.kind
        payload = request.payload

        try:
            if kind == "memory_fact_extract":
                facts = await self._memory.extract_facts(
                    user_message=payload.get("user_message", ""),
                    assistant_message=payload.get("assistant_message", ""),
                    conversation_id=payload.get("conversation_id"),
                )
                yield KernelEvent(
                    job_id=request.job_id,
                    stream_type="result",
                    payload={"facts": facts},
                    sequence=0,
                )

            elif kind == "memory_session_summary":
                summary = await self._memory.summarize_session(
                    session_id=payload.get("session_id", ""),
                    conversation_history=payload.get("history", []),
                    started_at=payload.get("started_at", 0),
                    turn_count=payload.get("turn_count", 0),
                    tools_used=payload.get("tools_used"),
                )
                yield KernelEvent(
                    job_id=request.job_id,
                    stream_type="result",
                    payload={"summary": summary},
                    sequence=0,
                )

            elif kind == "memory_action_compact":
                await self._memory.compact_actions()
                yield KernelEvent(
                    job_id=request.job_id,
                    stream_type="result",
                    payload={"compacted": True},
                    sequence=0,
                )

        except Exception as e:
            yield KernelEvent.error(
                request.job_id, str(e), sequence=0, code="memory_error"
            )

    async def _handle_notification(
        self, request: KernelRequest
    ) -> AsyncGenerator[KernelEvent, None]:
        """Route notification jobs to NotificationService."""
        if self._notification is None:
            yield KernelEvent.error(
                request.job_id,
                "NotificationService not configured",
                sequence=0,
                code="no_service",
            )
            return

        yield KernelEvent(
            job_id=request.job_id,
            stream_type="result",
            payload={"action": "archive", "notification": ""},
            sequence=0,
        )

    async def _handle_tool(
        self, request: KernelRequest
    ) -> AsyncGenerator[KernelEvent, None]:
        """Route tool execution jobs to ToolService."""
        if self._tool is None:
            yield KernelEvent.error(
                request.job_id,
                "ToolService not configured",
                sequence=0,
                code="no_service",
            )
            return

        payload = request.payload
        try:
            result = await self._tool.execute(
                tool_name=payload.get("tool_name", ""),
                arguments=payload.get("arguments", {}),
                call_id=payload.get("call_id", ""),
                plugin_context=payload.get("plugin_context"),
            )
            yield KernelEvent.tool_result(
                job_id=request.job_id,
                tool_name=result.tool_name,
                output=result.output,
                call_id=result.call_id,
                error=result.error,
                sequence=0,
            )
        except Exception as e:
            yield KernelEvent.error(
                request.job_id, str(e), sequence=0, code="tool_error"
            )
