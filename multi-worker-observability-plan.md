# Multi-Worker Kernel + Full Observability Plan

## Current State

- `KernelScheduler` exists with two priority queues (interactive/background), but runs `max_workers=1`
- `KernelCore` processes messages directly — scheduler is NOT wired into the main pipeline yet
- Transport → KernelCore.process_message() → pipeline (no job queue)
- Services exist (Reply, Memory, Notification, Tool) but are called directly, not through scheduler
- Two parallel code paths exist:
  - **Legacy (currently wired):** `LLMProcessor` with its own agentic loop, sentence splitting, tool execution via `ToolRegistry.dispatch()`
  - **New (built but unwired):** `LLMCore`, `ContextBuilder`, `ToolOrchestrator`, `ReplyService`, `MemoryService`, `NotificationService`, `ToolService`
- `PipelineTimer` tracks STT→Memory→LLM→TTS latency
- `health_check()` returns queue depths and job counts
- `trace` dict in `LLMRequestEnvelope` blueprinted but not populated

---

## Scope Boundary: What Goes Through the Scheduler

Not everything in `KernelCore` is a "job." The scheduler handles **discrete units of work** that can be queued, prioritized, and canceled. Persistent loops, control messages, and session lifecycle remain as direct handling.

### Through the Scheduler (jobs)

| Kind | Priority | Service |
|------|----------|---------|
| `reply_text` | INTERACTIVE | ReplyService |
| `reply_voice` | INTERACTIVE | ReplyService |
| `fact_extraction` | BACKGROUND | MemoryService |
| `session_summary` | BACKGROUND | MemoryService |
| `notification_decision` | BACKGROUND | NotificationService |
| `tool_execution` | INTERACTIVE | ToolService |

### Direct Handling (NOT jobs)

| Responsibility | Why Not a Job |
|----------------|---------------|
| STT streaming event loop (`_stt_event_loop`) | Persistent background listener — runs for the lifetime of a voice session, not a discrete unit of work |
| Session lifecycle (`_get_session`, `_ensure_session_started`, `_cleanup_session`) | Per-session state management — must complete synchronously before any job can run |
| Control events (`stream_start`, `stream_stop`, `mute`, `unmute`, `image`, `config`) | Control messages that modify session state — not work to be queued |
| Debounce logic (`_debounce_and_trigger`) | Accumulates transcript fragments, waits for silence — a coordination mechanism, not a job |
| Session greeting | Triggered on voice connect — a one-shot side effect of session start |
| TTS synthesis | Consumes text chunks from reply jobs — runs downstream of the scheduler, not inside it (see [TTS Integration](#tts-integration)) |

**Rule:** If it has a clear start, produces a result, and can be canceled — it's a job. If it's a persistent loop, a state mutation, or a coordination mechanism — it stays direct.

---

## Part A: Multi-Worker Kernel (2 P-Core + 2 E-Core)

### Phase A0: Wire Service Layer (Prerequisite)

Before the scheduler can route jobs to services, the new service layer must replace the legacy `LLMProcessor` pipeline. The scheduler routes to services; services use `LLMCore`, not `LLMProcessor`.

**Why this is a prerequisite:** The `ServiceRouter` in `scheduler.py` expects to call `ReplyService.generate_reply()`, `MemoryService.extract_facts()`, etc. These services exist and use `LLMCore` internally, but `KernelCore` currently bypasses them entirely and calls `LLMProcessor` directly. We must wire the services before wiring the scheduler.

#### A0.1: Add STATUS Event to LLMEventEnvelope

**File:** `app/src/aether/llm/contracts.py`

Currently `LLMProcessor` yields `status_frame()` before each tool execution (voice acknowledge like "Let me check that..."). `LLMCore.generate_with_tools()` does NOT yield status events — it only yields `tool_result`. Add a `STATUS` event type so the new pipeline can emit tool-acknowledge messages.

```python
# In LLMEventEnvelope, add to event_type options:
@dataclass
class LLMEventEnvelope:
    """Single event from LLM streaming."""
    kind: str                    # "reply_text", "reply_voice", etc.
    event_type: str              # "text_chunk", "tool_call", "tool_result", "status", "done", "error"
    data: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

# Factory method:
@classmethod
def status(cls, kind: str, message: str, tool_name: str | None = None) -> "LLMEventEnvelope":
    """Status event — tool acknowledge, thinking indicator, etc."""
    return cls(
        kind=kind,
        event_type="status",
        data={"message": message, "tool_name": tool_name},
    )
```

Wire into `LLMCore.generate_with_tools()`:

```python
# In the tool-calling loop, before executing each tool:
async for tool_call in tool_calls:
    # Emit status event for voice acknowledge
    status_text = tool_call.status_text or f"Using {tool_call.name}..."
    yield LLMEventEnvelope.status(
        kind=envelope.kind,
        message=status_text,
        tool_name=tool_call.name,
    )

    # Execute tool
    result = await self._tool_orchestrator.execute(tool_call)
    yield LLMEventEnvelope(
        kind=envelope.kind,
        event_type="tool_result",
        data={"tool_name": tool_call.name, "result": result},
    )
```

#### A0.2: Wire ReplyService Into KernelCore

**File:** `app/src/aether/kernel/core.py`

Replace direct `LLMProcessor` calls with `ReplyService` calls. This is a swap — same behavior, new code path.

```python
# Before (legacy):
async for frame in self._llm_processor.process(text, history, ...):
    yield frame

# After (new service layer):
async for event in self._reply_service.generate_reply(
    kind="reply_text",
    user_id=user_id,
    session_id=session_id,
    text=text,
    history=history,
    context=context,
):
    yield self._llm_event_to_core_msg(event, msg)
```

#### A0.3: Wire Background Services

**File:** `app/src/aether/kernel/core.py`

Replace direct `MemoryProcessor` and `EventProcessor` calls:

```python
# Fact extraction — was: self._memory_processor.extract(...)
# Now: submit to MemoryService (still called directly in A0, scheduler in A1)
asyncio.create_task(
    self._memory_service.extract_facts(
        user_id=user_id,
        session_id=session_id,
        text=text,
        response=response,
    )
)

# Notification decision — was: self._event_processor.process(...)
# Now: submit to NotificationService
asyncio.create_task(
    self._notification_service.process_event(
        user_id=user_id,
        session_id=session_id,
        event=event,
    )
)
```

#### A0.4: Verification (Phase Gate)

1. Send `/chat` request → response streams via `ReplyService` → `LLMCore` (not `LLMProcessor`)
2. Tool calls emit `status` events before execution (voice acknowledge works)
3. Fact extraction runs via `MemoryService`
4. Notification decisions run via `NotificationService`
5. SLO check: `/chat` TTFT p95 ≤ 1800ms (no regression from service layer swap)
6. **Do NOT proceed to A1 until A0 passes SLO gate**

---

### Phase A1: Wire Scheduler Into Main Pipeline

**File:** `app/src/aether/kernel/core.py`

The key change: `KernelCore.process_message()` stops calling services directly. Instead, it creates a `KernelRequest` and submits to the scheduler.

```python
# In KernelCore.__init__(), create scheduler
self.scheduler = KernelScheduler(
    service_router=self._service_router,
    max_interactive_workers=int(os.getenv("AETHER_KERNEL_WORKERS_INTERACTIVE", "2")),
    max_background_workers=int(os.getenv("AETHER_KERNEL_WORKERS_BACKGROUND", "2")),
)

# In startup
await self.scheduler.start()

# In shutdown
await self.scheduler.stop()
```

#### Feature Flag Rollback

**File:** `app/src/aether/kernel/core.py`

The `AETHER_KERNEL_ENABLED` flag controls whether `process_message()` routes through the scheduler or falls back to the direct service calls from Phase A0. This allows instant rollback if the scheduler introduces regressions.

```python
async def process_message(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
    if self._config.kernel_enabled:
        # NEW PATH: Submit to scheduler
        async for core_msg in self._process_via_scheduler(msg):
            yield core_msg
    else:
        # FALLBACK: Direct service calls (Phase A0 path)
        async for core_msg in self._process_direct(msg):
            yield core_msg

async def _process_via_scheduler(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
    """Route through KernelScheduler."""
    request = KernelRequest(
        kind="reply_voice" if msg.modality == "voice" else "reply_text",
        modality=msg.modality,
        user_id=msg.user_id,
        session_id=msg.session_id,
        payload={"text": msg.content.text, "history": ...},
        priority=JobPriority.INTERACTIVE.value,
    )

    job_id = await self.scheduler.submit(request)

    async for event in self.scheduler.stream(job_id):
        yield self._kernel_event_to_core_msg(event, msg)

    # Fire-and-forget background jobs
    await self._submit_background_jobs(msg)

async def _process_direct(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
    """Direct service calls — fallback when scheduler disabled."""
    async for event in self._reply_service.generate_reply(...):
        yield self._llm_event_to_core_msg(event, msg)

    # Background work via asyncio.create_task (Phase A0 pattern)
    asyncio.create_task(self._memory_service.extract_facts(...))
    asyncio.create_task(self._notification_service.process_event(...))
```

Add to `AetherConfig`:

```python
# In app/src/aether/core/config.py
kernel_enabled: bool = field(
    default_factory=lambda: os.getenv("AETHER_KERNEL_ENABLED", "false").lower() == "true"
)
```

#### ServiceRouter Event Mapping

**File:** `app/src/aether/kernel/scheduler.py`

The `ServiceRouter` must convert between `LLMEventEnvelope` (yielded by services) and `KernelEvent` (consumed by scheduler callers). This mapping is explicit:

```python
class ServiceRouter:
    """Routes KernelRequest to the correct service, converts events."""

    def _llm_event_to_kernel_event(
        self, llm_event: LLMEventEnvelope, job_id: str
    ) -> KernelEvent:
        """Convert service-layer events to kernel-layer events."""
        mapping = {
            "text_chunk":  "text_chunk",
            "tool_call":   "tool_call",
            "tool_result": "tool_result",
            "status":      "status",       # Tool acknowledge
            "done":        "done",
            "error":       "error",
        }
        return KernelEvent(
            job_id=job_id,
            stream_type=mapping.get(llm_event.event_type, llm_event.event_type),
            data=llm_event.data,
            trace=llm_event.trace,
        )

    async def route_streaming(
        self, job: Job
    ) -> AsyncGenerator[KernelEvent, None]:
        """Route job to service, yield KernelEvents."""
        if job.request.kind in ("reply_text", "reply_voice"):
            async for llm_event in self._reply_service.generate_reply(
                kind=job.request.kind,
                **job.request.payload,
            ):
                yield self._llm_event_to_kernel_event(llm_event, job.job_id)
        elif job.request.kind == "fact_extraction":
            result = await self._memory_service.extract_facts(**job.request.payload)
            yield KernelEvent(job_id=job.job_id, stream_type="done", data=result)
        # ... other kinds
```

#### Background Callback Integration

**File:** `app/src/aether/kernel/core.py`

**Problem:** The `scheduler.stream(job_id)` pattern assumes a caller is actively consuming events. But for STT-triggered voice responses, there is no caller — the STT event loop detects an utterance, triggers debounce, and eventually fires a voice response. Currently this pushes through `set_status_audio_callback()`.

**Solution:** For STT-triggered jobs, `KernelCore` spawns a background task that consumes the scheduler stream and pushes events through the existing callback. The scheduler doesn't need to know about the callback — `KernelCore` bridges the two.

```python
async def _trigger_voice_response(self, session_id: str, text: str):
    """Called by debounce logic when STT detects a complete utterance."""
    if not self._config.kernel_enabled:
        # Fallback: direct pipeline (existing behavior)
        async for frame in self._process_direct_voice(session_id, text):
            await self._status_audio_callback(frame)
        return

    # Submit to scheduler
    request = KernelRequest(
        kind="reply_voice",
        modality="voice",
        user_id=self._user_id,
        session_id=session_id,
        payload={"text": text, "history": ...},
        priority=JobPriority.INTERACTIVE.value,
    )
    job_id = await self.scheduler.submit(request)

    # Consume stream in background, push through callback
    async def _consume_and_push():
        async for event in self.scheduler.stream(job_id):
            core_msg = self._kernel_event_to_core_msg(event, session_id=session_id)
            await self._status_audio_callback(core_msg)

    asyncio.create_task(_consume_and_push())
```

**Key insight:** The scheduler is always pull-based (`stream()`). The push-to-callback bridge lives in `KernelCore`, not in the scheduler. This keeps the scheduler simple and the callback pattern contained.

#### TTS Integration

**Where TTS fits:** TTS remains **outside** the scheduler. The scheduler produces `KernelEvent(stream_type="text_chunk")` events. `KernelCore` (or a future VoiceAdapter) consumes these chunks and feeds them to TTS sentence-by-sentence, exactly as `_run_voice_pipeline()` does today.

```
Scheduler → KernelEvent(text_chunk) → KernelCore → sentence buffer → TTS → audio → callback
```

The scheduler does NOT know about TTS. It produces text; the voice pipeline consumes text and produces audio. This boundary is intentional — TTS is a transport concern, not a scheduling concern.

```python
# In _consume_and_push() for voice jobs:
async def _consume_and_push():
    sentence_buffer = ""
    async for event in self.scheduler.stream(job_id):
        if event.stream_type == "text_chunk":
            sentence_buffer += event.data.get("text", "")
            # Check for sentence boundary
            sentences = self._split_sentences(sentence_buffer)
            for sentence in sentences[:-1]:  # All complete sentences
                audio = await self._tts_processor.synthesize(sentence)
                await self._status_audio_callback(audio)
            sentence_buffer = sentences[-1] if sentences else ""
        elif event.stream_type == "status":
            # Tool acknowledge — send status text as TTS
            status_audio = await self._tts_processor.synthesize(event.data["message"])
            await self._status_audio_callback(status_audio)
        elif event.stream_type == "done":
            # Flush remaining buffer
            if sentence_buffer.strip():
                audio = await self._tts_processor.synthesize(sentence_buffer)
                await self._status_audio_callback(audio)
```

---

### Phase A2: Split Worker Pools (Isolated Lanes)

**File:** `app/src/aether/kernel/scheduler.py`

Replace single `max_workers` + single `_worker_loop` with two isolated pools:

```python
class KernelScheduler:
    def __init__(
        self,
        service_router: ServiceRouter,
        max_interactive_workers: int = 2,   # P-Cores
        max_background_workers: int = 2,    # E-Cores
    ):
        self._service_router = service_router

        # Separate queues (already exist)
        self._interactive_queue: deque[Job] = deque()
        self._background_queue: deque[Job] = deque()

        # Separate semaphores (NEW — replaces single semaphore)
        self._interactive_sem = asyncio.Semaphore(max_interactive_workers)
        self._background_sem = asyncio.Semaphore(max_background_workers)

        # Separate events
        self._interactive_event = asyncio.Event()
        self._background_event = asyncio.Event()

        # Worker config
        self._max_interactive = max_interactive_workers
        self._max_background = max_background_workers

        # Worker tasks
        self._interactive_workers: list[asyncio.Task] = []
        self._background_workers: list[asyncio.Task] = []
```

Start separate worker loops:
```python
async def start(self):
    self._running = True

    # Spawn P-Core workers (interactive only)
    for i in range(self._max_interactive):
        task = asyncio.create_task(
            self._worker_loop(
                name=f"P-Core-{i}",
                queue=self._interactive_queue,
                semaphore=self._interactive_sem,
                event=self._interactive_event,
            )
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
            )
        )
        self._background_workers.append(task)

    logger.info(
        f"Kernel started: {self._max_interactive} P-Cores, "
        f"{self._max_background} E-Cores"
    )
```

Updated worker loop (generic, works for both pools):
```python
async def _worker_loop(
    self,
    name: str,
    queue: deque,
    semaphore: asyncio.Semaphore,
    event: asyncio.Event,
):
    """Generic worker loop for a specific queue."""
    logger.info(f"Worker {name} started")

    while self._running:
        # Wait for work
        if not queue:
            event.clear()
            await event.wait()

        if not self._running:
            break

        # Dequeue
        try:
            job = queue.popleft()
        except IndexError:
            continue

        # Acquire worker slot
        async with semaphore:
            await self._execute_job(job, worker_name=name)

    logger.info(f"Worker {name} stopped")
```

Updated submit to signal correct queue:
```python
async def submit(self, request: KernelRequest) -> str:
    job = self._create_job(request)

    if request.priority == JobPriority.INTERACTIVE.value:
        self._interactive_queue.append(job)
        self._interactive_event.set()
    else:
        self._background_queue.append(job)
        self._background_event.set()

    return job.job_id
```

---

### Phase A3: Bounded Queues + Backpressure

**File:** `app/src/aether/kernel/scheduler.py`

Add queue limits:
```python
# Config
INTERACTIVE_QUEUE_LIMIT = int(os.getenv("AETHER_INTERACTIVE_QUEUE_LIMIT", "20"))
BACKGROUND_QUEUE_LIMIT = int(os.getenv("AETHER_BACKGROUND_QUEUE_LIMIT", "50"))

async def submit(self, request: KernelRequest) -> str:
    if request.priority == JobPriority.INTERACTIVE.value:
        if len(self._interactive_queue) >= INTERACTIVE_QUEUE_LIMIT:
            raise QueueFullError("Interactive queue at capacity")
        self._interactive_queue.append(job)
        self._interactive_event.set()
    else:
        if len(self._background_queue) >= BACKGROUND_QUEUE_LIMIT:
            # Shed oldest background job
            shed = self._background_queue.popleft()
            shed.status = JobStatus.CANCELED
            shed.result = KernelResult.failed(
                job_id=shed.job_id,
                error={"code": "queue_shed", "message": "Background queue full"},
            )
            logger.warning(f"Shed background job {shed.job_id} ({shed.request.kind})")
            self._metrics["jobs_shed"] += 1

        self._background_queue.append(job)
        self._background_event.set()

    return job.job_id
```

Rule: Interactive queue raises error (caller retries or fails fast). Background queue sheds oldest (best-effort work).

---

### Phase A4: Cancellation on Disconnect / Supersede

**File:** `app/src/aether/kernel/scheduler.py`

Add `cancel_by_session()` for when a user disconnects or sends a new message:
```python
async def cancel_by_session(self, session_id: str, kinds: list[str] | None = None) -> int:
    """Cancel all pending/running jobs for a session. Returns count canceled."""
    canceled = 0
    for job in list(self._jobs.values()):
        if job.request.session_id != session_id:
            continue
        if kinds and job.request.kind not in kinds:
            continue
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            success = await self.cancel(job.job_id)
            if success:
                canceled += 1
    return canceled
```

Wire into transport disconnect:
```python
# In WebSocketTransport on_disconnect:
await scheduler.cancel_by_session(session_id)

# In voice mode, when new utterance supersedes old:
await scheduler.cancel_by_session(session_id, kinds=["reply_voice"])
```

**Known limitation:** `cancel_by_session()` iterates all jobs — O(n) scan. Acceptable for single-user containers where job count is low. If this becomes a bottleneck, add a `session_id → set[job_id]` index:

```python
# Future optimization (not needed now):
self._session_jobs: dict[str, set[str]] = defaultdict(set)

# In submit():
self._session_jobs[request.session_id].add(job.job_id)

# In cancel_by_session():
for job_id in list(self._session_jobs.get(session_id, [])):
    await self.cancel(job_id)
```

---

### Phase A5: Environment Config

**New env vars:**
```env
AETHER_KERNEL_WORKERS_INTERACTIVE=2   # P-Cores
AETHER_KERNEL_WORKERS_BACKGROUND=2    # E-Cores
AETHER_INTERACTIVE_QUEUE_LIMIT=20
AETHER_BACKGROUND_QUEUE_LIMIT=50
AETHER_KERNEL_ENABLED=true            # Feature flag for rollback
```

---

## Part B: Full Observability

### Phase B1: Metrics Collector

**New file:** `app/src/aether/core/metrics.py`

In-process metrics collector. No external dependencies (Prometheus export can wrap this later).

```python
import time
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict


class MetricsCollector:
    """In-process metrics for kernel observability."""

    _instance: "MetricsCollector | None" = None

    @classmethod
    def get(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Counters
        self._counters: dict[str, int] = defaultdict(int)

        # Histograms (store raw values, compute percentiles on read)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._histogram_max_samples = 1000  # Rolling window

        # Gauges (current values)
        self._gauges: dict[str, float] = defaultdict(float)

        # Timestamps
        self._last_reset = time.time()

    # --- Counters ---
    def inc(self, name: str, value: int = 1, labels: dict | None = None):
        key = self._key(name, labels)
        self._counters[key] += value

    # --- Histograms ---
    def observe(self, name: str, value: float, labels: dict | None = None):
        key = self._key(name, labels)
        samples = self._histograms[key]
        samples.append(value)
        if len(samples) > self._histogram_max_samples:
            samples.pop(0)

    # --- Gauges ---
    def gauge_set(self, name: str, value: float, labels: dict | None = None):
        key = self._key(name, labels)
        self._gauges[key] = value

    def gauge_inc(self, name: str, value: float = 1, labels: dict | None = None):
        key = self._key(name, labels)
        self._gauges[key] += value

    def gauge_dec(self, name: str, value: float = 1, labels: dict | None = None):
        key = self._key(name, labels)
        self._gauges[key] -= value

    # --- Percentiles ---
    def percentile(self, name: str, p: float, labels: dict | None = None) -> float | None:
        key = self._key(name, labels)
        samples = sorted(self._histograms.get(key, []))
        if not samples:
            return None
        idx = int(len(samples) * p / 100)
        return samples[min(idx, len(samples) - 1)]

    # --- Snapshot ---
    def snapshot(self) -> dict:
        """Full metrics snapshot for /health or dashboard."""
        result = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
            "uptime_seconds": time.time() - self._last_reset,
        }
        for key, samples in self._histograms.items():
            if samples:
                sorted_s = sorted(samples)
                result["histograms"][key] = {
                    "count": len(sorted_s),
                    "p50": sorted_s[len(sorted_s) // 2],
                    "p95": sorted_s[int(len(sorted_s) * 0.95)],
                    "p99": sorted_s[int(len(sorted_s) * 0.99)],
                    "min": sorted_s[0],
                    "max": sorted_s[-1],
                }
        return result

    def _key(self, name: str, labels: dict | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
```

---

### Phase B2: Instrument the Kernel Scheduler

**File:** `app/src/aether/kernel/scheduler.py`

Add metrics at every decision point:

```python
from aether.core.metrics import MetricsCollector

metrics = MetricsCollector.get()

# In submit():
metrics.inc("kernel.jobs.submitted", labels={"kind": request.kind, "priority": priority_name})
metrics.gauge_inc("kernel.queue.depth", labels={"lane": priority_name})

# In _execute_job():
metrics.gauge_dec("kernel.queue.depth", labels={"lane": priority_name})
metrics.gauge_inc("kernel.workers.active", labels={"pool": worker_name.split("-")[0]})

start = time.time()
# ... execute ...
elapsed_ms = (time.time() - start) * 1000

metrics.observe("kernel.job.duration_ms", elapsed_ms, labels={"kind": job.request.kind})
metrics.gauge_dec("kernel.workers.active", labels={"pool": worker_name.split("-")[0]})

# On completion:
metrics.inc("kernel.jobs.completed", labels={"kind": job.request.kind, "status": job.status.value})

# On cancellation:
metrics.inc("kernel.jobs.canceled", labels={"kind": job.request.kind, "reason": reason})

# On shed:
metrics.inc("kernel.jobs.shed", labels={"kind": job.request.kind})

# Queue depth gauges (updated on submit/dequeue):
metrics.gauge_set("kernel.queue.interactive", len(self._interactive_queue))
metrics.gauge_set("kernel.queue.background", len(self._background_queue))
```

---

### Phase B3: Instrument Services

**File:** Each service file

```python
# ReplyService.generate_reply():
metrics.inc("service.reply.started", labels={"kind": kind})
start = time.time()
# ... generate ...
metrics.observe("service.reply.duration_ms", elapsed_ms, labels={"kind": kind})
metrics.inc("service.reply.completed", labels={"kind": kind})
metrics.observe("service.reply.tokens", token_count, labels={"kind": kind})

# MemoryService.extract_facts():
metrics.inc("service.memory.extraction.started")
# ... extract ...
metrics.inc("service.memory.facts_extracted", value=num_facts)
metrics.observe("service.memory.extraction_ms", elapsed_ms)

# NotificationService.process_event():
metrics.inc("service.notification.processed", labels={"decision": decision.action, "plugin": event.plugin})
metrics.observe("service.notification.decision_ms", elapsed_ms)

# ToolService.execute():
metrics.inc("service.tool.executed", labels={"tool": tool_name, "error": str(result.error)})
metrics.observe("service.tool.duration_ms", elapsed_ms, labels={"tool": tool_name})
```

---

### Phase B4: Instrument LLM Core

**File:** `app/src/aether/llm/core.py`

```python
# Per LLM call:
metrics.inc("llm.calls", labels={"kind": envelope.kind, "provider": envelope.policy.get("provider", "unknown")})
metrics.observe("llm.ttft_ms", time_to_first_token_ms, labels={"kind": envelope.kind})
metrics.observe("llm.duration_ms", total_ms, labels={"kind": envelope.kind})
metrics.observe("llm.tokens.prompt", usage.get("prompt_tokens", 0), labels={"kind": envelope.kind})
metrics.observe("llm.tokens.completion", usage.get("completion_tokens", 0), labels={"kind": envelope.kind})

# Tool calling iterations:
metrics.inc("llm.tool_iterations", labels={"kind": envelope.kind})
```

---

### Phase B5: Instrument Providers (STT/TTS)

**File:** Provider files

```python
# STT:
metrics.inc("provider.stt.connections")
metrics.observe("provider.stt.utterance_duration_ms", duration)
metrics.inc("provider.stt.errors")

# TTS:
metrics.inc("provider.tts.requests", labels={"provider": provider_name})
metrics.observe("provider.tts.latency_ms", elapsed_ms, labels={"provider": provider_name})
metrics.observe("provider.tts.audio_duration_ms", audio_duration)
metrics.inc("provider.tts.errors", labels={"provider": provider_name})
```

---

### Phase B6: Per-Job Tracing

**File:** `app/src/aether/core/tracing.py`

Lightweight structured tracing — each job gets a trace with spans:

```python
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Span:
    span_id: str
    name: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict = field(default_factory=dict)

    def finish(self, **attrs):
        self.end_time = time.time()
        self.attributes.update(attrs)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class JobTrace:
    """Collects spans for a single kernel job."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.spans: list[Span] = []
        self._active_span: Span | None = None

    def start_span(self, name: str, parent_id: str | None = None, **attrs) -> Span:
        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            parent_id=parent_id or (self._active_span.span_id if self._active_span else None),
            start_time=time.time(),
            attributes=attrs,
        )
        self.spans.append(span)
        self._active_span = span
        return span

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "parent_id": s.parent_id,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                }
                for s in self.spans
            ],
            "total_ms": sum(s.duration_ms or 0 for s in self.spans if s.parent_id is None),
        }
```

Usage in scheduler:
```python
async def _execute_job(self, job: Job, worker_name: str):
    trace = JobTrace()

    # Root span
    root = trace.start_span("kernel.job", kind=job.request.kind, worker=worker_name)

    # Enqueue delay
    enqueue_delay = time.time() - job.created_at
    metrics.observe("kernel.enqueue_delay_ms", enqueue_delay * 1000,
                    labels={"lane": "interactive" if job.request.priority == 0 else "background"})

    # Service execution span
    svc_span = trace.start_span("service.execute", parent_id=root.span_id)
    result = await self._service_router.route(job.request)
    svc_span.finish(status=result.status)

    root.finish()

    # Attach trace to result
    job.result.metrics["trace"] = trace.to_dict()

    # Log structured trace
    logger.info(f"Job {job.job_id} trace: {trace.to_dict()}")
```

---

### Phase B7: Health & Metrics Endpoints

**File:** `app/src/aether/main.py`

```python
@app.get("/health")
async def health():
    """Comprehensive health check with metrics."""
    m = MetricsCollector.get()
    scheduler_health = scheduler.health_check() if scheduler else {}

    return {
        "status": "healthy",
        "version": "0.09",
        "scheduler": scheduler_health,
        "metrics": m.snapshot(),
    }

@app.get("/metrics")
async def metrics_endpoint():
    """Full metrics snapshot."""
    return MetricsCollector.get().snapshot()

@app.get("/metrics/latency")
async def latency_metrics():
    """SLO-focused latency metrics."""
    m = MetricsCollector.get()
    return {
        "chat_ttft_p50_ms": m.percentile("llm.ttft_ms", 50, labels={"kind": "reply_text"}),
        "chat_ttft_p95_ms": m.percentile("llm.ttft_ms", 95, labels={"kind": "reply_text"}),
        "voice_ttft_p50_ms": m.percentile("llm.ttft_ms", 50, labels={"kind": "reply_voice"}),
        "voice_ttft_p95_ms": m.percentile("llm.ttft_ms", 95, labels={"kind": "reply_voice"}),
        "tts_latency_p50_ms": m.percentile("provider.tts.latency_ms", 50),
        "tts_latency_p95_ms": m.percentile("provider.tts.latency_ms", 95),
        "job_duration_p50_ms": m.percentile("kernel.job.duration_ms", 50),
        "job_duration_p95_ms": m.percentile("kernel.job.duration_ms", 95),
        "notification_decision_p95_ms": m.percentile("service.notification.decision_ms", 95),
        "tool_execution_p95_ms": m.percentile("service.tool.duration_ms", 95),
    }
```

---

### Phase B8: Structured Log Format

**File:** `app/src/aether/core/logging.py`

Add JSON structured logging mode for production:

```python
AETHER_LOG_FORMAT = os.getenv("AETHER_LOG_FORMAT", "text")  # "text" or "json"

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include extra fields (job_id, trace_id, kind, etc.)
        for key in ("job_id", "trace_id", "kind", "worker", "duration_ms", "status"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)
```

Enable via `AETHER_LOG_FORMAT=json`.

---

## Execution Order

Recommended sequencing to minimize risk and maximize early value:

```
Phase 1: Observability (zero risk, immediately useful)
├── B1: MetricsCollector
├── B6: Per-Job Tracing (JobTrace, Span)
├── B8: Structured Log Format
└── B7: Health & Metrics Endpoints

Phase 2: Service Layer Transition (prerequisite for scheduler)
├── A0.1: Add STATUS event to LLMEventEnvelope
├── A0.2: Wire ReplyService into KernelCore
├── A0.3: Wire Background Services
└── A0.4: SLO Gate — do NOT proceed if p95 regresses

Phase 3: Scheduler Wiring (the big change)
├── A1: Wire scheduler behind AETHER_KERNEL_ENABLED=true feature flag
├── B2: Instrument scheduler (metrics at every decision point)
└── SLO Gate — compare scheduler path vs direct path latency

Phase 4: Worker Pool Split
├── A2: Split into P-Core / E-Core pools
├── B3: Instrument services
├── B4: Instrument LLM Core
└── SLO Gate — verify background work doesn't starve interactive

Phase 5: Hardening
├── A3: Bounded queues + backpressure
├── A4: Cancellation on disconnect / supersede
├── B5: Instrument STT/TTS providers
└── A5: Environment config finalized
```

**Phase gate rule (from REFACTOR.md):** Do NOT advance to the next phase if SLO thresholds fail. Fix regressions before proceeding.

---

## Files Modified (in order)

| # | File | Phase | Change |
|---|------|-------|--------|
| 1 | `app/src/aether/core/metrics.py` | B1 | **NEW** — MetricsCollector (counters, histograms, gauges, percentiles) |
| 2 | `app/src/aether/core/tracing.py` | B6 | **NEW** — JobTrace, Span (lightweight structured tracing) |
| 3 | `app/src/aether/core/logging.py` | B8 | Add JSON structured log format |
| 4 | `app/src/aether/main.py` | B7 | Add /health, /metrics, /metrics/latency endpoints |
| 5 | `app/src/aether/llm/contracts.py` | A0.1 | Add `status` event type to LLMEventEnvelope |
| 6 | `app/src/aether/llm/core.py` | A0.1 | Yield status events before tool execution |
| 7 | `app/src/aether/kernel/core.py` | A0.2 | Replace LLMProcessor calls with ReplyService calls |
| 8 | `app/src/aether/kernel/core.py` | A0.3 | Replace MemoryProcessor/EventProcessor with service calls |
| 9 | `app/src/aether/core/config.py` | A1 | Add `kernel_enabled` flag |
| 10 | `app/src/aether/kernel/core.py` | A1 | Wire scheduler with feature flag, background callback bridge, TTS integration |
| 11 | `app/src/aether/kernel/scheduler.py` | A1 | Add ServiceRouter event mapping (`LLMEventEnvelope` → `KernelEvent`) |
| 12 | `app/src/aether/kernel/scheduler.py` | A2 | Split into 2 worker pools (P-Core/E-Core) |
| 13 | `app/src/aether/kernel/scheduler.py` | B2 | Add metrics instrumentation |
| 14 | `app/src/aether/services/reply_service.py` | B3 | Add metrics instrumentation |
| 15 | `app/src/aether/services/memory_service.py` | B3 | Add metrics instrumentation |
| 16 | `app/src/aether/services/notification_service.py` | B3 | Add metrics instrumentation |
| 17 | `app/src/aether/services/tool_service.py` | B3 | Add metrics instrumentation |
| 18 | `app/src/aether/llm/core.py` | B4 | Add metrics (TTFT, duration, tokens, tool iterations) |
| 19 | `app/src/aether/providers/*.py` | B5 | Add STT/TTS metrics |
| 20 | `app/src/aether/kernel/scheduler.py` | A3 | Bounded queues, backpressure |
| 21 | `app/src/aether/kernel/scheduler.py` | A4 | cancel_by_session() with O(n) note |
| 22 | `app/src/aether/transport/websocket.py` | A4 | Wire disconnect → cancel_by_session() |

---

## Env Config Summary

```env
# Multi-worker
AETHER_KERNEL_WORKERS_INTERACTIVE=2
AETHER_KERNEL_WORKERS_BACKGROUND=2
AETHER_INTERACTIVE_QUEUE_LIMIT=20
AETHER_BACKGROUND_QUEUE_LIMIT=50
AETHER_KERNEL_ENABLED=true

# Observability
AETHER_LOG_FORMAT=text          # "text" or "json"
AETHER_LOG_LEVEL=INFO
```

---

## SLO Thresholds (from REFACTOR.md)

| Metric | Pass | Fail |
|--------|------|------|
| /chat TTFT p50 | ≤ 700ms | — |
| /chat TTFT p95 | ≤ 1800ms | > 2000ms |
| Voice first token p50 | ≤ 900ms | — |
| Voice first token p95 | ≤ 2200ms | > 2500ms |
| Voice audio start p50 | ≤ 350ms | — |
| Voice audio start p95 | ≤ 900ms | > 1100ms |
| Tool execution p95 | ≤ +15% baseline | > +25% baseline |
| Notification decide p95 | ≤ 1200ms | > 1500ms |

---

## Verification

1. Start agent with `AETHER_KERNEL_WORKERS_INTERACTIVE=2, BACKGROUND=2`
2. Logs show: `Kernel started: 2 P-Cores, 2 E-Cores`
3. Send `/chat` request → job routed to P-Core, response streams normally
4. During active chat, trigger fact extraction → runs on E-Core, no blocking
5. Hit `/metrics/latency` → p50/p95 values populated
6. Hit `/health` → queue depths, active workers, job counts visible
7. Disconnect client → pending jobs canceled
8. Fill background queue → oldest jobs shed with log warning
9. Set `AETHER_LOG_FORMAT=json` → structured JSON logs
10. Set `AETHER_KERNEL_ENABLED=false` → falls back to direct service calls (Phase A0 path)
11. Tool calls in voice mode → status event emitted → TTS speaks acknowledge before tool runs
12. STT-triggered voice response → scheduler stream consumed via background callback → audio delivered
