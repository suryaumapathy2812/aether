# Transport Rethink + Multi-Worker + Observability Plan

## Vision

Two clean transports. No shared abstractions. Each transport owns its full I/O pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Aether Agent                             │
│                      (one per user)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  HTTP API        │  │  WebRTC Voice     │  │  WS Sidecar   │  │
│  │  /v1/chat/       │  │  VoiceSession     │  │  (notifs,     │  │
│  │  completions     │  │  owns STT + TTS   │  │   status,     │  │
│  │                  │  │  owns audio track  │  │   transcript) │  │
│  │  OpenAI-compat   │  │  per-session STT  │  │               │  │
│  │  stream + sync   │  │  data channel     │  │  push-only    │  │
│  └────────┬─────────┘  └────────┬──────────┘  └───────┬───────┘  │
│           │                     │                     │         │
│           └──────────┬──────────┘                     │         │
│                      ▼                                │         │
│           ┌─────────────────────┐                     │         │
│           │     AgentCore       │─────────────────────┘         │
│           │  (thin facade)      │                               │
│           └──────────┬──────────┘                               │
│                      │                                          │
│           ┌──────────▼──────────┐                               │
│           │  KernelScheduler    │                               │
│           │  2 P-Cores (inter.) │                               │
│           │  2 E-Cores (bg)     │                               │
│           └──────────┬──────────┘                               │
│                      │                                          │
│      ┌───────────────┼───────────────┐                          │
│      ▼               ▼               ▼                          │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────┐           │
│  │ Reply   │  │   Memory     │  │  Notification    │           │
│  │ Service │  │   Service    │  │  Service         │           │
│  └────┬────┘  └──────────────┘  └──────────────────┘           │
│       │                                                         │
│  ┌────▼──────────────────────────────────────────────────────┐  │
│  │  LLMCore + ContextBuilder + ToolOrchestrator              │  │
│  │  (memory retrieval, tool loop, plugin context injection)  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Providers: STT (Deepgram) | LLM (OpenAI) | TTS (multi)  │  │
│  │  Memory Store | Tools | Skills | Plugins                  │  │
│  │  MetricsCollector | JobTrace | Structured Logging         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Gets Removed

| Component | Why |
|-----------|-----|
| `CoreMsg` (`transport/core_msg.py`) | Replaced by direct function calls + KernelRequest |
| `CoreInterface` (`transport/interface.py`) | Transports talk to AgentCore directly |
| `TransportManager` (`transport/manager.py`) | Each transport manages itself |
| `transport/base.py` | No shared transport abstraction needed |
| `transport/pairing.py` | Modality adapters replaced by transport-owned pipelines |
| `modality/` directory | VoiceSession owns voice logic; HTTP owns text logic |
| `KernelCore` (`kernel/core.py`) | Replaced by AgentCore (thinner, no CoreMsg) |
| `processors/llm.py` (LLMProcessor) | Replaced by ReplyService → LLMCore |
| `processors/memory.py` (MemoryProcessor) | Replaced by MemoryService |
| `processors/stt.py` (STTProcessor wrapper) | VoiceSession talks to STT provider directly |

## What Gets Kept

| Component | Why |
|-----------|-----|
| `KernelScheduler` (`kernel/scheduler.py`) | 2P+2E worker pools, job lifecycle |
| `kernel/contracts.py` | KernelRequest, KernelEvent, KernelResult |
| `services/*` | ReplyService, MemoryService, NotificationService, ToolService |
| `llm/core.py` | LLMCore — single LLM entry point |
| `llm/context_builder.py` | Memory + skills + plugins injection |
| `tools/orchestrator.py` | Tool execution with plugin context |
| `tools/base.py`, `tools/registry.py` | Tool infrastructure |
| All providers (`providers/*`) | STT, LLM, TTS |
| `memory/store.py` | Four-tier memory |
| `plugins/*`, `skills/*` | Full plugin ecosystem |
| `core/config.py` | Configuration |
| `core/metrics.py` | MetricsCollector (new) |
| `core/logging.py` | Structured logging |
| `processors/event.py` | EventProcessor (used by NotificationService) |

---

## File Structure After Refactor

```
app/src/aether/
├── main.py                        # FastAPI routes only (slim)
├── agent.py                       # NEW: AgentCore facade
│
├── http/                          # NEW: OpenAI-compatible HTTP API
│   ├── __init__.py
│   └── openai_compat.py           # /v1/chat/completions, /v1/models
│
├── voice/                         # NEW: self-contained voice pipeline
│   ├── __init__.py
│   ├── session.py                 # VoiceSession (owns STT + TTS)
│   └── webrtc.py                  # WebRTC signaling + audio tracks
│
├── ws/                            # NEW: notification sidecar
│   ├── __init__.py
│   └── sidecar.py                 # Push-only WS for notifs/status/transcript
│
├── kernel/                        # KEEP: scheduler
│   ├── scheduler.py               # 2P+2E worker pools
│   └── contracts.py               # KernelRequest, KernelEvent, KernelResult
│
├── services/                      # KEEP
│   ├── reply_service.py
│   ├── memory_service.py
│   ├── notification_service.py
│   └── tool_service.py
│
├── llm/                           # KEEP
│   ├── core.py                    # LLMCore (generate, generate_with_tools)
│   ├── context_builder.py         # ContextBuilder
│   └── contracts.py               # LLMRequestEnvelope, LLMEventEnvelope
│
├── tools/                         # KEEP
│   ├── base.py
│   ├── registry.py
│   ├── orchestrator.py
│   └── *.py                       # Built-in tools
│
├── providers/                     # KEEP
│   ├── deepgram_stt.py
│   ├── openai_llm.py
│   ├── openai_tts.py
│   └── *.py                       # Other providers
│
├── memory/                        # KEEP
│   └── store.py
│
├── plugins/                       # KEEP
│   ├── loader.py
│   ├── event.py
│   └── context.py
│
├── skills/                        # KEEP
│   └── loader.py
│
├── core/                          # KEEP + NEW
│   ├── config.py
│   ├── logging.py                 # + JSON structured format
│   ├── metrics.py                 # NEW: MetricsCollector
│   └── tracing.py                 # NEW: JobTrace, Span
│
├── REMOVE: transport/             # Entire directory
├── REMOVE: modality/              # Entire directory
├── REMOVE: kernel/core.py         # Replaced by agent.py
├── REMOVE: kernel/interface.py    # No more KernelInterface ABC
├── REMOVE: processors/llm.py     # Replaced by ReplyService
├── REMOVE: processors/memory.py  # Replaced by MemoryService
└── REMOVE: processors/stt.py     # VoiceSession owns STT directly
```

---

## Migration Invariants (must never break)

- All current endpoints remain stable during migration (add new, don't remove old until Phase 5)
- Existing wire protocol (text_chunk, audio_chunk, transcript, status, stream_end, tool_result, notification) stays compatible for WS sidecar
- User config behavior preserved (STT/TTS/LLM provider/model selection via env vars)
- Memory writes (facts, actions, sessions) continue working
- Tool and plugin execution unchanged
- No voice roundtrip latency regression

---

## Phase 1: Observability Foundation (zero risk)

Add observability infrastructure first. No behavior changes. Immediately useful for measuring everything that follows.

### 1.1: MetricsCollector

**New file:** `app/src/aether/core/metrics.py`

Singleton in-process metrics. No external dependencies.

```python
class MetricsCollector:
    """In-process metrics: counters, histograms (p50/p95/p99), gauges."""

    _instance: "MetricsCollector | None" = None

    @classmethod
    def get(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._histogram_max_samples = 1000  # Rolling window
        self._gauges: dict[str, float] = defaultdict(float)
        self._last_reset = time.time()

    def inc(self, name: str, value: int = 1, labels: dict | None = None):
        self._counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, labels: dict | None = None):
        key = self._key(name, labels)
        samples = self._histograms[key]
        samples.append(value)
        if len(samples) > self._histogram_max_samples:
            samples.pop(0)

    def gauge_set(self, name: str, value: float, labels: dict | None = None):
        self._gauges[self._key(name, labels)] = value

    def gauge_inc(self, name: str, value: float = 1, labels: dict | None = None):
        self._gauges[self._key(name, labels)] += value

    def gauge_dec(self, name: str, value: float = 1, labels: dict | None = None):
        self._gauges[self._key(name, labels)] -= value

    def percentile(self, name: str, p: float, labels: dict | None = None) -> float | None:
        key = self._key(name, labels)
        samples = sorted(self._histograms.get(key, []))
        if not samples:
            return None
        idx = int(len(samples) * p / 100)
        return samples[min(idx, len(samples) - 1)]

    def snapshot(self) -> dict:
        result = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
            "uptime_seconds": time.time() - self._last_reset,
        }
        for key, samples in self._histograms.items():
            if samples:
                s = sorted(samples)
                result["histograms"][key] = {
                    "count": len(s),
                    "p50": s[len(s) // 2],
                    "p95": s[int(len(s) * 0.95)],
                    "p99": s[int(len(s) * 0.99)],
                    "min": s[0],
                    "max": s[-1],
                }
        return result

    def _key(self, name: str, labels: dict | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
```

### 1.2: JobTrace

**New file:** `app/src/aether/core/tracing.py`

Lightweight per-job tracing with spans.

```python
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
        }
```

### 1.3: Structured Logging

**File:** `app/src/aether/core/logging.py`

Add JSON structured logging mode alongside existing color formatter:

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
        for key in ("job_id", "trace_id", "kind", "worker", "duration_ms", "status"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)
```

Use `AETHER_LOG_FORMAT=json` for structured output. Default remains `text` (current behavior).

### 1.4: Health & Metrics Endpoints

**File:** `app/src/aether/main.py`

```python
@app.get("/health")
async def health():
    m = MetricsCollector.get()
    return {
        "status": "healthy",
        "version": "0.09",
        "metrics": m.snapshot(),
    }

@app.get("/metrics")
async def metrics_endpoint():
    return MetricsCollector.get().snapshot()

@app.get("/metrics/latency")
async def latency_metrics():
    m = MetricsCollector.get()
    return {
        "chat_ttft_p50_ms": m.percentile("llm.ttft_ms", 50, labels={"kind": "reply_text"}),
        "chat_ttft_p95_ms": m.percentile("llm.ttft_ms", 95, labels={"kind": "reply_text"}),
        "voice_ttft_p50_ms": m.percentile("llm.ttft_ms", 50, labels={"kind": "reply_voice"}),
        "voice_ttft_p95_ms": m.percentile("llm.ttft_ms", 95, labels={"kind": "reply_voice"}),
        "tts_latency_p50_ms": m.percentile("provider.tts.latency_ms", 50),
        "tts_latency_p95_ms": m.percentile("provider.tts.latency_ms", 95),
        "notification_decision_p95_ms": m.percentile("service.notification.decision_ms", 95),
    }
```

**Phase 1 exit criteria:** Endpoints return data. No behavior change. All existing tests pass.

---

## Phase 2: AgentCore + Service Layer Wiring

Create the `AgentCore` facade and wire services through it. This replaces `KernelCore`'s direct `LLMProcessor` calls with service calls. Old transport layer still works during this phase — we're changing the internals, not the surface.

### 2.1: AgentCore Facade

**New file:** `app/src/aether/agent.py`

```python
class AgentCore:
    """
    Single interface for all transports.
    Wraps the scheduler and services behind simple async methods.
    """

    def __init__(
        self,
        scheduler: KernelScheduler,
        reply_service: ReplyService,
        memory_service: MemoryService,
        notification_service: NotificationService,
        memory_store: MemoryStore,
        tool_registry: ToolRegistry,
        skill_loader: SkillLoader,
        plugin_context_store: PluginContextStore,
    ):
        self._scheduler = scheduler
        self._reply_service = reply_service
        self._memory_service = memory_service
        self._notification_service = notification_service
        self._memory_store = memory_store
        self._tool_registry = tool_registry
        self._skill_loader = skill_loader
        self._plugin_context = plugin_context_store

        # Conversation history per session (in-memory, single user)
        self._sessions: dict[str, list[dict]] = {}

        # Notification subscribers (WS sidecar connections)
        self._notification_subscribers: list[Callable] = []

        # Voice greeting flag
        self._has_greeted = False

    async def start(self):
        """Start scheduler."""
        await self._scheduler.start()

    async def stop(self):
        """Stop scheduler."""
        await self._scheduler.stop()

    # ─── Reply (used by HTTP + Voice) ───────────────────────────

    async def generate_reply(
        self,
        text: str,
        session_id: str,
        history: list[dict] | None = None,
        vision: dict | None = None,
    ) -> AsyncGenerator[KernelEvent, None]:
        """
        Submit a reply job to the scheduler. Yields KernelEvents.
        Caller (HTTP or VoiceSession) decides how to render them.
        """
        if history is None:
            history = self._sessions.get(session_id, [])

        request = KernelRequest(
            kind="reply_text",
            modality="text",
            user_id=os.getenv("AETHER_USER_ID", ""),
            session_id=session_id,
            payload={
                "text": text,
                "history": history,
                "vision": vision,
            },
            priority=JobPriority.INTERACTIVE.value,
        )

        job_id = await self._scheduler.submit(request)

        collected_text = []
        async for event in self._scheduler.stream(job_id):
            if event.stream_type == "text_chunk":
                collected_text.append(event.data.get("text", ""))
            yield event

        # Update session history
        full_response = "".join(collected_text).strip()
        session_history = self._sessions.setdefault(session_id, [])
        session_history.append({"role": "user", "content": text})
        if full_response:
            session_history.append({"role": "assistant", "content": full_response})

        # Fire background jobs on E-Cores
        await self._submit_background_jobs(text, full_response, session_id)

    async def generate_reply_voice(
        self,
        text: str,
        session_id: str,
    ) -> AsyncGenerator[KernelEvent, None]:
        """Same as generate_reply but tagged as voice for service routing."""
        history = self._sessions.get(session_id, [])

        request = KernelRequest(
            kind="reply_voice",
            modality="voice",
            user_id=os.getenv("AETHER_USER_ID", ""),
            session_id=session_id,
            payload={"text": text, "history": history},
            priority=JobPriority.INTERACTIVE.value,
        )

        job_id = await self._scheduler.submit(request)

        collected_text = []
        async for event in self._scheduler.stream(job_id):
            if event.stream_type == "text_chunk":
                collected_text.append(event.data.get("text", ""))
            yield event

        full_response = "".join(collected_text).strip()
        session_history = self._sessions.setdefault(session_id, [])
        session_history.append({"role": "user", "content": text})
        if full_response:
            session_history.append({"role": "assistant", "content": full_response})

        await self._submit_background_jobs(text, full_response, session_id)

    # ─── Background Jobs ────────────────────────────────────────

    async def _submit_background_jobs(
        self, user_text: str, assistant_text: str, session_id: str
    ):
        """Submit fact extraction and other background work to E-Cores."""
        # Fact extraction
        await self._scheduler.submit(KernelRequest(
            kind="fact_extraction",
            modality="system",
            user_id=os.getenv("AETHER_USER_ID", ""),
            session_id=session_id,
            payload={"user_text": user_text, "assistant_text": assistant_text},
            priority=JobPriority.BACKGROUND.value,
        ))

    async def process_notification(self, event) -> dict:
        """Submit notification decision to E-Core. Returns decision."""
        request = KernelRequest(
            kind="notification_decision",
            modality="system",
            user_id=os.getenv("AETHER_USER_ID", ""),
            session_id="notifications",
            payload={"event": event},
            priority=JobPriority.BACKGROUND.value,
        )
        job_id = await self._scheduler.submit(request)
        result = await self._scheduler.await_result(job_id, timeout_ms=10000)
        return result.payload

    # ─── Session Management ─────────────────────────────────────

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def clear_history(self, session_id: str):
        self._sessions.pop(session_id, None)

    # ─── Greeting ───────────────────────────────────────────────

    async def generate_greeting(self) -> str | None:
        """Generate voice greeting. Returns text or None if already greeted."""
        if self._has_greeted:
            return None
        self._has_greeted = True
        # Use LLM to generate personalized greeting based on memory
        # Implementation delegates to a simple LLM call via reply_service
        # or a dedicated greeting method
        return await self._reply_service.generate_greeting()

    # ─── Notification Subscribers ───────────────────────────────

    def subscribe_notifications(self, callback: Callable):
        self._notification_subscribers.append(callback)

    def unsubscribe_notifications(self, callback: Callable):
        self._notification_subscribers.remove(callback)

    async def broadcast_notification(self, notification: dict):
        for cb in self._notification_subscribers:
            try:
                await cb(notification)
            except Exception:
                pass
```

### 2.2: Wire STATUS Event Into LLMCore

**File:** `app/src/aether/llm/contracts.py`

Add `status` to LLMEventEnvelope event_type options.

**File:** `app/src/aether/llm/core.py`

In `generate_with_tools()`, yield status event before each tool execution:

```python
for tool_call in tool_calls:
    # Emit status for voice acknowledge ("Let me check that...")
    yield LLMEventEnvelope(
        kind=envelope.kind,
        event_type="status",
        data={"message": f"Using {tool_call.tool_name}...", "tool_name": tool_call.tool_name},
    )

    result = await self._tool_orchestrator.execute(tool_call, envelope.plugin_context)
    yield LLMEventEnvelope(
        kind=envelope.kind,
        event_type="tool_result",
        data={"tool_name": tool_call.tool_name, "output": result.output, "error": result.error},
    )
```

### 2.3: Instrument Services with Metrics

Add `MetricsCollector` calls to all services:

**ReplyService:**
```python
metrics.inc("service.reply.started", labels={"kind": kind})
# ... generate ...
metrics.observe("service.reply.duration_ms", elapsed_ms, labels={"kind": kind})
metrics.observe("service.reply.tokens", token_count)
```

**MemoryService:**
```python
metrics.inc("service.memory.extraction.started")
metrics.inc("service.memory.facts_extracted", value=num_facts)
metrics.observe("service.memory.extraction_ms", elapsed_ms)
```

**NotificationService:**
```python
metrics.inc("service.notification.processed", labels={"decision": decision})
metrics.observe("service.notification.decision_ms", elapsed_ms)
```

**ToolService:**
```python
metrics.inc("service.tool.executed", labels={"tool": tool_name, "error": str(error)})
metrics.observe("service.tool.duration_ms", elapsed_ms, labels={"tool": tool_name})
```

**LLMCore:**
```python
metrics.inc("llm.calls", labels={"kind": envelope.kind})
metrics.observe("llm.ttft_ms", time_to_first_token_ms, labels={"kind": envelope.kind})
metrics.observe("llm.duration_ms", total_ms, labels={"kind": envelope.kind})
metrics.observe("llm.tokens.prompt", prompt_tokens)
metrics.observe("llm.tokens.completion", completion_tokens)
```

**Providers:**
```python
# STT
metrics.inc("provider.stt.connections")
metrics.observe("provider.stt.utterance_duration_ms", duration)
metrics.inc("provider.stt.errors")

# TTS
metrics.inc("provider.tts.requests", labels={"provider": name})
metrics.observe("provider.tts.latency_ms", elapsed_ms)
metrics.inc("provider.tts.errors")
```

**Phase 2 exit criteria:** AgentCore exists. Services wired. Metrics populated. Old transport layer still works. All tests pass.

---

## Phase 3: OpenAI-Compatible HTTP API

### 3.1: Endpoint Implementation

**New file:** `app/src/aether/http/openai_compat.py`

```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    tools = body.get("tools", None)  # Optional: client-provided tools
    user = body.get("user", "")

    # Extract the latest user message
    # Support both OpenAI format and multimodal (vision)
    last_msg = _extract_last_user_message(messages)
    vision = _extract_vision(messages)

    # Session ID from user field or generate
    session_id = f"http-{user or 'anon'}"

    # Generate completion ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = os.getenv("AETHER_LLM_MODEL", "aether")

    if stream:
        return StreamingResponse(
            _stream_response(completion_id, created, model, last_msg, session_id, vision),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _sync_response(completion_id, created, model, last_msg, session_id, vision)


async def _stream_response(
    completion_id: str,
    created: int,
    model: str,
    text: str,
    session_id: str,
    vision: dict | None,
):
    """SSE stream in OpenAI format."""

    # Role chunk
    yield _sse({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    })

    # Stream from AgentCore
    finish_reason = "stop"
    async for event in agent_core.generate_reply(text, session_id, vision=vision):
        if event.stream_type == "text_chunk":
            chunk_text = event.data.get("text", "")
            if chunk_text:
                yield _sse({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }],
                })

        elif event.stream_type == "tool_call":
            # Stream tool calls in OpenAI format
            tc = event.data
            yield _sse({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": tc.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("tool_name", ""),
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            })
            finish_reason = "tool_calls"

        elif event.stream_type == "tool_result":
            # Tool results are internal — don't stream to client
            # (the LLM loop handles them and produces text output)
            pass

        elif event.stream_type == "status":
            # Status events (tool acknowledge) — skip in HTTP, optional to include
            pass

        elif event.stream_type == "done":
            pass

    # Final chunk with finish_reason
    yield _sse({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    })

    yield "data: [DONE]\n\n"


async def _sync_response(
    completion_id: str,
    created: int,
    model: str,
    text: str,
    session_id: str,
    vision: dict | None,
) -> JSONResponse:
    """Non-streaming response in OpenAI format."""
    collected = []
    async for event in agent_core.generate_reply(text, session_id, vision=vision):
        if event.stream_type == "text_chunk":
            collected.append(event.data.get("text", ""))

    content = "".join(collected)
    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,   # Fill from metrics if available
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


@router.get("/v1/models")
async def list_models():
    """Return configured model."""
    model = os.getenv("AETHER_LLM_MODEL", "aether")
    return {
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "aether",
        }],
    }


def _extract_last_user_message(messages: list[dict]) -> str:
    """Extract text from last user message. Handles OpenAI and multimodal formats."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multimodal: extract text parts
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            return " ".join(texts)
    return ""


def _extract_vision(messages: list[dict]) -> dict | None:
    """Extract base64 image from last user message if present."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # data:image/jpeg;base64,...
                    mime = url.split(";")[0].split(":")[1]
                    data = url.split(",", 1)[1]
                    return {"mime": mime, "data": data}
    return None


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
```

### 3.2: Mount in main.py

```python
from aether.http.openai_compat import router as openai_router
app.include_router(openai_router)
```

**Phase 3 exit criteria:**

1. `curl -X POST /v1/chat/completions -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'` → SSE stream in OpenAI format
2. `curl -X POST /v1/chat/completions -d '{"messages":[{"role":"user","content":"hello"}],"stream":false}'` → JSON response
3. OpenAI Python SDK works: `client = OpenAI(base_url="http://localhost:8000/v1"); client.chat.completions.create(...)`
4. Vision works: multimodal content array with base64 image
5. Tool calls appear in stream with correct OpenAI format
6. `/v1/models` returns configured model
7. Memory context injected (ask "what's my name" after telling it)

---

## Phase 4: Simplified WebRTC Voice Transport

### 4.1: VoiceSession

**New file:** `app/src/aether/voice/session.py`

```python
class VoiceSession:
    """
    Owns the full voice pipeline for one WebRTC connection.
    Each session creates its own STT instance (no shared singleton).
    """

    def __init__(
        self,
        agent: AgentCore,
        tts_provider: TTSProvider,
        stt_config: dict,
        session_id: str,
    ):
        self.agent = agent
        self.tts_provider = tts_provider
        self.session_id = session_id

        # Per-session STT (fixes shared singleton bug)
        self.stt = DeepgramSTTProvider(config=stt_config)

        # State
        self.is_streaming = False
        self.is_responding = False
        self.is_muted = False
        self.accumulated_transcript = ""
        self._debounce_task: asyncio.Task | None = None
        self._stt_event_task: asyncio.Task | None = None

        # Callbacks (set by WebRTC transport)
        self.on_audio_out: Callable[[bytes], Awaitable[None]] | None = None
        self.on_text_event: Callable[[dict], Awaitable[None]] | None = None

    # ─── Lifecycle ──────────────────────────────────────────

    async def start(self):
        """Called on stream_start. Connect STT, start listening."""
        self.is_streaming = True
        await self.stt.connect_stream()
        self._stt_event_task = asyncio.create_task(self._stt_event_loop())

        # Greeting (first connection only)
        greeting = await self.agent.generate_greeting()
        if greeting:
            await self._send_text_event("transcript", greeting, role="assistant")
            await self._synthesize_and_send(greeting)
            await self._send_text_event("stream_end", "")

    async def stop(self):
        """Called on disconnect. Cleanup everything."""
        self.is_streaming = False
        if self._debounce_task:
            self._debounce_task.cancel()
        if self._stt_event_task:
            self._stt_event_task.cancel()
        await self.stt.disconnect_stream()

    # ─── Audio Input ────────────────────────────────────────

    async def on_audio_in(self, pcm_bytes: bytes):
        """Raw PCM from WebRTC audio track → STT."""
        if not self.is_streaming or self.is_muted:
            return
        await self.stt.send_audio(pcm_bytes)

    # ─── STT Event Loop ────────────────────────────────────

    async def _stt_event_loop(self):
        """Listen to STT events, trigger response on utterance end."""
        try:
            async for event in self.stt.stream_events():
                if self.is_muted:
                    continue

                if event.type == FrameType.TEXT and event.metadata.get("interim"):
                    if not self.is_responding:
                        await self._send_text_event(
                            "transcript", event.data, interim=True
                        )

                elif event.type == FrameType.CONTROL:
                    action = event.data.get("action", "")
                    if action == "utterance_end":
                        transcript = event.data.get("transcript", "")
                        self.accumulated_transcript += " " + transcript
                        self.accumulated_transcript = self.accumulated_transcript.strip()

                        # Cancel old debounce, start new
                        if self._debounce_task:
                            self._debounce_task.cancel()
                        self._debounce_task = asyncio.create_task(
                            self._debounce_and_trigger()
                        )
        except asyncio.CancelledError:
            pass

    async def _debounce_and_trigger(self):
        """Wait for silence, then trigger voice response."""
        try:
            await asyncio.sleep(0.5)  # Configurable debounce delay

            text = self.accumulated_transcript.strip()
            self.accumulated_transcript = ""
            if not text:
                return

            await self._trigger_response(text)
        except asyncio.CancelledError:
            pass  # User still speaking, debounce reset

    # ─── Voice Response ─────────────────────────────────────

    async def _trigger_response(self, text: str):
        """Complete voice response: LLM → TTS → audio out."""
        self.is_responding = True
        metrics = MetricsCollector.get()

        try:
            # Send final transcript
            await self._send_text_event("transcript", text, interim=False)
            await self._send_text_event("status", "thinking...")

            # Stream from AgentCore
            sentence_buffer = ""
            response_start = time.time()

            async for event in self.agent.generate_reply_voice(text, self.session_id):
                if event.stream_type == "text_chunk":
                    chunk = event.data.get("text", "")
                    await self._send_text_event("text_chunk", chunk)

                    # Sentence-level TTS
                    sentence_buffer += chunk
                    sentences = _split_sentences(sentence_buffer)
                    for sentence in sentences[:-1]:
                        await self._synthesize_and_send(sentence)
                    sentence_buffer = sentences[-1] if sentences else ""

                elif event.stream_type == "status":
                    # Tool acknowledge — speak it
                    status_text = event.data.get("message", "")
                    await self._send_text_event("status", status_text)
                    await self._synthesize_and_send(status_text)

                elif event.stream_type == "tool_result":
                    tool_name = event.data.get("tool_name", "")
                    await self._send_text_event("tool_result", json.dumps(event.data))

                elif event.stream_type == "done":
                    pass

            # Flush remaining text
            if sentence_buffer.strip():
                await self._synthesize_and_send(sentence_buffer.strip())

            await self._send_text_event("stream_end", "")

            elapsed_ms = (time.time() - response_start) * 1000
            metrics.observe("voice.response_ms", elapsed_ms)

        finally:
            self.is_responding = False
            await self._send_text_event("status", "listening...")

    # ─── TTS ────────────────────────────────────────────────

    async def _synthesize_and_send(self, text: str):
        """Text → TTS → audio out via callback."""
        if not text.strip() or not self.on_audio_out:
            return
        try:
            audio_bytes = await self.tts_provider.synthesize(text)
            await self.on_audio_out(audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed: {e}")

    # ─── Event Helpers ──────────────────────────────────────

    async def _send_text_event(self, event_type: str, data: str, **kwargs):
        """Send event via data channel callback."""
        if self.on_text_event:
            await self.on_text_event({
                "type": event_type,
                "data": data,
                **kwargs,
            })

    # ─── Control ────────────────────────────────────────────

    def mute(self):
        self.is_muted = True

    def unmute(self):
        self.is_muted = False


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries for TTS chunking."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return parts if parts else [text]
```

### 4.2: Simplified WebRTC Transport

**New file:** `app/src/aether/voice/webrtc.py`

```python
class WebRTCTransport:
    """
    Simplified WebRTC transport.
    Creates a VoiceSession per connection.
    No CoreMsg, no TransportManager.
    """

    def __init__(self, agent: AgentCore, tts_provider: TTSProvider, stt_config: dict):
        self.agent = agent
        self.tts_provider = tts_provider
        self.stt_config = stt_config
        self._sessions: dict[str, WebRTCConnection] = {}

    async def handle_offer(self, sdp: str, user_id: str, pc_id: str | None = None) -> dict:
        """Handle SDP offer → create peer connection + VoiceSession."""

        pc_id = pc_id or f"pc-{uuid.uuid4().hex[:12]}"
        pc = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        ))

        # Create voice session
        voice_session = VoiceSession(
            agent=self.agent,
            tts_provider=self.tts_provider,
            stt_config=self.stt_config,
            session_id=f"webrtc-{pc_id}",
        )

        # Create outbound audio track
        audio_track = RawAudioTrack(sample_rate=48000)
        pc.addTrack(audio_track)

        # Wire voice session audio output → RawAudioTrack
        async def send_audio(pcm_bytes: bytes):
            # Resample if needed (TTS typically 24kHz → 48kHz for WebRTC)
            resampled = _resample_if_needed(pcm_bytes, src_rate=24000, dst_rate=48000)
            audio_track.add_audio_bytes(resampled)

        voice_session.on_audio_out = send_audio

        # Store connection
        conn = WebRTCConnection(
            pc_id=pc_id,
            pc=pc,
            voice_session=voice_session,
            audio_track=audio_track,
            data_channel=None,
        )
        self._sessions[pc_id] = conn

        # Setup event handlers
        self._setup_handlers(conn)

        # Set remote description and create answer
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "pc_id": pc_id,
        }

    async def handle_ice(self, pc_id: str, candidates: list[dict]):
        """Add ICE candidates to peer connection."""
        conn = self._sessions.get(pc_id)
        if not conn:
            return
        for c in candidates:
            candidate = RTCIceCandidate(
                sdpMid=c.get("sdpMid"),
                sdpMLineIndex=c.get("sdpMLineIndex"),
                candidate=c.get("candidate", ""),
            )
            await conn.pc.addIceCandidate(candidate)

    def _setup_handlers(self, conn: "WebRTCConnection"):
        pc = conn.pc

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            if state == "connected":
                logger.info(f"WebRTC connected: {conn.pc_id}")
            elif state in ("disconnected", "failed", "closed"):
                logger.info(f"WebRTC disconnected: {conn.pc_id}")
                await conn.voice_session.stop()
                self._sessions.pop(conn.pc_id, None)

        @pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                asyncio.create_task(self._read_audio_loop(conn, track))

        @pc.on("datachannel")
        async def on_datachannel(channel):
            conn.data_channel = channel

            # Wire text events from VoiceSession → data channel
            async def send_event(event: dict):
                if conn.data_channel and conn.data_channel.readyState == "open":
                    conn.data_channel.send(json.dumps(event))

            conn.voice_session.on_text_event = send_event

            @channel.on("message")
            async def on_message(message):
                await self._handle_data_message(conn, message)

    async def _read_audio_loop(self, conn: "WebRTCConnection", track):
        """Read audio frames from WebRTC → VoiceSession."""
        try:
            while True:
                frame = await track.recv()
                if not conn.voice_session.is_streaming:
                    continue  # Drop until stream_start

                # Convert to PCM16 mono
                pcm = frame.to_ndarray().flatten()
                if frame.sample_rate != 16000:
                    pcm = _resample(pcm, frame.sample_rate, 16000)
                pcm_bytes = pcm.astype(np.int16).tobytes()

                await conn.voice_session.on_audio_in(pcm_bytes)
        except Exception:
            pass  # Connection closed

    async def _handle_data_message(self, conn: "WebRTCConnection", message: str):
        """Handle data channel messages (same protocol as old WebSocket)."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "stream_start":
                await conn.voice_session.start()

            elif msg_type == "stream_stop":
                await conn.voice_session.stop()

            elif msg_type == "mute":
                conn.voice_session.mute()

            elif msg_type == "unmute":
                conn.voice_session.unmute()

            elif msg_type == "text":
                # Text input via data channel (rare but supported)
                text = data.get("data", "")
                if text:
                    await conn.voice_session._trigger_response(text)

        except json.JSONDecodeError:
            pass


@dataclass
class WebRTCConnection:
    pc_id: str
    pc: RTCPeerConnection
    voice_session: VoiceSession
    audio_track: RawAudioTrack
    data_channel: RTCDataChannel | None
```

### 4.3: Mount in main.py

```python
from aether.voice.webrtc import WebRTCTransport

webrtc_transport = WebRTCTransport(
    agent=agent_core,
    tts_provider=tts_provider,
    stt_config={"model": config.stt.model, "language": config.stt.language},
)

@app.post("/webrtc/offer")
async def webrtc_offer(request: Request):
    body = await request.json()
    result = await webrtc_transport.handle_offer(
        sdp=body["sdp"],
        user_id=body.get("user_id", ""),
        pc_id=body.get("pc_id"),
    )
    return JSONResponse(result)

@app.patch("/webrtc/ice")
async def webrtc_ice(request: Request):
    body = await request.json()
    await webrtc_transport.handle_ice(body["pc_id"], body.get("candidates", []))
    return JSONResponse({"status": "ok"})
```

**Phase 4 exit criteria:**

1. WebRTC offer/answer exchange works
2. Audio from mic → STT → transcript appears on data channel
3. Transcript → LLM → TTS → audio plays on speaker
4. Greeting speaks on first connection only
5. Mute/unmute work
6. Data channel receives: transcript, text_chunk, status, tool_result, stream_end
7. Disconnect cleanly stops STT, no leaked connections
8. Two WebRTC sessions simultaneously don't interfere (separate STT instances)

---

## Phase 5: WebSocket Notification Sidecar

### 5.1: Sidecar Implementation

**New file:** `app/src/aether/ws/sidecar.py`

```python
class WSSidecar:
    """
    Push-only WebSocket for dashboard.
    Receives: notifications, status updates, transcript display.
    No voice, no text chat.
    """

    def __init__(self, agent: AgentCore):
        self.agent = agent
        self._connections: list[WebSocket] = []

    async def handle_connection(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)

        # Subscribe to agent notifications
        async def on_notification(notif: dict):
            await self._send(ws, "notification", notif)

        self.agent.subscribe_notifications(on_notification)

        try:
            while True:
                # Listen for client messages (notification_feedback, etc.)
                data = await ws.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "notification_feedback":
                    feedback = data.get("data", {})
                    # Store preference in memory
                    await self._handle_feedback(feedback)

        except WebSocketDisconnect:
            pass
        finally:
            self._connections.remove(ws)
            self.agent.unsubscribe_notifications(on_notification)

    async def push(self, event_type: str, data: dict):
        """Push event to all connected sidecar clients."""
        for ws in list(self._connections):
            try:
                await self._send(ws, event_type, data)
            except Exception:
                self._connections.remove(ws)

    async def _send(self, ws: WebSocket, event_type: str, data):
        await ws.send_json({"type": event_type, "data": data})

    async def _handle_feedback(self, feedback: dict):
        action = feedback.get("action", "")
        plugin = feedback.get("plugin", "unknown")
        sender = feedback.get("sender", "")

        fact = ""
        if action == "engaged":
            fact = f"User immediately reads {plugin} notifications from {sender}"
        elif action == "dismissed":
            fact = f"User dismisses {plugin} notifications from {sender}"
        elif action == "muted":
            fact = f"User wants to mute all {plugin} notifications from {sender}"

        if fact:
            await self.agent._memory_store.store_preference(fact)
```

### 5.2: Mount in main.py

```python
from aether.ws.sidecar import WSSidecar

ws_sidecar = WSSidecar(agent=agent_core)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_sidecar.handle_connection(websocket)
```

**Phase 5 exit criteria:** Dashboard connects via WS, receives notifications. Feedback stored in memory.

---

## Phase 6: Multi-Worker Scheduler (2P + 2E)

### 6.1: Split Worker Pools

**File:** `app/src/aether/kernel/scheduler.py`

Replace single worker pool with isolated P-Core and E-Core pools:

```python
class KernelScheduler:
    def __init__(
        self,
        service_router: ServiceRouter,
        max_interactive_workers: int = 2,   # P-Cores
        max_background_workers: int = 2,    # E-Cores
    ):
        self._service_router = service_router

        # Separate queues
        self._interactive_queue: deque[Job] = deque()
        self._background_queue: deque[Job] = deque()

        # Separate semaphores
        self._interactive_sem = asyncio.Semaphore(max_interactive_workers)
        self._background_sem = asyncio.Semaphore(max_background_workers)

        # Separate signals
        self._interactive_event = asyncio.Event()
        self._background_event = asyncio.Event()

        self._max_interactive = max_interactive_workers
        self._max_background = max_background_workers

    async def start(self):
        self._running = True
        self._workers = []

        for i in range(self._max_interactive):
            task = asyncio.create_task(self._worker_loop(
                name=f"P-Core-{i}",
                queue=self._interactive_queue,
                semaphore=self._interactive_sem,
                event=self._interactive_event,
            ))
            self._workers.append(task)

        for i in range(self._max_background):
            task = asyncio.create_task(self._worker_loop(
                name=f"E-Core-{i}",
                queue=self._background_queue,
                semaphore=self._background_sem,
                event=self._background_event,
            ))
            self._workers.append(task)

        logger.info(
            f"Kernel started: {self._max_interactive} P-Cores, "
            f"{self._max_background} E-Cores"
        )

    async def _worker_loop(self, name, queue, semaphore, event):
        """Generic worker loop. P-Cores only touch interactive queue.
        E-Cores only touch background queue. Complete isolation."""
        while self._running:
            if not queue:
                event.clear()
                await event.wait()
            if not self._running:
                break
            try:
                job = queue.popleft()
            except IndexError:
                continue
            async with semaphore:
                await self._execute_job(job, worker_name=name)

    async def submit(self, request: KernelRequest) -> str:
        job = self._create_job(request)
        metrics = MetricsCollector.get()

        if request.priority == JobPriority.INTERACTIVE.value:
            self._interactive_queue.append(job)
            self._interactive_event.set()
            metrics.gauge_set("kernel.queue.interactive", len(self._interactive_queue))
        else:
            self._background_queue.append(job)
            self._background_event.set()
            metrics.gauge_set("kernel.queue.background", len(self._background_queue))

        metrics.inc("kernel.jobs.submitted", labels={"kind": request.kind})
        return job.job_id

    async def _execute_job(self, job: Job, worker_name: str):
        metrics = MetricsCollector.get()
        metrics.gauge_inc("kernel.workers.active", labels={"pool": worker_name.split("-")[0]})

        start = time.time()
        try:
            async for event in self._service_router.route_streaming(job):
                job.events.append(event)
                job.done_event.set()

            elapsed_ms = (time.time() - start) * 1000
            job.status = JobStatus.COMPLETED
            metrics.observe("kernel.job.duration_ms", elapsed_ms, labels={"kind": job.request.kind})
            metrics.inc("kernel.jobs.completed", labels={"kind": job.request.kind})

        except Exception as e:
            job.status = JobStatus.FAILED
            metrics.inc("kernel.jobs.failed", labels={"kind": job.request.kind})
            logger.error(f"Job {job.job_id} failed: {e}")

        finally:
            metrics.gauge_dec("kernel.workers.active", labels={"pool": worker_name.split("-")[0]})

    async def cancel_by_session(self, session_id: str, kinds: list[str] | None = None) -> int:
        """Cancel all pending/running jobs for a session."""
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

### 6.2: Environment Config

```env
AETHER_KERNEL_WORKERS_INTERACTIVE=2   # P-Cores
AETHER_KERNEL_WORKERS_BACKGROUND=2    # E-Cores
AETHER_LOG_FORMAT=text                # "text" or "json"
AETHER_LOG_LEVEL=INFO
```

**Phase 6 exit criteria:**

1. Logs: `Kernel started: 2 P-Cores, 2 E-Cores`
2. Chat reply → routed to P-Core
3. Fact extraction → routed to E-Core, doesn't block reply
4. `/health` shows queue depths, active workers
5. `/metrics/latency` shows p50/p95 for all paths

---

## Phase 7: Cleanup

Remove old files:

```
DELETE: app/src/aether/transport/          # Entire directory
DELETE: app/src/aether/modality/           # Entire directory
DELETE: app/src/aether/kernel/core.py      # Replaced by agent.py
DELETE: app/src/aether/kernel/interface.py  # No longer needed
DELETE: app/src/aether/processors/llm.py   # Replaced by ReplyService
DELETE: app/src/aether/processors/memory.py # Replaced by MemoryService
DELETE: app/src/aether/processors/stt.py   # VoiceSession owns STT
```

Keep `processors/event.py` (used by NotificationService).

Update imports across the codebase. Run full test suite.

---

## Execution Order

```
Phase 1: Observability (zero risk, no behavior change)
├── 1.1: MetricsCollector
├── 1.2: JobTrace + Span
├── 1.3: Structured logging (JSON mode)
└── 1.4: /health, /metrics, /metrics/latency endpoints

Phase 2: AgentCore + Service Wiring (internal change, old transport still works)
├── 2.1: AgentCore facade
├── 2.2: STATUS event in LLMCore
└── 2.3: Instrument services + providers with metrics

Phase 3: OpenAI HTTP API (additive, new endpoint)
├── 3.1: /v1/chat/completions (stream + sync)
├── 3.2: /v1/models
└── Test: OpenAI Python SDK works against it

Phase 4: Simplified WebRTC (new voice transport)
├── 4.1: VoiceSession (per-session STT, owns TTS)
├── 4.2: WebRTCTransport (signaling + audio)
└── Test: Full voice roundtrip, greeting, mute, disconnect

Phase 5: WS Notification Sidecar
├── 5.1: WSSidecar (push-only)
└── Test: Dashboard receives notifications

Phase 6: Multi-Worker Scheduler (2P + 2E)
├── 6.1: Split worker pools
├── 6.2: Instrument scheduler
└── Test: Background jobs don't block interactive

Phase 7: Cleanup
├── Delete old transport/, modality/, kernel/core.py
├── Delete processors/llm.py, processors/memory.py, processors/stt.py
└── Full test suite green
```

---

## SLO Thresholds

| Metric | Pass | Fail |
|--------|------|------|
| /v1/chat/completions TTFT p50 | ≤ 700ms | — |
| /v1/chat/completions TTFT p95 | ≤ 1800ms | > 2000ms |
| Voice first token p50 | ≤ 900ms | — |
| Voice first token p95 | ≤ 2200ms | > 2500ms |
| Voice audio start p50 | ≤ 350ms | — |
| Voice audio start p95 | ≤ 900ms | > 1100ms |
| Tool execution p95 | ≤ +15% baseline | > +25% baseline |
| Notification decide p95 | ≤ 1200ms | > 1500ms |

**Phase gate rule:** Do NOT advance to next phase if SLO fails. Fix first.

---

## Verification Checklist

- [ ] Phase 1: `/health` and `/metrics` return data
- [ ] Phase 2: AgentCore works, services produce metrics
- [ ] Phase 3: `openai.Client(base_url=...).chat.completions.create()` works (stream + sync)
- [ ] Phase 3: Vision (base64 image in content array) works
- [ ] Phase 3: Tool calls appear in SSE stream
- [ ] Phase 4: WebRTC audio → STT → LLM → TTS → audio roundtrip
- [ ] Phase 4: Greeting on first connect only
- [ ] Phase 4: Mute/unmute work
- [ ] Phase 4: Two simultaneous WebRTC sessions don't interfere
- [ ] Phase 5: Dashboard WS receives notifications
- [ ] Phase 5: Notification feedback stored as memory fact
- [ ] Phase 6: Background jobs run on E-Cores without blocking P-Cores
- [ ] Phase 6: `/metrics/latency` shows all SLO metrics
- [ ] Phase 7: All old files removed, full test suite green
