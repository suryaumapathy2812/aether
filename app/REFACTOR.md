Before (Current in app/)
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                               CLIENTS                                                         │
│  Dashboard (HTTP /chat, WS text)             iOS (WebRTC voice + data channel)            Plugin webhooks   │
└───────────────┬───────────────────────────────────────┬───────────────────────────────────────────────┬──────┘
                │                                       │                                               │
                ▼                                       ▼                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      FastAPI Entrypoints (main.py)                                          │
│  /chat   /ws   /webrtc/offer   /webrtc/ice   /plugin_event   /plugin_event/batch   /memory/*   /config     │
└───────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         TRANSPORT LAYER                                                       │
│  WebSocketTransport                      WebRTCTransport                                                     │
│  - normalize wire -> CoreMsg             - signaling + pc lifecycle                                          │
│  - serialize CoreMsg -> wire             - audio track in/out + data channel                                │
└───────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┘
                │ CoreMsg
                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                TransportManager + CoreHandler (BIG CENTER)                                   │
│                                                                                                              │
│  CoreHandler currently mixes:                                                                                │
│  - session state                                                                                             │
│  - mode switching (text/voice)                                                                               │
│  - STT stream loop + debounce                                                                                │
│  - text pipeline (Memory -> LLM -> text_chunk)                                                               │
│  - voice pipeline (Memory -> LLM -> TTS -> audio_chunk)                                                      │
│  - status events + stream_end                                                                                │
│  - notification feedback                                                                                      │
│  - session summary trigger                                                                                    │
└───────┬───────────────────────────────┬───────────────────────────────┬──────────────────────────────────────┘
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐               ┌───────────────┐               ┌──────────────────────┐
│   PROCESSORS  │               │   PROVIDERS   │               │   TOOLS / SKILLS     │
│ MemoryRetr.   │◄─────────────►│ LLM (OpenAI)  │◄─────────────►│ ToolRegistry         │
│ LLMProcessor  │               │ STT (Deepgram)│               │ TaskRunner           │
│ STTProcessor  │               │ TTS (OpenAI/…)│               │ Plugin tools         │
│ TTSProcessor  │               └───────────────┘               │ SkillLoader          │
│ EventProcessor│                                                   └──────────────────────┘
└───────┬───────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             MEMORY STORE                                                      │
│  conversations, facts, actions, session_summaries, embeddings (SQLite + aiosqlite)                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
---
After (Target Kernel Architecture)

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            CLIENTS                                                           │
│   Dashboard (/chat)              iOS (WebRTC)                 Plugin Webhooks          Internal Timers     │
└──────────────┬───────────────────────────┬────────────────────────────┬────────────────────┬───────────────┘
               │                           │                            │                    │
               ▼                           ▼                            ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TRANSPORT LAYER (protocol + pairing)                                       │
│                                                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────────┐     │
│  │   HTTPTransport     │    │  WebSocketTransport │    │ WebRTCTransport │    │  InternalScheduler  │     │
│  │   (for /chat)       │    │  (text or voice)    │    │  (voice only)   │    │  (background jobs)  │     │
│  └──────────┬──────────┘    └──────────┬──────────┘    └────────┬────────┘    └──────────┬──────────┘     │
│             │                          │                        │                        │                 │
│             │     ┌────────────────────┴────────────────────┐   │                        │                 │
│             │     │         EXPLICIT PAIRING RULE           │   │                        │                 │
│             │     │  transport_type → adapter_type          │   │                        │                 │
│             │     │  ─────────────────────────────────────  │   │                        │                 │
│             │     │  http          → TextAdapter            │   │                        │                 │
│             │     │  websocket+text → TextAdapter           │   │                        │                 │
│             │     │  websocket+voice→ VoiceAdapter          │   │                        │                 │
│             │     │  webrtc        → VoiceAdapter           │   │                        │                 │
│             │     │  internal      → SystemAdapter          │   │                        │                 │
│             │     └────────────────────┬────────────────────┘   │                        │                 │
│             │                          │                        │                        │                 │
│             ▼                          ▼                        ▼                        ▼                 │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐  │
│  │    TextAdapter      │    │    VoiceAdapter     │    │    VoiceAdapter     │    │   SystemAdapter     │  │
│  │                     │    │                     │    │                     │    │                     │  │
│  │  • text shaping     │    │  • STT stream loop  │    │  • STT stream loop  │    │  • no user I/O      │  │
│  │  • stream framing   │    │  • debounce logic   │    │  • debounce logic   │    │  • fire-and-forget  │  │
│  │                     │    │  • TTS output mux   │    │  • TTS output mux   │    │                     │  │
│  │  NO STT/TTS         │    │  • utterance detect │    │  • utterance detect │    │  NO STT/TTS         │  │
│  └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘  │
└─────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────┘
              │                          │                          │                          │
              └──────────────────────────┼──────────────────────────┼──────────────────────────┘
                                         │                          │
                                         ▼                          ▼
                              LLMRequestEnvelope (serialized)  KernelRequest
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              LLM PROCESSOR (SINGLE ENTRY POINT)                                              │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              ContextBuilder                                                            │ │
│  │                                                                                                       │ │
│  │  build_llm_input(user_message, session, memory, tools, skills, plugins) → LLMRequestEnvelope        │ │
│  │                                                                                                       │ │
│  │  Steps:                                                                                               │ │
│  │  1. Build system prompt = base_style + injected_skills + plugin_instructions                        │ │
│  │  2. Build conversation = history + [{role: user, content: user_message}]                             │ │
│  │  3. Inject memory context = retrieved_memories (facts, actions, sessions)                            │ │
│  │  4. Build tool schemas = [t.to_schema() for t in enabled_tools]                                      │ │
│  │  5. Build plugin context = {plugin_name: {access_token, config, ...}}                                │ │
│  │  6. Resolve policy = provider/model/voice for this user/session                                      │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                    │                                                        │
│                                                    ▼ LLMRequestEnvelope                                    │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              LLM Core                                                                  │ │
│  │                                                                                                       │ │
│  │  generate(envelope: LLMRequestEnvelope) → AsyncGenerator[LLMEventEnvelope | ToolCallRequest]         │ │
│  │                                                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                        TOOL CALLING LOOP (agentic)                                               │ │ │
│  │  │                                                                                                 │ │ │
│  │  │  while True:                                                                                    │ │ │
│  │  │      response = await provider.generate(envelope)                                               │ │ │
│  │  │                                                                                                 │ │ │
│  │  │      # Stream content chunks                                                                   │ │ │
│  │  │      for chunk in response.stream:                                                             │ │ │
│  │  │          yield LLMEventEnvelope(event_type="text_chunk", payload=chunk)                        │ │ │
│  │  │                                                                                                 │ │ │
│  │  │      # Handle tool calls                                                                       │ │ │
│  │  │      if response.tool_calls:                                                                   │ │ │
│  │  │          for tc in response.tool_calls:                                                        │ │ │
│  │  │              yield ToolCallRequest(tool_name, arguments, call_id)                              │ │ │
│  │  │          # Caller executes tools, appends results to envelope.messages                         │ │ │
│  │  │          # Loop continues with updated conversation                                            │ │ │
│  │  │                                                                                                 │ │ │
│  │  │      if response.finish_reason == "stop":                                                      │ │ │
│  │  │          yield LLMEventEnvelope(event_type="stream_end")                                       │ │ │
│  │  │          break                                                                                 │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                    │                                                        │
│                         ┌──────────────────────────┼──────────────────────────┐                           │
│                         │                          │                          │                           │
│                         ▼                          ▼                          ▼                           │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐        │
│  │      Reply Service          │  │      Memory Service         │  │   Notification Service      │        │
│  │      kind: reply_*          │  │      kind: memory_*         │  │      kind: notif_*          │        │
│  │                             │  │                             │  │                             │        │
│  │  • user-facing responses    │  │  • fact extraction          │  │  • event classification     │        │
│  │  • tool follow-up handling  │  │  • session summaries        │  │  • notification compose     │        │
│  │  • TTS coordination         │  │  • action compaction        │  │  • urgency decisions        │        │
│  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼ ToolCallRequest
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TOOL ORCHESTRATOR                                                               │
│                                                                                                             │
│  execute(tool_name: str, arguments: dict, context: PluginContext) → ToolResult                             │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Tool Registry                                                                                         │ │
│  │                                                                                                       │ │
│  │  Built-in Tools:          Plugin Tools:              Sub-agent:                                       │ │
│  │  • read_file              • gmail/list_unread        • run_task (delegates to TaskRunner)            │ │
│  │  • write_file             • gmail/send_reply                                                        │ │
│  │  • list_directory         • spotify/play                                                             │ │
│  │  • run_command            • spotify/search                                                           │ │
│  │  • web_search             • calendar/list_events                                                     │ │
│  │                           • contacts/search                                                          │ │
│  │  Context Injection:                                                                                   │ │
│  │  • Plugin tools receive context = {access_token, config, ...} from PluginContextStore               │ │
│  │  • Built-in tools receive context = {working_dir, ...}                                              │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                    │                                                        │
│                                                    ▼ ToolResult → back to LLM Core                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PROVIDER LAYER (unchanged contracts)                                            │
│                                                                                                             │
│  LLMProvider              STTProvider               TTSProvider                                             │
│  • OpenAI GPT-4o          • Deepgram (streaming)    • OpenAI TTS                                            │
│  • generate_stream()      • connect_stream()        • synthesize()                                          │
│  • health_check()         • send_audio()            • health_check()                                        │
│                           • stream_events()                                                                │
│                                                                                                             │
│  Policy Resolver: resolves provider/model/voice per user/session from config + user preferences            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY STORE (SQLite + embeddings)                                              │
│                                                                                                             │
│  conversations    facts           actions           sessions                                               │
│  (raw exchanges)  (extracted)     (tool calls)      (summaries)                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

---
Delta View (what really changes)
BEFORE:
Transport -> CoreHandler (monolith) -> processors/providers/tools/memory

AFTER:
Transport -> ModalityAdapter -> LLMProcessor (single entry) -> ToolOrchestrator -> Providers/Memory
                                      │
                                      └── Services (Reply, Memory, Notification) all use same LLM interface

---
Key Architectural Decisions

1. LLM IS INDEPENDENT FROM PIPELINE
   - LLMProcessor is a standalone component, not owned by any service
   - Voice pipeline (STT → LLM → TTS) is orchestrated by VoiceAdapter, not embedded in LLM
   - LLM has no knowledge of STT/TTS — it only sees text in, text out

2. SINGLE LLM ENTRY POINT
   - ALL LLM calls go through LLMProcessor.generate()
   - Input: LLMRequestEnvelope (serialized, validated)
   - Output: AsyncGenerator[LLMEventEnvelope | ToolCallRequest]
   - No direct provider calls from services/adapters

3. TRANSPORT-ADAPTER PAIRING
   - WebRTCTransport ↔ VoiceAdapter (STT/TTS integrated)
   - WebSocketTransport + voice mode ↔ VoiceAdapter
   - WebSocketTransport + text mode ↔ TextAdapter
   - HTTPTransport (/chat) ↔ TextAdapter
   - InternalScheduler ↔ SystemAdapter (background jobs)

4. TOOLS/PLUGINS/SKILLS COMPATIBILITY
   - Tool calling loop is explicit in LLM Core
   - Skills injected via ContextBuilder into system prompt
   - Plugin context (OAuth tokens) passed to tool execution
   - ToolRegistry unchanged, only orchestration moved
---
Layer Responsibilities (final intent)
- Transport: protocol + connection only.
- Modality Adapter: convert text/audio events <-> kernel jobs, handle STT/TTS orchestration details.
- Kernel: job routing, prioritization, concurrency, fairness, cancellation.
- Services: domain logic by workload type (reply, memory, notification, tools).
- Providers: external model APIs only (no orchestration).
- Store: persistence only.

===============================================================================


here is a production-safe refactor plan for moving toward your “LLM-kernel OS” model without losing working behavior.
Updated target (must-win outcomes)
1. LLM orchestration is independent from the STT->LLM->TTS pipeline.
2. Any workflow that needs an LLM goes through one shared LLM processor.
3. Every LLM input/output is serialized and validated against a fixed structure.
4. STT/TTS are paired with voice transports (WebRTC/voice WS paths), text shaping/streaming is paired with text transports.
5. Tools, plugins, skills, and tool-calling behavior remain fully compatible.

Anti-over-abstraction rule
- Add a new layer only when it removes a measured production pain (latency, fairness, operability, correctness).
- No speculative infrastructure without a current failing invariant, test gap, or measurable bottleneck.
- During migration, keep a single source of truth per concern; do not duplicate orchestration logic in legacy and new paths.
- Keep CoreHandler compatibility shell thin and temporary, with explicit decommission criteria.

Refactor Goal
- Keep transports responsible for protocol + modality adaptation.
- Move orchestration to a kernel scheduler.
- Treat LLM as shared compute for multiple workload types, not a single session-bound pipeline.
- Preserve all current contracts and behavior during migration.
---
0) Migration Invariants (must never break)
- app/src/aether/main.py endpoints remain stable:
  - /chat
  - /ws
  - /webrtc/offer
  - /webrtc/ice
  - /plugin_event, /plugin_event/batch
- Existing wire protocol stays compatible:
  - text_chunk, status, transcript, audio_chunk, stream_end, tool_result, notification
- User config behavior preserved:
  - STT provider/model/language (AETHER_STT_*)
  - TTS provider/model/voice (AETHER_TTS_*, Sarvam/ElevenLabs variants)
  - LLM model/style settings
- No regression in:
  - voice roundtrip latency
  - chat streaming smoothness
  - memory writes (facts/actions/sessions)
  - tool/plugin execution
---
1) Baseline & Inventory Phase (mandatory, no architecture changes)
1.1 Build the Work Kind Catalog from current code paths
From today’s code, canonical kinds should start as:
- reply_text — /chat text response path
- reply_voice — STT-triggered voice response path in CoreHandler._trigger_voice_response
- memory_fact_extract — MemoryStore.add async extraction
- memory_session_summary — _summarize_session
- memory_action_compact — _compact_old_actions
- notification_decide — EventProcessor._classify
- notification_compose — EventProcessor._generate_notification
- tool_execute — ToolRegistry.dispatch
- subagent_task — TaskRunner.run
1.2 Clarify modality semantics (strict)
- text: user-facing textual I/O (/chat, WS text sessions)
- voice: user-facing audio-in/audio-out paths (WebRTC/WS voice mode)
- system: internal/background jobs (summaries, fact extraction, notification decisions, action compaction)
1.3 Capture latency baseline (p50/p95)
Measure current for:
- /chat: request -> first byte, total completion
- voice: utterance_end -> first text_chunk, first audio_chunk, stream_end
- tool job latency
- notification decision latency
Define SLO pass/fail thresholds in this document before any phase rollout:
- /chat TTFT (request -> first text chunk)
  - pass: p50 <= 700ms, p95 <= 1800ms
  - fail: p95 > 2000ms
- voice first token (utterance_end -> first assistant token)
  - pass: p50 <= 900ms, p95 <= 2200ms
  - fail: p95 > 2500ms
- voice audio start (first token -> first audio_chunk)
  - pass: p50 <= 350ms, p95 <= 900ms
  - fail: p95 > 1100ms
- tool_execute latency
  - pass: p95 within +15% of baseline by tool family
  - fail: sustained > +25% regression
- notification_decide latency
  - pass: p95 <= 1200ms
  - fail: p95 > 1500ms

Phase gate rule:
- Do not advance phases on SLO fail. Roll back by feature flag and fix before continuing.
1.4 Test matrix lock
Ensure these test suites are green and add missing smoke tests if needed:
- app/tests/test_transport.py
- app/tests/test_webrtc_transport.py
- app/tests/test_agent_endpoints.py
- add a voice-path regression smoke test for STT event loop + TTS emission ordering
---
2) Kernel Contracts Phase (adapter mode, behavior unchanged)
Create new package: app/src/aether/kernel/
2.1 New contracts
- kernel/contracts.py

```python
# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL CONTRACTS (job orchestration)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KernelRequest:
    kind: str                    # reply_text, reply_voice, memory_fact_extract, etc.
    modality: str                # text, voice, system
    user_id: str
    session_id: str
    payload: dict
    priority: int = 0            # 0=interactive, 1=background
    deadline_ms: int | None = None

@dataclass
class KernelEvent:
    kind: str
    stream_type: str             # text_chunk, audio_chunk, tool_result, stream_end
    payload: dict
    sequence: int                # monotonic per job

@dataclass
class KernelResult:
    status: str                  # success, failed, canceled, timeout
    payload: dict
    metrics: dict


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ENVELOPES (single shared format for ALL LLM consumers)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMRequestEnvelope:
    """Fixed input structure for ALL LLM calls. Every field is validated."""
    
    # Identity & routing
    schema_version: str = "1.0"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    kind: str = ""               # reply_text, memory_fact_extract, etc.
    modality: str = ""           # text, voice, system
    user_id: str = ""
    session_id: str = ""
    
    # Core LLM input
    messages: list[dict] = field(default_factory=list)
    # [{role: "system"|"user"|"assistant"|"tool", content: str, tool_call_id?: str}]
    
    # Tool context
    tools: list[dict] = field(default_factory=list)
    # [{name, description, parameters: {type, properties, required}}]
    tool_choice: str = "auto"    # auto, none, required, or specific tool
    
    # Plugin context (injected into tool execution, not LLM prompt)
    plugin_context: dict = field(default_factory=dict)
    # {plugin_name: {access_token, refresh_token, config: {...}}}
    
    # Policy (provider/model selection)
    policy: dict = field(default_factory=dict)
    # {provider: "openai", model: "gpt-4o", temperature: 0.7, max_tokens: 4096}
    
    # Tracing
    trace: dict = field(default_factory=dict)
    # {trace_id, span_id, parent_span_id}


@dataclass
class LLMEventEnvelope:
    """Streamed event from LLM. Every event is independently processable."""
    
    schema_version: str = "1.0"
    request_id: str = ""
    job_id: str = ""
    
    event_type: str = ""         # text_chunk, tool_call, tool_result, stream_end, error
    sequence: int = 0            # monotonic per request
    idempotency_key: str = ""    # request_id + sequence for dedup
    
    payload: dict = field(default_factory=dict)
    # text_chunk: {text: str, role: "assistant"}
    # tool_call: {tool_name, arguments, call_id}
    # tool_result: {tool_name, output, call_id, error: bool}
    # stream_end: {finish_reason: "stop"|"tool_calls"|"length"}
    # error: {code, message, recoverable: bool}
    
    metrics: dict = field(default_factory=dict)
    # {latency_ms, tokens_generated, ...}


@dataclass
class LLMResultEnvelope:
    """Terminal result for non-streaming or summary of streaming call."""
    
    schema_version: str = "1.0"
    request_id: str = ""
    job_id: str = ""
    
    status: str = ""             # success, failed, canceled, timeout
    output: dict = field(default_factory=dict)
    # {content: str, tool_calls: [...], finish_reason: str}
    
    usage: dict = field(default_factory=dict)
    # {prompt_tokens, completion_tokens, total_tokens}
    
    error: dict | None = None
    # {code, message, recoverable: bool}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL CALLING CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCallRequest:
    """Emitted by LLM Core when LLM wants to call a tool."""
    tool_name: str
    arguments: dict
    call_id: str                 # OpenAI-style tool call ID

@dataclass
class ToolResult:
    """Result of tool execution, fed back to LLM."""
    tool_name: str
    output: str
    call_id: str
    error: bool = False
    metadata: dict = field(default_factory=dict)
```
2.2 Kernel interface
- kernel/interface.py

```python
class KernelInterface(ABC):
    @abstractmethod
    async def submit(self, request: KernelRequest) -> str:
        """Submit a job, return job_id."""
        ...

    @abstractmethod
    async def stream(self, job_id: str) -> AsyncGenerator[KernelEvent, None]:
        """Stream events from a job."""
        ...

    @abstractmethod
    async def await_result(self, job_id: str, timeout_ms: int | None = None) -> KernelResult:
        """Wait for job completion, return result."""
        ...

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Cancel a job. Returns True if canceled, False if already complete."""
        ...
```
2.3 ContextBuilder (skill/plugin injection)
- llm/context_builder.py

```python
class ContextBuilder:
    """
    Builds LLMRequestEnvelope from various sources.
    This is where skills and plugins get injected into the LLM context.
    """

    def __init__(
        self,
        skill_loader: SkillLoader,
        plugin_context_store: PluginContextStore,
        tool_registry: ToolRegistry,
        memory_store: MemoryStore,
        policy_resolver: PolicyResolver,
    ):
        self.skill_loader = skill_loader
        self.plugin_context_store = plugin_context_store
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.policy_resolver = policy_resolver

    async def build(
        self,
        user_message: str,
        session: SessionState,
        enabled_plugins: list[str],
    ) -> LLMRequestEnvelope:
        """
        Build a complete LLM request envelope.
        
        Injection order (matters for prompt construction):
        1. Base system prompt (from config.personality.base_style)
        2. Injected skills (matched by keyword overlap with user_message)
        3. Plugin instructions (from plugin SKILL.md files)
        4. Memory context (retrieved facts, actions, sessions)
        5. Conversation history
        6. Tool schemas (from tool_registry + plugin tools)
        7. Plugin context (OAuth tokens, configs - NOT in prompt, for tool execution)
        """

        # 1. Base system prompt
        base_prompt = config.personality.base_style
        custom_instructions = config.personality.custom_instructions or ""

        # 2. Inject skills (keyword matching)
        matched_skills = self.skill_loader.match(user_message)
        skill_instructions = "\n\n".join(
            f"[Skill: {s.name}]\n{s.content}" for s in matched_skills
        )

        # 3. Plugin instructions (from SKILL.md)
        plugin_instructions = ""
        for plugin_name in enabled_plugins:
            skill_content = self._get_plugin_skill_content(plugin_name)
            if skill_content:
                plugin_instructions += f"\n\n[Plugin: {plugin_name}]\n{skill_content}"

        # 4. Memory retrieval
        memories = await self.memory_store.search(user_message, limit=5)
        memory_context = self._format_memories(memories)

        # 5. Build full system prompt
        system_prompt = self._build_system_prompt(
            base_prompt=base_prompt,
            custom_instructions=custom_instructions,
            skill_instructions=skill_instructions,
            plugin_instructions=plugin_instructions,
            memory_context=memory_context,
        )

        # 6. Conversation history
        messages = session.history + [{"role": "user", "content": user_message}]

        # 7. Tool schemas (built-in + plugin tools)
        tools = self._build_tool_schemas(enabled_plugins)

        # 8. Plugin context (for tool execution, NOT in prompt)
        plugin_context = self._build_plugin_context(enabled_plugins)

        # 9. Policy resolution
        policy = self.policy_resolver.resolve(session.user_id)

        return LLMRequestEnvelope(
            kind="reply_text" if session.mode == "text" else "reply_voice",
            modality=session.mode,
            user_id=session.user_id,
            session_id=session.session_id,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            tools=tools,
            plugin_context=plugin_context,
            policy=policy,
        )

    def _build_system_prompt(
        self,
        base_prompt: str,
        custom_instructions: str,
        skill_instructions: str,
        plugin_instructions: str,
        memory_context: str,
    ) -> str:
        """Assemble the full system prompt."""
        parts = [base_prompt]

        if custom_instructions:
            parts.append(f"\n\nCustom Instructions:\n{custom_instructions}")

        if skill_instructions:
            parts.append(f"\n\n--- Active Skills ---{skill_instructions}")

        if plugin_instructions:
            parts.append(f"\n\n--- Active Plugins ---{plugin_instructions}")

        if memory_context:
            parts.append(f"\n\n--- Relevant Context ---\n{memory_context}")

        return "\n".join(parts)

    def _build_tool_schemas(self, enabled_plugins: list[str]) -> list[dict]:
        """Build tool schemas from registry + plugin tools."""
        tools = []

        # Built-in tools
        for tool in self.tool_registry.all():
            tools.append(tool.to_openai_schema())

        # Plugin tools (already registered in tool_registry with plugin_name)
        # Filter to only enabled plugins
        for plugin_name in enabled_plugins:
            for tool in self.tool_registry.get_plugin_tools(plugin_name):
                tools.append(tool.to_openai_schema())

        return tools

    def _build_plugin_context(self, enabled_plugins: list[str]) -> dict:
        """Build plugin context for tool execution (OAuth tokens, configs)."""
        context = {}
        for plugin_name in enabled_plugins:
            plugin_ctx = self.plugin_context_store.get(plugin_name)
            if plugin_ctx:
                context[plugin_name] = plugin_ctx
        return context
```
2.4 LLM Core (single entry point with tool calling loop)
- llm/core.py

```python
class LLMCore:
    """
    Single entry point for ALL LLM calls.
    Handles the agentic tool-calling loop internally.
    """

    def __init__(self, provider: LLMProvider, tool_orchestrator: ToolOrchestrator):
        self.provider = provider
        self.tool_orchestrator = tool_orchestrator

    async def generate(
        self,
        envelope: LLMRequestEnvelope,
    ) -> AsyncGenerator[LLMEventEnvelope | ToolCallRequest, None]:
        """
        Generate LLM response with automatic tool calling loop.

        Yields:
        - LLMEventEnvelope (text_chunk, stream_end, error)
        - ToolCallRequest (when LLM wants to call a tool)

        The caller can either:
        1. Execute tools themselves and call generate() again with results
        2. Use generate_with_tools() which handles the loop automatically
        """
        sequence = 0
        max_tool_iterations = 5  # Prevent infinite loops

        for iteration in range(max_tool_iterations):
            # Call provider
            response = await self.provider.generate_stream(
                messages=envelope.messages,
                tools=envelope.tools,
                tool_choice=envelope.tool_choice,
                **envelope.policy,
            )

            # Stream content chunks
            content_chunks = []
            tool_calls = []

            async for chunk in response:
                if chunk.type == "text":
                    content_chunks.append(chunk.text)
                    sequence += 1
                    yield LLMEventEnvelope(
                        request_id=envelope.request_id,
                        job_id=envelope.job_id,
                        event_type="text_chunk",
                        sequence=sequence,
                        idempotency_key=f"{envelope.request_id}:{sequence}",
                        payload={"text": chunk.text, "role": "assistant"},
                    )

                elif chunk.type == "tool_call":
                    tool_calls.append(chunk)

                elif chunk.type == "finish":
                    finish_reason = chunk.reason

            # Handle tool calls
            if tool_calls:
                for tc in tool_calls:
                    yield ToolCallRequest(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        call_id=tc.id,
                    )

                # If caller is handling tools, they'll call us again
                # For now, signal that tools are pending
                if finish_reason == "tool_calls":
                    # Append assistant message with tool calls to history
                    envelope.messages.append({
                        "role": "assistant",
                        "content": "".join(content_chunks),
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in tool_calls
                        ],
                    })
                    continue  # Loop will continue when caller provides tool results

            # Stream end
            sequence += 1
            yield LLMEventEnvelope(
                request_id=envelope.request_id,
                job_id=envelope.job_id,
                event_type="stream_end",
                sequence=sequence,
                idempotency_key=f"{envelope.request_id}:{sequence}",
                payload={"finish_reason": finish_reason},
            )
            break

    async def generate_with_tools(
        self,
        envelope: LLMRequestEnvelope,
    ) -> AsyncGenerator[LLMEventEnvelope, None]:
        """
        Generate with automatic tool execution.
        Handles the full agentic loop internally.
        """
        max_iterations = 5

        for iteration in range(max_iterations):
            tool_results = []

            async for event in self.generate(envelope):
                if isinstance(event, ToolCallRequest):
                    # Execute tool
                    result = await self.tool_orchestrator.execute(
                        tool_name=event.tool_name,
                        arguments=event.arguments,
                        call_id=event.call_id,
                        plugin_context=envelope.plugin_context,
                    )

                    # Emit tool result event
                    yield LLMEventEnvelope(
                        request_id=envelope.request_id,
                        job_id=envelope.job_id,
                        event_type="tool_result",
                        payload={
                            "tool_name": result.tool_name,
                            "output": result.output[:500],  # Truncated for streaming
                            "call_id": result.call_id,
                            "error": result.error,
                        },
                    )

                    tool_results.append(result)

                elif isinstance(event, LLMEventEnvelope):
                    yield event

                    # If stream_end and no pending tools, we're done
                    if event.event_type == "stream_end":
                        return

            # If we have tool results, feed them back to LLM
            if tool_results:
                for result in tool_results:
                    envelope.messages.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.output,
                    })
            else:
                break
```
2.5 Tool Orchestrator (tool execution with plugin context)
- tools/orchestrator.py

```python
class ToolOrchestrator:
    """
    Executes tools with proper context injection.
    Plugin tools receive OAuth tokens and config from plugin_context.
    """

    def __init__(self, tool_registry: ToolRegistry, task_runner: TaskRunner):
        self.tool_registry = tool_registry
        self.task_runner = task_runner

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        call_id: str,
        plugin_context: dict,
    ) -> ToolResult:
        """
        Execute a tool with context injection.

        Plugin context flow:
        1. Tool is looked up in registry
        2. If tool belongs to a plugin, inject plugin_context[plugin_name]
        3. Tool receives context via safe_execute(context=...)
        """

        tool = self.tool_registry.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                output=f"Unknown tool: {tool_name}",
                call_id=call_id,
                error=True,
            )

        # Determine if this is a plugin tool
        plugin_name = self.tool_registry.get_plugin_for_tool(tool_name)

        # Build context for tool execution
        if plugin_name:
            # Plugin tool: inject OAuth tokens and config
            context = plugin_context.get(plugin_name, {})
        else:
            # Built-in tool: inject working_dir, etc.
            context = {"working_dir": config.server.working_dir}

        # Execute with context
        result = await tool.safe_execute(context=context, **arguments)
        result.call_id = call_id

        return result

    async def execute_subagent(
        self,
        task_description: str,
        session_id: str,
        plugin_context: dict,
    ) -> ToolResult:
        """
        Execute a sub-agent task (run_task tool).
        Sub-agents get their own LLM context but share plugin credentials.
        """
        return await self.task_runner.run(
            task_description=task_description,
            session_id=session_id,
            plugin_context=plugin_context,
        )
```
2.6 Legacy adapter (critical)
- kernel/legacy_adapter.py

```python
class LegacyAdapter:
    """
    Wraps current CoreHandler logic unchanged.
    Maps existing CoreMsg + internal flows to/from Kernel* contracts.
    
    This is the bridge that allows incremental migration:
    - Phase 2: CoreHandler wrapped behind kernel interface
    - Phase 4: Adapters replace CoreHandler logic piece by piece
    - Phase 7: Legacy adapter removed entirely
    """

    def __init__(self, core_handler: CoreHandler):
        self.core_handler = core_handler

    async def submit(self, request: KernelRequest) -> str:
        """Convert KernelRequest to CoreMsg, submit to CoreHandler."""
        msg = self._kernel_to_core_msg(request)
        # CoreHandler processes and we collect events
        job_id = request.job_id or str(uuid.uuid4())
        self._pending_jobs[job_id] = {
            "request": request,
            "events": [],
            "status": "running",
        }

        # Process in background
        asyncio.create_task(self._process_job(job_id, msg))
        return job_id

    async def stream(self, job_id: str) -> AsyncGenerator[KernelEvent, None]:
        """Stream events from a pending job."""
        while True:
            job = self._pending_jobs.get(job_id)
            if not job:
                break

            # Yield any new events
            while job["events"]:
                event = job["events"].pop(0)
                yield event

            if job["status"] in ("complete", "failed", "canceled"):
                break

            await asyncio.sleep(0.01)  # Small delay to avoid busy loop

    def _kernel_to_core_msg(self, request: KernelRequest) -> CoreMsg:
        """Convert KernelRequest to CoreMsg format."""
        if request.kind.startswith("reply_"):
            return CoreMsg.text(
                text=request.payload.get("text", ""),
                user_id=request.user_id,
                session_id=request.session_id,
            )
        elif request.kind.startswith("memory_"):
            return CoreMsg.event(
                event_type=request.kind,
                user_id=request.user_id,
                session_id=request.session_id,
                payload=request.payload,
            )
        # ... other mappings
```

2.7 Transport-Adapter Pairing (explicit)
- transport/pairing.py

```python
class TransportPairing:
    """
    Explicit pairing of transports with modality adapters.
    Each transport knows which adapter handles its I/O.
    """

    @staticmethod
    def get_adapter(
        transport_type: str,
        session_mode: str,
    ) -> ModalityAdapter:
        """
        transport_type: "http", "websocket", "webrtc", "internal"
        session_mode: "text", "voice", "system"

        Pairing rules:
        - http + text → TextAdapter
        - websocket + text → TextAdapter
        - websocket + voice → VoiceAdapter
        - webrtc + voice → VoiceAdapter
        - internal + system → SystemAdapter
        """
        if transport_type == "webrtc":
            return VoiceAdapter()
        elif transport_type == "http":
            return TextAdapter()
        elif transport_type == "websocket":
            if session_mode == "voice":
                return VoiceAdapter()
            else:
                return TextAdapter()
        elif transport_type == "internal":
            return SystemAdapter()
        else:
            raise ValueError(f"Unknown transport: {transport_type}")


class ModalityAdapter(ABC):
    """Base class for modality adapters."""

    @abstractmethod
    async def handle_input(
        self,
        input: CoreMsg | dict,
        kernel: KernelInterface,
    ) -> AsyncGenerator[CoreMsg, None]:
        """Handle input, submit to kernel, yield output messages."""
        ...


class TextAdapter(ModalityAdapter):
    """Text I/O shaping. NO STT/TTS."""

    async def handle_input(
        self,
        input: CoreMsg,
        kernel: KernelInterface,
    ) -> AsyncGenerator[CoreMsg, None]:
        # Submit text job to kernel
        job_id = await kernel.submit(KernelRequest(
            kind="reply_text",
            modality="text",
            user_id=input.user_id,
            session_id=input.session_id,
            payload={"text": input.content.text},
        ))

        # Stream events back
        async for event in kernel.stream(job_id):
            if event.stream_type == "text_chunk":
                yield CoreMsg.text(
                    text=event.payload["text"],
                    user_id=input.user_id,
                    session_id=input.session_id,
                    role="assistant",
                )
            elif event.stream_type == "stream_end":
                yield CoreMsg.event(
                    event_type="stream_end",
                    user_id=input.user_id,
                    session_id=input.session_id,
                )


class VoiceAdapter(ModalityAdapter):
    """Voice I/O with STT/TTS orchestration."""

    def __init__(self, stt_provider: STTProvider, tts_provider: TTSProvider):
        self.stt = stt_provider
        self.tts = tts_provider

    async def handle_input(
        self,
        input: CoreMsg,
        kernel: KernelInterface,
    ) -> AsyncGenerator[CoreMsg, None]:
        """
        Handles:
        - stream_start: connect STT, start event loop
        - audio_chunk: forward to STT
        - stream_stop: disconnect STT, trigger response
        - text (from STT): submit to kernel, stream response, TTS output
        """
        if input.content.event_type == "stream_start":
            await self.stt.connect_stream()
            asyncio.create_task(self._stt_event_loop(input.user_id, input.session_id, kernel))
            yield CoreMsg.text(text="listening...", role="system", transport="status")

        elif input.content.event_type == "audio_chunk":
            audio_data = base64.b64decode(input.content.payload["data"])
            await self.stt.send_audio(audio_data)

        elif input.content.event_type == "stream_stop":
            await self.stt.disconnect_stream()

    async def _stt_event_loop(self, user_id: str, session_id: str, kernel: KernelInterface):
        """Listen to STT events, trigger LLM on utterance end."""
        accumulated = ""
        async for event in self.stt.stream_events():
            if event.type == FrameType.TEXT and event.metadata.get("interim"):
                # Forward interim transcript
                yield CoreMsg.text(text=event.data, role="system", transport="transcript_interim")

            elif event.type == FrameType.CONTROL and event.data.get("action") == "utterance_end":
                accumulated += event.data.get("transcript", "")

                # Debounce: wait for silence
                await asyncio.sleep(config.server.debounce_delay)

                if accumulated:
                    # Submit to kernel
                    job_id = await kernel.submit(KernelRequest(
                        kind="reply_voice",
                        modality="voice",
                        user_id=user_id,
                        session_id=session_id,
                        payload={"text": accumulated},
                    ))

                    # Stream response + TTS
                    async for kernel_event in kernel.stream(job_id):
                        if kernel_event.stream_type == "text_chunk":
                            # TTS synthesis
                            async for audio_frame in self.tts.synthesize(kernel_event.payload["text"]):
                                yield CoreMsg.audio(audio_data=audio_frame.data, ...)

                    accumulated = ""
```
2.8 Single LLM processor requirement (mandatory)
- All LLM calls go through LLMCore.generate() or generate_with_tools()
- No direct provider calls from services/adapters
- Provider access is encapsulated behind LLMCore

Consumers of LLM:
| Kind                    | Service            | Uses LLMCore |
|-------------------------|--------------------|--------------| 
| reply_text              | ReplyService       | ✅           |
| reply_voice             | ReplyService       | ✅           |
| memory_fact_extract     | MemoryService      | ✅           |
| memory_session_summary  | MemoryService      | ✅           |
| memory_action_compact   | MemoryService      | ✅           |
| notification_decide     | NotificationService| ✅           |
| notification_compose    | NotificationService| ✅           |
| subagent_task           | TaskRunner         | ✅           |

Exit criteria for Phase 2:
- No behavior changes observed.
- All tests pass.
- One feature flag: AETHER_KERNEL_ENABLED=false|true (default false initially).
- LLMRequestEnvelope validation passes for all code paths.
- ContextBuilder produces correct system prompts with skills/plugins injected.
- Tool calling loop produces same results as current LLMProcessor.

---
3) Scheduler Introduction Phase (single worker first)
3.1 Add scheduler scaffolding
- kernel/scheduler.py with:
  - queue per priority class (interactive, background)
  - per-user fairness token bucket
  - queue depth metrics + reject policy
3.2 Start with single worker mode
- kernel/workers.py runs one worker initially (parity mode).
- This keeps semantics close to current while introducing job abstraction.
3.3 Route only /chat first
- In app/src/aether/main.py, /chat submits reply_text to kernel.
- Keep existing /chat response stream format unchanged.
Exit criteria
- /chat parity and latency within agreed budget (<10% regression).
- rollback via feature flag.
---
4) Modality Adapter Phase (decouple STT/TTS from monolithic handler)
Introduce app/src/aether/modality/:
- text_adapter.py
- voice_adapter.py
4.1 What moves
From CoreHandler into adapters:
- STT stream management loop (_stt_event_loop)
- debounce/utterance trigger (_debounce_and_trigger)
- status/audio bridging concerns
- TTS chunk emission strategy per transport mode
4.1.1 Transport-paired modality rule
- Voice transport paths own STT/TTS orchestration lifecycle.
- Text transport paths own text stream shaping.
- Kernel/services must remain transport-agnostic (no direct protocol/media handling).
4.2 What stays in kernel
- memory retrieval orchestration
- LLM job dispatch
- tool/sub-agent orchestration
- memory writes and session-level bookkeeping
4.3 Audio output & latency safeguards
- preserve current RawAudioTrack push behavior in transport/webrtc.py
- preserve sentence chunking strategy initially (for TTS timing parity)
- add latency checkpoints:
  - utterance_end -> first assistant token
  - first token -> first audio chunk
4.4 Model selection support
Create policy/provider_policy.py:
- resolves STT/TTS/LLM provider/model per user/session
- source of truth remains current config/env behavior (core/config.py, /config/reload)
- adapters call policy resolver each session start and on config reload
Exit criteria
- iOS WebRTC voice parity (state transitions + audio flow)
- no regression in provider/model switch behavior
---
5) Multi-worker Kernel Phase (real “CPU cores”)
5.1 Increase concurrency safely
- configurable worker pool:
  - AETHER_KERNEL_WORKERS_INTERACTIVE
  - AETHER_KERNEL_WORKERS_BACKGROUND
- interactive kinds (reply_text, reply_voice, tool follow-up) prioritized over background.
5.2 Cancellation & backpressure
- cancel jobs on disconnect or superseding turn (text mode optional)
- bounded queues
- explicit responses for overload (drop/defer background first)
5.2.1 Side-effectful tool cancellation semantics
- pre-dispatch cancel: tool is not executed; job status = canceled.
- in-flight cancel (before side effect commit): tool receives cancel signal when supported; status = canceled or timed_out.
- post-commit cancel: side effect is preserved; emit terminal event with status = completed_with_cancellation and include side_effect_committed=true.
- retries are idempotency-key based; never re-run non-idempotent tool actions without explicit policy allow.

5.2.2 Stream ordering and idempotency contract
- all streamed events must include sequence (monotonic per job) and idempotency_key.
- ordering rules:
  - text_chunk: strictly increasing sequence.
  - audio_chunk: strictly increasing sequence per sentence/channel.
  - tool_result: emitted after corresponding tool call state transition.
  - stream_end: emitted exactly once and only after all prior events for that job are flushed.
- consumers must de-duplicate by idempotency_key and ignore out-of-order duplicates.
5.3 Non-reply workloads through kernel
Route these into scheduler:
- notification_decide, notification_compose
- memory_fact_extract
- memory_session_summary
- memory_action_compact
- subagent_task
Exit criteria
- background work no longer blocks user reply paths
- measurable throughput improvement under mixed load
---
6) Service Extraction Phase (clean bounded contexts)
Split current mixed logic into services:
- services/reply_service.py
- services/memory_service.py
- services/notification_service.py
- services/tool_service.py
Each service consumes kernel contracts and emits typed events.
This replaces deep coupling in transport/handler.py with narrow APIs.

6.1 Provider concurrency and rate-limit policy (mandatory)
- Introduce centralized provider capacity controls (LLM/STT/TTS):
  - max concurrent requests per provider/model
  - per-user and global token/request budgets
  - queue timeout + shed policy for overload
  - adaptive fallback policy (only if explicitly configured)
- Scheduler admission control must consider provider headroom before dispatch.
- Backpressure must prefer preserving interactive reply paths over background jobs.
---
7) Legacy Path Decommission Phase
Only after parity + burn-in:
- reduce CoreHandler to a compatibility shell or remove legacy internals
- remove direct internal calls from /chat (_get_session, _gather_pre_frames)
- transport layer only talks kernel; kernel only talks services/providers
---
Feature Flags & Rollback Strategy
- AETHER_KERNEL_ENABLED
- AETHER_KERNEL_SCHEDULER_ENABLED
- AETHER_VOICE_ADAPTER_V2_ENABLED
- AETHER_KERNEL_MULTIWORKER_ENABLED
Rollback always means:
- flip one flag
- restart agent
- no endpoint/protocol changes required
---
Exact Risks You Raised, and Mitigation
- “Will STT/TTS move break output audio stream?”
  - Keep RawAudioTrack path untouched in first adapter cut; only move orchestration, not transport plumbing.
- “What about latency?”
  - phase gate with explicit p50/p95 budgets before progressing.
- “What about model selection options?”
  - add provider policy resolver; preserve existing config source and /config/reload.
- “What other kinds do we perform now?”
  - inventory above comes from existing code paths, not speculative architecture.
- “system modality unclear”
  - now formalized: non-user-facing internal jobs only.

Observability as first-class deliverable
- Every kernel job has a stable job_id and request_id propagated end-to-end.
- Per-kind tracing spans:
  - adapter receive
  - kernel enqueue/dequeue
  - service start/end
  - llm/tool/provider call boundaries
  - stream completion
- Required metrics dashboards:
  - queue depth by priority lane
  - enqueue delay and run duration by kind
  - dropped/deferred/canceled counts by reason
  - provider concurrency saturation and rate-limit events
  - stream ordering/duplicate counters
- Promotion gate: no phase promotion without dashboards and alert thresholds configured.
---
First Implementation Sprint (safe start)
1. Add kernel/contracts.py and kernel/interface.py
2. Implement legacy_adapter.py with no behavior change
3. Wire /chat through kernel in adapter mode only
4. Add tests:
   - kind mapping correctness
   - /chat parity stream test
   - stream ordering + idempotency test (text_chunk/tool_result/stream_end)
   - llm envelope schema validation test (request/event/result)
   - cancellation semantics test for side-effectful tool flows
   - rollback path test (KERNEL_ENABLED=false)
5. No STT/TTS movement yet

---

## Summary: How This Plan Addresses Your Requirements

### Requirement 1: LLM Independent from STT-LLM-TTS Pipeline
**Addressed by:**
- `LLMCore` is a standalone component (section 2.4)
- `VoiceAdapter` orchestrates STT → LLM → TTS, but LLM doesn't know about STT/TTS
- LLM only sees `LLMRequestEnvelope` with text messages

### Requirement 2: Single LLM Processor with Serialized I/O
**Addressed by:**
- `LLMRequestEnvelope` and `LLMEventEnvelope` (section 2.1) define fixed structures
- All LLM consumers use the same contracts (table in section 2.8)
- No direct provider calls from services/adapters

### Requirement 3: STT/TTS Paired with Transport
**Addressed by:**
- `TransportPairing` class (section 2.7) defines explicit pairing rules
- `VoiceAdapter` contains STT/TTS orchestration
- `TextAdapter` has NO STT/TTS
- WebRTC → VoiceAdapter, HTTP/WebSocket+text → TextAdapter

### Requirement 4: Tools/Plugins/Skills Work Without Issues
**Addressed by:**
- `ContextBuilder` (section 2.3) injects skills and plugins into system prompt
- `ToolOrchestrator` (section 2.5) handles tool execution with plugin context
- `LLMCore.generate_with_tools()` (section 2.4) handles the tool calling loop
- Plugin OAuth tokens flow through `plugin_context` to tool execution

---

## File Structure After Refactor

```
app/src/aether/
├── main.py                     # FastAPI endpoints (unchanged)
│
├── kernel/                     # NEW: Job orchestration
│   ├── contracts.py            # KernelRequest, KernelEvent, KernelResult
│   ├── interface.py            # KernelInterface ABC
│   ├── scheduler.py            # Priority queues, fairness
│   ├── workers.py              # Worker pool
│   └── legacy_adapter.py       # Bridge to CoreHandler (temporary)
│
├── llm/                        # NEW: Single LLM entry point
│   ├── contracts.py            # LLMRequestEnvelope, LLMEventEnvelope, LLMResultEnvelope
│   ├── core.py                 # LLMCore (generate, generate_with_tools)
│   └── context_builder.py      # ContextBuilder (skill/plugin injection)
│
├── tools/                      # UPDATED: Tool orchestration
│   ├── base.py                 # AetherTool, ToolResult (unchanged)
│   ├── registry.py             # ToolRegistry (unchanged)
│   └── orchestrator.py         # NEW: ToolOrchestrator
│
├── transport/                  # UPDATED: Explicit pairing
│   ├── interface.py            # CoreInterface (unchanged)
│   ├── core_msg.py             # CoreMsg (unchanged)
│   ├── manager.py              # TransportManager (updated)
│   ├── websocket.py            # WebSocketTransport (unchanged)
│   ├── webrtc.py               # WebRTCTransport (unchanged)
│   └── pairing.py              # NEW: TransportPairing, ModalityAdapter
│
├── modality/                   # NEW: Modality adapters
│   ├── base.py                 # ModalityAdapter ABC
│   ├── text_adapter.py         # TextAdapter
│   ├── voice_adapter.py        # VoiceAdapter (STT/TTS)
│   └── system_adapter.py       # SystemAdapter (background)
│
├── services/                   # NEW: Domain services
│   ├── reply_service.py        # User-facing responses
│   ├── memory_service.py       # Fact extraction, summaries
│   ├── notification_service.py # Event classification
│   └── tool_service.py         # Tool execution coordination
│
├── providers/                  # UNCHANGED
├── memory/                     # UNCHANGED
├── processors/                 # DEPRECATED (logic moves to services/adapters)
├── skills/                     # UNCHANGED
├── plugins/                    # UNCHANGED
└── agents/                     # UNCHANGED
```

---

## Migration Checklist

- [ ] Phase 0: Baseline metrics captured, tests green
- [ ] Phase 1: Work kind catalog complete
- [ ] Phase 2: Kernel contracts defined, legacy adapter working
- [ ] Phase 2: LLMCore with tool calling loop implemented
- [ ] Phase 2: ContextBuilder with skill/plugin injection working
- [ ] Phase 3: Scheduler routing /chat
- [ ] Phase 4: VoiceAdapter with STT/TTS
- [ ] Phase 4: TextAdapter for /chat
- [ ] Phase 5: Multi-worker kernel
- [ ] Phase 6: Services extracted
- [ ] Phase 7: CoreHandler decommissioned



