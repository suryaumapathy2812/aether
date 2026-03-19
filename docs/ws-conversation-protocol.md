Goal
- Move all conversation runtime (text + voice instruction mode + interrupts + streaming assistant events) to WebSocket.
- Keep HTTP for everything else: session CRUD, plugins, channels, settings, auth/admin/uploads/health.
- No chat fallback path.
Research basis (current system)
- Dashboard chat runtime + SSE/event handling: dashboard/src/lib/chat-runtime.ts
- Current WS notification channel: dashboard/src/components/RealtimeProvider.tsx, agent/internal/ws/handler.go
- Current conversation SSE + session persistence: agent/internal/conversation/httpapi/handler.go
- Orchestrator auth/proxy contract and required routes: orchestrator/internal/server/server.go, orchestrator/CONTRACT.md
- Existing session model and persistence behavior: agent/internal/db/store.go, dashboard/src/components/SessionSync.tsx
Hard scope boundaries (no creep)
- In scope:
  - WS protocol for conversation
  - Text chat over WS
  - Voice instruction mode over WS (audio input -> LLM direct -> text output)
  - Interrupt/cancel over WS
- Out of scope:
  - TTS output
  - VAD/smart turn detection
  - WebRTC/realtime model connection
  - Tool/plugin CRUD over WS
  - Reworking auth/session CRUD APIs
---
Phase 1: Protocol + Contract Freeze
Deliverable: docs/ws-conversation-protocol.md (single source of truth)
- Define one envelope for all chat events:
  - Required fields: v, type, event_id, session_id, turn_id, seq, ts, payload
- Define client -> server events:
  - session.start
  - turn.start (mode: text|voice)
  - turn.input.text
  - turn.input.audio.chunk
  - turn.commit
  - turn.cancel
  - session.stop
  - ack
- Define server -> client events:
  - session.ready
  - turn.accepted
  - assistant.text.delta
  - assistant.tool.status
  - assistant.done
  - turn.cancelled
  - error
  - ack
- Define error codes (stable, finite): auth, validation, rate_limit, turn_conflict, unsupported_audio, internal
- Define ordering/idempotency rules:
  - event_id dedupe window per connection
  - monotonic seq per turn_id
- Define one-turn-at-a-time rule per session_id
Acceptance gate:
- Protocol reviewed and frozen before coding starts.
---
Phase 2: Agent WS Conversation Runtime
Deliverable: new WS conversation handler in agent, reusing existing core runtime
- Add conversation WS endpoint in agent (parallel to existing notification WS path):
  - likely under agent/internal/conversation/wsapi/*
- Reuse existing conversation execution pipeline:
  - keep existing prompt/context/tools behavior from agent/internal/conversation/runtime.go + agent/internal/llm/core.go
- Move current SSE event emission mapping to WS event emission
- Text mode path:
  - turn.input.text -> same processing as current HTTP turn path
- Voice instruction mode path:
  - collect turn.input.audio.chunk until turn.commit
  - pass audio directly as model input content (no separate STT service)
  - assistant response streamed as text events
- Persistence:
  - keep same session/message store behavior via agent/internal/db/store.go
  - persist user turn with metadata input_mode: voice|text
- Interrupt:
  - turn.cancel cancels in-flight generation context and emits turn.cancelled
Acceptance gate:
- WS text turn and WS voice turn both produce correct persisted messages and streamed assistant deltas.
---
Phase 3: Orchestrator WS Proxy for Conversation
Deliverable: authenticated WS pass-through route for conversation
- Add orchestrator WS route for conversation (similar model to notifications WS proxy)
- Enforce auth and user scoping exactly as existing HTTP proxy rules
- Inject/validate user context before forwarding
- Ensure heartbeat/ping support and close-code mapping
Acceptance gate:
- Dashboard connects to orchestrator WS endpoint only; no direct agent socket from browser.
---
Phase 4: Dashboard Runtime Unification (WS-only conversation)
Deliverable: one realtime chat transport in dashboard for text + voice
- Replace SSE turn execution in dashboard/src/lib/chat-runtime.ts with WS protocol client
- Add connection manager:
  - connect/authenticate
  - reconnect backoff
  - in-flight turn protection
  - ack handling
- Text mode:
  - composer sends turn.start + turn.input.text + turn.commit
- Voice mode:
  - capture mic chunks
  - send turn.input.audio.chunk
  - turn.commit
- Shared UI timeline:
  - same session/thread for both modes
  - user messages tagged by input_mode metadata
- Remove chat fallback logic intentionally (as requested)
Acceptance gate:
- User can switch text/voice in same session with identical assistant stream behavior.
---
Phase 5: Reliability + Observability Hardening
Deliverable: WS production readiness checklist complete
- Metrics:
  - connection count, reconnect rate, auth failures, turn latency, cancel latency, error code frequency
- Structured logs with keys:
  - user_id, session_id, turn_id, event_id, type, latency_ms
- Backpressure limits:
  - max audio chunk size
  - max turn audio duration
  - max buffered bytes per turn
- Explicit close/error UX states in dashboard for WS outage (no fallback)
Acceptance gate:
- Can diagnose failures from logs/metrics without packet inspection.
---
Phase 6: Cutover + Cleanup
Deliverable: conversation officially WS-first and WS-only
- Keep old HTTP conversation endpoint temporarily behind internal flag for rollback window only
- Production cutover to WS route from dashboard
- After stabilization window, remove unused SSE chat path code
- Keep HTTP session CRUD/plugins/channels/settings untouched
Acceptance gate:
- No conversation traffic via old HTTP turn path in normal operation.
---
Test Plan (must-pass before cutover)
- Unit:
  - protocol validation, dedupe logic, turn FSM transitions, cancel behavior
- Integration:
  - orchestrator-authenticated WS text turn
  - orchestrator-authenticated WS voice turn
  - same-session text -> voice -> text continuity
- Failure:
  - dropped socket mid-turn
  - duplicate event_id
  - oversized audio chunk
  - cancel during tool execution
- Regression:
  - session CRUD unaffected
  - plugins/channels/settings unaffected
---
Execution order (recommended)
1. Protocol freeze  
2. Agent WS runtime  
3. Orchestrator WS proxy  
4. Dashboard WS client migration  
5. Hardening + cutover
This sequence minimizes risk and avoids scope bleed.
