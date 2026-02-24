# Aether — One-Shot Implementation Plan (Final Product Completion)

> This plan is optimized for a single execution wave to complete all remaining gaps against `Product.md` and `Requirements.md` across `app/`, `orchestrator/`, and `dashboard/`.

---

## 1. One-Shot Objective

Ship the remaining work in one coordinated run so Aether meets product-level expectations:
- One coherent P Worker experience across voice/text/video
- Non-blocking E Worker delegation with seamless user continuity
- Multi-device canonical session behavior
- Always-on proactive engine with feedback learning loop
- Resilience, observability, security, and dashboard parity
- Measured compliance with key latency/reliability requirements

This run is complete only when all final gates (Section 10) are green.

---

## 2. One-Shot Rules

1. No partial rollout inside this run. Changes are integrated as one release candidate.
2. Parallelize aggressively, but merge through defined checkpoints.
3. Any failed validation triggers stop-report-fix-resume.
4. No API contract regressions for existing clients (`/chat`, `/v1/chat/completions`, dashboard routes).
5. `reference_projects/` remain out of scope and excluded from validation.

---

## 3. Scope In / Scope Out

### In scope
- `app/src/aether/**`
- `orchestrator/src/**`
- `dashboard/src/**` (+ supporting API/proxy wiring)
- tests, metrics, traceability, reliability, security policy behavior

### Out of scope
- Go/Rust runtime rewrite
- external cloned reference repos
- non-product experimental branches

---

## 4. Workstream Model (Parallel)

## Workstream A — P Worker UX and Delegation Continuity (app)
Goal: ensure user never experiences blocking or dead air.

Deliverables:
- Non-blocking `delegate_to_agent` everywhere
- Immediate acknowledgement + async completion callbacks
- Structured completion narration in ongoing conversations
- Stable text/voice parity for task lifecycle updates

Primary files:
- `app/src/aether/voice/backends/gemini/bridge.py`
- `app/src/aether/voice/backends/gemini/tool_bridge.py`
- `app/src/aether/agent.py`
- `app/src/aether/main.py`

---

## Workstream B — Canonical Multi-Device Session Semantics (app + orchestrator)
Goal: one coherent session across all active devices.

Deliverables:
- Canonical session ownership + device attachment model
- Output fan-out policy (audio/text/notification)
- Input arbitration (`last-write-wins` with deterministic tie handling)
- Notification sync state across devices

Primary files:
- `app/src/aether/session/store.py`
- `app/src/aether/voice/session.py`
- `app/src/aether/voice/webrtc.py`
- `app/src/aether/voice/telephony.py`
- `orchestrator/src/main.py`

---

## Workstream C — Realtime Resilience and Failure Recovery (app)
Goal: graceful behavior under disconnects/outages.

Deliverables:
- Bounded reconnect buffers (audio/text) + replay policy
- Durable resumption tokens and restart continuity
- Backoff/circuit behavior for provider outages
- User-facing degraded-mode responses

Primary files:
- `app/src/aether/voice/backends/gemini/session.py`
- `app/src/aether/voice/session.py`
- `app/src/aether/session/store.py`
- `app/src/aether/core/config.py`

---

## Workstream D — Proactive Engine End-to-End Completion (app + orchestrator)
Goal: complete the always-on proactive promise.

Deliverables:
- End-to-end event -> decision -> queue -> delivery -> feedback -> learning loop
- Notification type policy enforcement (`suppress/queue/nudge/surface/interrupt`)
- Delivery timing with user/context awareness
- Nightly analysis outputs wired into runtime behavior

Primary files:
- `app/src/aether/agent.py`
- `app/src/aether/services/nightly_analysis.py`
- `app/src/aether/memory/store.py`
- `orchestrator/src/main.py`

---

## Workstream E — Observability + SLO Enforcement (app + orchestrator + dashboard)
Goal: prove behavior with measurable budgets and traces.

Deliverables:
- Correlation IDs across P/E/task/tool/notification events
- Realtime metrics: connect, first-token, first-audio, delegation latency, tool timeout rates
- SLO checks in CI and dashboards
- End-to-end trace drill-down in dashboard

Primary files:
- `app/src/aether/kernel/event_bus.py`
- `app/src/aether/session/ledger.py`
- `app/src/aether/core/metrics.py`
- `orchestrator/src/main.py`
- `dashboard/src/**` (observability views)

---

## Workstream F — Security + Policy Hardening (app + orchestrator)
Goal: satisfy security and policy requirements with enforceable controls.

Deliverables:
- Tool policy checks by capability/context/entitlement
- timeout/cancel enforcement by tool category
- sandbox/allowlist verification for high-risk tools
- secret/token handling and logging redaction audit

Primary files:
- `app/src/aether/tools/**`
- `app/src/aether/core/config.py`
- `orchestrator/src/auth.py`
- `orchestrator/src/main.py`

---

## Workstream G — Dashboard Product Parity (dashboard)
Goal: expose controls and transparency required by product vision.

Deliverables:
- Session timeline and task lifecycle surfaces
- Proactive notification controls/history/feedback
- Memory inspect/edit/delete UX
- Device presence + plugin health/reconnect UX

Primary files:
- `dashboard/src/app/**`
- `dashboard/src/components/**`

---

## Workstream H — Multimodal/Video Completion (app)
Goal: complete voice+text+video front-door behavior.

Deliverables:
- practical video input path for P Worker
- multimodal response behavior and fallback handling
- transport/API support for multimodal payloads

Primary files:
- `app/src/aether/voice/**`
- related endpoint handlers in `app/src/aether/main.py`

---

## 5. Dependency Graph and Critical Path

Critical path:
1. Workstream A (non-blocking continuity)
2. Workstream C (resilience)
3. Workstream B (multi-device semantics)
4. Workstream D (proactive completion)
5. Workstream E/F (SLO + security hardening)
6. Workstream G/H (dashboard parity + multimodal completion)

Parallel start allowed:
- E, F can begin once A/C interfaces are stable
- G can begin once API contracts from A/B/D are stable

---

## 6. Integration Checkpoints (Mandatory)

Checkpoint 1: Core behavior merge
- Merge A + C first
- Validate no regression in app realtime/text paths

Checkpoint 2: Session coherence merge
- Merge B
- Validate cross-device behavior and notification sync

Checkpoint 3: Proactive merge
- Merge D
- Validate full proactive lifecycle and feedback learning

Checkpoint 4: Hardening merge
- Merge E + F
- Validate traceability, SLO instrumentation, policy/security tests

Checkpoint 5: Product surface merge
- Merge G + H
- Validate dashboard parity and multimodal behavior

---

## 7. Validation Matrix (Run in One-Shot)

## A. Automated tests
- `app` suite: pass
- `orchestrator` suite: pass
- `dashboard` lint/type/build tests: pass

## B. Integration scenarios
1. Long delegated task while user continues conversation (no blocking)
2. Two active devices with synchronized notification state
3. Realtime disconnect + reconnect + replay behavior
4. Proactive webhook event to user notification with feedback loop write-back
5. Memory extraction after each turn and retrievability in subsequent turns

## C. SLO checks (must be measured)
- text first token < 800ms (p95 target)
- voice first audio < 500ms (p95 target)
- memory retrieval < 100ms (p95 target)
- notification delivery < 3s (p95 target)

## D. Security/policy checks
- unauthorized tool paths denied as expected
- timeout/cancellation behavior for over-budget tools
- token/secret redaction in logs confirmed

---

## 8. Stop-on-Failure Protocol

For any failed gate:
1. Stop execution wave
2. Publish failure report (what failed, where, impact)
3. Propose exact remediation patch set
4. Apply fix
5. Re-run failed gate + dependent gates

No silent fallback for failed critical gates.

---

## 9. Release and Rollback Plan

Release readiness:
- All gates in Section 10 pass
- migration notes documented for ops
- feature flags/config toggles validated

Rollback readiness:
- keep prior stable tag + config rollback path
- verify rollback can restore prior behavior within one deployment cycle

---

## 10. Final Acceptance Gates (All Required)

1. App tests green
2. Orchestrator tests green
3. Dashboard checks green
4. End-to-end non-blocking delegation scenario green
5. Multi-device coherence scenario green
6. Reconnect/outage recovery scenario green
7. Proactive engine scenario green
8. SLO metrics captured and within thresholds
9. Security/policy checks green
10. No contract regressions in existing public endpoints

If any gate fails, one-shot is not complete.

---

## 11. Execution Command Center (for implementation run)

When executing this one-shot:
- Use parallel subagents per workstream
- Merge only at mandatory checkpoints
- Keep a live checklist tied to Section 10 gates
- Do not close until all gates pass in one final validation pass

---

*This document replaces roadmap-style planning and is intended for one execution wave.*
