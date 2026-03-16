# Reliability v1

## Purpose
Raise task reliability for high-value workflows before broadening capability scope.

## Scope (v1)
- Calendar: today/tomorrow/this week queries
- Gmail: count/list/read/reply (draft-first)
- Chat sessions: create/list/load continuity
- Generic web/info: search plus summarize

## Non-Goals
- Full autonomous long-horizon planning
- Broad plugin expansion
- Advanced agent swarms

## Reliability Targets (Release Gates)
- Task success rate >= 95% on golden evals
- Missing or wrong required tool call <= 2%
- Out-of-bounds answer rate = 0%
- Recovery quality >= 90% on injected tool failures
- P95 latency:
  - Calendar/Gmail intent: < 8s
  - Session operations: < 3s

## Core Architecture Requirements

### 1) Deterministic Tool Policies
- Relative date intent (`today`, `tomorrow`, `this week`, `next <weekday>`) must call `current_time` first.
- Calendar window intent must call `upcoming_events` with explicit window bounds (`timeMin`, `timeMax`) or normalized equivalent.
- Final response must only reference events in the validated window.

### 2) Execution State Machine
1. Classify intent.
2. Build plan.
3. Execute required tools.
4. Validate tool outputs against policy.
5. Generate response from validated data only.

### 3) Output Validator (Pre-Response Gate)
Reject/regenerate if:
- Response contains entities not present in tool output.
- Response includes date/time outside requested window.
- Required tool call is missing for intent class.

### 4) Failure Policy
- On tool/API error:
  - Report failure plainly.
  - Retry once if safe.
  - Use bounded fallback path.
  - Never fabricate data.
- If unresolved:
  - Ask one targeted question with a recommended default.

## Prompt and Policy Versioning
- Store orchestration prompts and policy rules as versioned files.
- Every prompt/policy change requires:
  - Version bump.
  - Changelog note.
  - Eval diff report.
  - Rollback reference.

## Observability
Capture per request:
- `intent_class`
- `selected_tools` (ordered)
- `tool_args` (redacted)
- `tool_results` hash/summary
- `validator` verdict and reason
- `final_response` hash
- Latency breakdown

Dashboard counters:
- Tool-policy compliance
- Validator reject rate
- Fallback/repair rate
- Top failing intents
- Regression trend per release

## Test and Eval Strategy

### Golden Evals
- Real transcript based with deterministic assertions.
- Categories:
  - Happy path
  - Edge windows/timezones
  - Tool failure/retry
  - Ambiguous intent requiring one clarification

### CI Gates
- Any change in orchestration/tool routing/policy must run eval suite.
- Merge blocked on threshold failure.

## Rollout Plan (2 Weeks)

### Week 1
- Implement policy engine for calendar plus gmail.
- Add output validator.
- Build golden eval harness.
- Fix failing cases until gate pass.

### Week 2
- Add session continuity evals.
- Canary rollout (10% traffic).
- Monitor metrics plus fast rollback.
- Ramp to 100% only after 48h stable.

## Definition of Done (v1)
- CI eval gates passing at thresholds above.
- Seven-day production stability within targets.
- No recurrence of relative-date failure class.
- Runbook published (triage, rollback, hotfix flow).
