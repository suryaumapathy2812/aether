# Aether — Implementation Plan

> This document defines the phased implementation roadmap for closing the architectural gap between Aether's current Python baseline and the target architecture described in `Requirements.md`. Each phase is independently testable and deployable. Phases are ordered by dependency — no phase can begin until the phase it depends on is complete.

---

## Table of Contents

1. [The Gap](#1-the-gap)
2. [What Stays Unchanged](#2-what-stays-unchanged)
3. [Phase Overview](#3-phase-overview)
4. [Phase 1 — Task Ledger + Persistent Message Store](#4-phase-1--task-ledger--persistent-message-store)
5. [Phase 2 — Client Event Stream](#5-phase-2--client-event-stream)
6. [Phase 3 — Session Agent Loop (E Worker)](#6-phase-3--session-agent-loop-e-worker)
7. [Phase 4 — Non-Blocking Sub-Agents](#7-phase-4--non-blocking-sub-agents)
8. [Phase 5 — Async Task Endpoints](#8-phase-5--async-task-endpoints)
9. [Phase 6 — Context Compaction](#9-phase-6--context-compaction)
10. [Phase 7 — Agent Type System](#10-phase-7--agent-type-system)
11. [Summary](#11-summary)

---

## 1. The Gap

The current agent operates as a request-response system. Every user message produces exactly one response, after which the agent stops working. This is the fundamental architectural gap between Aether today and the target architecture.

### Current Model

```
User message → LLM call → (tool loop, max 10) → Response → DONE
                                                      ↑
                                                One request.
                                                One response.
                                                Agent stops after responding.
```

### Target Model

```
User message → P Worker → Task Ledger → E Worker → tools → LLM → tools → ... → Result
                  ↑            ↑             ↑                                      │
                  │            │             └── Session Loop (outer loop):          │
                  │            │                 persists state, manages context,    │
                  │            │                 handles compaction, spawns          │
                  │            │                 sub-agents, extracts memory         │
                  │            │                                                    │
                  │            └── SQLite-backed task store:                         │
                  │                P writes tasks, E picks up, LLM reads status     │
                  │                                                                 │
                  └── Always responsive.                                            │
                      Delegates to E Worker via Task Ledger.                        │
                      Never blocked.                                                │
                                                                                    │
                  Sub-agents run as independent sessions ←──────────────────────────┘
                  with their own loops, own state, own tools.
                  Parent doesn't block. Checks status via Task Ledger.
```

This maps to the P Worker / E Worker split defined in `Requirements.md` §2.2. The P Worker stays responsive. The E Worker runs the session loop. The Task Ledger (§2.2.1) is the single communication channel between them — SQLite-backed, persistent, traceable, language-agnostic.

### The Six Missing Capabilities

**1. Task Ledger (P↔E Communication)**
The P Worker and E Worker need a communication channel. `Requirements.md` §2.2.1 defines the Task Ledger — a SQLite-backed, in-process task store where the P Worker writes tasks, the E Worker picks them up, and the LLM can query status at any time. This does not exist yet.

**2. Persistent Message Store**
Sessions live in `self._sessions: dict[str, list[dict]]` — lost on restart. For long-running tasks, messages must survive crashes. Sub-agents need their own independent message histories. The Task Ledger needs session context to be durable alongside it.

**3. Session-Level Agent Loop (the outer loop)**
The existing `LLMCore` tool loop is the inner loop — it handles `LLM → tool → LLM` within a single turn. There is no outer loop that persists messages between turns, manages context window size, allows the agent to keep working across multiple LLM calls, triggers memory extraction, or handles sub-agent results feeding back into the session. This is the E Worker's core responsibility.

**4. Non-Blocking Sub-Agents**
`TaskRunner.run()` blocks the parent for up to 60 seconds. Real sub-agents need independent sessions with their own agent loops, fire-and-forget spawning (P Worker delegates via Task Ledger, continues immediately), status checking via the ledger, and result retrieval.

**5. Client Event Stream**
Clients (WebSocket, SSE) need to observe agent work in real time — text chunks, tool calls, status changes. Currently there is no internal pub/sub for client-facing streaming. This is distinct from P↔E communication (which uses the Task Ledger). The event stream is for real-time observation only, not for durable delivery.

**6. Async Task Endpoints**
`POST /v1/chat/completions` blocks until the response is complete. Long-running agent work requires fire-and-forget endpoints and SSE event streams for status observation.

---

## 2. What Stays Unchanged

These components are not modified in any phase. They are the stable foundation the new architecture is built on top of.

| Component | Location | Why Unchanged |
|---|---|---|
| `LLMCore.generate_with_tools()` | `src/aether/llm/` | Becomes the inner loop — already correct |
| `KernelScheduler` | `src/aether/kernel/` | P-Core/E-Core pools are reused as-is |
| `ToolRegistry` / `ToolOrchestrator` | `src/aether/tools/` | Tool execution is correct |
| `ContextBuilder` | `src/aether/llm/` | Extended in Phase 3, not rewritten |
| All existing tools | `src/aether/tools/` | Unchanged |
| All existing plugins | `src/aether/plugins/` | Unchanged |
| All existing skills | `src/aether/skills/` | Unchanged |
| Skill System (all 5 tools, loader) | `src/aether/tools/`, `src/aether/skills/` | Already implemented — out of scope |
| Voice pipeline (WebRTC, TTS, STT) | `src/aether/voice/` | Unchanged |
| `POST /v1/chat/completions` | `src/aether/main.py` | Backward compatible, still works |

---

## 3. Phase Overview

| Phase | Name | Depends On | New Lines | Modified Lines | Status |
|---|---|---|---|---|---|
| 1 | Task Ledger + Persistent Message Store | — | ~400 | ~50 | Not started |
| 2 | Client Event Stream | Phase 1 | ~100 | — | Not started |
| 3 | Session Agent Loop (E Worker) | Phases 1, 2 | ~400 | ~80 | Not started |
| 4 | Non-Blocking Sub-Agents | Phases 1, 2, 3 | ~300 | ~100 | Not started |
| 5 | Async Task Endpoints | Phases 1, 2, 3 | ~250 | ~30 | Not started |
| 6 | Context Compaction | Phase 3 | ~200 | ~50 | Not started |
| 7 | Agent Type System | Phases 3, 4 | ~200 | ~50 | Not started |
| | **Total** | | **~1,850** | **~360** | |

**Files touched**: ~15 modified, ~13 new  
**Existing code preserved**: All components listed in Section 2  
**Already implemented (out of scope)**: Skill System (§9 of Requirements.md) — all 5 tools, loader, 3 directories, progressive loading, marketplace integration

---

## 4. Phase 1 — Task Ledger + Persistent Message Store

**Why first**: Every other phase depends on durable state. The Task Ledger is the P↔E communication channel defined in `Requirements.md` §2.2.1. The message store provides session history. Nothing else can be built until both survive restarts.

### New Files

```
src/aether/session/
├── __init__.py
├── store.py        — SessionStore: all reads/writes for sessions, messages, and tasks
├── models.py       — Session, Message, MessagePart, Task dataclasses
└── ledger.py       — TaskLedger: thin wrapper over SessionStore for P↔E task operations
```

### Modified Files

```
src/aether/memory/store.py   — Add sessions, messages, message_parts, tasks tables to existing SQLite
src/aether/agent.py          — Replace self._sessions dict with SessionStore calls
```

### Schema

**`sessions` table**

| Column | Type | Description |
|---|---|---|
| `session_id` | TEXT PRIMARY KEY | UUID |
| `parent_session_id` | TEXT | NULL for root sessions; set for sub-agent sessions |
| `agent_type` | TEXT | `"main"`, `"general"`, `"explorer"`, `"planner"` |
| `status` | TEXT | `idle` / `busy` / `done` / `error` |
| `created_at` | INTEGER | Unix timestamp |
| `updated_at` | INTEGER | Unix timestamp |

**`messages` table**

| Column | Type | Description |
|---|---|---|
| `message_id` | TEXT PRIMARY KEY | UUID |
| `session_id` | TEXT | Foreign key → sessions |
| `role` | TEXT | `user` / `assistant` / `tool` |
| `content` | TEXT | JSON-encoded content |
| `sequence` | INTEGER | Ordering within session |
| `created_at` | INTEGER | Unix timestamp |

**`message_parts` table**

| Column | Type | Description |
|---|---|---|
| `part_id` | TEXT PRIMARY KEY | UUID |
| `message_id` | TEXT | Foreign key → messages |
| `part_type` | TEXT | `text` / `tool_call` / `tool_result` / `status` |
| `content` | TEXT | JSON-encoded content |
| `status` | TEXT | `pending` / `running` / `completed` / `error` |

**`tasks` table (the Task Ledger)**

This is the implementation of `Requirements.md` §2.2.1. It is the single communication channel between the P Worker and E Worker.

| Column | Type | Description |
|---|---|---|
| `task_id` | TEXT PRIMARY KEY | UUID |
| `session_id` | TEXT | Foreign key → sessions (which session spawned this task) |
| `type` | TEXT | `tool_call` / `memory_extract` / `proactive_check` / `scheduled` / `sub_agent` |
| `status` | TEXT | `pending` / `running` / `complete` / `error` |
| `priority` | TEXT | `high` / `normal` / `low` |
| `payload` | TEXT | JSON-encoded input to the task |
| `result` | TEXT | JSON-encoded output (NULL until complete) |
| `error` | TEXT | Error message (NULL unless status is error) |
| `submitted_at` | INTEGER | Unix timestamp — when P Worker created the task |
| `started_at` | INTEGER | Unix timestamp — when E Worker picked it up (NULL until running) |
| `completed_at` | INTEGER | Unix timestamp — when E Worker finished (NULL until complete/error) |

**Status transitions** (enforced by `TaskLedger`):
```
pending → running → complete
                  → error
```
No other transitions are valid. A failed task is retried by creating a new task, not by resetting status.

**No pruning.** Tasks are retained indefinitely. The LLM can reference past work ("I sent that email 3 hours ago"). Storage is negligible — thousands of tasks = kilobytes.

### Interface

```python
class SessionStore:
    # --- Session operations ---
    async def create_session(
        self,
        agent_type: str = "main",
        parent_id: str | None = None,
    ) -> str: ...                                          # Returns session_id

    async def get_session(self, session_id: str) -> Session: ...

    async def update_status(self, session_id: str, status: str) -> None: ...

    # --- Message operations ---
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: dict | str,
    ) -> str: ...                                          # Returns message_id

    async def get_messages(self, session_id: str) -> list[Message]: ...

    async def save_part(self, session_id: str, part: MessagePart) -> None: ...


class TaskLedger:
    """P↔E communication channel. Wraps SessionStore task operations.
    
    Read/write rules (from Requirements.md §2.2.1):
    - P Worker: create tasks (pending), read any task's status/result
    - E Worker: pick up pending tasks, set to running, set to complete/error
    - LLM: read the ledger via check_tasks tool
    """

    async def submit(
        self,
        session_id: str,
        task_type: str,
        payload: dict,
        priority: str = "normal",
    ) -> str: ...                                          # Returns task_id

    async def pick_next(self) -> Task | None: ...          # E Worker picks FIFO, priority-weighted

    async def set_running(self, task_id: str) -> None: ...

    async def set_complete(self, task_id: str, result: dict) -> None: ...

    async def set_error(self, task_id: str, error: str) -> None: ...

    async def get_task(self, task_id: str) -> Task: ...

    async def get_tasks(
        self,
        session_id: str | None = None,
        status: str | None = None,
    ) -> list[Task]: ...                                   # LLM reads via check_tasks tool

    async def get_pending_on_restart(self) -> list[Task]: ...  # Re-queue running → pending on restart
```

### On Restart

When the agent restarts, the `TaskLedger` scans for tasks with status `running` — these were in-flight when the agent died. They are re-queued as `pending` so the E Worker picks them up again. This is the resumability guarantee from `Requirements.md` §2.2.1.

### Acceptance Criteria

- [ ] Agent restart does not lose any session messages or tasks
- [ ] `get_messages(session_id)` returns messages in correct sequence order
- [ ] Sub-agent sessions have `parent_session_id` set correctly
- [ ] `self._sessions` dict is fully removed from `AgentCore`
- [ ] Task status transitions are enforced (`pending → running → complete|error` only)
- [ ] `get_pending_on_restart()` correctly re-queues `running` tasks as `pending`
- [ ] `get_tasks()` returns tasks filterable by session and status
- [ ] All existing `/v1/chat/completions` behavior is unchanged

### Estimated Size

~400 new lines, ~50 modified lines

---

## 5. Phase 2 — Client Event Stream

**Why second**: Clients (WebSocket, SSE) need to observe agent work in real time — text chunks streaming, tool calls executing, status changes. This is the **client-facing observation layer**, not the P↔E communication channel. P↔E communication uses the Task Ledger (Phase 1). The event stream is for real-time display only.

### New Files

```
src/aether/kernel/event_stream.py   — EventStream: async pub/sub backed by asyncio.Queue
```

### Modified Files

None. The event stream is wired into consumers in later phases.

### Design

```python
class EventStream:
    async def publish(self, topic: str, event: dict) -> None: ...

    def subscribe(self, topic: str) -> AsyncGenerator[dict, None]: ...
```

**Topic conventions**:

| Topic | Published by | Consumed by |
|---|---|---|
| `session.{id}.event` | E Worker / SessionLoop (Phase 3) | WS sidecar, SSE endpoint (Phase 5) |
| `session.{id}.status` | E Worker / SessionLoop, SubAgentManager | P Worker / AgentCore, SSE endpoint |

**What this is NOT**:

This is not the P↔E communication channel. The Task Ledger (Phase 1) handles all P↔E communication — task delegation, status tracking, result delivery. The event stream is a separate concern: it broadcasts ephemeral events to connected clients for real-time UI updates. If no client is listening, events are dropped. That's fine — durable state lives in `SessionStore` and `TaskLedger`.

**Implementation notes**:
- One `asyncio.Queue` per subscriber per topic
- Subscribers are registered at startup; topics are created on first publish
- No persistence — events are in-memory only. If a subscriber is not listening, the event is dropped. This is intentional: events are for real-time observation, not durable delivery.
- Pattern matches the existing `KernelScheduler` job queue pattern — no new dependencies

### Acceptance Criteria

- [ ] `publish()` delivers to all active subscribers on that topic
- [ ] `subscribe()` returns an async generator that yields events as they arrive
- [ ] A subscriber that disconnects does not block publishers
- [ ] Topics with no subscribers do not accumulate memory

### Estimated Size

~100 new lines, 0 modified lines

---

## 6. Phase 3 — Session Agent Loop (E Worker)

**Why third**: This is the core capability. The session loop is the E Worker's main responsibility — the outer loop that makes the agent autonomous. It keeps working, manages context, triggers memory extraction, and decides when it is done.

### P/E Worker Mapping

| Responsibility | Worker | How |
|---|---|---|
| Receive user message, create task | P Worker | Writes `tool_call` task to Task Ledger |
| Run the session loop | E Worker | Picks up task, runs `SessionLoop.run()` |
| Stream events to client | E Worker → EventStream | Publishes events as they are generated |
| Trigger memory extraction | E Worker | Writes `memory_extract` task to Task Ledger after each turn |
| Inform user of completion | P Worker | Reads task result from Task Ledger, delivers to client |

### New Files

```
src/aether/session/
├── loop.py         — SessionLoop: the outer agent loop (E Worker's core)
└── compaction.py   — Stub only; full implementation in Phase 6
```

### Modified Files

```
src/aether/agent.py                      — Add run_session() method, wire P→E delegation via TaskLedger
src/aether/kernel/contracts.py           — Add SessionStatus enum, new event types
src/aether/services/reply_service.py     — Add path to build context from persisted session
```

### Design

```python
class SessionLoop:
    """The E Worker's outer loop. Runs autonomously until the task is complete."""

    async def run(self, session_id: str, abort: asyncio.Event) -> None:
        while not abort.is_set():
            # 1. Load messages from SessionStore
            messages = await self.session_store.get_messages(session_id)

            # 2. Exit condition: last assistant message has no pending tool calls
            if self._should_exit(messages):
                await self.session_store.update_status(session_id, "done")
                break

            # 3. Context overflow check → compact (stub in Phase 3, full in Phase 6)
            if self._needs_compaction(messages):
                await self.compactor.compact(session_id)
                continue

            # 4. Build context and call LLM (uses existing LLMCore — unchanged)
            envelope = await self.context_builder.build_from_session(session_id)
            async for event in self.llm_core.generate_with_tools(envelope):
                await self.event_stream.publish(f"session.{session_id}.event", event)
                await self.session_store.save_part(session_id, event)

            # 5. Trigger memory extraction (async — does not block the loop)
            await self.task_ledger.submit(
                session_id=session_id,
                task_type="memory_extract",
                payload={"session_id": session_id, "turn": len(messages)},
                priority="low",
            )

            # 6. Loop back
```

**Exit conditions** (`_should_exit`):
- Last message is from `assistant` role
- That message contains no `tool_call` parts with status `pending` or `running`
- No abort signal received

**Memory extraction** (step 5):
After each turn, the E Worker writes a `memory_extract` task to the Task Ledger. This is picked up by a separate E Worker coroutine that extracts facts, memories, and decisions from the conversation (see `Requirements.md` §5.1.1 and §7.5). Extraction is fire-and-forget — it never blocks the session loop. If extraction fails, it is retried via the Task Ledger's normal retry mechanism (create a new task).

**Relationship to existing code**:
- `LLMCore.generate_with_tools()` is the **inner loop** — unchanged
- `SessionLoop.run()` is the **outer loop** — new, runs on E Worker
- `ReplyService` gets a new code path: build context from `SessionStore` instead of from scratch. The existing path remains for backward compatibility with `/v1/chat/completions`.

**How P Worker delegates to E Worker**:
```python
# In AgentCore (P Worker side):
async def handle_user_message(self, session_id: str, message: str):
    # 1. Persist the user message
    await self.session_store.add_message(session_id, role="user", content=message)

    # 2. Delegate to E Worker via Task Ledger
    task_id = await self.task_ledger.submit(
        session_id=session_id,
        task_type="tool_call",
        payload={"action": "run_session"},
        priority="high",
    )

    # 3. P Worker returns immediately — user sees "thinking..." or similar
    return task_id
```

### Acceptance Criteria

- [ ] Agent continues working after a response if tool calls are pending
- [ ] Agent stops when no tool calls are pending in the last assistant message
- [ ] Abort signal (`asyncio.Event`) stops the loop cleanly within one iteration
- [ ] All events are published to `EventStream` as they are generated
- [ ] All events are persisted to `SessionStore` as they are generated
- [ ] Memory extraction task is written to Task Ledger after each turn
- [ ] P Worker delegates to E Worker via Task Ledger (not direct function call)
- [ ] `/v1/chat/completions` behavior is unchanged (uses existing ReplyService path)

### Estimated Size

~400 new lines, ~80 modified lines

---

## 7. Phase 4 — Non-Blocking Sub-Agents

**Why fourth**: Depends on `SessionLoop` (Phase 3), `EventStream` (Phase 2), `SessionStore` and `TaskLedger` (Phase 1). Sub-agents are sessions with their own loops — they need all three.

### P/E Worker Mapping

| Responsibility | Worker | How |
|---|---|---|
| LLM decides to spawn a sub-agent | E Worker (parent session loop) | LLM calls `spawn_task` tool |
| Create child session + task | `spawn_task` tool | Writes `sub_agent` task to Task Ledger |
| Run child session loop | E Worker (new coroutine) | Picks up task, runs `SessionLoop.run()` for child |
| Check sub-agent status | E Worker (parent session loop) | LLM calls `check_tasks` tool, reads from Task Ledger |
| Notify user of completion | P Worker | Reads completed task from Task Ledger, delivers to client |

### New Files

```
src/aether/agents/
├── manager.py       — SubAgentManager: spawn, check status, get result
└── agent_types.py   — Agent type definitions (general, explorer, planner)

src/aether/tools/
└── check_tasks.py   — check_tasks tool: LLM reads the Task Ledger
```

### Modified Files

```
src/aether/agents/task_runner.py   — Rewrite to use SessionLoop (non-blocking)
src/aether/tools/run_task.py       — Rewrite as spawn_task tool (fire-and-forget)
```

### Design

```python
class SubAgentManager:
    async def spawn(
        self,
        prompt: str,
        agent_type: str,
        parent_session_id: str,
    ) -> str:
        """Create child session, write task to ledger, return task_id immediately."""
        child_id = await self.session_store.create_session(
            agent_type=agent_type,
            parent_id=parent_session_id,
        )
        await self.session_store.add_message(child_id, role="user", content=prompt)

        # Write to Task Ledger — E Worker picks this up
        task_id = await self.task_ledger.submit(
            session_id=parent_session_id,
            task_type="sub_agent",
            payload={"child_session_id": child_id, "agent_type": agent_type},
            priority="normal",
        )

        # Start the child session loop (E Worker coroutine)
        task = asyncio.create_task(
            self._run_child(task_id, child_id)
        )
        self._running[child_id] = task

        return task_id  # Returns IMMEDIATELY — parent is not blocked

    async def _run_child(self, task_id: str, child_session_id: str) -> None:
        """Run child session loop, update Task Ledger on completion."""
        await self.task_ledger.set_running(task_id)
        try:
            await self.session_loop.run(child_session_id, abort=asyncio.Event())
            messages = await self.session_store.get_messages(child_session_id)
            result = self._extract_final_text(messages)
            await self.task_ledger.set_complete(task_id, {"result": result})
        except Exception as e:
            await self.task_ledger.set_error(task_id, str(e))
```

**LLM-facing tools**:

| Tool | Description |
|---|---|
| `spawn_task(prompt, agent_type)` | Delegate work to a sub-agent. Returns `task_id` immediately. |
| `check_tasks(task_id?, session_id?, status?)` | Read the Task Ledger. Poll status, list tasks, retrieve results. This is the single tool for all task introspection — the LLM uses it to answer "what's happening with X?" |

**`check_tasks` is the Task Ledger reader** defined in `Requirements.md` §2.2.1. It reads directly from the `tasks` SQLite table. It is not specific to sub-agents — it can query any task (tool calls, memory extractions, sub-agents, scheduled tasks). This makes the entire E Worker's work traceable and queryable by the LLM.

**Sub-agent execution**:
- Sub-agents run on the E-Core pool (background workers) — they never block interactive P-Core work
- Sub-agents cannot spawn their own sub-agents by default (configurable per agent type in Phase 7)
- On completion, the Task Ledger entry is updated to `complete` — the P Worker reads this and notifies the user
- On restart, in-flight sub-agent tasks are re-queued via `TaskLedger.get_pending_on_restart()`

### Acceptance Criteria

- [ ] `spawn()` returns immediately — parent session is not blocked
- [ ] Sub-agent runs its full `SessionLoop` independently
- [ ] Sub-agent progress is tracked in the Task Ledger (`pending → running → complete|error`)
- [ ] `check_tasks(task_id=X)` returns current status and result from the Task Ledger
- [ ] `check_tasks(session_id=X)` returns all tasks for a session
- [ ] Sub-agent failure sets Task Ledger status to `error` — does not crash the parent session
- [ ] Agent restart re-queues in-flight sub-agent tasks

### Estimated Size

~300 new lines, ~100 modified lines

---

## 8. Phase 5 — Async Task Endpoints

**Why fifth**: Depends on `SessionLoop` (Phase 3) and `EventStream` (Phase 2) for streaming. Provides the HTTP surface that clients use to start long-running work and observe it.

### P/E Worker Mapping

| Responsibility | Worker | How |
|---|---|---|
| Accept HTTP request, create session | P Worker | Handles `POST /v1/sessions/{id}/prompt` |
| Delegate to E Worker | P Worker | Writes task to Task Ledger, returns `202` immediately |
| Stream events to SSE client | P Worker | Subscribes to `EventStream`, forwards to HTTP response |
| Serve task status | P Worker | Reads from Task Ledger, returns JSON |

The async endpoints are the P Worker's HTTP surface. They never block — they delegate to the E Worker via the Task Ledger and return immediately. Clients observe progress via SSE (backed by `EventStream`) or by polling the Task Ledger via REST.

### New Files

```
src/aether/http/
├── sessions.py   — Session management endpoints
└── events.py     — SSE event stream endpoint
```

### Modified Files

```
src/aether/main.py    — Mount new routers
src/aether/agent.py   — Add async_prompt() and session management methods
```

### New Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/sessions` | Create a new session. Returns `session_id`. |
| `POST` | `/v1/sessions/{id}/prompt` | Submit a prompt. P Worker writes task to Task Ledger. Returns `202 Accepted` immediately. |
| `GET` | `/v1/sessions/{id}/events` | SSE stream of all events for this session (from `EventStream`). |
| `GET` | `/v1/sessions/{id}/status` | Current session status (`idle` / `busy` / `done` / `error`). |
| `GET` | `/v1/sessions/{id}/messages` | All messages in the session, in sequence order. |
| `POST` | `/v1/sessions/{id}/cancel` | Cancel a running session. |
| `GET` | `/v1/tasks` | List all tasks from the Task Ledger (filterable by session, status, type). |
| `GET` | `/v1/tasks/{id}` | Status and result of a specific task from the Task Ledger. |

**SSE event types** (published on `GET /v1/sessions/{id}/events`):

| Event type | Payload |
|---|---|
| `text_chunk` | `{"content": "..."}` |
| `tool_call` | `{"tool": "...", "args": {...}}` |
| `tool_result` | `{"tool": "...", "result": {...}}` |
| `status` | `{"status": "busy" \| "done" \| "error"}` |
| `error` | `{"message": "..."}` |

**Backward compatibility**: `POST /v1/chat/completions` is unchanged. It continues to work as a synchronous endpoint. The new endpoints are additive.

### Acceptance Criteria

- [ ] `POST /v1/sessions/{id}/prompt` returns `202` immediately, before the agent starts working
- [ ] `GET /v1/sessions/{id}/events` streams events in real time as the agent works
- [ ] SSE stream closes cleanly when the session reaches `done` or `error` status
- [ ] `POST /v1/sessions/{id}/cancel` stops the session loop within one iteration
- [ ] `GET /v1/tasks` returns tasks from the Task Ledger (not a separate data source)
- [ ] `POST /v1/chat/completions` behavior is unchanged

### Estimated Size

~250 new lines, ~30 modified lines

---

## 9. Phase 6 — Context Compaction

**Why sixth**: Needed for sessions that run long enough to exceed the model's context window. Without compaction, long-running sessions will fail with context overflow errors.

This is purely E Worker work — compaction runs inside the session loop when the context window is approaching its limit. The P Worker is not involved.

### New Files

```
src/aether/session/compaction.py   — Full implementation (stubbed in Phase 3)
```

### Modified Files

```
src/aether/session/loop.py          — Wire compaction into the loop (stub → real call)
src/aether/llm/context_builder.py   — Build context correctly from compacted messages
```

### Design

**Trigger**: When the estimated token count of all messages in a session exceeds 80% of the model's context window limit.

**Algorithm**:
1. Identify the compaction boundary: preserve the last N turns (default: 10) and all messages with `pending` or `running` tool calls
2. Take all messages before the boundary
3. Call the LLM with a summarization prompt: "Summarize the following conversation history concisely, preserving all facts, decisions, and context needed to continue the work."
4. Replace the pre-boundary messages with a single `[compacted_summary]` message containing the summary
5. Update `SessionStore` — the compacted messages are deleted, the summary message is inserted at sequence 0

**Invariants**:
- Pending tool calls are never compacted — they must remain in the context
- The summary message is marked with `part_type: "compacted_summary"` so it can be identified
- Compaction is idempotent — running it twice on the same session produces the same result

### Acceptance Criteria

- [ ] Sessions that exceed 80% context window trigger compaction automatically
- [ ] After compaction, the session loop continues without error
- [ ] Pending tool calls are never lost during compaction
- [ ] The compacted summary is stored in `SessionStore` and survives restarts
- [ ] Compaction does not cause duplicate messages

### Estimated Size

~200 new lines, ~50 modified lines

---

## 10. Phase 7 — Agent Type System

**Why last**: This is the polish layer. It defines specialized E Worker behaviors and tool restrictions per agent type. Everything it depends on (SessionLoop, SubAgentManager) must be working first.

### New Files

```
src/aether/agents/
└── definitions/
    ├── general.py    — General-purpose sub-agent (all tools allowed)
    ├── explorer.py   — Read-only exploration (no write tools)
    └── planner.py    — Planning agent (read-only, produces structured plans)
```

### Modified Files

```
src/aether/agents/agent_types.py       — Load definitions, resolve by name
src/aether/llm/context_builder.py      — Support agent-specific system prompts
```

### Design

Each agent type is a Python dataclass (or simple dict) defining:

```python
@dataclass
class AgentTypeDefinition:
    name: str                        # "general", "explorer", "planner"
    description: str                 # Shown to the LLM when choosing agent type
    system_prompt: str               # Injected as the system message for this agent
    allowed_tools: list[str] | None  # None = all tools; list = whitelist
    model: str | None                # None = use default model
    max_iterations: int              # Maximum loop iterations before forced exit
    can_spawn_subagents: bool        # Default: False
```

**Built-in agent types**:

| Type | Allowed Tools | Can Spawn | Use Case |
|---|---|---|---|
| `general` | All | No | General delegation — research, writing, analysis |
| `explorer` | Read-only tools only | No | Codebase exploration, file reading, search |
| `planner` | Read-only tools only | No | Produces structured plans, does not execute |

**`spawn_task` tool update**: Accepts `agent_type` parameter. Validates that the requested type exists before spawning.

### Acceptance Criteria

- [ ] `spawn_task(prompt, agent_type="explorer")` spawns a sub-agent with only read-only tools
- [ ] `spawn_task(prompt, agent_type="unknown")` returns a clear error to the LLM
- [ ] Agent type system prompt is correctly injected as the session's system message
- [ ] `can_spawn_subagents: False` prevents recursive spawning

### Estimated Size

~200 new lines, ~50 modified lines

---

## 11. Summary

### Size Estimate

| Phase | New Lines | Modified Lines | Total |
|---|---|---|---|
| 1 — Task Ledger + Persistent Message Store | ~400 | ~50 | ~450 |
| 2 — Client Event Stream | ~100 | 0 | ~100 |
| 3 — Session Agent Loop (E Worker) | ~400 | ~80 | ~480 |
| 4 — Non-Blocking Sub-Agents | ~300 | ~100 | ~400 |
| 5 — Async Task Endpoints | ~250 | ~30 | ~280 |
| 6 — Context Compaction | ~200 | ~50 | ~250 |
| 7 — Agent Type System | ~200 | ~50 | ~250 |
| **Total** | **~1,850** | **~360** | **~2,210** |

### What Aether Can Do After Each Phase

| After Phase | New Capability |
|---|---|
| Phase 1 | Sessions and tasks survive restarts. P↔E communication via Task Ledger. LLM can query task status. |
| Phase 2 | Clients can observe agent work in real time via event streaming. |
| Phase 3 | E Worker keeps working autonomously until the task is complete. Memory extraction after each turn. |
| Phase 4 | Agent can delegate work to sub-agents without blocking the user. All tasks traceable via Task Ledger. |
| Phase 5 | Clients can start long-running work via HTTP and observe it via SSE. |
| Phase 6 | Sessions can run indefinitely without hitting context window limits. |
| Phase 7 | Sub-agents have specialized behaviors and restricted tool access. |

### Architecture Alignment

This plan implements the P Worker / E Worker split defined in `Requirements.md` §2.2:

| Component | Maps to | Phase |
|---|---|---|
| Task Ledger (`tasks` table in SQLite) | §2.2.1 — P↔E communication channel | Phase 1 |
| `SessionStore` (sessions, messages) | §7.3 — Conversational memory (session log) | Phase 1 |
| `EventStream` | Client-facing observation (not P↔E) | Phase 2 |
| `SessionLoop` | E Worker's outer loop | Phase 3 |
| Memory extraction (fire-and-forget) | §5.1.1 — E Worker extracts facts/memories/decisions | Phase 3 |
| `SubAgentManager` + `check_tasks` | §2.2.1 — LLM reads Task Ledger | Phase 4 |
| Async HTTP endpoints | P Worker's HTTP surface | Phase 5 |
| Context compaction | §5.3 — Context window management | Phase 6 |
| Agent type definitions | E Worker specialization | Phase 7 |

**Not in scope** (already implemented or deferred):
- Skill System (§9) — already implemented: 5 tools, loader, 3 directories, marketplace
- Proactive Engine nightly analysis (§6.2) — requires memory system implementation first
- Memory System three-bucket model (§7) — separate implementation plan needed
- Go rewrite — explicitly deferred to a future phase

### Three Scenarios After All Phases

**Scenario 1 — Quick interaction** (same as today)
User asks a question → P Worker responds → Done. No change to this path.

**Scenario 2 — Background task**
User asks agent to do something long → P Worker writes task to Task Ledger → Immediately acknowledges user → E Worker picks up task, runs session loop → Task Ledger updated to `complete` → P Worker notifies user. User can ask "is that done yet?" at any time — LLM reads Task Ledger via `check_tasks`.

**Scenario 3 — Long-running autonomous work**
P Worker receives a complex task → E Worker runs session loop → Compacts context when needed → Spawns sub-agents for parallel work (each tracked in Task Ledger) → Extracts memory after each turn → All progress traceable via `check_tasks` → P Worker notifies user on completion. Same capability as Claude Code / OpenCode, with full restart resumability.

---

*This document is a living specification. Update the Status column in Section 3 as phases are completed.*
