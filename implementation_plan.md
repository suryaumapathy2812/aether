---
The Real Gap: What Claude Code / OpenCode Have That Aether Doesn't
After reading every line of your code, here's the precise architectural gap:
Your current model:
User message → LLM call → (tool loop max 10) → Response → DONE
                                                    ↑
                                              One request, one response.
                                              Agent stops thinking after responding.
Claude Code / OpenCode model:
User message → Session Loop → LLM call → tools → LLM call → tools → ... → Response
                    ↑              ↑                                            │
                    │              └── Inner loop: LLM ↔ tools (unlimited)      │
                    │                                                           │
                    └── Outer loop: persists state, manages context,            │
                        handles compaction, spawns sub-agents,                  │
                        checks if agent should keep working                     │
                                                                                │
                    Sub-agents run as independent sessions ←────────────────────┘
                    with their own loops, own state, own tools
                    Parent doesn't block. Checks status via tool.
There are 5 fundamental capabilities you need:
1. Session-Level Agent Loop (the outer loop)
Your LLMCore tool loop is the inner loop — it handles LLM → tool → LLM within a single turn. But there's no outer loop that:
- Persists messages to DB between turns
- Manages context window (compaction when full)
- Allows the agent to keep working across multiple LLM calls
- Handles sub-agent results feeding back into the loop
2. Persistent Message Store
Sessions live in self._sessions: dict[str, list[dict]] — lost on restart. For long-running tasks, messages must survive crashes, and sub-agents need their own message histories.
3. Non-Blocking Sub-Agents
TaskRunner.run() blocks the parent for 60s. Real sub-agents need:
- Independent sessions with their own agent loops
- Fire-and-forget spawning (parent continues immediately)
- Status checking (parent can ask "is it done?")
- Result retrieval (parent gets output when ready)
4. Event Bus
Background work (sub-agents, long tasks) needs to notify the main thread when done. Currently there's no internal pub/sub — notifications only flow outward to WS clients.
5. Async Task Endpoints
/v1/chat/completions blocks until the response is complete. For long-running agent work, you need fire-and-forget endpoints + SSE event streams for status.
---
Implementation Plan: 7 Phases
Here's the plan, ordered by dependency. Each phase is independently testable and deployable.
Phase 1: Persistent Message Store
Why first: Everything else depends on durable session state.
NEW: src/aether/session/
├── store.py          — SessionStore (SQLite: sessions, messages, parts tables)
├── models.py         — Session, Message, MessagePart dataclasses
└── __init__.py
MODIFIED:
├── src/aether/memory/store.py    — Add sessions/messages tables to existing SQLite
├── src/aether/agent.py           — Replace self._sessions dict with SessionStore reads/writes
What it does:
- sessions table: session_id, parent_session_id, agent_type, status (idle/busy/done/error), created_at, updated_at
- messages table: message_id, session_id, role, content (JSON), sequence, created_at
- message_parts table: part_id, message_id, part_type (text/tool_call/tool_result/status), content (JSON), status (pending/running/completed/error)
- SessionStore wraps aiosqlite with create_session(), add_message(), get_messages(), update_status()
- AgentCore switches from self._sessions[id] to await session_store.get_messages(id)
Lines: ~300 new, ~50 modified
---
Phase 2: Event Bus
Why second: Sub-agents and the session loop need to publish events.
NEW: src/aether/kernel/event_bus.py    — Simple async pub/sub
What it does:
- EventBus class with publish(topic, event) and subscribe(topic) → AsyncGenerator
- Topics: session.{id}.event, session.{id}.status, task.completed
- Backed by asyncio.Queue per subscriber (same pattern as your scheduler's job queues)
- AgentCore subscribes to task.completed to push notifications
- WS sidecar subscribes to session events for real-time updates
Lines: ~100 new
---
Phase 3: Session Agent Loop
Why third: This is the core capability — the outer loop that makes the agent autonomous.
NEW: src/aether/session/
├── loop.py           — SessionLoop (the outer agent loop)
└── compaction.py     — Context compaction (summarize old messages when context is full)
MODIFIED:
├── src/aether/agent.py               — New method: run_session() that starts a SessionLoop
├── src/aether/kernel/contracts.py     — Add SessionStatus enum, new event types
├── src/aether/services/reply_service.py — Adapt to work within SessionLoop
What it does:
class SessionLoop:
    async def run(self, session_id: str, abort: asyncio.Event):
        while not abort.is_set():
            # 1. Load messages from SessionStore
            messages = await self.session_store.get_messages(session_id)
            
            # 2. Check exit: last assistant message has no tool calls
            if self._should_exit(messages):
                break
            
            # 3. Check context overflow → compact
            if self._needs_compaction(messages):
                await self.compactor.compact(session_id)
                continue
            
            # 4. Build context + call LLM with tools (uses existing LLMCore)
            envelope = await self.context_builder.build_from_session(session_id)
            async for event in self.llm_core.generate_with_tools(envelope):
                await self.event_bus.publish(f"session.{session_id}.event", event)
                await self.session_store.save_part(session_id, event)
            
            # 5. Loop back
- The existing LLMCore.generate_with_tools() becomes the inner loop (unchanged)
- SessionLoop.run() is the outer loop that manages session-level concerns
- ReplyService gets a new path: instead of building context from scratch, it can build from a session's persisted messages
- Compaction: when token count exceeds threshold, use LLM to summarize old messages, replace them with a summary message
Lines: ~350 new, ~80 modified
---
Phase 4: Non-Blocking Sub-Agents
Why fourth: Depends on SessionLoop + EventBus + SessionStore.
NEW: src/aether/agents/
├── manager.py        — SubAgentManager (spawn, check, get_result)
└── agent_types.py    — Agent type definitions (general, explore, etc.)
MODIFIED:
├── src/aether/agents/task_runner.py   — Rewrite to use SessionLoop (non-blocking)
├── src/aether/tools/run_task.py       — spawn_task tool (fire-and-forget)
NEW: src/aether/tools/
├── check_task.py     — check_task tool (status + result retrieval)
What it does:
class SubAgentManager:
    async def spawn(self, prompt, agent_type, parent_session_id) -> str:
        """Create child session, start its loop as asyncio.Task, return session_id."""
        child_id = await self.session_store.create_session(
            parent_id=parent_session_id,
            agent_type=agent_type,
        )
        await self.session_store.add_message(child_id, role="user", content=prompt)
        
        task = asyncio.create_task(
            self.session_loop.run(child_id, abort=asyncio.Event())
        )
        self._tasks[child_id] = task
        
        # When done, publish event
        task.add_done_callback(lambda t: self._on_complete(child_id))
        return child_id  # Returns IMMEDIATELY
    
    async def get_status(self, session_id) -> dict:
        session = await self.session_store.get_session(session_id)
        return {"status": session.status, "agent_type": session.agent_type}
    
    async def get_result(self, session_id) -> str | None:
        if self._tasks[session_id].done():
            messages = await self.session_store.get_messages(session_id)
            return self._extract_final_text(messages)
        return None
- spawn_task tool: LLM calls this to delegate work. Returns immediately with task_id.
- check_task tool: LLM calls this to check status or get results.
- Agent types define: system prompt, allowed tools, model override, max iterations
- Sub-agents run on E-Core pool (background workers) — don't block interactive work
- On completion, publishes to EventBus → AgentCore picks up → notifies user
Lines: ~300 new, ~100 modified (task_runner.py rewrite)
---
Phase 5: Async Task Endpoints
Why fifth: Depends on SessionLoop + EventBus for streaming status.
NEW: src/aether/http/
├── sessions.py       — Session management endpoints
└── events.py         — SSE event stream endpoint
MODIFIED:
├── src/aether/main.py                — Mount new routers
├── src/aether/agent.py               — New methods for async prompt + session management
What it does:
POST /v1/sessions                     → Create a new session
POST /v1/sessions/{id}/prompt         → Start agent loop (fire-and-forget, returns 202)
GET  /v1/sessions/{id}/events         → SSE stream of all session events
GET  /v1/sessions/{id}/status         → Current session status
GET  /v1/sessions/{id}/messages       → All messages in session
POST /v1/sessions/{id}/cancel         → Cancel running session
GET  /v1/tasks                        → List all sub-agent tasks
GET  /v1/tasks/{id}/status            → Sub-agent status
- /v1/chat/completions continues to work as-is (synchronous, backward compatible)
- New endpoints enable clients to start long-running work and observe via SSE
- SSE stream publishes: text chunks, tool calls, tool results, status updates, completion
Lines: ~250 new, ~30 modified
---
Phase 6: Context Compaction
Why sixth: Needed for truly long-running tasks that exceed context window.
NEW: src/aether/session/compaction.py  (already stubbed in Phase 3)
MODIFIED:
├── src/aether/session/loop.py         — Wire compaction into the loop
├── src/aether/llm/context_builder.py  — Build context from compacted messages
What it does:
- When messages exceed ~80% of model's context window, trigger compaction
- Use LLM to summarize old messages into a compact summary
- Replace old messages with [compacted_summary] message
- Preserve recent messages (last N turns) + all pending tool results
- Similar to OpenCode's SessionCompaction and Claude Code's auto-compact
Lines: ~200 new, ~50 modified
---
Phase 7: Agent Type System
Why last: Polish layer — defines specialized agent behaviors.
NEW: src/aether/agents/
├── definitions/          — Agent definition files (like Claude Code's agents/*.md)
│   ├── general.py        — General-purpose sub-agent
│   ├── explorer.py       — Read-only codebase exploration
│   └── planner.py        — Planning agent (read-only, creates plans)
MODIFIED:
├── src/aether/agents/agent_types.py   — Load definitions, resolve by name
├── src/aether/llm/context_builder.py  — Support agent-specific system prompts
What it does:
- Each agent type defines: name, description, system_prompt, allowed_tools, model, max_iterations
- spawn_task tool accepts agent_type parameter
- Agent types restrict tool access (explorer can only read, planner can only plan)
- Prevents recursive spawning (sub-agents can't spawn sub-agents by default)
Lines: ~200 new, ~50 modified
---
Summary
PHASE 1: Persistent Message Store     ~350 lines   (foundation)
PHASE 2: Event Bus                    ~100 lines   (communication)
PHASE 3: Session Agent Loop           ~430 lines   (core capability)
PHASE 4: Non-Blocking Sub-Agents      ~400 lines   (delegation)
PHASE 5: Async Task Endpoints         ~280 lines   (API surface)
PHASE 6: Context Compaction           ~250 lines   (long-running support)
PHASE 7: Agent Type System            ~250 lines   (specialization)
                                     ─────────
                              TOTAL: ~2,060 lines
Files touched: ~15 modified, ~12 new
Existing code preserved: LLMCore, Scheduler, ToolRegistry, ContextBuilder, all tools, all plugins
What Stays Unchanged
- LLMCore.generate_with_tools() — your inner tool loop, untouched
- KernelScheduler — P-Core/E-Core pools, untouched (sub-agents use E-Core)
- ToolRegistry / ToolOrchestrator — tool execution, untouched
- ContextBuilder — extended but not rewritten
- All existing tools, plugins, skills — unchanged
- Voice pipeline (WebRTC, TTS, STT) — unchanged
- /v1/chat/completions — backward compatible, still works
After All Phases, Aether Can:
1. Quick interactions (Scenario 1): Same as today, no change needed
2. Background tasks (Scenario 2): LLM spawns sub-agent → acknowledges user → sub-agent runs independently → notifies when done
3. Long-running autonomous work (Scenario 3): Session loop runs indefinitely, compacts context, spawns sub-agents, tracks progress — same capability as Claude Code/OpenCode