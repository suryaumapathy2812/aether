# Aether — System Requirements

> This document defines what the system must do, how fast it must do it, how reliable it must be, and what the architecture must look like. It is intentionally technology-agnostic. Stack decisions are made separately, informed by these requirements.

---

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [System Overview](#2-system-overview)
3. [Performance Requirements](#3-performance-requirements)
4. [The Orchestrator](#4-the-orchestrator)
5. [The Agent](#5-the-agent)
6. [The Proactive Engine](#6-the-proactive-engine)
7. [The Memory System](#7-the-memory-system)
8. [The Plugin System](#8-the-plugin-system)
9. [The Skill System](#9-the-skill-system)
10. [The Device Protocol](#10-the-device-protocol)
11. [Observability and Tracing](#11-observability-and-tracing)
12. [Security Requirements](#12-security-requirements)
13. [Scalability Requirements](#13-scalability-requirements)
14. [Data Requirements](#14-data-requirements)
15. [Failure Modes and Recovery](#15-failure-modes-and-recovery)

---

## 1. Guiding Principles

These principles govern every architectural and implementation decision. When requirements conflict, these principles resolve the conflict.

### 1.1 Simplicity Over Cleverness
The system must be composed of simple components that do one thing well. Complex architectures introduce failure modes that are hard to debug, hard to scale, and hard to reason about. Every component must be explainable in one sentence.

### 1.2 Fast by Default, Not by Optimization
Performance must be designed in from the start, not added later. A system that is slow by design cannot be optimized into being fast. Every component must have a defined latency budget and must be built to meet it without heroic effort.

### 1.3 Failure Is Expected, Not Exceptional
Every component will fail. The system must be designed assuming failure at every layer. No single component failure should result in data loss or user-visible outage. Recovery must be automatic and fast.

### 1.4 The User Sees None of This
Every technical decision must be evaluated against the user experience. Infrastructure complexity must never surface to the user. The system must feel like magic — which means all complexity is hidden, all failures are handled invisibly, and all latency is below the threshold of perception.

### 1.5 Correctness Before Performance
A fast system that produces wrong results is worse than a slow system that produces correct results. Correctness — of data, of state, of behavior — is non-negotiable. Performance is optimized within the constraints of correctness.

### 1.6 Observable by Default
Every component must emit structured events that allow the system's behavior to be understood at any point in time. Debugging must be possible without reproducing the problem. Every job, every event, every decision must be traceable.

---

## 2. System Overview

The system consists of five logical layers. Each layer has a defined responsibility and communicates with adjacent layers through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: Device Layer                                          │
│  Any device implementing the Aether Device Protocol            │
│  iOS, Android, Web, Earphones, Smart Speakers, Wearables       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Aether Device Protocol
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 4: Agent Layer (one per user)                            │
│  ┌─────────────────────┐   ┌──────────────────────────────┐    │
│  │  P Worker            │   │  E Worker(s)                  │    │
│  │  (Conversational LLM)│──▶│  (Tool execution, long tasks) │    │
│  │  Always responsive   │◀──│  Async, never blocks P        │    │
│  └─────────────────────┘   └──────────────────────────────┘    │
│  Session state, memory interface, proactive notifications       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Internal API
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 3: Orchestrator Layer                                    │
│  Auth, routing, agent lifecycle, plugin management, billing     │
│  The coordination plane — never in the data path                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Storage APIs
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 2: Storage Layer                                         │
│  Per-user isolated storage: sessions, memory, workspace, traces │
│  Shared storage: accounts, node registry, plugin catalog        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 What Each Layer Owns

| Layer | Owns | Does Not Own |
|---|---|---|
| Device | Audio/text I/O, notification display | Intelligence, memory, state |
| Agent (P Worker) | Session state, realtime connection, conversation, memory interface | Tool execution, long-running tasks |
| Agent (E Worker) | Tool execution, API calls, file ops, code execution | Session state, user connections |
| Orchestrator | Auth, routing, agent lifecycle, billing | Conversation content, user data |
| Storage | Durability, consistency, query | Business logic |

### 2.2 The P Worker / E Worker Split

The agent is internally split into two roles that run within the same agent process:

**P Worker (Performance Worker)** — the conversational LLM. Its only job is to stay responsive to the user at all times. It must never be blocked by a tool call, a slow API, or a long-running task. When the user speaks or types, the P worker responds immediately. When a tool needs to run, the P worker delegates to the E worker and continues the conversation.

**E Worker (Execution Worker)** — the task executor. It picks up delegated work from the P worker, executes it (tool calls, API calls, file operations, code execution, browser automation), and notifies the P worker when done. The P worker then informs the user of the result.

This split is critical for voice. A voice agent that goes silent for 30 seconds while a tool runs is broken. The P worker says "I'm on it" and keeps the conversation alive while the E worker does the heavy lifting.

```
User: "Check my email and summarize the important ones"

P Worker: "Checking your email now — give me a moment."
  │
  ├──▶ E Worker: [fetch emails, classify, summarize]
  │                    (takes 15 seconds)
  │
  │    User: "Also, what's on my calendar today?"
  │
P Worker: "You have three meetings today — 10am standup, 1pm design review, 3pm 1:1 with Alex."
  │
  ◀── E Worker: [done — 4 important emails summarized]
  │
P Worker: "Your email summary is ready. You have 4 important emails..."
```

The P worker is never blocked. The user never waits.

#### 2.2.1 The Task Ledger (P↔E Communication)

The P worker and E worker communicate through a **Task Ledger** — an in-process, persistent data structure backed by SQLite. It is not a queue, not an event bus, and has no external dependencies. It lives inside the agent process.

The Task Ledger serves three purposes:
1. **Delegation**: P worker writes a task, E worker picks it up
2. **Traceability**: The LLM can query the ledger at any time to check task status ("is that email sent yet?")
3. **Resumability**: On agent restart, the ledger is intact — pending and running tasks are resumed, completed tasks are available for context

**The Task record:**

```
{
  "task_id":      "uuid",
  "type":         "tool_call | memory_extract | proactive_check | scheduled",
  "status":       "pending | running | complete | error",
  "payload":      { ... },          // input to the task
  "result":       { ... },          // output from the task (null until complete)
  "error":        "string | null",  // error message if status is error
  "submitted_at": "iso8601",
  "started_at":   "iso8601 | null",
  "completed_at": "iso8601 | null"
}
```

**Status transitions:**

```
pending → running → complete
                  → error
```

No other transitions are valid. A task cannot go backwards. A failed task can be retried by creating a new task.

**Read/write rules:**

| Actor | Can do |
|---|---|
| P Worker | Create new tasks (status: `pending`), read any task's status and result |
| E Worker | Pick up `pending` tasks, set to `running`, set to `complete` or `error` with result |
| LLM | Read the ledger via a tool (e.g., `check_tasks`) — used to answer "what's happening with X?" |

**Why SQLite, not an in-memory structure:**

- Agent restarts are expected (crashes, updates, host migrations). An in-memory ledger is lost on restart. SQLite survives.
- On restart, the E worker scans for tasks that were `running` when the agent died and re-queues them as `pending`.
- Completed tasks are retained indefinitely — the LLM can reference past work ("I sent that email 3 hours ago, here's the result").
- No pruning. The ledger is append-only with status updates. Storage is negligible (thousands of tasks = kilobytes).

**Why this is language-agnostic:**

The contract is the Task record schema and the status transition rules. Every language has SQLite bindings. Every language can read and write JSON. The concurrency primitive (mutex, lock, atomic) that protects concurrent access is a runtime detail — every language has one. If the agent is rewritten from Python to Go to Rust, the Task Ledger works identically because the contract is data, not code.

**Ordering and priority:**

The E worker processes tasks FIFO by default (`submitted_at` ascending). High-priority tasks (e.g., user-initiated tool calls) are picked up before low-priority tasks (e.g., background memory extraction). Priority is a field on the Task record, not a separate queue.

### 2.3 Data Flow Principles

- Conversation content never passes through the orchestrator
- API keys never leave the agent's runtime environment
- Memory is always written to persistent storage before being considered committed
- All events are append-only — nothing is deleted, only superseded
- The orchestrator sees metadata (agent online, session count, health) but never content
- Tool execution never blocks the conversational LLM — all tool calls are delegated to E workers via the Task Ledger
- The Task Ledger is the single communication channel between P and E workers — no direct function calls, no shared mutable state beyond the ledger

---

## 3. Performance Requirements

These are hard requirements, not targets. The system must meet these under normal operating conditions.

### 3.1 User-Facing Latency

| Interaction | Requirement | Notes |
|---|---|---|
| Sign up → agent ready | < 2 seconds | Full agent provisioned and ready to receive messages |
| First token of text response | < 800ms | From message received to first character streaming |
| First audio of voice response | < 500ms | From end of user speech to first audio byte |
| Notification delivery | < 3 seconds | From event trigger to device notification |
| Plugin connection (OAuth) | < 5 seconds | From user tap to agent having access |
| Memory retrieval | < 100ms | Relevant memories available before LLM call |
| Tool execution (simple) | < 2 seconds | File read, web search, simple command |
| Tool execution (complex) | < 30 seconds | Multi-step, code execution, browser automation |

### 3.2 System Throughput

| Metric | Requirement |
|---|---|
| Concurrent active sessions per node | 10,000+ |
| Messages processed per second (system-wide) | 100,000+ |
| Events processed per second (proactive engine) | 1,000,000+ |
| Plugin webhook events per second | 500,000+ |
| Memory writes per second | 50,000+ |
| Notification deliveries per second | 100,000+ |

### 3.3 Resource Efficiency

| Component | Idle RAM | Active RAM | Boot Time |
|---|---|---|---|
| Agent process (target) | < 20MB | < 60MB | < 200ms |
| Agent process (current — Python) | ~80MB | ~150MB | ~2 seconds |
| Orchestrator | < 200MB | < 500MB | < 2 seconds |

These requirements exist because the system must be economically viable at scale. An agent that costs 500MB of RAM per user cannot serve 10,000 users on reasonable hardware. An agent that costs 20MB can serve 400 users per 8GB of RAM.

> **Implementation note**: The target RAM figures (20MB idle / 60MB active) require a compiled runtime (Go or Rust). The current Python implementation exceeds these targets. A Go rewrite of the agent runtime is planned for a future phase. The architecture is designed to make this rewrite straightforward — the agent is a single process with well-defined interfaces to the orchestrator, storage, and device layers. The rewrite changes the runtime, not the architecture.

### 3.4 Availability

| Component | Required Uptime | Recovery Time |
|---|---|---|
| Orchestrator | 99.99% | < 30 seconds |
| Agent (per user) | 99.9% | < 5 seconds |
| Event Bus | 99.99% | < 10 seconds |
| Storage | 99.999% | < 60 seconds |
| Proactive Engine | 99.9% | < 60 seconds |

---

## 4. The Orchestrator

The orchestrator is the coordination plane. It knows where every agent is, who every user is, and how to route requests. It does not process conversation content.

### 4.1 Responsibilities

**Authentication and Authorization**
- User signup, signin, session management
- Device registration and token issuance
- API key management (encrypted at rest, never logged)
- Permission management for plugins and capabilities
- Multi-device session coordination

**Agent Lifecycle Management**
- Provision a new agent when a user signs up (< 2 seconds)
- Monitor agent health via heartbeat
- Restart failed agents automatically (< 5 seconds)
- Scale agent capacity based on active user count
- Drain and terminate idle agents after configurable timeout
- Maintain agent registry: agent ID → endpoint mapping

**Request Routing**
- Route incoming device connections to the correct agent
- Handle agent failover transparently (user does not notice)
- Load balance across agent instances for the same user (multi-device)
- Maintain reverse tunnel connections for self-hosted nodes (future)

**Plugin Management**
- Maintain plugin catalog (metadata, capabilities, OAuth config)
- Handle OAuth flows on behalf of agents
- Store plugin credentials (encrypted, per user)
- Deliver plugin configuration to agents at startup
- Receive and route plugin webhooks to the correct agent

**Billing and Entitlements**
- Track usage per user
- Enforce plan limits
- Manage feature flags per subscription tier
- Handle payment events

**System Configuration**
- Distribute configuration to agents (model preferences, feature flags)
- Manage update rollouts
- Maintain capability registry (what each agent version supports)

### 4.2 What the Orchestrator Must Not Do

- Process conversation content
- Store conversation history
- Make LLM calls on behalf of users
- Hold user memory or facts
- Log message content
- Be in the critical path of a conversation once it is established

The orchestrator establishes the connection and then steps aside. The agent and the device communicate directly (or via a thin tunnel that the orchestrator does not inspect).

### 4.3 Orchestrator API Surface

**Public API (authenticated by user session)**
```
POST   /auth/signup
POST   /auth/signin
POST   /auth/signout
GET    /auth/session

GET    /devices
POST   /devices/pair
DELETE /devices/{device_id}

GET    /plugins
POST   /plugins/{name}/install
POST   /plugins/{name}/connect     (initiates OAuth)
DELETE /plugins/{name}

GET    /preferences
PUT    /preferences

GET    /account
PUT    /account
DELETE /account                    (full data deletion)
```

**Agent API (authenticated by agent secret)**
```
POST   /internal/agents/register
POST   /internal/agents/{id}/heartbeat
POST   /internal/agents/{id}/keep_alive
GET    /internal/agents/{id}/config
GET    /internal/plugins?user_id=
GET    /internal/plugins/{name}/config?user_id=
GET    /internal/preferences?user_id=
POST   /internal/events/{id}/decision
```

**Webhook Receiver (authenticated by plugin secret)**
```
POST   /webhooks/{plugin_name}
```

### 4.4 Orchestrator Performance Requirements

| Operation | Latency |
|---|---|
| Auth token validation | < 5ms |
| Agent lookup by user ID | < 10ms |
| Plugin config fetch | < 20ms |
| Webhook ingestion | < 50ms |
| Agent provisioning | < 2000ms |
| OAuth flow initiation | < 200ms |

### 4.5 Orchestrator Scalability

The orchestrator must be horizontally scalable. Multiple orchestrator instances must be able to run simultaneously with no shared in-process state. All state lives in the database. Any orchestrator instance can handle any request.

---

## 5. The Agent

The agent is the core of the system. It is the intelligence that the user interacts with. One agent runs per user, always.

The agent is internally composed of two roles: the **P Worker** (conversational LLM — always responsive) and the **E Worker** (task executor — handles all tool calls and long-running work). Both run within the same agent process. The P worker is never blocked by the E worker.

### 5.1 P Worker Responsibilities (Conversational LLM)

The P worker's only job is to stay responsive to the user. It must respond to text within 800ms and to voice within 500ms, regardless of what else is happening.

**Session Management**
- Maintain the active conversation state for this user
- Handle multiple simultaneous device connections (phone + laptop + earphone)
- Persist session history to storage after each turn
- Resume sessions correctly after restart
- Manage session context window (compaction when approaching limits)

**Realtime Connection Management**
- Maintain a persistent connection to the realtime AI model (voice sessions)
- Handle connection drops and reconnects transparently
- Buffer audio/text during reconnection
- Route audio from device to model and back

**Memory Interface**
- Retrieve relevant memories (facts, memories, decisions) before each LLM call (< 100ms)
- Trigger memory extraction after each conversation turn (async — does not block response)
- Cache frequently accessed memories in-process

**Task Delegation**
- When the LLM requests a tool call, the P worker delegates it to the E worker immediately
- The P worker acknowledges the delegation to the user ("I'm checking that now")
- The P worker continues accepting user input while the E worker executes
- When the E worker completes, the P worker receives the result and informs the user

**Proactive Notification Delivery**
- Receive notifications from the proactive engine
- Generate the notification message (one LLM call)
- Deliver to the user's active device(s)
- Handle user response to the notification

**Event Emission**
- Emit a structured trace event for every significant action
- Events are append-only and written asynchronously (never block the main path)
- Events include: message received, tool delegated, tool result received, LLM response, memory retrieved, notification sent

### 5.1.1 E Worker Responsibilities (Task Executor)

The E worker executes all tool calls and long-running tasks. It is the only component that performs side effects (API calls, file operations, code execution, browser automation). It never interacts with the user directly.

**Tool Execution**
- Pick up delegated tool calls from the P worker
- Execute tools in response to model requests
- Enforce tool policy (what tools are allowed, in what context)
- Sandbox tool execution (no tool can affect other users' data)
- Return results to the P worker when complete
- Handle tool failures gracefully (retry, fallback, report failure to P worker)

**Execution Pipeline**
- Multiple E worker tasks can run concurrently (e.g., fetching email while searching the web)
- Each task has a timeout based on its category (see §5.4)
- If a task exceeds its timeout, it is cancelled and the P worker is informed
- The E worker emits trace events for every tool call (start, complete, fail)

**Memory Extraction**
- After each conversation turn, the E worker extracts facts, memories, and decisions from the conversation
- Extraction is async — the P worker fires and forgets
- Extracted items are written to the memory store (see §7)

### 5.2 Agent Lifecycle

```
Provisioning (< 2 seconds):
  1. Orchestrator creates agent record in database
  2. Agent process starts (container or process)
  3. Agent fetches config from orchestrator (preferences, plugin configs)
  4. Agent registers with orchestrator (endpoint, capabilities)
  5. Agent is ready to receive connections

Active (user is using the agent):
  - Realtime connection open
  - Memory warm in cache
  - Tools available
  - Responding in < 800ms

Idle (user not active, agent still running):
  - Realtime connection closed
  - Memory cache cleared
  - Proactive engine still running (checking for events)
  - RAM: < 20MB
  - CPU: near zero

Restart (after crash or update):
  - Agent restarts in < 5 seconds
  - Loads last session state from storage
  - Re-registers with orchestrator
  - User does not notice (if restart < 5 seconds)

Termination (user deleted account or long-term inactivity):
  - Graceful shutdown: flush all pending writes
  - Deregister from orchestrator
  - Archive session data per retention policy
```

### 5.3 Session Management

**Context Window Management**

Every conversation has a context window limit. The agent must manage this automatically:

- Track token count of the current session
- When approaching the limit (at 80% capacity), trigger compaction
- Compaction: summarize older turns, preserve decisions, commitments, and open questions
- The user never sees "context limit reached" — the agent handles it silently

**Multi-Device Sessions**

A user may have multiple devices connected simultaneously. The agent must:

- Maintain one canonical session state
- Broadcast responses to all connected devices
- Handle input from any device (last-write-wins for concurrent input)
- Sync notification state across devices (dismiss on one, dismiss on all)

**Session Persistence**

Every turn is persisted before the response is sent. If the agent crashes mid-response, the user can reconnect and the session resumes from the last committed state. No conversation is ever lost.

### 5.4 Tool Execution Requirements

**Tool Policy**

Tools are gated by a policy engine. The policy engine evaluates:
- Is this tool allowed for this user's plan?
- Is this tool allowed in this context (voice vs text, foreground vs background)?
- Has the user granted permission for this tool's capabilities?
- Is this tool's execution within the user's configured limits?

**Tool Sandboxing**

All tool execution is sandboxed:
- File operations are confined to the user's workspace volume
- Network calls are logged and rate-limited
- Command execution is confined to an allowlist of safe operations
- No tool can read or write another user's data

**Tool Execution Latency Budget**

| Tool Category | Latency Budget |
|---|---|
| Memory read | < 100ms |
| File read/write | < 200ms |
| Web search | < 3 seconds |
| Web fetch (single page) | < 5 seconds |
| Code execution | < 30 seconds |
| External API call | < 10 seconds |
| Browser automation | < 60 seconds |

If a tool exceeds its budget, it is cancelled and the model is informed.

### 5.5 Agent Configuration

At startup, the agent fetches its configuration from the orchestrator. Configuration includes:

- User preferences (model selection, voice, language, personality)
- Plugin credentials (OAuth tokens, API keys — encrypted in transit)
- Feature flags (what capabilities are enabled for this user's plan)
- Tool policy (what tools are allowed)
- Memory configuration (retrieval limits, similarity thresholds)

Configuration is refreshed on a configurable interval (default: 5 minutes) and on explicit reload signal from the orchestrator.

---

## 6. The Proactive Engine

The proactive engine is what makes Aether different from every other AI assistant. It runs continuously, per user, watching for events and deciding when to reach out.

**The proactive engine does not have its own decision model.** It uses the LLM reading from the three memory buckets (facts, memories, decisions) to make every proactive decision. See §7.2 for how this works.

### 6.1 Architecture

The proactive engine is a pipeline within the agent:

```
Event Sources → Agent (P Worker) → Memory Retrieval → LLM Decision → Notification → Device
```

When an event arrives, the P worker retrieves relevant facts, memories, and decisions from the memory store, then asks the LLM: "Given what I know about this user, should I surface this, and how?" The LLM's answer is informed by the accumulated decisions bucket — which grows with every interaction.

### 6.2 Event Sources

**Time-based (Cron)**
- Scheduled checks run on a configurable interval per user
- Morning briefing (daily, at user's configured wake time)
- Commitment tracking (daily, checks for unresolved commitments)
- Goal tracking (configurable, checks for stalled goals)
- Relationship tracking (weekly, checks for lapsed connections)

**Plugin Webhooks**
- Gmail: new email, email reply, email flagged
- Calendar: event created, event modified, event starting soon
- Linear/Notion: task assigned, task updated, deadline approaching
- GitHub: PR review requested, CI failed, issue assigned
- Any plugin can register webhook event types

**Memory Pattern Analysis (Nightly)**

The agent is always running — it does not shut down when the user is inactive. The agent must be available to receive plugin webhooks (new emails, calendar changes), process scheduled events, and deliver queued notifications at any time. Because the agent is always running, nightly analysis is simply a periodic E worker task on a timer — not a separate scheduler or external cron job.

Nightly analysis is a batch LLM call that scans accumulated memories and extracts higher-order patterns that are invisible in any single conversation. It performs four functions:

1. **Decision extraction from patterns**: Look across recent memories and notice behavioral patterns that span multiple interactions. "User has dismissed 4 evening calendar notifications this week" → write a **decision**: "Don't surface calendar notifications after 9pm." This is the learning loop (§6.4) running in batch mode — catching patterns that real-time extraction missed because they only emerge over days or weeks.

2. **Proactive opportunity detection**: Cross-reference facts and memories to identify time-sensitive opportunities. "User's mom's birthday is in 3 days" (fact) + "User bought flowers last year for mom's birthday" (memory) → write a candidate notification to the queue for tomorrow morning. "User has a flight at 6am Friday" (fact) + "User usually wakes at 8am" (memory) → schedule an interrupt alarm.

3. **Fact consolidation and decay**: Merge duplicate or overlapping facts into richer single entries. Flag stale facts that may no longer be true ("User is in Tokyo" was 3 weeks ago — mark as uncertain). Identify contradictory facts and resolve them (newer supersedes older, or flag for user confirmation).

4. **Memory health**: Track memory store metrics — total items per bucket, storage usage, extraction success rate, decision coverage (what percentage of recurring events have a learned decision). This is operational, not user-facing.

Nightly analysis must complete within 60 seconds per user (see §6.6). It writes its outputs (new decisions, candidate notifications, updated facts) through the same Task Ledger as any other E worker task — fully traceable and resumable.

**Agent-Generated Events**
- During a conversation, the agent can schedule future events
- "Remind me about this in three days" → creates a scheduled event
- Task completion events trigger follow-up checks

### 6.3 Notification Types

Every proactive notification has a type that determines how it is delivered. The LLM chooses the type based on the content's importance and the user's decisions bucket.

| Type | Meaning | Delivery |
|---|---|---|
| `suppress` | Not worth surfacing | Discard — do not notify |
| `queue` | Worth surfacing, but not now | Schedule for a better moment |
| `nudge` | Low-priority, non-interrupting | Soft notification (badge, ambient sound) |
| `surface` | Normal priority | Standard push notification |
| `interrupt` | Urgent, time-sensitive | Break through focus/DND |

The LLM decides the type by reading the decisions bucket. Examples:

- Decision: "Don't notify about calendar changes after 9pm" → calendar change at 10pm → `suppress` or `queue` for morning
- Decision: "User always responds to emails from John within minutes" → new email from John → `surface`
- Decision: "User prefers morning briefings to be brief" → morning briefing → `nudge` with a short summary
- No relevant decision exists (new user) → fall back to sensible defaults based on event urgency

### 6.4 The Learning Loop

The proactive engine improves because the memory system improves:

1. Agent sends a notification (type: `surface`)
2. User dismisses it without reading
3. This interaction is recorded as a **memory**: "User dismissed calendar notification at 9:47pm"
4. After several similar memories accumulate, the E worker (or the LLM during conversation) notices the pattern
5. A **decision** is written: "Don't surface calendar notifications after 9pm"
6. Next time a calendar event arrives at 9:30pm, the LLM reads this decision and chooses `queue` instead

The loop is: **event → notification → user reaction → memory → pattern → decision → better notification**. This is the compounding advantage. A new user gets generic notifications. A six-month user gets precisely timed, precisely targeted notifications.

### 6.5 Notification Queue

Notifications that are queued (not immediately delivered) are stored in a durable queue:

- Persisted to storage (survives agent restart)
- Indexed by `deliver_at` timestamp
- Swept every minute by the P worker
- Delivered when: `deliver_at` has passed AND user context is appropriate
- Expired if not delivered within a configurable window (default: 4 hours)

### 6.6 Proactive Engine Performance Requirements

| Operation | Requirement |
|---|---|
| Event ingestion (webhook → agent) | < 100ms |
| Decision classification (LLM + memory retrieval) | < 500ms |
| Notification delivery (agent → device) | < 3 seconds |
| Morning briefing generation | < 10 seconds |
| Nightly memory analysis (per user) | < 60 seconds |

### 6.7 Proactive Engine Reliability

- Event loss is not acceptable. Every webhook event must be acknowledged only after it is durably stored.
- Decision failures must not drop events. If the LLM call fails, events are retried with backoff.
- Notification delivery must be at-least-once. Duplicate notifications are acceptable; missed notifications are not.

---

## 7. The Memory System

Memory is what makes the agent feel like it knows the user. It must be fast, accurate, and invisible.

**The memory system is also the decision engine.** There is no separate ML model or classifier for proactive decisions. The LLM reads from the memory buckets and reasons over accumulated knowledge to make decisions. The more the user interacts, the better the decisions become.

### 7.1 The Three Memory Buckets

Every interaction with the platform is scanned for three types of information. New entries are created in the database after every message/response exchange. This is the core extraction loop — it runs on every turn, always.

#### Facts

Objective, stable information about the user and their world. Facts are things that are true and unlikely to change frequently.

```
"User's name is Surya"
"User prefers Python for scripting"
"User has a standup every Monday at 10am"
"User is learning Spanish"
"User's sister's name is Priya"
"User works at Acme Corp as a senior engineer"
"User's timezone is Asia/Kolkata"
```

Facts are extracted from conversations by the E worker after each turn. They are stored with embeddings for semantic search. Facts are versioned — newer facts supersede older ones on the same topic ("User moved from Bangalore to London" supersedes "User lives in Bangalore").

#### Memories

Contextual, episodic information — what happened, what was discussed, what the user was feeling or working on. Memories are time-bound and situational.

```
"Last week user was stressed about the Q3 deadline"
"User had a productive brainstorming session about the new feature on Tuesday"
"User mentioned wanting to call their sister but kept forgetting"
"User was excited about the new apartment they found"
"User dismissed 3 calendar notifications in a row on Thursday evening"
"User always responds to emails from John within minutes"
```

Memories capture the narrative of the user's life. They are what make the agent feel like it was "there" — it remembers not just facts, but experiences. Memories are also where behavioral patterns are recorded: what the user engaged with, what they dismissed, when they were active, what topics they care about.

#### Decisions

Learned rules about how the agent should behave for this specific user. Decisions are the output of the agent observing patterns in facts and memories and codifying them into actionable guidance.

```
"Don't notify about calendar changes after 9pm"
"User prefers bullet points over paragraphs"
"Surface GitHub PR reviews in the morning, not evening"
"When user says 'handle it', they mean send the email without asking for confirmation"
"User likes a brief morning summary, not a detailed one"
"Don't suggest Spanish practice on weekdays — user only has time on weekends"
"User prefers voice responses to be concise (< 30 seconds)"
```

**What triggers a write to decisions:**
- The LLM itself noticing a behavioral pattern ("user has dismissed evening notifications 5 times → record a decision")
- Explicit user feedback ("stop notifying me about this" → decision recorded)
- Both. The agent is always watching for patterns and the user can always override.

Decisions are the most powerful bucket. They are what make the agent feel personalized. A new user has zero decisions. A user after six months has hundreds. This is the compounding advantage — the switching cost — described in the product vision.

### 7.2 How Memory Powers Decisions (No Separate Decision Engine)

The proactive engine (§6) does not use a separate ML model or classifier. It uses the LLM reading from all three memory buckets.

When an event arrives (a webhook, a cron trigger, a calendar change), the agent:

1. Retrieves relevant **facts** (who is involved, what is this about)
2. Retrieves relevant **memories** (what happened last time, how did the user react)
3. Retrieves relevant **decisions** (what has the agent learned about how to handle this)
4. The LLM reasons over all three and decides: suppress, queue, nudge, surface, or interrupt

This is why the memory system IS the decision engine. Better memory → better decisions. More interactions → more memory → more decisions → better agent. The flywheel is built into the data model.

### 7.3 Conversational Memory (Session Log)

In addition to the three buckets, raw conversation history is stored as an append-only log. Every turn, every message, every tool call and result. This is used for:

- Session continuity (resume after disconnect)
- Context reconstruction (rebuild the conversation window after restart)
- Memory extraction source (the E worker reads this to extract facts, memories, and decisions)
- Audit trail (what was said, when)

Session summaries are generated when a session ends — what was discussed, what was decided, what was left open. These summaries are stored as memories.

### 7.4 Memory Retrieval Requirements

Before every LLM call, the agent retrieves relevant items from all three buckets. This must complete in < 100ms.

Retrieval strategy:
1. Semantic search across **facts** (embedding similarity to current query)
2. Recent **memories** (last N sessions + semantically relevant episodes)
3. Relevant **decisions** (what rules apply to this context)
4. Current context (time of day, active device, calendar state)

The retrieval must return the most relevant items, not all items. The context window is finite. Quality of retrieval matters more than quantity. The LLM should receive 10 highly relevant items, not 100 vaguely related ones.

### 7.5 Memory Extraction Requirements

After each conversation turn, the E worker extracts facts, memories, and decisions. Requirements:

- Extraction must not block the conversation (async — P worker fires and forgets)
- Extraction must complete within 30 seconds of the turn ending
- Extracted items must be deduplicated (no duplicate facts/memories/decisions)
- Facts and decisions must be versioned (newer entries supersede older ones on the same topic)
- Extraction failures must be retried (at least 3 attempts before giving up)
- Every user message and every agent response is scanned — no interaction is skipped

### 7.6 Memory Storage Requirements

- Per-user isolation: one user's memory cannot be accessed by another user's agent
- Semantic search: all three buckets must support vector similarity search
- Full-text search: all three buckets must support keyword search as a fallback
- Temporal ordering: items must be retrievable in chronological order
- Type filtering: queries must be able to target a specific bucket (facts only, decisions only, etc.)
- Soft deletion: items can be marked deleted by the user but are retained for 30 days before hard deletion
- Export: all memory must be exportable in a standard format (JSON)

### 7.7 Memory Privacy Requirements

- Memory is never used to train models without explicit user consent
- Memory is never shared between users
- Memory is never accessible to the orchestrator (only to the agent)
- Users can view, edit, and delete any memory item at any time
- Full memory deletion must complete within 24 hours of request

---

## 8. The Plugin System

Plugins extend the agent's capabilities by connecting to external services. The plugin system must be secure, reliable, and easy to build on.

### 8.1 Plugin Types

**OAuth Plugins**
Connect to services using OAuth 2.0. Examples: Gmail, Google Calendar, GitHub, Spotify, Notion. The orchestrator handles the OAuth flow. The resulting token is stored encrypted and delivered to the agent.

**API Key Plugins**
Connect to services using API keys. Examples: custom APIs, internal tools. The user provides the key, it is stored encrypted, delivered to the agent.

**Webhook Plugins**
Receive events from external services. Examples: Gmail push notifications, GitHub webhooks, Linear webhooks. The orchestrator receives the webhook and routes it to the correct agent via the event bus.

**Capability Plugins**
Add new tools to the agent. Examples: browser automation, image generation, code execution environments. These are code that runs in the agent's tool executor.

### 8.2 Plugin Security Requirements

- Plugin credentials are encrypted at rest using a key derived from the user's account secret
- Plugin credentials are never logged
- Plugin credentials are transmitted to the agent over an encrypted channel
- Plugins can only access the user's data with explicit user permission
- Plugin webhook endpoints are authenticated (each plugin has a unique secret)
- Plugins cannot access other users' data or credentials
- Plugin capability code runs in a sandbox (cannot affect the agent's core state)

### 8.3 Plugin Reliability Requirements

- Plugin failures must not crash the agent
- Plugin failures must be reported to the user in a useful way ("Gmail is not responding — I'll try again in a few minutes")
- Plugin credential expiry must be detected and the user prompted to reconnect
- Plugin webhooks that fail to deliver must be retried with exponential backoff

### 8.4 Plugin Developer Requirements

Third-party developers must be able to build plugins. The plugin SDK must:

- Be documented clearly enough that a developer can build a working plugin in one day
- Provide a local development environment (no need to deploy to test)
- Provide a testing framework (mock the agent, test plugin behavior)
- Define a clear security review process for marketplace listing
- Support versioning (plugins can be updated without breaking existing users)

---

## 9. The Skill System

Skills teach the agent **how** to do something. A plugin connects the agent to an external service (Gmail, Spotify, Calendar). A skill teaches the agent a methodology, a workflow, or domain expertise. Plugins provide capabilities. Skills provide knowledge.

Examples of the distinction:

| Plugin | Skill |
|---|---|
| Gmail plugin: can read/send email | Email triage skill: how to prioritize emails, what to summarize, when to flag |
| Google Calendar plugin: can read/write events | Scheduling skill: how to find optimal meeting times, how to handle conflicts |
| Brave Search plugin: can search the web | Research skill: how to decompose a question, cross-reference sources, synthesize findings |
| — (no plugin needed) | Soul skill: the agent's personality, tone, communication style |
| — (no plugin needed) | Tool-calling skill: when to use tools vs. answer from memory, how to chain tool calls |

A plugin without a skill is a raw capability. A skill without a plugin is pure knowledge. The most powerful combinations pair both — the Gmail plugin provides access, the email triage skill provides judgment.

### 9.1 Skill Architecture

Skills are Markdown files (`SKILL.md`) that contain structured instructions for the LLM. They are not code — they are prompts. A skill file tells the agent what to do, when to do it, and how to reason about a domain.

Skills are **never auto-injected** into the LLM context. The agent's base prompt is kept lean (~50 lines). Skills are loaded on demand — the LLM sees a list of available skill names and fetches the ones it needs for the current task. This keeps the context window clean and avoids wasting tokens on irrelevant instructions.

**Three skill directories:**

| Directory | Purpose | Managed by | Tracked in git |
|---|---|---|---|
| `app/skills/` | Built-in skills shipped with the agent | Developers | Yes |
| `app/.skills/` | User-installed skills (marketplace or custom) | User / LLM at runtime | No (gitignored) |
| `app/plugins/{plugin}/SKILL.md` | Plugin-specific skills | Plugin developers | Yes (with plugin) |

Built-in skills define the agent's core behavior (personality, tool-calling strategy, file management). Plugin skills teach the agent how to use a specific plugin effectively. User-installed skills extend the agent's knowledge in any direction — the user (or the LLM itself) can install skills from the marketplace or create custom ones at runtime.

### 9.2 Skill Loading (Progressive)

The Skill Loader discovers and indexes all skills at agent startup. It does not load skill content into memory — it builds an index of skill names, descriptions, and keywords.

**Startup sequence:**
1. Scan all three skill directories
2. Parse each `SKILL.md` for metadata (name, description, keywords)
3. Build an in-memory index: skill name → file path + metadata
4. Register the index with the LLM context builder (names only, not content)

**Runtime loading (on demand):**
1. LLM sees the list of available skill names in its system prompt
2. LLM decides it needs a skill for the current task (e.g., "I need the email-triage skill to handle this inbox")
3. LLM calls `search_skill` (keyword match) or `read_skill` (by name) to fetch the skill content
4. Skill content is injected into the current conversation context
5. LLM applies the skill's instructions to the task

This is progressive loading — skills are discovered at startup but loaded at runtime, only when needed. A user with 50 installed skills pays zero context cost until the LLM actually needs one.

**Live registration:**

Skills created or installed during a session are immediately registered in the index — no restart required. The LLM can create a skill, install it, and use it in the same conversation.

### 9.3 Skill Tools

The agent has five tools for managing skills:

| Tool | Purpose |
|---|---|
| `search_skill` | Search installed skills by keyword. Returns matching skill names and descriptions. |
| `read_skill` | Read a skill's full content by name. Returns the SKILL.md content for injection into context. |
| `create_skill` | Create a new skill at runtime. Writes a SKILL.md to `app/.skills/` and registers it live. |
| `install_skill` | Install a skill from GitHub. Fetches SKILL.md from a repository and writes to `app/.skills/`. |
| `search_marketplace` | Search the skills marketplace (skills.sh) for available skills by keyword. |

**Skill addressing (marketplace):**

Skills from GitHub are addressed using the shorthand `owner/repo@skill-name`, which resolves to:
```
https://raw.githubusercontent.com/{owner}/{repo}/main/{skill-name}/SKILL.md
```

This mirrors the skills.sh marketplace convention. The agent can discover skills via `search_marketplace`, then install them via `install_skill` with the shorthand address.

**Skill name sanitization:**

Skill names are sanitized on create/install: lowercased, non-alphanumeric characters replaced with hyphens, leading/trailing dots and hyphens stripped, maximum 255 characters. This ensures consistent addressing across platforms.

### 9.4 Skill vs Plugin Lifecycle

| Aspect | Plugin | Skill |
|---|---|---|
| Installation | Via orchestrator (OAuth flow, credential storage) | Via agent tools (file write, no credentials) |
| Authentication | OAuth tokens, API keys (encrypted, managed) | None — skills are plain text |
| Runtime cost | Network calls, API rate limits, webhook processing | Zero until loaded — then just context tokens |
| Failure mode | External service down → graceful degradation | Skill file missing → fallback to base behavior |
| Who creates them | Developers (plugin SDK, security review) | Anyone — developers, users, or the LLM itself |
| Update mechanism | Plugin catalog via orchestrator | Re-install from marketplace or edit file directly |

### 9.5 Skill System Requirements

**Discovery and Loading**
- Skill index must be built at startup in < 1 second (for up to 100 skills)
- Skill search must return results in < 50ms (keyword matching against index)
- Skill content must be readable in < 100ms (file read from disk)
- Live registration of new skills must complete in < 200ms

**Creation and Installation**
- Skill creation must validate the SKILL.md structure before writing
- Skill installation must validate the source URL and fetch content securely (HTTPS only)
- Installed skills must be immediately available without agent restart
- Skill names must be unique within each directory — installing a skill with an existing name overwrites it

**Context Management**
- Skills must never be auto-injected into the LLM context
- The system prompt includes only skill names and one-line descriptions (not content)
- Full skill content is loaded only when the LLM explicitly requests it via `read_skill`
- Multiple skills can be active in a single conversation (the LLM manages its own context budget)

**Reliability**
- Missing or corrupted skill files must not crash the agent — they are skipped during indexing with a warning
- Marketplace fetch failures must be reported to the user, not silently swallowed
- Skills from untrusted sources are plain text (Markdown) — they cannot execute code, access the filesystem, or modify agent behavior outside of the LLM's reasoning

---

## 10. The Device Protocol

The Aether Device Protocol defines how any device connects to a user's agent. It is intentionally minimal.

### 10.1 Protocol Requirements

**Authentication**
- Device registers with the user's Aether account via the orchestrator
- Device receives a scoped token (bound to this device, this user)
- All subsequent communication is authenticated with this token
- Tokens expire and must be refreshed (configurable, default: 30 days)
- Token revocation must propagate to the device within 60 seconds

**Connection**
- Devices connect via WebSocket (persistent, bidirectional)
- Connection must survive network interruptions (automatic reconnect with backoff)
- Connection must work on constrained networks (2G, high latency)
- Maximum message size: 1MB (larger payloads use chunked transfer)

**Audio (Voice Devices)**
- Device streams raw audio to the agent (PCM, 16kHz, 16-bit, mono)
- Agent streams audio back to the device (same format or device-specified format)
- Audio streaming must have < 200ms end-to-end latency on a good network
- Audio must be resilient to packet loss (graceful degradation, not silence)

**Text (Screen Devices)**
- Device sends text messages as UTF-8 JSON
- Agent streams text response as chunks (server-sent events style over WebSocket)
- First chunk must arrive within 800ms of message sent

**Notifications**
- Agent pushes notifications to all connected devices for this user
- Notification payload: type, title, body, actions, priority, expiry
- Device acknowledges receipt
- Device reports user action (engaged, dismissed, snoozed)
- Acknowledgement and action reports are used by the proactive engine to learn

**State Reporting**
- Device reports its state: active, idle, do-not-disturb, background
- State is used by the proactive engine for notification timing
- State updates must be delivered to the agent within 1 second

### 10.2 Protocol Versioning

The protocol is versioned. Devices declare their supported protocol version on connection. The agent negotiates the highest mutually supported version. Old devices continue to work with older protocol versions until the version is deprecated (minimum 12 months notice).

### 10.3 Reference Implementations

Aether provides reference implementations of the device protocol for:
- iOS (Swift)
- Android (Kotlin)
- Web (TypeScript)
- Embedded Linux (C, for hardware manufacturers)

Third parties may implement the protocol in any language. The protocol specification is public.

---

## 11. Observability and Tracing

Every action in the system must be traceable. This is not optional — it is a core requirement.

### 11.1 What Must Be Traced

Every significant event in the system emits a structured trace event:

**Session Events**
- Session started (device, user, timestamp)
- Message received (session, timestamp, message length — not content)
- LLM call started (session, model, token count)
- LLM call completed (session, model, tokens used, latency)
- Tool called (session, tool name, timestamp)
- Tool completed (session, tool name, duration, success/failure)
- Session ended (session, duration, turn count)

**Agent Events**
- Agent started (agent ID, version, timestamp)
- Agent registered (agent ID, orchestrator response)
- Config loaded (agent ID, config version)
- Memory retrieved (agent ID, query, result count, latency)
- Notification received (agent ID, notification type)
- Notification delivered (agent ID, device, latency)

**Worker Events**
- Job received (job ID, type, user ID)
- Job started (job ID, worker ID, timestamp)
- Job completed (job ID, duration, success/failure)
- Job failed (job ID, error, retry count)

**Orchestrator Events**
- Auth event (type: signup/signin/signout, success/failure — no credentials)
- Agent provisioned (agent ID, duration)
- Agent restarted (agent ID, reason)
- Plugin connected (plugin name, user ID — no credentials)
- Webhook received (plugin name, event type, routing latency)

### 11.2 Trace Storage Requirements

- Traces are append-only (never modified after write)
- Traces are queryable by: user ID, session ID, agent ID, time range, event type
- Traces are retained for 90 days by default (configurable per plan)
- Traces must be writable at < 1ms overhead (never block the main execution path)
- Traces must be readable within 1 second for any query covering < 24 hours of data

### 11.3 What Must Not Be Traced

- Message content (what the user said)
- LLM response content (what the agent said)
- Tool inputs and outputs (what data was processed)
- Memory content (what facts are stored)
- API keys or credentials of any kind

Traces contain metadata about what happened, not the content of what happened. This is both a privacy requirement and a security requirement.

### 11.4 Operational Dashboards

The system must provide operational dashboards showing:

- Active agents (count, by region)
- Active sessions (count, by device type)
- System latency (p50, p95, p99 for all key operations)
- Error rates (by component, by error type)
- Queue depths (event bus, job queue, notification queue)
- Worker utilization (P workers, E workers)
- Storage utilization (per component)

Dashboards must update in near-real-time (< 30 second lag).

---

## 12. Security Requirements

### 12.1 Authentication

- All user-facing endpoints require authentication
- All agent-facing endpoints require agent authentication (separate from user auth)
- All device connections require device token authentication
- All webhook endpoints require plugin secret authentication
- Authentication tokens must be short-lived (user sessions: 30 days, device tokens: 30 days, agent tokens: 24 hours)
- Token refresh must be automatic and transparent

### 12.2 Authorization

- Users can only access their own data
- Agents can only access their assigned user's data
- Devices can only access the user's data they are registered to
- Plugins can only access the data the user has explicitly granted
- No cross-user data access is possible at any layer

### 12.3 Encryption

- All data in transit is encrypted (TLS 1.3 minimum)
- All data at rest is encrypted (AES-256 minimum)
- API keys and OAuth tokens are encrypted with a key derived from the user's account secret
- Encryption keys are rotated on a configurable schedule
- Key material is never logged

### 12.4 Input Validation

- All inputs are validated before processing
- Tool inputs are sanitized before execution
- Command execution uses explicit argument lists (never shell interpolation)
- File paths are validated against the user's workspace boundary (no path traversal)
- Webhook payloads are validated against the plugin's declared schema

### 12.5 Rate Limiting

- All public endpoints are rate-limited per IP and per user
- LLM calls are rate-limited per user (configurable per plan)
- Tool execution is rate-limited per user
- Webhook ingestion is rate-limited per plugin
- Rate limit violations return 429 with a Retry-After header

### 12.6 Audit Logging

- All authentication events are logged (success and failure)
- All authorization failures are logged
- All admin actions are logged
- All data deletion requests are logged
- Audit logs are immutable and retained for 1 year

---

## 13. Scalability Requirements

### 13.1 User Scale

The system must scale from 1 user to 10,000,000 users without architectural changes. Scaling must be achieved by adding capacity, not by redesigning the system.

| Scale | Agents | Active Sessions | Events/sec |
|---|---|---|---|
| 1,000 users | 1,000 | 100 | 10,000 |
| 100,000 users | 100,000 | 10,000 | 1,000,000 |
| 10,000,000 users | 10,000,000 | 1,000,000 | 100,000,000 |

### 13.2 Horizontal Scaling

Every component must be horizontally scalable:

- **Orchestrator**: stateless, multiple instances behind a load balancer
- **Agents**: one per user, distributed across agent hosts
- **P/E Workers**: stateless, scale by adding workers
- **Event Bus**: partitioned by user ID, scale by adding partitions
- **Storage**: sharded by user ID, scale by adding shards

### 13.3 Agent Density

The system must achieve the following agent density targets:

| Hardware | Idle Agents | Active Agents |
|---|---|---|
| 1GB RAM | 40 | 15 |
| 8GB RAM | 400 | 150 |
| 32GB RAM | 1,600 | 600 |
| 128GB RAM | 6,400 | 2,400 |

These targets require the agent to use < 20MB RAM idle and < 60MB RAM active.

### 13.4 Geographic Distribution

The system must support deployment in multiple geographic regions:

- Agents are deployed in the region closest to the user
- The orchestrator is globally distributed with regional failover
- Storage is replicated across regions with configurable consistency
- The event bus is regional (events do not cross regions unnecessarily)

---

## 14. Data Requirements

### 14.1 Data Isolation

Every user's data is isolated from every other user's data. Isolation must be enforced at the storage layer, not just the application layer. A bug in the application must not be able to expose one user's data to another.

### 14.2 Data Retention

| Data Type | Default Retention | User Configurable |
|---|---|---|
| Conversation history | Indefinite | Yes (min: 30 days) |
| Factual memory | Indefinite | Yes (min: 30 days) |
| Session summaries | Indefinite | Yes (min: 30 days) |
| Action logs | 1 year | No |
| Trace events | 90 days | No |
| Audit logs | 1 year | No |
| Deleted data | 30 days (soft delete) | No |

### 14.3 Data Export

Users must be able to export all of their data at any time:

- Export format: JSON (human-readable and machine-readable)
- Export must include: all conversations, all memories, all session summaries, all action logs
- Export must complete within 24 hours of request
- Export must be delivered securely (encrypted download link, expires in 7 days)

### 14.4 Data Deletion

Users must be able to delete all of their data:

- Deletion request triggers immediate soft delete (data inaccessible within 1 minute)
- Hard deletion completes within 30 days
- Deletion is irreversible after the soft-delete window
- Deletion confirmation is sent to the user's email
- Audit log of the deletion is retained for 1 year (contains no user data, only metadata)

### 14.5 Data Portability

The data export format must be documented and stable. A user must be able to take their exported data and import it into a future version of Aether, or into a compatible system. The format must not change in a breaking way without a migration path.

---

## 15. Failure Modes and Recovery

### 15.1 Agent Crash

**What happens**: The agent process crashes unexpectedly.

**Recovery**:
1. The orchestrator detects the crash via missed heartbeat (within 30 seconds)
2. The orchestrator restarts the agent (within 5 seconds of detection)
3. The agent loads its last committed session state from storage
4. The agent re-registers with the orchestrator
5. Active device connections are re-established (devices reconnect automatically)

**User impact**: < 35 seconds of unavailability. Active voice sessions are interrupted and must be restarted. Text sessions resume automatically.

### 15.2 Orchestrator Crash

**What happens**: The orchestrator process crashes.

**Recovery**:
1. Load balancer detects the crash (within 10 seconds)
2. Traffic is routed to other orchestrator instances
3. Crashed instance restarts and rejoins the pool

**User impact**: None (if multiple orchestrator instances are running). New connections may fail for < 10 seconds.

### 15.3 Storage Failure

**What happens**: The primary storage node becomes unavailable.

**Recovery**:
1. Automatic failover to replica (within 60 seconds)
2. Agents continue operating with cached state during failover
3. Writes that occurred during failover are replayed from the event log

**User impact**: Read operations may be slightly stale during failover. Write operations are queued and replayed. No data loss.

### 15.4 Event Bus Failure

**What happens**: The event bus becomes unavailable.

**Recovery**:
1. Producers buffer events locally (in-memory, bounded)
2. Event bus restarts and consumers resume from last committed offset
3. Buffered events are flushed to the event bus

**User impact**: Proactive notifications may be delayed. Conversations are not affected (conversations do not depend on the event bus).

### 15.5 LLM Provider Failure

**What happens**: The LLM provider (Gemini, OpenAI, etc.) becomes unavailable or returns errors.

**Recovery**:
1. Agent detects failure (timeout or error response)
2. Agent retries with exponential backoff (3 attempts, max 30 seconds)
3. If all retries fail, agent informs the user: "I'm having trouble reaching my AI provider. I'll try again in a moment."
4. Agent queues the request and retries when the provider recovers

**User impact**: Response latency increases. User is informed of the issue. No data loss.

### 15.6 Plugin Failure

**What happens**: A plugin (Gmail, Calendar, etc.) becomes unavailable.

**Recovery**:
1. Agent detects failure (timeout or error response from plugin)
2. Agent informs the user: "Gmail is not responding right now. I'll try again in a few minutes."
3. Agent retries the plugin call with backoff
4. Webhook events from the plugin are queued and processed when the plugin recovers

**User impact**: Plugin-dependent features are temporarily unavailable. Core agent functionality is not affected.

### 15.7 Network Partition

**What happens**: The agent loses connectivity to the orchestrator.

**Recovery**:
1. Agent continues operating with cached config
2. Agent attempts to reconnect to orchestrator with exponential backoff
3. When connectivity is restored, agent re-registers and syncs state

**User impact**: None for active sessions. New device connections may fail until connectivity is restored.

---

## Appendix A: Key Metrics Summary

| Metric | Target | Current (Python) |
|---|---|---|
| Sign up → agent ready | < 2 seconds | ~5–15 seconds (cold container) |
| First text token | < 800ms | ✅ Met |
| First voice audio | < 500ms | ✅ Met |
| Notification delivery | < 3 seconds | ✅ Met |
| Memory retrieval | < 100ms | ✅ Met |
| Agent idle RAM | < 20MB | ~80MB (Go rewrite planned) |
| Agent active RAM | < 60MB | ~150MB (Go rewrite planned) |
| Agent boot time | < 200ms | ~2 seconds |
| Agent restart after crash | < 35 seconds | ✅ Met |
| System availability | 99.99% | — |
| Data loss tolerance | Zero | ✅ Met |

---

## Appendix B: Non-Requirements

These are things the system explicitly does not need to do, to keep scope clear:

- **On-device AI inference**: The agent does not run LLM inference locally. All LLM calls go to cloud providers.
- **Self-hosted orchestrator**: The orchestrator is not designed to be self-hosted by users. It is a managed service.
- **Multi-tenant agents**: One agent per user. Agents are not shared between users.
- **Real-time collaboration**: Multiple users cannot share a single agent session.
- **Offline operation**: The agent requires internet connectivity to function. Offline mode is not a requirement.
- **Model training**: The system does not train models. It uses existing models via APIs.
- **Custom model fine-tuning**: Users cannot fine-tune models. The system uses base models with prompt engineering and memory.

---

*This document is a living specification. It will be updated as the product evolves. All changes must be reviewed against the Guiding Principles in Section 1.*
