# Project Aether — Roundtable Summary

**Date:** February 17, 2026
**Participants:** Surya (Founder), Steve Jobs (Product), Jony Ive (Design), Linus Torvalds (Architecture), Maya (System Architect), Dev (Engineer — Optimist), Priya (Engineer — Skeptic)

---

## Vision

A cloud-first, voice-primary AI assistant where conversation is the standard mode of interaction, text is fallback, and vision is integrated. Memory makes it personal over time. Built for consumers — iPhone-level simplicity. Software first, hardware later.

## Core Philosophy (Surya)

- **UX over DX.** Optimize for the user first. Developer experience follows once the product works.
- **Hono, not NestJS.** Don't overcomplicate. Keep it light. One file energy. Use only what you need.
- **Code improves daily.** Ship working software. Abstract first, stabilize later. Clean boundaries enable this.
- **Clean boundaries are non-negotiable.** The interfaces between components must be well-defined even if implementations are rough.

## Product Requirements (Steve Jobs & Jony Ive)

1. **Zero-config onboarding** — Voice-guided, under 60 seconds. "Hi, what should I call you?" No settings, no API keys, no provider selection.
2. **Voice is primary** — No beeps, no buttons to start. Natural conversational rhythm. Designed pauses. Mirror the user's energy.
3. **Consistent voice identity** — One voice, same personality, always. The voice IS the brand.
4. **Proactive memory** — "You usually call your mom on Sundays." The system surfaces knowledge without being asked. Memory feels casual, not clinical.
5. **Vision feels natural** — "Oh that looks like a nice restaurant" not "I can see an image containing a restaurant facade." Vision output is digested into conversation.
6. **Three qualities:** Fast (<500ms), Remembers (like a friend), Sees (naturally).
7. **Hardware topology:** Voice = primary interface, phone app = visual/text fallback, future hardware = ambient presence.

## Architecture (Linus & Maya)

### Core Principle: Pipeline of Processors

```
AudioIn → STT → Memory → LLM → TTS → AudioOut
              ↑
           VisionIn
```

### The Three Boundaries (Interfaces)

1. **Processor Interface** — `start()`, `process(frame)`, `stop()`. Takes input frames, produces output frames. That's it.
2. **Model Interface** — `generate()`, `stream()`. Provider-agnostic. Factory functions create provider-specific implementations. (Pattern from Vercel AI SDK)
3. **Memory Interface** — `add()`, `search()`, `get()`. Implementations evolve behind it. (Pattern from mem0)

### What is a Frame?

- Type + data + metadata + timestamp
- Types: Audio, Text, Vision, Memory, Control
- Protobuf for wire format, plain objects in-process

### What is a Processor?

- Takes input frames, produces output frames
- Has `start()`, `process()`, `stop()`
- Knows nothing about the pipeline topology
- Pipeline is just a list of processors chained together

## Technology Decisions

- **Language:** Python (entire voice AI ecosystem is Python — Pipecat, LiveKit, mem0, Hume)
- **Package Manager:** UV
- **Transport:** WebSocket for v0.01, WebRTC (LiveKit) for v0.1+
- **No local-first:** Cloud models only. Smaller models aren't smart enough yet.
- **Protocol:** Aether Frame Protocol (AFP) — complements MCP/A2A, doesn't compete. Protobuf serialized, gRPC for service-to-service.

## Reference Projects Studied

| Project | Key Takeaway |
|---------|-------------|
| **Pipecat** | Frame-based pipeline model. 50-line voice agent. Pipeline = list of processors. Elegant simplicity. |
| **LiveKit Agents** | WebRTC transport at scale (3B+ calls/year for ChatGPT). Hybrid WebRTC + WebSocket pattern. |
| **OpenClaw** | Memory system (hybrid BM25 + vector search), model fallback chains, multi-provider abstraction. Complex but comprehensive. |
| **Mem0** | Dual-store memory (vectors + knowledge graph). Simple API: `add()`, `search()`. Entity extraction via LLM. |
| **Vercel AI SDK** | Provider abstraction done right. `LanguageModelV3` interface. One-line model swap. |
| **Vocode** | Parallel pipeline lesson: synthesize sentence N+1 while playing sentence N. Endpointing is the hardest problem. |
| **Hume** | 48-emotion prosody analysis. Emotion as data alongside audio. WebSocket bidirectional streaming. |

## Latency & Cost Research (2026)

- **Target:** <500ms voice-to-voice for natural feel
- **Best-in-class STT:** Sub-200ms (Deepgram streaming)
- **Best-in-class TTS:** 75ms (ElevenLabs Flash v2.5)
- **LLM TTFT:** 200-500ms
- **Total achievable:** ~500ms with parallel streaming pipeline
- **Cost at scale:** $0.06-0.24/min (OpenAI Realtime), $0.08/min (ElevenLabs Conv AI)

## Phasing (Linus's Kernel Approach)

### v0.01 — The Bootloader
- One pipeline: mic → STT → LLM → TTS → speaker
- Memory: embed conversation chunks, inject into context
- Vision: send image alongside voice to multimodal LLM
- Transport: WebSocket
- Client: web page with mic button
- One file pipeline runner. One file per processor.

### v0.1 — The Interfaces
- Formalize Processor, Model, Memory interfaces
- Add model router (route by task type)
- Parallel streaming pipeline
- Upgrade to WebRTC via LiveKit
- Mobile app shell

### v0.5 — The Modules
- Knowledge graph memory (dual-store)
- Proactive memory surfacing
- Emotion-aware responses (Hume)
- Multi-provider fallback chains
- Background processing between conversations

### v1.0 — The Product
- Frozen interfaces, stable API
- Hardware-ready IPC boundary
- Zero-config onboarding
- Consistent voice identity
- "It just works"

## Key Risks (Priya)

- **Cost at scale:** 1M minutes/month = $80K-$300K in API costs alone
- **Latency consistency:** 500ms target is tight, network variance can break it
- **Memory quality:** Raw chunk embedding without LLM extraction may produce noisy results
- **Provider reliability:** Single provider dependency is a risk. Fallback chains needed by v0.1.
- **Complexity creep:** OpenClaw's 3,200 issues came from surface area growth. Stay narrow.

---

## Round 9 — Deep Research: Tools, Skills, Sub-Agents & Status Patterns

### Research Scope

Studied three codebases in depth: **Claude Code**, **OpenClaw**, and **OpenCode** (the SST open-source coding agent with 100K+ GitHub stars). Focused on four areas Surya requested: Tools, Skills, Sub-agents, and the status/acknowledge UX pattern.

---

### 1. TOOLS — How Actions Get Executed

**Steve:** Every system does the same thing at the core. Define a tool with name + description + parameter schema + async execute function. Hand the schemas to the LLM. LLM picks a tool and returns args. Execute. Feed result back. Repeat until done. The differences are in *where* provider abstraction happens.

**Maya — Three Approaches Compared:**

| Aspect | OpenClaw | OpenCode | What Aether Should Do |
|--------|----------|----------|----------------------|
| **Definition** | `AgentTool<TParams, TResult>` — name, description, inputSchema, execute | `Tool.Info` with Zod schema — lazy-initialized via `init()` | **Zod-based like OpenCode.** Type-safe, runtime validation built-in |
| **Registration** | Dynamic per-session via factory function, policy-filtered | File-based discovery (SKILL.md) + config paths | **Code-based registry** — simpler than file discovery for v0.05 |
| **Execution** | Wrapper chain: abort → validation → workspace guard → execute | Built into `Tool.define()` — auto-validates, auto-truncates output | **Single wrapper** — validate args, execute, truncate output |
| **Provider Agnosticism** | Same tools everywhere, *policy* filters per-provider | ai-sdk adapters transform tools *per-provider API format* | **OpenCode's approach** — let the SDK handle provider differences |
| **Result Format** | `{ content: [{type, text}], details }` | `{ title, metadata, output, attachments }` | **Simple:** `{ output: string, metadata?: any }` |

**Dev:** OpenCode supports 15+ providers out of the box through Vercel's ai-sdk. Same tool definition works across OpenAI, Anthropic, Google, Bedrock, Mistral, Groq, xAI — the SDK transforms tool schemas to each provider's format automatically. That's what we want.

**Priya:** OpenClaw has something OpenCode doesn't — a **policy pipeline**. Tool groups like `group:fs`, `group:runtime`, `group:web` with profiles like `minimal`, `coding`, `full`. This matters when you have sub-agents that shouldn't have access to everything. We'll need this eventually.

**Linus:** Build OpenCode's approach first — it's simpler. Tools are just `Tool.define(id, { description, parameters: z.object({...}), execute })`. Add OpenClaw's policy groups when we add sub-agents.

---

### 2. SKILLS — Higher-Level Capabilities

**Steve:** Skills are instructions, not code. They're markdown documents that teach the LLM *how* to use tools for specific workflows. Think of them as specialized knowledge that gets loaded when relevant.

**Maya — Three Approaches Compared:**

| Aspect | Claude Code | OpenClaw | OpenCode |
|--------|------------|----------|----------|
| **What is a Skill** | SKILL.md + bundled scripts/references/assets | SKILL.md + wrapper shell scripts + agent configs | SKILL.md with metadata, accessed as a tool |
| **Discovery** | Plugin filesystem scan | Multi-source: bundled → workspace → plugin dirs | Multi-source + remote URLs + config paths |
| **Invocation** | *Implicit* — description matching triggers loading | *Explicit* — slash command (`/review-pr`) | *Tool-based* — agent calls `skill("name")` |
| **Composition** | Skills teach how to use tools (context injection) | Skills orchestrate multi-agent workflows | Skills accessed via tool interface with permissions |
| **Lifecycle** | Progressive: metadata → body → resources (lazy) | Script-first with artifact contracts | State caching with on-demand loading |

**Dev:** OpenCode's approach is cleanest for us. Skills are just another tool — the LLM calls `skill("name")`, gets the SKILL.md content injected, and follows the instructions. No special skill engine needed.

**Priya:** But Claude Code's *implicit* triggering is better UX. The user says "create a folder structure" and the system auto-detects that the "file-management" skill is relevant and loads it. No one has to explicitly invoke anything.

**Steve:** For voice, implicit is the only option. You can't say "slash review-pr" naturally. The system needs to detect intent and load the right skill. Claude Code's description-matching approach is right for Aether.

**Linus — Proposed Design for Aether:**
- Skills are SKILL.md files with `name` and `description` in frontmatter
- Scanned from a `skills/` directory at startup
- Description matching determines which skill to load (like Claude Code)
- Loaded skill content gets injected into LLM context
- Skills can reference tools by name in their instructions
- Progressive loading: metadata always, full body only when triggered

---

### 3. SUB-AGENTS — Parallel Task Execution

**Steve:** This is the biggest architectural difference between the three. OpenClaw has a full sub-agent system with process isolation, concurrency limits, and cascade killing. OpenCode has agent *switching* — same session, different permissions. Claude Code has background tasks.

**Maya — Three Approaches Compared:**

| Aspect | OpenClaw (Full Sub-Agents) | OpenCode (Role Switching) | Claude Code (Background Tasks) |
|--------|---------------------------|---------------------------|-------------------------------|
| **Process Model** | Separate LLM instance per sub-agent | Same session, different permission set | Background task queue |
| **Isolation** | Full session isolation + sandbox | Permission-based in same session | Task-level isolation |
| **Concurrency** | Lane-based (default 8 parallel), 5 per parent | Serial only (no parallel) | Queue-based |
| **Nesting** | 2 levels: main → orchestrator → workers | None | None |
| **Communication** | Announce flow (async result delivery to parent) | Synthetic message injection | Task notification |
| **Management** | list/kill/steer commands | None | `/tasks` command |
| **Lifecycle** | Event-driven + 60min auto-archive sweeper | Immediate | Timer-based |

**Dev:** OpenClaw's model is what we need. When the user says "create a project structure with 5 files," the main agent should acknowledge immediately, spawn a sub-agent to do the work, and continue the conversation. No blocking.

**Priya:** But OpenClaw's sub-agent system is 1,500+ lines across 4 files. That's complex. We need to start simpler.

**Linus — Proposed Design for Aether:**

**Phase 1 (v0.05):** Simple background tasks (like Claude Code)
- Main agent calls `spawn_task(prompt, tools)`
- Task runs as a separate LLM call with its own tool set
- Result returned as a message when complete
- Max 3 concurrent tasks, 60s timeout
- No nesting

**Phase 2 (v0.1):** Orchestrator pattern (like OpenClaw)
- Agent roles: `main` (conversation), `worker` (tasks), `planner` (read-only analysis)
- Workers get restricted tool permissions
- Registry tracks active workers
- Kill/steer support

**Steve — Voice UX for Sub-Agents:**
The key insight: in voice, sub-agents are *invisible*. The user says "organize my photos by date." The main agent says "On it, I'll sort those for you." A sub-agent does the work. When it's done, the main agent says "All done — organized 47 photos into 12 folders." The user never knows there was a sub-agent. They just experienced a helpful assistant that didn't make them wait.

---

### 4. STATUS / ACKNOWLEDGE PATTERN

**Steve:** This is a small thing with massive UX impact. When you ask Claude Code to do something, you immediately see "Reading..." or "Editing..." with a spinner. You know it's working. In voice, this is even more critical — silence = broken.

**Maya — OpenCode's Implementation (Fully Documented):**

Each tool has a **pending status text** shown while running:

| Tool | Status Text |
|------|------------|
| Glob | "Finding files..." |
| Read | "Reading file..." |
| Grep | "Searching content..." |
| Bash | "Writing command..." |
| Write | "Preparing write..." |
| Edit | "Preparing edit..." |
| WebSearch | "Searching web..." |
| Task | "Delegating..." |
| Skill | "Loading skill..." |

**State machine:** `pending → running → completed/error`

**Rendering:** Braille spinner (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) at 80ms intervals + status text. On completion, spinner stops, shows icon + result preview.

**Multiple tools show simultaneously:**
```
⠋ Finding files...
⠹ Reading file...
⠼ Searching code...
```

**Dev:** For Aether's voice mode, these status words become the **acknowledge phase**:
1. LLM decides to call a tool
2. System speaks the acknowledge phrase: "Let me find that..." / "Reading that now..." / "Searching for that..."
3. Tool executes in background
4. LLM gets result, speaks the answer

**Linus — Proposed Design for Aether:**
- Each tool definition includes a `status_text` field: `"Finding files..."`, `"Running command..."`, etc.
- In text mode: show spinner + status (like OpenCode)
- In voice mode: status text becomes TTS acknowledgment
- The acknowledge is spoken *before* tool execution starts — no dead air
- Keep it natural: "Let me check..." not "Executing file_read tool..."

---

### Summary — What We're Stealing From Each

| From | What | Why |
|------|------|-----|
| **OpenCode** | Zod-based tool definitions + ai-sdk provider abstraction | Type-safe, 15+ providers free, lazy init |
| **Claude Code** | Implicit skill triggering via description matching | Best for voice — no slash commands needed |
| **OpenClaw** | Sub-agent registry + lifecycle management pattern | Need parallel execution without blocking conversation |
| **OpenCode** | Status text per tool + spinner state machine | Direct mapping to voice acknowledge pattern |
| **OpenClaw** | Tool policy groups + profiles | Security for sub-agents (restricted permissions) |
| **Claude Code** | Progressive skill loading (metadata → body → resources) | Context efficiency — don't load what you don't need |

---

## Proposed v0.05: "Hands" — Updated Architecture

Based on research, here's the refined plan:

1. **Tool Interface** — `AetherTool`: name, description, parameters (Zod), status_text, async execute
2. **Tool Registry** — Code-based registration, returns schemas for any LLM provider
3. **5 Core Tools** — read_file, write_file, list_directory, run_command, web_search
4. **Provider-Agnostic Execution** — Use ai-sdk pattern: one tool definition, SDK adapts per provider
5. **Skill System** — SKILL.md files, description-matching trigger, progressive context loading
6. **Background Tasks** — Simple `spawn_task()` with max 3 concurrent, 60s timeout
7. **Status/Acknowledge** — Each tool has status_text, spoken as TTS in voice mode, spinner in text mode
8. **Safety** — Working directory jail, command timeout, basic tool groups (will expand with sub-agents)

## Next Steps

1. Write v0.01 detailed requirements
2. Set up Python project with UV in `/app` folder
3. Build bootloader: pipeline runner + core processors
4. Get a working voice conversation end-to-end
