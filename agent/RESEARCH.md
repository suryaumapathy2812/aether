# Aether Agent — Reference Architecture Research

> Research findings from analyzing OpenCode, Claude Code, and OpenClaw codebases.
> These three projects implement the "agentic loop" pattern — where an LLM receives a user message, decides to use tools, executes them, feeds results back, and loops until it has a final answer.

---

## Table of Contents

1. [The Agentic Loop](#1-the-agentic-loop)
2. [Tool System](#2-tool-system)
3. [Message Management](#3-message-management)
4. [Streaming](#4-streaming)
5. [Error Handling & Retry](#5-error-handling--retry)
6. [Provider Abstraction](#6-provider-abstraction)
7. [Skills System](#7-skills-system)
8. [Plugins System](#8-plugins-system)
9. [Key Takeaways for Aether](#9-key-takeaways-for-aether)
10. [Voice Pipeline Architecture](#10-voice-pipeline-architecture)
11. [Voice Pipeline Deep Dives](#11-voice-pipeline-deep-dives)
12. [Voice Pipeline Takeaways for Aether](#12-voice-pipeline-takeaways-for-aether)

---

## 1. The Agentic Loop

### Universal Pattern (All Three Follow This)

```
while (true) {
    1. Build messages array (system prompt + history + tool results)
    2. Call LLM (streaming)
    3. Process stream events:
       - Text chunks → emit to UI
       - Tool calls → execute tool → append tool_result to messages
    4. Check exit conditions:
       - No more tool calls? → BREAK (done)
       - Context overflow? → Compact/summarize → CONTINUE
       - Error? → Retry with backoff or fail
    5. CONTINUE (loop back to step 1 with tool results appended)
}
```

### OpenCode — Two Nested Loops

**Architecture:** Outer loop manages session state, inner loop handles streaming + tool execution.

| Layer | File | Purpose |
|-------|------|---------|
| **Outer loop** | `packages/opencode/src/session/prompt.ts` → `loop()` (line 294) | `while(true)` — reads messages from DB, checks exit conditions (assistant finished + no pending tool calls), creates processor, handles compaction/overflow |
| **Inner loop** | `packages/opencode/src/session/processor.ts` → `process()` (line 49) | `while(true)` — calls `LLM.stream()`, iterates over stream events (`tool-call`, `tool-result`, `text-delta`, etc.), retry on failure |
| **LLM layer** | `packages/opencode/src/session/llm.ts` → `stream()` (line 46) | Wraps Vercel AI SDK `streamText()` — the SDK handles tool execution internally when tools with `execute` functions are passed |

**Outer loop flow (`prompt.ts`):**
1. User message created and saved to DB (line 158)
2. Enter `loop()` (line 184) — `while(true)` at line 294
3. Read messages from DB via `MessageV2.filterCompacted()` (line 298)
4. Find last user message and last assistant message (lines 300-315)
5. Exit condition: assistant finished AND no pending tool calls (lines 318-325)
6. Handle special cases: pending subtasks (line 352), compaction (line 529), context overflow (line 541)
7. Create `SessionProcessor` (line 566), resolve tools (line 602), build system prompts (line 653)
8. Call `processor.process()` (line 659)
9. Process result: `"stop"` → break, `"compact"` → trigger compaction, `"continue"` → loop again (lines 705-714)

**Inner loop flow (`processor.ts`):**
1. Enter `process()` — `while(true)` at line 49 for retry logic
2. Call `LLM.stream()` (line 53)
3. Iterate over `stream.fullStream` (line 55) handling events:
   - `tool-input-start` (line 111): create pending tool part
   - `tool-call` (line 134): mark tool running, check doom loop (3 identical consecutive calls, lines 152-176)
   - `tool-result` (line 180): mark tool completed with output
   - `tool-error` (line 204): mark tool as error, check permission rejection (lines 220-225)
   - `text-start/delta/end`, `reasoning-start/delta/end`: streaming text/reasoning
   - `start-step/finish-step`: track snapshots, usage, cost
4. Return `"continue"`, `"stop"`, or `"compact"` (lines 412-415)

### Claude Code — Hook-Driven Loop (Architecture Inferred)

> Note: Claude Code source is closed. Architecture inferred from plugin SDK, hooks system, changelog, and example plugins.

**Architecture:** Single loop with hook intercept points at every stage.

```
SessionStart
  └→ while (true) {
       UserPromptSubmit  ← hook: can modify prompt
       ↓
       LLM call (streaming)
       ↓
       For each tool_use in response:
         PreToolUse       ← hook: can block/modify (allow|deny|ask)
         Execute tool
         PostToolUse      ← hook: can add context
       ↓
       Stop              ← hook: can BLOCK stop → forces continuation!
     }
SessionEnd
```

**Hook events (in loop order):**

| Hook Event | When | Can Do |
|------------|------|--------|
| `SessionStart` | Session begins | Inject initial context |
| `UserPromptSubmit` | User sends message | Modify prompt |
| `PreToolUse` | Before tool execution | Block (`deny`), allow, ask, modify input |
| `PostToolUse` | After tool completes | Add context, observe results |
| `Stop` | Agent wants to stop | Block stop → continue loop (`approve`\|`block` + reason) |
| `SubagentStop` | Subagent completing | Same as Stop for child agents |
| `PreCompact` | Before context compaction | Inject must-preserve context |
| `SessionEnd` | Session ends | Cleanup |
| `Notification` | Notification sent | Observe/modify |

**Key insight:** The `Stop` hook is particularly powerful — a hook can reject the agent's decision to stop, feeding a new prompt back and forcing the loop to continue. This enables workflows like "don't stop until tests pass."

### OpenClaw — Two-Layer Loop (Outer Retry + Inner SDK)

**Architecture:** Inner agentic loop lives in an external SDK. OpenClaw wraps it with retry/failover.

| Layer | File | Purpose |
|-------|------|---------|
| **Outer loop** | `src/agents/pi-embedded-runner/run.ts` (line 499) | `while(true)` — retry/failover: auth profile rotation, context overflow recovery, compaction, thinking-level fallbacks |
| **Inner loop** | SDK's `session.prompt()` (called at `attempt.ts` line 1092) | Core agent cycle inside `@mariozechner/pi-agent-core` — sends messages → detects tool calls → executes → feeds back → loops |
| **Event observation** | `src/agents/pi-embedded-subscribe.ts` | Subscribes to SDK events: `tool_execution_start/end`, `message_start/update/end`, `auto_compaction` |

**Outer loop flow (`run.ts`):**
1. Enter `runEmbeddedPiAgent()` (line 187)
2. `while(true)` at line 499 with retry limit (32-160 iterations based on auth profiles, lines 106-116)
3. Call `runEmbeddedAttempt()` (line 536) for each attempt
4. Handle failures:
   - Context overflow → up to 3 compaction attempts, then tool result truncation (lines 648-810)
   - Auth failure → rotate to next auth profile (lines 919-1016)
   - Thinking level error → fall back to lower thinking level (lines 907-917)
   - Prompt error → retry (lines 812-904)
   - Timeout → abort (lines 1058-1080)

**Attempt flow (`attempt.ts`):**
1. Create tools via `createOpenClawCodingTools()` (line 296)
2. Split tools into built-in and custom via `splitSdkTools()` (line 572)
3. Create agent session via SDK's `createAgentSession()` (line 599)
4. Subscribe to events via `subscribeEmbeddedPiSession()` (line 803)
5. Submit prompt via `activeSession.prompt()` (lines 1092-1094) — inner loop runs here
6. Return result or error

---

## 2. Tool System

### OpenCode

**Tool definition** (`packages/opencode/src/tool/tool.ts`):
- `Tool.define()` (line 48) wraps tools with:
  - Zod parameter validation (line 59)
  - Automatic output truncation (line 74)
  - Metadata (name, description, parameters schema)

**Tool registry** (`packages/opencode/src/tool/registry.ts`):
- `ToolRegistry.tools()` (line 127) returns all available tools
- Built-in tools: `bash`, `read`, `write`, `edit`, `glob`, `grep`, `task` (subagent), `batch`, `websearch`, `webfetch`, `codesearch`, `skill`, `question`, `plan_enter`, `plan_exit`, `apply_patch`, `lsp`, `invalid`, `todo`

**Tool resolution for LLM** (`prompt.ts` → `resolveTools()` line 736):
- Converts internal `Tool.Info` objects into Vercel AI SDK `tool()` objects
- Attaches `execute` functions
- Wraps with plugin hooks (`tool.execute.before`/`after`)
- Adds permission checking via `ctx.ask()`

**Special tools:**
- **Batch tool** (`tool/batch.ts`): Enables parallel tool execution via `Promise.all()`, max 25 calls
- **Task/Subagent tool** (`tool/task.ts`): Creates child session, calls `SessionPrompt.prompt()` recursively (line 128)
- **Invalid tool** (`tool/invalid.ts`): Catch-all for malformed tool calls, returns error message

**Tool repair:**
- `experimental_repairToolCall` in `llm.ts` (line 182) — lowercases malformed tool names or redirects to `invalid` tool

**Doom loop detection:**
- In `processor.ts` (lines 152-176): tracks last 3 tool calls. If all identical → injects error, prevents infinite loops.

### Claude Code

**Built-in tools** (inferred from settings, permissions, plugin docs):

| Category | Tools |
|----------|-------|
| File | `Read`, `Write`, `Edit`, `MultiEdit`, `Glob`, `Grep`, `LS`, `NotebookRead`, `NotebookEdit` |
| Execution | `Bash`, `BashOutput`, `KillShell` |
| Agent | `Task` (subagents), `TaskUpdate`, `TaskStop`, `TaskOutputTool` |
| Web | `WebFetch`, `WebSearch` |
| Interaction | `AskUserQuestion`, `SlashCommand`, `Skill`, `MCPSearch` |
| Memory | `TodoWrite`, memory system |
| Code Intel | `LSP` (Language Server Protocol) |
| External | `mcp__<server>__<tool>` (MCP integration) |

**Permission system:**
- Rules: `allow` | `ask` | `deny`
- Glob patterns: `Bash(git:*)`, `Bash(npm *)`, `mcp__server__*`
- Hierarchy: managed settings > user settings > project settings > session
- Hook-based: `PreToolUse` hook can return `permissionDecision: "allow"|"deny"|"ask"`

### OpenClaw

**Tool creation** (`src/agents/pi-tools.ts`):
- `createOpenClawCodingTools()` creates all tools
- Uses SDK's `codingTools` as base, adds custom tools on top
- Tools: bash/exec, read, write, edit, apply_patch, channel tools, openclaw-specific tools

**Before-tool-call hook** (`src/agents/pi-tools.before-tool-call.ts`):
- `wrapToolWithBeforeToolCallHook()` (line 175) wraps every tool
- Can **block** tool calls: returns `{blocked: true, reason}`
- Can **modify** parameters before execution
- Includes **tool loop detection** (lines 83-133): detects repeated identical calls

**Tool policy** (`src/agents/pi-tools.policy.ts`):
- Allow/deny lists with glob patterns
- Subagent-specific tool restrictions
- Sandbox policies for execution isolation

---

## 3. Message Management

### OpenCode

**Message types** (`packages/opencode/src/session/message-v2.ts`):
- User messages and Assistant messages
- Rich Part types: text, tool, reasoning, file, step-start, step-finish, patch, compaction, subtask

**Message → LLM format** (`MessageV2.toModelMessages()` line 491):
- Converts internal format to Vercel AI SDK's `ModelMessage[]`
- Uses `convertToModelMessages()` from the AI SDK

**Storage:**
- Messages stored in **SQLite** via Drizzle ORM
- Streamed in reverse chronological order via `MessageV2.stream()` (line 716)

**Context window management** (`packages/opencode/src/session/compaction.ts`):
- `isOverflow()` (line 32): checks if tokens exceed usable context
- `prune()` (line 58): erases old tool outputs, keeps last 40K tokens
- `process()` (line 101): uses a **compaction agent** to generate conversation summary
- `filterCompacted()` (line 794): stops reading history when it hits a compaction boundary

### Claude Code

**Context building:**
- `CLAUDE.md` files at multiple directory levels provide persistent instructions
- Skills (`.claude/skills/`) auto-discovered and loaded, ~2% of context window
- Agent definitions (`.claude/agents/`) are markdown with frontmatter
- Commands (`.claude/commands/`) are slash-command prompts with YAML frontmatter

**Storage:**
- Transcript stored as **JSONL** format at `transcript_path`
- Each line: JSON object with `role`, `message.content[]`
- Session resume supported (`--resume`)

**Context management:**
- Auto-compact triggers when context window fills
- Manual `/compact` command available
- `PreCompact` hook allows injecting must-preserve context

### OpenClaw

**Context window guard** (`src/agents/context-window-guard.ts`):
- Hard minimum: 16K tokens
- Warning threshold: below 32K tokens

**Compaction** (`src/agents/compaction.ts`):
- Uses SDK's `generateSummary()`
- Chunks messages by token share, merges summaries
- Up to 3 compaction attempts on context overflow
- Fallback: truncate tool results if compaction isn't enough

---

## 4. Streaming

### OpenCode

- Uses Vercel AI SDK's `streamText()` which returns a `fullStream` async iterable
- Events: `text-start`, `text-delta`, `text-end`, `reasoning-start`, `reasoning-delta`, `reasoning-end`, `tool-input-start`, `tool-call`, `tool-result`, `tool-error`, `start-step`, `finish-step`
- Tool calls and text can be interleaved in the stream
- Each event is processed in `processor.ts` and persisted to DB in real-time

### Claude Code

- Streaming is the default API path
- Non-streaming fallback exists
- `--include-partial-messages` flag for partial message streaming (SDK)
- Thinking blocks stream in real-time (visible in transcript mode Ctrl+O)
- MCP tool streaming supported

### OpenClaw

- Stream function is provider-specific: SDK's `streamSimple` for most, custom for Ollama (attempt.ts lines 650-662)
- Event subscription system observes stream events:
  - `message_start`, `message_update`, `message_end`
  - `tool_execution_start`, `tool_execution_update`, `tool_execution_end`
  - `agent_start`, `agent_end`
  - `auto_compaction_start`, `auto_compaction_end`
- Events emitted to UI in real-time via handler chain

---

## 5. Error Handling & Retry

### OpenCode

**Tool errors** (`processor.ts` lines 204-228):
- Recorded as `ToolStateError`
- Permission rejections can block the loop

**LLM API errors** (`processor.ts` lines 350-377):
- Parsed via `MessageV2.fromError()` (message-v2.ts line 811)
- Checked for retryability via `SessionRetry.retryable()` (retry.ts line 61)

**Retry logic** (`packages/opencode/src/session/retry.ts`):
- Initial delay: 2000ms
- Backoff factor: 2x
- Respects `retry-after` headers (lines 28-58)
- Retries: rate limits, overloaded errors, connection resets

**Context overflow** (`packages/opencode/src/provider/error.ts` lines 8-22):
- Detected via regex patterns
- Results in `ContextOverflowError` — NOT retried, triggers compaction instead

### Claude Code

- Inferred: retry on transient errors, auto-compact on context overflow
- Provider-specific token counting (optimized for Bedrock)
- Prompt caching with cache invalidation tracking

### OpenClaw

**Outer loop error handling** (`run.ts`):
- Context overflow: up to 3 compaction attempts, then tool result truncation (lines 648-810)
- Auth failure: rotate to next auth profile (lines 919-1016)
- Thinking level error: fall back to lower level (lines 907-917)
- Prompt error: retry (lines 812-904)
- Timeout: abort (lines 1058-1080)
- Max retry iterations: 32-160 depending on auth profile count (lines 108-116)

**Tool-specific:**
- `FailoverError` class for provider failover (`src/agents/failover-error.ts`)
- Anthropic: refusal magic scrubbing (run.ts lines 64-76)
- Google: turn ordering fixes and tool sanitization

---

## 6. Provider Abstraction

### OpenCode

**File:** `packages/opencode/src/provider/provider.ts`

**Supported providers (20+):** OpenAI, Anthropic, Google, Azure, Bedrock, OpenRouter, xAI, Mistral, Groq, DeepInfra, Cerebras, Cohere, Vercel, GitLab, GitHub Copilot, and more.

**Common interface:** All providers loaded via Vercel AI SDK's `Provider` interface (`LanguageModelV2`).

**Model resolution:** `Provider.getLanguage()` (line 1160) returns a `LanguageModelV2` used by `LLM.stream()`.

**Custom loaders:** `CUSTOM_LOADERS` (line 116) per provider — handle auth, custom model loading (e.g., OpenAI uses `sdk.responses()`, Bedrock adds region prefixes).

**System prompts:** Different system prompts per provider/model family (Claude, GPT, Gemini) in `system.ts`.

### Claude Code

**Providers:** Anthropic (direct), AWS Bedrock, Google Vertex AI, Microsoft Azure AI Foundry.

**API proxy:** via `ANTHROPIC_BASE_URL`, HTTP_PROXY, HTTPS_PROXY, mTLS.

**Model aliases:** `sonnet`, `opus`, `haiku` → resolve to provider-specific model IDs.

**Auth:** API key, OAuth (Claude Max/Pro subscribers).

### OpenClaw

**Model resolution:** `src/agents/pi-embedded-runner/model.ts` → `resolveModel()`.

**Auth profiles:** `src/agents/model-auth.ts` — multiple API keys per provider, rotation on failure.

**Provider-specific handling:**
- Ollama: custom stream function (attempt.ts lines 650-662)
- Anthropic: refusal magic scrubbing
- Google: turn ordering fixes, tool sanitization
- GitHub Copilot: custom provider (`src/providers/`)

---

## 7. Skills System

### What is a Skill?

A **skill** is a self-contained knowledge package — a markdown file with instructions that gets injected into the LLM's system prompt to give it domain-specific capabilities. All three projects implement skills similarly: a `SKILL.md` file with YAML frontmatter for metadata, plus optional bundled resources.

Skills are NOT tools. Tools are functions the LLM can call. Skills are **context/instructions** that teach the LLM how to approach a specific domain.

### Comparison Table

| Aspect | OpenCode | Claude Code | OpenClaw |
|--------|----------|-------------|----------|
| File format | `SKILL.md` with YAML frontmatter | `SKILL.md` with YAML frontmatter | `SKILL.md` with YAML frontmatter |
| Discovery locations | `.opencode/skills/`, workspace `skills/`, custom dirs, marketplace | `.claude/skills/`, plugin `skills/` dirs | 6 sources: bundled, managed (`~/.openclaw/skills`), personal (`~/.agents/skills`), project (`<workspace>/.agents/skills`), workspace (`<workspace>/skills`), plugin-contributed, extra dirs |
| Loading | Progressive (metadata → full content → resources) | Progressive 3-level (metadata ~100 words → body <5k words → bundled resources) | Progressive with budget (max 150 skills, max 30K chars in prompt) |
| LLM tool | `skill` tool for search/read/create/install | `Skill` tool (confirmed via allowed-tools) | Skills injected via `formatSkillsForPrompt()` |
| Install support | Yes — marketplace, create, install | Yes — via plugins | Yes — brew, npm, go, uv, download; `.skill` package format (zip) |
| ~70+ bundled | No (user-created) | No (plugin-contributed) | Yes (1password, github, tmux, voice-call, obsidian, notion, slack, discord, etc.) |

### OpenCode Skills

**Source files:**
- `packages/opencode/src/tool/skill.ts` — Skill tool implementation
- `packages/opencode/src/agent/agent.ts` — Agent definitions reference skills

**Skill tool capabilities:**
The LLM has a `skill` tool it can invoke to:
- **Search** for skills by keyword
- **Read** a skill's full content
- **Create** new skills from instructions
- **Install** skills from a marketplace/registry

**How skills get into context:**
1. Agent definitions specify which skills are relevant
2. Skills are discovered from `.opencode/skills/` and workspace `skills/` directories
3. Skill metadata (name + description) is always included in the system prompt
4. Full skill content is loaded on-demand when the LLM invokes the `skill` tool

**Skill file structure:**
```
skills/
  my-skill/
    SKILL.md          # Required: instructions + frontmatter
    scripts/          # Optional: executable helpers
    references/       # Optional: docs loaded as needed
    assets/           # Optional: files for output (not loaded into context)
```

### Claude Code Skills

**Source files:**
- `plugins/plugin-dev/skills/skill-development/SKILL.md` — Complete skill authoring guide
- `plugins/plugin-dev/skills/skill-development/references/skill-creator-original.md` — Detailed reference

**Skill definition — YAML frontmatter schema:**
```yaml
---
name: string          # Required — skill identifier
description: string   # Required — third-person, with trigger phrases in quotes
version: string       # Optional — semver
license: string       # Optional
---
```

**Progressive disclosure — 3 levels:**

| Level | What | When loaded | Size target |
|-------|------|-------------|-------------|
| 1. Metadata | `name` + `description` | Always in context | ~100 words |
| 2. SKILL.md body | Full instructions | When skill triggers | <5K words (ideal 1,500-2,000) |
| 3. Bundled resources | Scripts, references, assets | On demand by LLM | Unlimited |

**Skill content best practices:**
- Write in imperative/infinitive form (NOT second person)
- Description must use third person with specific trigger phrases in quotes: *"This skill should be used when the user asks to..."*
- Body should be actionable instructions, not documentation
- Scripts in `scripts/` can execute without being loaded into context (~0 context cost)
- References in `references/` are loaded as-needed into context
- Assets in `assets/` are NEVER loaded into context (used in output only)

**Skill as a tool:**
- `Skill` tool is available (confirmed via `allowed-tools: ["Skill"]` in command definitions)
- The LLM can invoke it to load a skill on-demand
- Skills also auto-activate based on description trigger phrases matching user intent

**Skill within plugins:**
```
plugin-name/
  skills/
    my-skill/
      SKILL.md
      scripts/
      references/
      assets/
```

### OpenClaw Skills

**Source files:**
- `src/agents/skills/types.ts` — `SkillEntry`, `OpenClawSkillMetadata`, `SkillInstallSpec` types
- `src/agents/skills/workspace.ts` — Discovery, loading, filtering, prompt building (778 lines)
- `src/agents/skills/config.ts` — `shouldIncludeSkill()`, eligibility evaluation
- `src/agents/skills/frontmatter.ts` — YAML frontmatter parsing
- `src/agents/skills/bundled-dir.ts` — Bundled skills directory resolution
- `src/agents/skills/plugin-skills.ts` — Plugin-contributed skill directories
- `src/agents/skills-install.ts` — Full skill installation engine (470 lines)
- `src/cli/skills-cli.ts` — CLI: `openclaw skills list|info|check`
- `skills/skill-creator/SKILL.md` — Meta-skill for creating new skills

**Skill metadata schema** (`OpenClawSkillMetadata` in `types.ts` lines 19-33):
```yaml
---
name: string
description: string
always: boolean           # If true, always include (bypass eligibility checks)
skillKey: string          # Unique key
primaryEnv: string        # Primary environment
emoji: string
homepage: string
os: string[]              # OS restrictions (e.g., ["macos", "linux"])
requires:
  bins: string[]          # Required binaries (ALL must exist)
  anyBins: string[]       # Required binaries (ANY must exist)
  env: string[]           # Required environment variables
  config: string[]        # Required config file paths
install:                  # Installation specs
  - kind: brew|node|go|uv|download
    package: string
    args: string[]
---
```

**Discovery — 6 sources (lowest to highest precedence):**
1. Extra dirs (from config/env)
2. Bundled skills (`skills/` in package — ~70+ skills)
3. Managed skills (`~/.openclaw/skills`)
4. Personal agent skills (`~/.agents/skills`)
5. Project agent skills (`<workspace>/.agents/skills`)
6. Workspace skills (`<workspace>/skills`)
7. Plugin-contributed skill dirs (via `resolvePluginSkillDirs()`)

Higher precedence sources override lower ones if same skill name.

**Loading into LLM context** (`workspace.ts` lines 446-499):
- `buildWorkspaceSkillSnapshot()` filters eligible skills
- Applies limits: max 150 skills, max 30K chars in prompt
- Uses **binary search** when char budget exceeded to fit as many skills as possible
- Compacts file paths with `~` for brevity
- Formats via `formatSkillsForPrompt()` from the SDK library

**Skill eligibility** (`config.ts` lines 70-112):
- `shouldIncludeSkill()` checks:
  1. Not disabled in config
  2. Passes bundled allowlist
  3. OS matches (for local vs remote execution)
  4. If `always: true` → include unconditionally
  5. Evaluate runtime requires: check bins exist, env vars set, config paths exist

**Skill commands:**
- Skills can register as slash commands via frontmatter fields: `command-dispatch`, `command-tool`, `command-arg-mode`
- `buildWorkspaceSkillCommandSpecs()` builds sanitized command specs
- Skills with `user-invocable: false` excluded from commands
- Skills with `disable-model-invocation: true` excluded from LLM prompt

**Skill installation** (`skills-install.ts`):
- Supports: brew, node (npm/pnpm/yarn/bun), go, uv, download
- Auto-installs prerequisite tools (e.g., uv via brew, go via brew or apt)
- Security scanning via `scanDirectoryWithSummary()` before install
- Configurable preferences: `skills.install.preferBrew`, `skills.install.nodeManager`

**Skill packaging:**
- Skills can be packaged as `.skill` files (zip with .skill extension)
- `skill-creator` meta-skill includes `scripts/init_skill.py`, `package_skill.py`, `quick_validate.py`

**CLI:**
- `openclaw skills list` — list all available skills
- `openclaw skills info <name>` — show skill details
- `openclaw skills check` — verify skill requirements

---

## 8. Plugins System

### What is a Plugin?

A **plugin** extends the agent with new capabilities: hooks (intercept points in the loop), tools, commands, agents (subagent definitions), skills, and MCP servers. All three projects support plugins, but with different architectures.

### Comparison Table

| Aspect | OpenCode | Claude Code | OpenClaw |
|--------|----------|-------------|----------|
| Manifest | Plugin hooks in config | `plugin.json` in `.claude-plugin/` | `openclaw.plugin.json` |
| Entry point | Hook functions in config | Hooks (bash/python scripts) + markdown files | TypeScript module with `register(api)` |
| Can add tools | Via MCP servers | Via MCP servers | Yes — `registerTool()` API |
| Can add hooks | `tool.execute.before/after` | 9 hook events via `hooks.json` | 22 hook types via `registerHook()` |
| Can add commands | No | Yes — `.md` files in `commands/` | Yes — `registerCommand()` API |
| Can add agents | No | Yes — `.md` files in `agents/` | No (but can register services) |
| Can add skills | No | Yes — `skills/` subdirectories | Yes — via manifest `skills` paths |
| MCP support | Yes — MCP tool servers | Yes — stdio, SSE, HTTP, WebSocket | No MCP found |
| Marketplace | No | Yes — `.claude-plugin/marketplace.json` | No |
| Hook execution | Inline (JS functions) | External process (bash/python, stdin/stdout JSON) | Inline (TypeScript, priority-ordered) |

### OpenCode Plugins

**Source files:**
- `packages/opencode/src/session/prompt.ts` → `resolveTools()` (line 736) — plugin hook integration
- Plugin hooks are integrated at the tool resolution level

**Plugin architecture:**
OpenCode has a lighter plugin model compared to Claude Code and OpenClaw. Plugins primarily work through:

1. **Tool execution hooks** — `tool.execute.before` and `tool.execute.after` callbacks wrap every tool
2. **MCP server integration** — external tool servers connected via Model Context Protocol
3. **Custom tool registration** — plugins can register additional tools via the tool registry

**MCP tools in OpenCode** (`prompt.ts` lines 830-916):
- MCP tools are resolved alongside built-in tools in `resolveTools()`
- Wrapped with permission checks and output truncation
- Named as MCP server tools and presented to the LLM like any other tool

### Claude Code Plugins

**Source files:**
- `plugins/plugin-dev/skills/plugin-structure/SKILL.md` — Complete plugin structure guide
- `plugins/plugin-dev/skills/hook-development/SKILL.md` — Complete hook reference
- `plugins/plugin-dev/skills/agent-development/SKILL.md` — Agent definition spec
- `plugins/plugin-dev/skills/command-development/SKILL.md` — Command format spec
- `plugins/plugin-dev/skills/mcp-integration/SKILL.md` — MCP server setup
- `plugins/plugin-dev/skills/plugin-settings/SKILL.md` — Plugin settings pattern
- `plugins/README.md` — Plugin system overview
- `.claude-plugin/marketplace.json` — Marketplace manifest

**Plugin directory structure:**
```
plugin-name/
├── .claude-plugin/
│   └── plugin.json          # Required: plugin manifest
├── commands/                 # Slash commands (.md files)
├── agents/                   # Subagent definitions (.md files)
├── skills/                   # Skills (subdirectories with SKILL.md)
├── hooks/
│   └── hooks.json           # Event handler configuration
├── .mcp.json                # MCP server definitions
├── scripts/                 # Helper scripts and utilities
└── README.md
```

**plugin.json schema:**
```json
{
  "name": "string (required, kebab-case)",
  "version": "string (semver, recommended)",
  "description": "string (recommended)",
  "author": {
    "name": "string",
    "email": "string",
    "url": "string"
  },
  "homepage": "string (URL)",
  "repository": "string | { type, url }",
  "license": "string",
  "keywords": ["string"],
  "commands": "./custom-path | [array of paths]",
  "agents": "./custom-path | [array of paths]",
  "hooks": "./path-to-hooks.json",
  "mcpServers": "./.mcp.json"
}
```

**hooks.json format:**
```json
{
  "description": "optional description",
  "hooks": {
    "EventName": [
      {
        "matcher": "regex | *",
        "hooks": [
          {
            "type": "command | prompt",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/script.sh",
            "prompt": "LLM prompt text",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

**Hook input (JSON via stdin to command hooks):**
```json
{
  "session_id": "string",
  "transcript_path": "string",
  "cwd": "string",
  "permission_mode": "ask | allow",
  "hook_event_name": "string",
  "tool_name": "string",
  "tool_input": {},
  "tool_result": "string",
  "user_prompt": "string",
  "reason": "string"
}
```

**Hook output formats:**

| Event | Output Format |
|-------|--------------|
| Standard | `{ "continue": bool, "suppressOutput": bool, "systemMessage": "string" }` |
| PreToolUse | `{ "hookSpecificOutput": { "permissionDecision": "allow\|deny\|ask", "updatedInput": {} }, "systemMessage": "string" }` |
| Stop / SubagentStop | `{ "decision": "approve\|block", "reason": "string", "systemMessage": "string" }` |

**Exit codes:** 0 = success, 2 = blocking error (stderr fed to Claude), other = non-blocking error.

**Hook types:**
- `command` — runs an external process (bash, python, etc.). Input via stdin JSON, output via stdout JSON.
- `prompt` — sends a prompt to the LLM. Used for `PreToolUse`, `Stop`, `SubagentStop`, `UserPromptSubmit`.

**All hooks within a matcher group run in parallel** — they don't see each other's output.

**Environment variables available to hooks:**
- `$CLAUDE_PROJECT_DIR` — workspace root
- `$CLAUDE_PLUGIN_ROOT` — plugin directory
- `$CLAUDE_ENV_FILE` — env file path (SessionStart only)
- `$CLAUDE_CODE_REMOTE` — remote mode flag

**Agent definitions (`.md` files in `agents/`):**
```yaml
---
name: string       # Required, lowercase-hyphens, 3-50 chars
description: string # Required, 10-5000 chars, with <example> blocks
model: inherit | sonnet | opus | haiku  # Required
color: blue | cyan | green | yellow | magenta | red  # Required
tools: ["Tool1", "Tool2"]  # Optional, omit for all tools
---

System prompt / instructions in markdown body...
```

The `description` field with `<example>` blocks determines when Claude auto-triggers the agent. The LLM decides which agent to use based on matching the user's request to example blocks.

**Command definitions (`.md` files in `commands/`):**
```yaml
---
name: string                    # Optional
description: string             # Shown in /help
allowed-tools: string | array   # Tool restrictions
model: sonnet | opus | haiku    # Model override
argument-hint: string           # Hint for arguments
disable-model-invocation: boolean
---

Command prompt/instructions in markdown body...

Supports:
- $ARGUMENTS — all arguments as string
- $1, $2, ... — positional arguments
- @file — file reference (content injected)
- `!command` — bash execution (output injected)
```

**MCP server types:**
- `stdio` — local process (most common)
- `sse` — hosted with OAuth
- `http` — REST with tokens
- `websocket` — real-time

MCP tool naming: `mcp__plugin_<plugin-name>_<server-name>__<tool-name>`

**Plugin settings pattern:**
- `.claude/plugin-name.local.md` files with YAML frontmatter + markdown body
- Read by hooks, commands, and agents at runtime
- Local (`.local.md`) files are gitignored for user-specific settings

**Marketplace format** (`.claude-plugin/marketplace.json`):
```json
{
  "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
  "name": "string",
  "version": "string",
  "description": "string",
  "owner": { "name": "string", "email": "string" },
  "plugins": [
    {
      "name": "string",
      "description": "string",
      "version": "string",
      "author": { "name": "string", "email": "string" },
      "source": "./relative/path",
      "category": "development | productivity | learning | security"
    }
  ]
}
```

**12 plugins in marketplace:** agent-sdk-dev, claude-opus-4-5-migration, code-review, commit-commands, explanatory-output-style, feature-dev, frontend-design, hookify, learning-output-style, plugin-dev, pr-review-toolkit, ralph-wiggum, security-guidance.

### OpenClaw Plugins (Extensions)

**Source files:**
- `src/plugins/types.ts` — `OpenClawPluginApi`, all hook types, `PluginDefinition` (762 lines)
- `src/plugins/loader.ts` — Plugin loading with jiti TypeScript transpilation (672 lines)
- `src/plugins/discovery.ts` — Plugin candidate discovery + security checks (605 lines)
- `src/plugins/manifest.ts` — `openclaw.plugin.json` parsing (151 lines)
- `src/plugins/registry.ts` — Central plugin registry (519 lines)
- `src/plugins/tools.ts` — Plugin tool resolution for agent context (134 lines)
- `src/plugins/hooks.ts` — Hook runner with void/modifying patterns (748 lines)
- `src/hooks/plugin-hooks.ts` — Plugin hook loading from directories (116 lines)
- `extensions/voice-call/` — Full plugin example

**Plugin manifest (`openclaw.plugin.json`):**
```json
{
  "id": "string (required)",
  "name": "string",
  "description": "string",
  "version": "string",
  "kind": "memory",
  "channels": ["string"],
  "providers": ["string"],
  "skills": ["./relative/path"],
  "uiHints": {},
  "configSchema": {
    "type": "object",
    "properties": { ... }
  }
}
```

Only `id` and `configSchema` are truly required. The `kind: "memory"` designates memory-slot plugins with deduplication.

**Plugin entry point:**
```typescript
// index.ts
import type { OpenClawPluginDefinition } from "openclaw/plugin-sdk";

export default {
  register(api) {
    // Register tools, hooks, services, channels, etc.
    api.registerTool({ ... });
    api.registerHook("before_tool_call", handler);
    api.registerService("my-service", serviceFactory);
  }
} satisfies OpenClawPluginDefinition;
```

**Plugin API — 10 registration methods** (`OpenClawPluginApi` in `types.ts` lines 244-283):

| Method | Purpose |
|--------|---------|
| `registerTool()` | Add a new tool the LLM can call |
| `registerHook()` | Intercept agent lifecycle events |
| `registerHttpHandler()` | Add HTTP request handler |
| `registerHttpRoute()` | Add HTTP route |
| `registerChannel()` | Add messaging channel (Discord, Slack, etc.) |
| `registerGatewayMethod()` | Add gateway method |
| `registerCli()` | Add CLI commands |
| `registerService()` | Add background service |
| `registerProvider()` | Add LLM provider |
| `registerCommand()` | Add slash command |
| `on()` | Listen to typed lifecycle events |

**22 hook types** (`types.ts` lines 298-322):

| Category | Hooks |
|----------|-------|
| Agent lifecycle | `before_agent_start`, `agent_end`, `before_reset` |
| LLM | `before_model_resolve`, `before_prompt_build`, `llm_input`, `llm_output` |
| Compaction | `before_compaction`, `after_compaction` |
| Tools | `before_tool_call`, `after_tool_call`, `tool_result_persist` |
| Messages | `message_received`, `message_sending`, `message_sent`, `before_message_write` |
| Sessions | `session_start`, `session_end` |
| Subagents | `subagent_spawning`, `subagent_delivery_target`, `subagent_spawned`, `subagent_ended` |
| Gateway | `gateway_start`, `gateway_stop` |

**Hook execution model:**
- **Void hooks** (e.g., `agent_end`, `message_sent`): run in **parallel**, fire-and-forget
- **Modifying hooks** (e.g., `before_tool_call`, `llm_input`): run **sequentially**, each can modify the payload
- Priority-ordered execution within each type

**Plugin discovery — 4 sources** (`discovery.ts` lines 537-605):
1. Config `loadPaths` (explicit paths)
2. Workspace `.openclaw/extensions/`
3. Global `~/.openclaw/extensions/`
4. Bundled plugins directory

**Security checks:**
- Source can't escape root directory
- No world-writable paths
- Ownership verification

**Plugin loading** (`loader.ts` lines 334-672):
- Uses **jiti** for TypeScript transpilation (no build step needed)
- Resolves `openclaw/plugin-sdk` via alias
- Validates config against JSON Schema from manifest
- Checks enable state
- Handles memory slot deduplication
- Creates API instance, calls `register()`
- Results cached by workspace + config key
- Global registry stored via `Symbol.for("openclaw.pluginRegistryState")`

**Plugin-contributed skills:**
- Plugins declare `skills` paths in their manifest
- These are merged into skill discovery as additional directories
- Resolved via `resolvePluginSkillDirs()` in `src/agents/skills/plugin-skills.ts`

**Example plugins found in `extensions/`:**
voice-call, synology-chat, nostr, memory-core, copilot-proxy, imessage, twitch, matrix, msteams, zalo, zalouser

---

## 9. Key Takeaways for Aether

### Core Loop Pattern

```go
func (a *Agent) Run(ctx context.Context, userMsg string) <-chan Event {
    messages := a.loadHistory()
    messages = append(messages, UserMessage(userMsg))

    for {
        stream := a.provider.Stream(ctx, messages, a.tools)
        assistantMsg, toolCalls := a.consumeStream(stream)  // emits SSE events
        messages = append(messages, assistantMsg)

        if len(toolCalls) == 0 {
            break  // done — pure text response
        }

        results := a.executeTools(ctx, toolCalls)  // parallel via goroutines
        messages = append(messages, ToolResults(results)...)
    }
}
```

### Design Decisions Matrix

| Concern | Pattern to Adopt | Source |
|---------|-----------------|--------|
| Core loop | `while(true)` — send → stream → detect tool_use → execute → append → check exit → loop | All three |
| Exit condition | Break when assistant response has NO tool calls (pure text = done) | All three |
| Tool execution | Tools are functions with defined input schemas. LLM returns `tool_use` blocks with name + JSON args. | All three |
| Parallel tools | Multiple tool_use blocks in one response → execute concurrently (goroutines) | OpenCode (batch), Claude Code |
| Context management | Track token count. Near limit → compact (summarize old messages, prune tool outputs) | All three |
| Streaming | Stream SSE events to client. Handle text chunks and tool calls interleaved. | All three |
| Error handling | Exponential backoff for rate limits/transient. Compact (don't retry) for context overflow. | OpenCode, OpenClaw |
| Provider abstraction | Common interface: `Stream(messages, tools, config) → EventStream`. Provider-specific adapters. | All three |
| Doom loop prevention | Track last N tool calls. If identical call 3+ times → inject error, break. | OpenCode, OpenClaw |
| Hook/middleware | Before/after tool execution hooks for permissions, modification, observation. | Claude Code, OpenClaw |
| Subagents | Task tool creates child session with own context, tools, and model. | All three |

### Skills Design for Aether

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Skill format | `SKILL.md` with YAML frontmatter | Universal across all three — proven, simple, human-editable |
| Loading strategy | Progressive: metadata always → full content on demand | All three use this — keeps base prompt small |
| Discovery | Multi-source filesystem scan (bundled → user → workspace) | OpenClaw's 6-source model is most mature |
| Skill tool | Yes — LLM can search/read/load skills on demand | OpenCode + Claude Code both have this |
| Bundled skills | Yes — ship common skills (like OpenClaw's ~70) | Aether's plugins (gmail, weather, etc.) map well to skills |
| Eligibility checks | OS, required binaries, env vars, config paths | OpenClaw's `shouldIncludeSkill()` pattern |
| Context budget | Max token/char budget with binary search fitting | OpenClaw's approach (max 150 skills, 30K chars) |

### Plugins Design for Aether

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Plugin model | Go interface with `Register(api PluginAPI)` | OpenClaw's pattern adapted for Go |
| Plugin manifest | JSON manifest (`aether.plugin.json`) | All three use JSON manifests |
| Hook system | Typed hooks: `before_tool_call`, `after_tool_call`, `before_prompt`, `session_start/end` | OpenClaw has the richest (22 hooks) — start with essential subset |
| Hook execution | Void hooks parallel, modifying hooks sequential | OpenClaw's proven pattern |
| Plugin capabilities | Register tools, hooks, HTTP handlers, services | OpenClaw's `registerX()` API is cleanest |
| Plugin discovery | Config paths → workspace dir → global dir → bundled | Standard across all three |
| Security | Path validation, no world-writable dirs | OpenClaw's security checks |
| Plugin-contributed skills | Plugins declare skill dirs in manifest | Both OpenCode and OpenClaw support this |

### Aether-Specific Considerations

Since Aether is a **personal assistant** (not a coding agent), the skills/plugins system should emphasize:

1. **Domain skills** over coding skills — calendar management, email drafting, smart home control, health tracking
2. **Plugin = integration** — each external service (Gmail, Calendar, Weather, Contacts) is a plugin that registers tools + skills
3. **Skills = persona/behavior** — teach the LLM how to interact with the user (preferences, communication style, routines)
4. **Proactive skills** — skills that trigger on schedule (morning briefing, notification digest) rather than only on user request
5. **Device-aware skills** — skills that know which device the user is on (phone vs desktop vs voice) and adapt behavior

---

## 10. Voice Pipeline Architecture

> Research findings from analyzing LiveKit Agents, Pipecat, and Vocode — three Python frameworks for building voice AI agents. Each implements the same core problem: bidirectional voice conversation with an LLM, including tool calling during speech.

### The Two Modes Every Framework Supports

All three frameworks support two fundamentally different voice architectures:

**Cascade (STT → LLM → TTS):**
```
Microphone → VAD → STT → [text] → LLM → [text] → TTS → Speaker
                                      ↓
                                 Tool calls → execute → feed back → LLM continues
```
- User speech is transcribed to text, sent to a text-based LLM, response synthesized to speech
- Higher latency (~1-3s round trip) but works with ANY LLM
- Tool calling is standard text-based function calling
- Each component (STT, LLM, TTS) is independently swappable

**Realtime (Native Multimodal):**
```
Microphone → [audio] → Realtime Model → [audio] → Speaker
                              ↓
                         Tool calls → execute → feed back → model continues
```
- Audio sent directly to a multimodal model (OpenAI Realtime API, Gemini Multimodal Live)
- Lower latency (~300-800ms) but limited to models with native audio support
- Tool calling happens mid-conversation with audio still flowing
- The model handles VAD, transcription, and synthesis internally

### Framework Comparison

| Aspect | LiveKit Agents | Pipecat | Vocode |
|--------|---------------|---------|--------|
| **Architecture** | Object-oriented — `AgentActivity` class with dual pipeline methods | Frame-based — data flows as `Frame` objects through `FrameProcessor` chains | Worker-based — `AsyncWorker` instances communicate via async queues |
| **Cascade mode** | `_pipeline_reply_task()` in `agent_activity.py` | Pipeline of `STTService` → `LLMService` → `TTSService` processors | `StreamingConversation` orchestrator with Transcriber → Agent → Synthesizer workers |
| **Realtime mode** | `_realtime_reply_task()` with `RealtimeModel`/`RealtimeSession` abstract interface | Single `OpenAIRealtimeLLMService` or `GeminiMultimodalLiveLLMService` replacing entire cascade | Not supported (cascade only) |
| **Transport** | LiveKit WebRTC rooms (`RoomIO`) | LiveKit, Daily, SmallWebRTC, WebSocket, Local | Twilio WebSocket, LiveKit, Vonage, Local |
| **Turn detection** | Multiple modes: `"stt"`, `"vad"`, `"realtime_llm"`, `"manual"`, custom `_TurnDetector` | Pluggable strategies: VAD start + `SmartTurnAnalyzerV3` stop | VAD-based with backchannel detection |
| **Tool calling** | `@function_tool` decorator, concurrent execution via goroutines, tools run alongside audio playout | `llm.register_function(name, handler)`, parallel execution, 10s timeout | `BaseAction` with `ActionTrigger` (function call or phrase-based), `ActionsWorker` queue |
| **Interruption** | `SpeechHandle` state machine with priority system (LOW=0, NORMAL=5, HIGH=10) | `SystemFrame` (uninterruptible) vs `DataFrame` (cancellable) vs `ControlFrame` | `InterruptibleEvent` wrapper with threading.Event flag, `broadcast_interrupt()` |
| **Maturity** | Production (LiveKit ecosystem) | Production (framework-agnostic) | Moderate (telephony-focused) |
| **Language** | Python | Python | Python |

### Pipeline Data Flow Patterns

**LiveKit Agents — Method-based:**
```
AgentSession.say() / generate_reply()
  → AgentActivity._pipeline_reply_task()
    → perform_llm_inference()     → LLM stream
    → text tee'd to:
      → transcription callback    → emit transcript events
      → perform_tts_inference()   → audio output via RoomIO
    → if tool calls detected:
      → perform_tool_executions() → concurrent goroutines
      → tool results appended     → trigger new reply task
```

**Pipecat — Frame-based:**
```
Transport.input
  → AudioRawFrame → STTService → TranscriptionFrame
  → LLMUserAggregator → LLMContext → LLMService
  → TextFrame / FunctionCallInProgressFrame
  → LLMAssistantAggregator → context update
  → TTSService → TTSAudioRawFrame
  → Transport.output
```

**Vocode — Queue-based:**
```
InputDevice → audio chunks → Transcriber (AsyncWorker)
  → TranscriptionsWorker → partial/final transcripts
  → RespondAgent (InterruptibleWorker) → LLM response chunks
  → AgentResponsesWorker → synthesis requests
  → Synthesizer → SynthesisResult (audio)
  → SynthesisResultsWorker → OutputDevice
  
  Side channel: ActionsWorker → tool execution → ActionResultAgentInput → Agent
```

---

## 11. Voice Pipeline Deep Dives

### 11.1 LiveKit Agents

**Core source:** `reference_projects/agents/livekit-agents/livekit/agents/voice/`

#### Agent Lifecycle

| Component | File | Purpose |
|-----------|------|---------|
| `AgentSession` | `agent_session.py` | Top-level session — owns activity, manages start/stop/shutdown |
| `AgentActivity` | `agent_activity.py` (2848 lines) | Core pipeline — dual mode (cascade + realtime), tool execution, speech handles |
| `VoiceAgent` | `agent.py` | User-facing config — model, tools, instructions, turn detection, hooks |
| `RoomIO` | `room_io/` | WebRTC bridge — subscribes to participant audio tracks, publishes agent audio |

**Startup flow:**
1. `AgentSession.start()` → creates `AgentActivity`
2. `RoomIO` connects to LiveKit room, subscribes to participant audio tracks
3. `AudioRecognition` starts VAD + STT pipeline
4. User speaks → VAD detects voice → STT transcribes → triggers reply

#### Cascade Pipeline (`_pipeline_reply_task`, lines 1866-2255)

```python
async def _pipeline_reply_task(self, handle: SpeechHandle):
    # 1. Build LLM input from chat context
    llm_input = self._build_llm_input()
    
    # 2. Stream LLM response
    llm_stream = self.perform_llm_inference(llm_input)
    
    # 3. Tee text output: one copy for transcription, one for TTS
    async for chunk in llm_stream:
        if chunk.is_text:
            tts_input.push(chunk.text)        # → TTS pipeline
            transcript.append(chunk.text)      # → transcript events
        elif chunk.is_tool_call:
            pending_tools.append(chunk)
    
    # 4. TTS synthesis runs concurrently
    tts_stream = self.perform_tts_inference(tts_input)
    # Audio frames pushed to RoomIO output
    
    # 5. Execute tool calls (concurrent)
    if pending_tools:
        results = await self.perform_tool_executions(pending_tools)
        # Append tool results to context
        # Trigger new reply task (loop!)
```

**Key details:**
- LLM text is "tee'd" — simultaneously fed to TTS for synthesis AND collected for transcript events
- TTS runs concurrently with LLM streaming (synthesis starts before LLM finishes)
- Tool calls detected during streaming trigger `perform_tool_executions()`
- Tool results get appended to chat context and trigger a NEW reply task (recursive loop)
- Max tool steps configurable to prevent infinite tool loops

#### Realtime Pipeline (`_realtime_reply_task`, lines 2257-2768)

```python
async def _realtime_reply_task(self, handle: SpeechHandle):
    # Realtime model handles everything — audio in, audio out, tool calls
    session: RealtimeSession = self._realtime_session
    
    # Audio from participant pushed directly to model
    session.push_audio(audio_frame)
    
    # Model generates events:
    async for event in session.events:
        if isinstance(event, GenerationCreatedEvent):
            # event.message_stream → text/audio chunks
            # event.function_stream → tool calls
            pass
```

**RealtimeModel interface** (`llm/realtime.py`):

```python
class RealtimeCapabilities:
    message_truncation: bool    # Can truncate messages for context management
    turn_detection: bool        # Model does its own VAD
    user_transcription: bool    # Model transcribes user speech
    audio_output: bool          # Model generates audio directly
    manual_function_calls: bool # Tools need manual result feeding

class RealtimeSession(ABC):
    async def push_audio(self, frame: AudioRawFrame): ...
    async def generate_reply(self, instructions: str = ""): ...
    async def interrupt(self): ...
    async def truncate(self, *, message_id: str, audio_end_ms: int): ...
    async def update_instructions(self, instructions: str): ...
    async def update_tools(self, tools: list): ...
```

**OpenAI Realtime plugin** (`livekit-plugins/livekit-plugins-openai/realtime/realtime_model.py`, 1762 lines):
- WebSocket connection to `wss://api.openai.com/v1/realtime`
- Auto-reconnection: OpenAI sessions expire after ~20 minutes
- Audio format: 24kHz, 16-bit PCM, mono, base64-encoded
- Server-side VAD with configurable silence threshold
- `GenerationCreatedEvent` provides `message_stream` (text/audio) + `function_stream` (tool calls)
- Tool results fed back via `conversation.item.create` + `response.create` WebSocket messages

#### Tool System

**`@function_tool` decorator** (`llm/tool_context.py`):
```python
@function_tool()
async def get_weather(location: str) -> str:
    """Get weather for a location.
    
    Args:
        location: City name or zip code
    """
    return await weather_api.fetch(location)
```
- Auto-generates JSON Schema from function signature + docstring
- Docstring parsed for parameter descriptions
- Return type used for response format
- Tools registered on `VoiceAgent` and passed to LLM

**Concurrent tool execution** (`agent_activity.py`):
- Multiple tool calls in one LLM response execute concurrently via `asyncio.gather()`
- Tools run ALONGSIDE audio playout (agent keeps speaking while tools execute)
- Tool results trigger a new reply task — the agent speaks the tool result
- `SpeechHandle` manages priority: tool-result speech can interrupt current speech

#### SpeechHandle State Machine (`speech_handle.py`)

```
IDLE → SPEAKING → {INTERRUPTED | DONE}
```

| Property | Purpose |
|----------|---------|
| `priority` | LOW=0, NORMAL=5, HIGH=10 — higher priority interrupts lower |
| `interrupt_future` | Resolves when speech is interrupted |
| `done_future` | Resolves when speech completes |
| `scheduled_future` | Resolves when speech is scheduled for playout |
| `interruption_timeout` | 5 seconds — prevents stuck handles |

**Priority system:** If a new speech handle has higher priority than the current one, the current speech is interrupted. Tool-result speech typically gets NORMAL priority, while explicit `agent.say()` can specify priority.

**Deadlock prevention:** If a tool calls `wait_for_playout()` on its own speech handle, it would deadlock (tool must complete before speech plays, but speech waits for tool). LiveKit detects this and logs a warning.

#### Turn Detection (`audio_recognition.py`)

| Mode | How It Works |
|------|-------------|
| `"stt"` | STT service detects end-of-utterance (most common for cascade) |
| `"vad"` | Voice Activity Detection — silence after speech triggers turn end |
| `"realtime_llm"` | The realtime model itself decides when user is done (server-side VAD) |
| `"manual"` | Application code explicitly signals turn boundaries |
| Custom | Implement `_TurnDetector` interface |

`AudioRecognition` orchestrates VAD + STT:
1. VAD detects voice activity start → `_on_start_of_speech()`
2. Audio streamed to STT during voice activity
3. VAD detects silence → start end-of-turn timer
4. Timer expires OR STT signals end-of-utterance → `_on_end_of_turn()`
5. Final transcript emitted → triggers reply task

#### Transport — RoomIO (`room_io/`)

**Input** (`_ParticipantAudioInputStream`):
- Subscribes to LiveKit participant's audio track via `rtc.AudioStream.from_track()`
- Resamples audio to match agent's expected sample rate
- Feeds audio frames to `AudioRecognition` pipeline

**Output** (`_ParticipantAudioOutput`):
- Creates a local `rtc.AudioSource`
- Publishes as a LiveKit audio track
- TTS audio frames pushed to this source
- Supports interruption (stops pushing frames mid-utterance)

---

### 11.2 Pipecat

**Core source:** `reference_projects/pipecat/src/pipecat/`

#### Frame Architecture

Everything in Pipecat is a **Frame**. Frames flow through a pipeline of **FrameProcessors**, each transforming or routing frames.

**Frame hierarchy** (`frames/frames.py`):

| Category | Base Class | Behavior | Examples |
|----------|-----------|----------|----------|
| **System** | `SystemFrame` | High priority, uninterruptible, bypass queues | `StartFrame`, `EndFrame`, `CancelFrame`, `MetricsFrame` |
| **Control** | `ControlFrame` | Ordered, not cancelled by interruptions | `EndOfTurnFrame`, `StartInterruptionFrame`, `StopInterruptionFrame`, `UserStartedSpeakingFrame` |
| **Data** | `DataFrame` | Ordered, cancelled by interruptions | `TextFrame`, `AudioRawFrame`, `TranscriptionFrame`, `FunctionCallInProgressFrame` |

**Key data frames:**
```python
AudioRawFrame(audio: bytes, sample_rate: int, num_channels: int)
TranscriptionFrame(text: str, user_id: str, timestamp: str)
TextFrame(text: str)
TTSAudioRawFrame(audio: bytes, sample_rate: int, num_channels: int)
FunctionCallInProgressFrame(function_name: str, tool_call_id: str, arguments: str)
FunctionCallResultFrame(function_name: str, tool_call_id: str, arguments: str, result: str)
# FunctionCallResultFrame extends UninterruptibleFrame — tool results can't be lost
```

#### Pipeline Construction

```python
pipeline = Pipeline([
    transport.input(),           # → AudioRawFrame, UserStartedSpeakingFrame
    stt_service,                 # AudioRawFrame → TranscriptionFrame
    user_aggregator,             # TranscriptionFrame → LLMMessagesAppendFrame
    llm_service,                 # LLMContext → TextFrame, FunctionCallInProgressFrame
    tts_service,                 # TextFrame → TTSAudioRawFrame
    transport.output(),          # TTSAudioRawFrame → speaker/WebRTC
    assistant_aggregator,        # Tracks assistant response in context
])

task = PipelineTask(pipeline)
runner = PipelineRunner()
await runner.run(task)
```

**Processor linking:** `PipelineSource` → processors linked via `prev.link(curr)` → `PipelineSink`. Each processor has `process_frame(frame, direction)` — frames flow downstream by default but can flow upstream (e.g., interruption signals).

**Parallel pipelines:** `ParallelPipeline` runs multiple processor chains concurrently, merging outputs.

#### Services

All AI services extend `AIService` → `FrameProcessor`:

**STT** (`services/stt_service.py`):
- Input: `AudioRawFrame`
- Output: `TranscriptionFrame` (interim/final)
- Implementations: Deepgram, Whisper, Google, Azure, AssemblyAI, etc.

**LLM** (`services/llm_service.py`):
- Input: `LLMContext` (messages array)
- Output: `TextFrame` chunks + `FunctionCallInProgressFrame` for tool calls
- Implementations: OpenAI, Anthropic, Google, Together, Fireworks, etc.
- `register_function(name, handler)` for tool callbacks

**TTS** (`services/tts_service.py`):
- Input: `TextFrame`
- Output: `TTSAudioRawFrame`
- Implementations: ElevenLabs, Cartesia, Deepgram, Azure, Google, PlayHT, XTTS, etc.
- Sentence-level aggregation before synthesis (configurable)

#### OpenAI Realtime Service (`services/openai/realtime/llm.py`)

Replaces the entire STT → LLM → TTS cascade with a single service:

```python
class OpenAIRealtimeLLMService(LLMService):
    # WebSocket connection to OpenAI Realtime API
    # Sends audio frames directly, receives audio + text + tool calls
    
    async def _connect(self):
        self._ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
            extra_headers={"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}
        )
    
    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            # Encode to base64, send as input_audio_buffer.append
            await self._send_audio(frame)
        elif isinstance(frame, LLMMessagesFrame):
            # Send conversation context update
            await self._update_conversation(frame)
```

**Audio handling:**
- Input: 24kHz, 16-bit PCM, mono → base64 encoded
- Output: base64 decoded → `TTSAudioRawFrame`
- Server-side VAD detects speech boundaries

**Tool calls:**
- Tracked via `_pending_function_calls` dict
- `response.function_call_arguments.done` → execute handler → send `conversation.item.create` with result
- `FunctionCallResultFrame` is `UninterruptibleFrame` (won't be lost on interruption)

#### Gemini Multimodal Live Service (`services/google/gemini_live/llm.py`)

```python
class GeminiMultimodalLiveLLMService(LLMService):
    # Uses Google genai SDK: client.aio.live.connect()
    # Supports: thinking, affective dialog, proactivity, session resumption
    
    async def _connect(self):
        async with self._client.aio.live.connect(model=self._model, config=config) as session:
            self._session = session
```

**Unique capabilities:**
- **Context window compression** (`SlidingWindow`): automatically compresses old context
- **Thinking/reasoning**: `thinking_config` with configurable budget
- **Affective dialog**: emotional awareness in responses
- **Proactivity**: model can initiate conversation
- **Session resumption**: reconnect to existing session
- **Grounding with Google Search**: model can search the web mid-response
- **Reconnection logic**: max 3 consecutive failures before giving up

#### Context Aggregators (`processors/aggregators/llm_response_universal.py`)

The glue between STT/LLM/TTS — maintains conversation context:

**`LLMContextAggregatorPair`** creates two linked aggregators:

| Aggregator | Role |
|-----------|------|
| `LLMUserAggregator` | Collects transcription frames → builds user message → appends to `LLMContext` → triggers LLM |
| `LLMAssistantAggregator` | Collects LLM response text → builds assistant message → appends to `LLMContext` → manages function call lifecycle |

**`LLMContext`** — universal context container:
```python
class LLMContext:
    messages: list[dict]          # Conversation history
    tools: list[dict]             # Available tools (JSON Schema)
    system_prompt: str            # System instructions
    tool_choice: str | dict       # "auto" | "required" | {"type": "function", "function": {"name": "..."}}
```

#### Turn Management (`turns/`)

**Pluggable strategy pattern:**

```python
class TurnStrategy:
    start_strategy: StartStrategy      # When to start listening
    stop_strategy: StopStrategy        # When user turn ends
    mute_strategy: MuteStrategy        # When to mute input
```

| Strategy | Behavior |
|----------|----------|
| `VADStartStrategy` | Start on voice activity detection |
| `SmartTurnAnalyzerV3` | ML-based end-of-turn detection (considers pauses, sentence completeness) |
| `ExternalStartStrategy` | For realtime APIs — server decides |
| `ExternalStopStrategy` | For realtime APIs — server signals turn end |
| `ManualStopStrategy` | Application code signals turn end |

**Interruption flow:**
1. User starts speaking → `UserStartedSpeakingFrame` (upstream)
2. `StartInterruptionFrame` cancels all in-flight `DataFrame`s
3. TTS stops, LLM generation cancelled
4. When user stops → new turn begins → fresh LLM call with updated context

#### Transports

**Three-layer architecture:**

```
Transport (public API)
  ├── InputTransport (mic/WebRTC → frames)
  ├── OutputTransport (frames → speaker/WebRTC)
  └── Client (protocol-specific connection)
```

| Transport | Client | Use Case |
|-----------|--------|----------|
| `LiveKitTransport` | LiveKit SDK | WebRTC rooms |
| `DailyTransport` | Daily SDK | WebRTC rooms |
| `SmallWebRTCTransport` | Custom | Lightweight WebRTC (no SFU) |
| `WebSocketTransport` | aiohttp | Browser/telephony WebSocket |
| `LocalTransport` | PyAudio | Local microphone/speaker |

---

### 11.3 Vocode

**Core source:** `reference_projects/vocode-core/vocode/streaming/`

#### Worker Architecture

Vocode uses an `AsyncWorker` pattern — each pipeline stage is a worker with input/output queues:

```python
class AsyncWorker:
    input_queue: asyncio.Queue
    output_queue: asyncio.Queue
    
    async def _run_loop(self):
        while True:
            item = await self.input_queue.get()
            await self.process(item)

class InterruptibleWorker(AsyncWorker):
    # Wraps queue items in InterruptibleEvent
    # Items can be cancelled mid-processing via threading.Event
    
class AsyncQueueWorker(AsyncWorker):
    # Convenience: output is always a queue
```

#### StreamingConversation — Central Orchestrator

`streaming_conversation.py` is the heart of Vocode. It creates and wires all workers:

```
┌─────────────┐     ┌──────────────────────┐     ┌──────────────┐
│ InputDevice  │────→│ TranscriptionsWorker │────→│ RespondAgent │
│ (mic/WebRTC) │     │ (partial → final)    │     │ (LLM calls)  │
└─────────────┘     └──────────────────────┘     └──────┬───────┘
                                                         │
                    ┌──────────────────────┐             │
                    │ AgentResponsesWorker │←────────────┘
                    │ (chunk → synth req)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────────────────────┐
                    │ SynthesisResultsWorker   │
                    │ (audio → output device)  │
                    └──────────┬───────────────┘
                               │
                    ┌──────────────────┐
                    │ OutputDevice     │
                    │ (speaker/Twilio) │
                    └──────────────────┘

  Side channel:
  ┌───────────────┐
  │ ActionsWorker │ ← tool calls from Agent
  │ (execute)     │ → ActionResultAgentInput back to Agent
  └───────────────┘
```

**Conversation lifecycle:**
1. `StreamingConversation.start()` → creates all workers, starts transcriber
2. Audio arrives from input device → transcriber produces partial/final transcripts
3. `TranscriptionsWorker` filters partials, sends finals to agent
4. `RespondAgent` calls LLM, streams response chunks
5. `AgentResponsesWorker` batches chunks, sends to synthesizer
6. `SynthesisResultsWorker` plays audio through output device
7. On interruption: `broadcast_interrupt()` drains queues, cancels in-flight work

#### Actions (Tool Calling)

Vocode's equivalent of tools — called "Actions":

```python
class BaseAction(Generic[ActionConfigType, ActionParamsType, ActionResponseType]):
    action_config: ActionConfigType
    
    async def run(self, params: ActionParamsType) -> ActionResponse:
        # Execute the action
        ...

class ActionTrigger:
    # Two trigger modes:
    FunctionCallActionTrigger  # LLM calls a function → action executes
    PhraseBasedActionTrigger   # User says a phrase → action triggers
```

**Built-in actions:**
| Action | Purpose |
|--------|---------|
| `EndConversation` | Gracefully end the call |
| `TransferCall` | Transfer to another phone number (telephony) |
| `DTMFAction` | Send DTMF tones (telephony) |
| `ExecuteExternalAction` | Call an external HTTP endpoint |

**Action execution flow:**
1. LLM response contains function call → `ActionsWorker` picks it up
2. Action's `run()` method executes
3. Result wrapped as `ActionResultAgentInput`
4. Fed back to `RespondAgent` as a new input
5. Agent incorporates result and continues responding

**Phrase-based triggers** — unique to Vocode:
- Instead of LLM function calling, detect specific phrases in user speech
- Example: user says "transfer me to a human" → `TransferCall` action triggers
- Useful for telephony where latency of LLM function calling is too slow

#### Interruption System

```python
class InterruptibleEvent(Generic[T]):
    payload: T
    is_interruptible: bool
    interrupt_event: threading.Event  # Set this to cancel
    
    def interrupt(self):
        self.interrupt_event.set()
```

**`broadcast_interrupt()` flow:**
1. User starts speaking mid-response → `broadcast_interrupt()` called
2. All worker queues drained
3. All in-flight `InterruptibleEvent`s have their `interrupt_event` set
4. Workers check `interrupt_event` before processing → skip cancelled items
5. Current synthesis/playout cancelled
6. Agent receives interruption signal → stops generating

**Backchannel detection:**
- Short utterances ("uh-huh", "yeah", "ok") during agent speech → NOT treated as interruptions
- Prevents premature interruption on listener feedback
- Configurable via `backchannel_config`

#### Filler Audio (`output_device/filler_audio_worker.py`)

Unique to Vocode — fills silence while waiting for LLM response:

| Filler Type | When |
|-------------|------|
| "um...", "uh..." | Thinking delay |
| "uh-huh" | Acknowledgment |
| Typing noise | Processing indicator |
| Custom audio | Configurable |

- `FillerAudioWorker` tracks time since last agent audio
- If threshold exceeded → plays filler audio
- Stops immediately when real response arrives
- Creates more natural conversation feel for telephony

#### Transports

**Twilio (`telephony/`):**
- WebSocket connection: Twilio streams μ-law audio at 8kHz
- Mark-based playback tracking: Twilio sends "mark" events when audio segments finish playing
- `TwilioOutputDevice` tracks marks for accurate interruption (knows exactly what user heard)
- `TelephonyServer` handles incoming calls, creates `StreamingConversation` per call

**LiveKit (`output_device/livekit_output_device.py`):**
- WebRTC transport via LiveKit SDK
- Audio published as LiveKit track
- Participant audio subscribed as input

**Vonage:**
- WebSocket-based, similar to Twilio
- Different audio encoding (PCM vs μ-law)

**Local (`output_device/speaker_output.py`):**
- PyAudio for local microphone/speaker
- Used for development/testing

---

## 12. Voice Pipeline Takeaways for Aether

### Architecture Decision Matrix

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| **Dual-mode support** | Yes — both cascade AND realtime | All three frameworks support both (Vocode only cascade). Realtime for low-latency, cascade for flexibility. Start with cascade, add realtime. |
| **Pipeline pattern** | Frame/event-based (Pipecat-inspired) | Most composable. Frames flow through processor chains. Easy to add/swap components. LiveKit's method-based is simpler but less flexible. Vocode's worker queues add complexity without benefit. |
| **Transport abstraction** | Interface-based with WebRTC primary | LiveKit's `RoomIO` pattern for WebRTC. Abstract transport so HTTP/WebSocket/telephony can be added later. |
| **Turn detection** | Pluggable strategy pattern | Pipecat's `TurnStrategy` (start + stop + mute strategies) is cleanest. Support VAD, STT-based, realtime model, and manual modes. |
| **Tool execution** | Concurrent, alongside audio playout | LiveKit's pattern: tools execute while agent speech continues. Tool results trigger new reply. Don't block audio for tools. |
| **Interruption** | Priority-based speech handles | LiveKit's `SpeechHandle` with priority levels. Higher priority speech interrupts lower. Tool results get NORMAL priority, explicit `say()` gets configurable priority. |
| **Context management** | Universal context container | Pipecat's `LLMContext` pattern — single object holding messages, tools, system prompt. Aggregators update it from both user (STT) and assistant (LLM response) sides. |
| **Filler audio** | Optional, for telephony | Vocode's filler audio is great UX for phone calls. Not needed for WebRTC (lower latency). Implement as optional processor in pipeline. |

### Realtime Model Interface for Go

Based on LiveKit's `RealtimeModel`/`RealtimeSession` — the cleanest abstraction found:

```go
type RealtimeCapabilities struct {
    MessageTruncation   bool  // Can truncate messages for context management
    TurnDetection       bool  // Model handles its own VAD
    UserTranscription   bool  // Model transcribes user speech
    AudioOutput         bool  // Model generates audio directly
    ManualFunctionCalls bool  // Tools need manual result feeding
}

type RealtimeSession interface {
    PushAudio(ctx context.Context, frame AudioFrame) error
    GenerateReply(ctx context.Context, instructions string) error
    Interrupt(ctx context.Context) error
    Truncate(ctx context.Context, messageID string, audioEndMs int) error
    UpdateInstructions(ctx context.Context, instructions string) error
    UpdateTools(ctx context.Context, tools []ToolDef) error
    Events() <-chan RealtimeEvent  // GenerationCreated, FunctionCall, Error, etc.
    Capabilities() RealtimeCapabilities
    Close() error
}

type RealtimeModel interface {
    NewSession(ctx context.Context, opts SessionOpts) (RealtimeSession, error)
}
```

### Cascade Pipeline Interface for Go

Based on Pipecat's processor chain pattern:

```go
type Frame interface {
    FrameType() FrameCategory  // System, Control, Data
}

type FrameProcessor interface {
    ProcessFrame(ctx context.Context, frame Frame, dir Direction) error
    Link(next FrameProcessor)
}

type Pipeline struct {
    processors []FrameProcessor
}

// Services implement FrameProcessor
type STTService interface {
    FrameProcessor
    // AudioRawFrame → TranscriptionFrame
}

type LLMService interface {
    FrameProcessor
    RegisterFunction(name string, handler ToolHandler)
    // LLMContext → TextFrame + FunctionCallFrame
}

type TTSService interface {
    FrameProcessor
    // TextFrame → AudioRawFrame
}
```

### Tool Calling During Voice — The Critical Pattern

All three frameworks solve the same problem: what happens when the LLM wants to call a tool mid-conversation?

```
User: "What's the weather in Tokyo?"

Cascade mode:
  STT: "What's the weather in Tokyo?" → text
  LLM: function_call(get_weather, {location: "Tokyo"}) → tool call detected
  Tool: execute get_weather("Tokyo") → "72°F, partly cloudy"
  LLM: "The weather in Tokyo is 72 degrees and partly cloudy." → text
  TTS: synthesize → audio
  
Realtime mode:
  Audio → Realtime Model: detects intent, emits function_call
  Tool: execute get_weather("Tokyo") → "72°F, partly cloudy"  
  Feed result back → Model continues generating audio response
  (Model may speak filler like "Let me check..." while tool executes)
```

**Key patterns across all three:**
1. **Tool results loop back** — they don't go to TTS directly; they go back to the LLM which formulates a natural language response
2. **Tools execute concurrently** with audio playout (LiveKit, Pipecat) or in a side channel (Vocode)
3. **Tool result frames are uninterruptible** (Pipecat) — if user interrupts during tool execution, the result is preserved and fed to LLM on next turn
4. **Max tool steps** prevent infinite tool loops (same doom-loop pattern as text agents)
5. **Agent can speak while tools execute** — "Let me look that up..." filler or continued speech

### Transport Layer for Aether

Aether needs WebRTC as primary transport (real-time voice with clients):

```go
type Transport interface {
    // Input: audio from user
    AudioInput() <-chan AudioFrame
    
    // Output: audio to user  
    SendAudio(frame AudioFrame) error
    
    // Lifecycle
    Start(ctx context.Context) error
    Close() error
    
    // Events
    OnParticipantConnected(handler func(participantID string))
    OnParticipantDisconnected(handler func(participantID string))
}

// Primary implementation
type LiveKitTransport struct {
    room   *lksdk.Room
    // Subscribes to participant audio tracks
    // Publishes agent audio track
}

// Future implementations
type WebSocketTransport struct { ... }   // Browser/telephony
type TwilioTransport struct { ... }      // Phone calls
```

### What Aether Should Build (Phase 1 Voice)

Based on all research, the minimum viable voice pipeline:

1. **WebRTC transport** — LiveKit room connection, audio track subscribe/publish
2. **Cascade pipeline** — STT (Deepgram/Whisper) → LLM (any text model) → TTS (ElevenLabs/Cartesia)
3. **Turn detection** — VAD-based (simple, reliable)
4. **Tool calling** — concurrent execution, results loop back to LLM
5. **Interruption** — user speech cancels in-flight TTS, fresh LLM call
6. **Context management** — shared with text agentic loop (same message format)

**Phase 2 additions:**
- Realtime model support (OpenAI Realtime API)
- Smart turn detection (ML-based end-of-turn)
- Filler audio for telephony
- Multiple transport types

---

## Reference File Paths

### OpenCode (`reference_projects/opencode_sst/`)
- `packages/opencode/src/session/prompt.ts` — Main loop entry
- `packages/opencode/src/session/processor.ts` — Stream processing loop
- `packages/opencode/src/session/llm.ts` — LLM streaming wrapper
- `packages/opencode/src/tool/tool.ts` — Tool definition
- `packages/opencode/src/tool/registry.ts` — Tool registry
- `packages/opencode/src/tool/batch.ts` — Parallel execution
- `packages/opencode/src/tool/task.ts` — Subagent tool
- `packages/opencode/src/tool/skill.ts` — Skill tool (search/read/create/install)
- `packages/opencode/src/skill/skill.ts` — Skill discovery + state
- `packages/opencode/src/skill/discovery.ts` — Remote skill registry protocol
- `packages/opencode/src/plugin/index.ts` — Plugin loader + 16 hook points
- `packages/plugin/src/index.ts` — Plugin SDK (`@opencode-ai/plugin`)
- `packages/opencode/src/mcp/index.ts` — MCP client (937 lines)
- `packages/opencode/src/session/message-v2.ts` — Message types
- `packages/opencode/src/session/compaction.ts` — Context compaction
- `packages/opencode/src/session/retry.ts` — Retry logic
- `packages/opencode/src/provider/provider.ts` — Provider abstraction
- `packages/opencode/src/agent/agent.ts` — Agent definitions

### Claude Code (`reference_projects/claude-code/`)
- `plugins/plugin-dev/skills/hook-development/SKILL.md` — Hook event reference
- `plugins/plugin-dev/skills/agent-development/SKILL.md` — Agent spec
- `plugins/plugin-dev/skills/command-development/SKILL.md` — Command format
- `plugins/plugin-dev/skills/plugin-structure/SKILL.md` — Plugin structure
- `plugins/plugin-dev/skills/skill-development/SKILL.md` — Skill authoring guide
- `plugins/plugin-dev/skills/mcp-integration/SKILL.md` — MCP server setup
- `plugins/plugin-dev/skills/plugin-settings/SKILL.md` — Plugin settings pattern
- `.claude-plugin/marketplace.json` — Marketplace manifest
- `CHANGELOG.md` — Feature/architecture changelog

### OpenClaw (`reference_projects/openclaw/`)
- `src/agents/pi-embedded-runner/run.ts` — Outer retry loop
- `src/agents/pi-embedded-runner/run/attempt.ts` — Attempt execution
- `src/agents/pi-embedded-subscribe.ts` — Event subscription
- `src/agents/pi-tools.ts` — Tool creation
- `src/agents/pi-tools.before-tool-call.ts` — Tool approval hooks
- `src/agents/pi-tools.policy.ts` — Tool policies
- `src/agents/skills/types.ts` — Skill types (SkillEntry, OpenClawSkillMetadata)
- `src/agents/skills/workspace.ts` — Skill discovery, loading, prompt building
- `src/agents/skills/config.ts` — Skill eligibility evaluation
- `src/agents/skills-install.ts` — Skill installation engine
- `src/plugins/types.ts` — Plugin API + 22 hook types
- `src/plugins/loader.ts` — Plugin loading with jiti
- `src/plugins/discovery.ts` — Plugin discovery + security
- `src/plugins/registry.ts` — Central plugin registry
- `src/plugins/hooks.ts` — Hook runner (void/modifying patterns)
- `src/agents/context-window-guard.ts` — Context guard
- `src/agents/compaction.ts` — Compaction
- `AGENTS.md` — Architecture guidelines

### LiveKit Agents (`reference_projects/agents/`)
- `livekit-agents/livekit/agents/voice/agent_activity.py` — Core dual pipeline (2848 lines): cascade `_pipeline_reply_task` + realtime `_realtime_reply_task`
- `livekit-agents/livekit/agents/voice/agent.py` — `VoiceAgent` user-facing config (model, tools, instructions, turn detection, hooks)
- `livekit-agents/livekit/agents/voice/agent_session.py` — `AgentSession` top-level session lifecycle
- `livekit-agents/livekit/agents/voice/generation.py` — Generation tracking (LLM inference + tool execution)
- `livekit-agents/livekit/agents/voice/audio_recognition.py` — `AudioRecognition` — VAD + STT orchestration, turn detection
- `livekit-agents/livekit/agents/voice/speech_handle.py` — `SpeechHandle` state machine (priority, interrupt, done futures)
- `livekit-agents/livekit/agents/voice/io.py` — I/O abstractions for agent session
- `livekit-agents/livekit/agents/voice/room_io/` — `RoomIO` WebRTC bridge (subscribe participant audio, publish agent audio)
- `livekit-agents/livekit/agents/llm/realtime.py` — `RealtimeModel`/`RealtimeSession`/`RealtimeCapabilities` abstract interface
- `livekit-agents/livekit/agents/llm/llm.py` — `LLM` base class for text-based models
- `livekit-agents/livekit/agents/llm/tool_context.py` — `@function_tool` decorator, schema generation from docstrings
- `livekit-agents/livekit/agents/stt/stt.py` — `STT` base class
- `livekit-agents/livekit/agents/tts/tts.py` — `TTS` base class
- `livekit-plugins/livekit-plugins-openai/livekit/plugins/openai/realtime/realtime_model.py` — OpenAI Realtime WebSocket implementation (1762 lines)

### Pipecat (`reference_projects/pipecat/`)
- `src/pipecat/frames/frames.py` — Frame hierarchy (SystemFrame, ControlFrame, DataFrame, AudioRawFrame, TranscriptionFrame, TextFrame, FunctionCallFrames)
- `src/pipecat/pipeline/pipeline.py` — `Pipeline` — processor chain construction and linking
- `src/pipecat/pipeline/task.py` — `PipelineTask` — orchestrates pipeline execution
- `src/pipecat/services/llm_service.py` — `LLMService` base class, `register_function()` for tool callbacks
- `src/pipecat/services/stt_service.py` — `STTService` base class (AudioRawFrame → TranscriptionFrame)
- `src/pipecat/services/tts_service.py` — `TTSService` base class (TextFrame → TTSAudioRawFrame)
- `src/pipecat/services/openai/realtime/llm.py` — `OpenAIRealtimeLLMService` — WebSocket, base64 audio, server-side VAD, tool call tracking
- `src/pipecat/services/google/gemini_live/llm.py` — `GeminiMultimodalLiveLLMService` — genai SDK, thinking, affective dialog, proactivity, session resumption, grounding
- `src/pipecat/transports/livekit/transport.py` — LiveKit WebRTC transport
- `src/pipecat/transports/daily/transport.py` — Daily WebRTC transport
- `src/pipecat/processors/aggregators/llm_response_universal.py` — `LLMContextAggregatorPair` (user + assistant aggregators), `LLMContext` universal container
- `src/pipecat/turns/` — Turn strategies: `TurnStrategy`, `VADStartStrategy`, `SmartTurnAnalyzerV3`, external/manual strategies

### Vocode (`reference_projects/vocode-core/`)
- `vocode/streaming/streaming_conversation.py` — `StreamingConversation` central orchestrator (wires all workers, manages lifecycle)
- `vocode/streaming/utils/worker.py` — `AsyncWorker`, `InterruptibleWorker`, `AsyncQueueWorker`, `InterruptibleEvent`
- `vocode/streaming/agent/base_agent.py` — `RespondAgent` base class
- `vocode/streaming/agent/chat_gpt_agent.py` — OpenAI-based agent implementation
- `vocode/streaming/action/base_action.py` — `BaseAction` generic, `ActionTrigger` (function call vs phrase-based)
- `vocode/streaming/action/end_conversation.py` — `EndConversation` action
- `vocode/streaming/action/transfer_call.py` — `TransferCall` action (telephony)
- `vocode/streaming/action/dtmf.py` — `DTMFAction` (telephony)
- `vocode/streaming/action/execute_external_action.py` — HTTP endpoint action
- `vocode/streaming/telephony/` — Twilio/Vonage telephony integration
- `vocode/streaming/output_device/twilio_output_device.py` — Twilio WebSocket with mark-based playback tracking
- `vocode/streaming/output_device/livekit_output_device.py` — LiveKit WebRTC output
- `vocode/streaming/output_device/speaker_output.py` — Local PyAudio output
- `vocode/streaming/output_device/filler_audio_worker.py` — Filler audio ("um...", typing noise) during LLM thinking
