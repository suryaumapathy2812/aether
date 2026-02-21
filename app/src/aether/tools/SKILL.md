# Aether Built-in Tools

These tools are always available — no plugin required. They give you the ability to work with the local filesystem, run shell commands, search the web, manage long-term memory, and delegate work to background agents.

---

## Filesystem Tools

### `read_file`

Read the full text content of a file.

**Parameters:**
- `path` (required) — Absolute or relative path to the file. Relative paths resolve from the working directory.

**Limits:** 100KB max file size. Binary files are read with UTF-8 encoding (errors replaced).

**When to use:**
- Before editing a file — always read it first to understand the current content
- When the user asks what's in a file
- To check configuration, logs, or source code

**When NOT to use:**
- For directories — use `list_directory` instead
- For files over 100KB — inform the user and suggest alternatives (e.g. `run_command` with `head` or `grep`)

**Examples:**
```
read_file path="src/main.py"
read_file path="/etc/hosts"
read_file path="README.md"
```

---

### `write_file`

Create or overwrite a file with new content. Parent directories are created automatically.

**Parameters:**
- `path` (required) — File path to write to
- `content` (required) — Full content to write (overwrites existing content entirely)

**Important:** This overwrites the entire file. Always `read_file` first if you need to preserve existing content and only change part of it.

**When to use:**
- Creating new files from scratch
- Replacing a file's entire content after reading and modifying it
- Writing generated output (code, configs, reports) to disk

**When NOT to use:**
- When you only need to change a few lines — read first, modify in memory, then write the full updated content
- Without confirming with the user when writing to important files (configs, production code)

**Examples:**
```
write_file path="hello.py" content="print('Hello, world!')"
write_file path="config/settings.json" content="{\"debug\": true}"
```

---

### `list_directory`

List files and subdirectories in a given path. Shows names, types (file/dir), and file sizes.

**Parameters:**
- `path` (optional, default `.`) — Directory path to list. Relative paths resolve from the working directory.

**Limits:** Returns up to 200 entries. Truncated directories show a notice.

**When to use:**
- To understand the structure of a project before reading files
- When the user asks "what files are here?" or "show me the project structure"
- Before writing a file, to confirm the target directory exists

**Examples:**
```
list_directory path="."
list_directory path="src/aether"
list_directory path="app/plugins"
```

---

### `run_command`

Execute a shell command in the working directory. Returns stdout/stderr output.

**Parameters:**
- `command` (required) — The shell command to run

**Constraints:**
- 30-second timeout — long-running processes will be killed
- 50KB output limit — large outputs are truncated
- Only these commands are allowed: `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `echo`, `rm`, `mkdir`, `cp`, `mv`, `touch`, `date`, `pwd`, `whoami`, `python`, `python3`, `pip`, `node`, `npm`, `npx`, `git`, `curl`, `wget`, `jq`, `sort`, `uniq`, `tr`, `cut`, `du`, `df`, `tree`, `file`, `which`, `env`, `uname`

**When to use:**
- Running scripts or tests (`python3 script.py`, `npm test`)
- Git operations (`git status`, `git log`, `git diff`)
- Searching file contents (`grep -r "pattern" src/`)
- Checking system state (`date`, `pwd`, `env`)
- Installing packages (`pip install requests`, `npm install`)

**When NOT to use:**
- For reading files — use `read_file` (it's safer and has better error messages)
- For listing directories — use `list_directory`
- For commands not in the allowed list — inform the user it's not permitted

**Decision rules:**
- Prefer `read_file` over `cat` for reading files
- Prefer `list_directory` over `ls` for browsing
- Use `grep` for searching within files when you know the pattern
- Always show the user the command output, don't silently discard it

**Examples:**
```
run_command command="git status"
run_command command="python3 -m pytest tests/ -v"
run_command command="grep -r 'TODO' src/ --include='*.py'"
run_command command="npm install"
run_command command="git log --oneline -10"
```

---

## Web Search

### `web_search`

Search the web using DuckDuckGo and return a summary of top results.

**Parameters:**
- `query` (required) — The search query string

**Returns:** Abstract summary + related topics with URLs. Returns up to 5 results.

**When to use:**
- When you need current information (news, prices, recent events)
- When you're unsure about a fact and want to verify it
- When the user asks about something outside your training data
- When you need documentation, API references, or tutorials

**When NOT to use:**
- For things you already know confidently
- For information already in the conversation or memory
- For private/internal information (it searches the public web)

**Decision rules:**
- Be specific in queries — "FastAPI async background tasks" beats "python async"
- If the first search returns nothing useful, try rephrasing with different keywords
- Always cite the source URL when sharing information from search results
- Don't search for the same thing twice in one conversation

**Examples:**
```
web_search query="Python asyncio gather vs wait difference"
web_search query="OpenAI embeddings API pricing 2025"
web_search query="Chennai weather forecast this week"
```

---

## Memory Tools

### `save_memory`

Save a fact, preference, or instruction to long-term memory so it persists across sessions.

**Parameters:**
- `content` (required) — The fact or preference to remember. Write as a clear, self-contained statement.

**When to use:**
- When the user tells you something about themselves (name, location, preferences, timezone)
- When the user gives you a standing instruction ("always reply in bullet points")
- When you discover something important during a task (project stack, key decisions made)
- When the user explicitly says "remember this" or "keep this in mind"

**When NOT to use:**
- For temporary context that only matters in this conversation
- For things already in memory (search first to avoid duplicates)
- For assistant responses or advice — only save user-stated facts

**Format rules:**
- Write in third-person: "User prefers..." / "User's project uses..."
- Be specific and canonical: "User's timezone is Asia/Kolkata (IST)" not "user is in India"
- Keep it short: 6–18 words per fact
- One fact per call — don't bundle multiple facts into one string

**Examples:**
```
save_memory content="User's name is Surya"
save_memory content="User prefers concise bullet-point responses over long paragraphs"
save_memory content="User's primary project is Aether, a personal AI agent OS built with FastAPI and Next.js"
save_memory content="User's timezone is Asia/Kolkata (IST, UTC+5:30)"
save_memory content="User prefers dark mode across all apps"
```

---

### `search_memory`

Search long-term memory for relevant facts, preferences, past decisions, and prior context using semantic similarity.

**Parameters:**
- `query` (required) — What to search for. Be specific and descriptive.
- `limit` (optional, default `5`) — Maximum number of results to return (1–20).

**Returns:** Ranked list of matching memories by type:
- `[fact]` — Distilled facts and preferences (highest value, boosted in ranking)
- `[conversation]` — Raw past conversation turns
- `[action]` — Previous tool calls and their results
- `[session]` — Summaries of past sessions

**When to use:**
- Before answering "what do you know about me?" or "what are my preferences?"
- When the user references something from a past session ("last time we discussed...")
- Before making decisions that depend on user preferences (style, tools, timezone)
- When you're unsure if you already know something — search before asking
- At the start of a new task to load relevant prior context

**When NOT to use:**
- For information already in the current conversation — it's in your context window
- For real-time or current information — use `web_search` instead
- For file contents — use `read_file`

**Decision rules:**
- Search before asking the user for information you might already have
- Use specific queries: "user timezone preference" beats "user settings"
- If results are irrelevant, try a different angle: "project tech stack" vs "what framework does user use"
- Facts are the most reliable results — prioritize `[fact]` type results
- A result with no matches means the information was never saved — ask the user

**Examples:**
```
search_memory query="user timezone and location"
search_memory query="user's preferred coding language and style"
search_memory query="Aether project architecture decisions"
search_memory query="previous conversation about deployment setup"
search_memory query="user communication preferences" limit=3
```

---

## Agent Delegation Tools

### `spawn_task`

Delegate a task to a background worker agent. Returns immediately with a `task_id`. The worker runs independently while the conversation continues.

**Parameters:**
- `prompt` (required) — Clear, self-contained instructions for the worker agent
- `agent_type` (optional, default `general`) — Type of agent: `general`, `explore`, `planner`

**Agent types:**
- `general` — All-purpose agent. Good for most tasks: writing, analysis, code generation
- `explore` — Optimized for codebase exploration, file search, and pattern discovery
- `planner` — Optimized for breaking down complex tasks into steps and planning

**Returns:** A `task_id` string. Use `check_task` to monitor progress.

**When to use:**
- For long-running tasks that would block the conversation (e.g. "analyze all 50 files in this repo")
- When you want to run multiple tasks in parallel
- When the user asks you to do something in the background while they continue chatting
- For tasks that require deep exploration without blocking a response

**When NOT to use:**
- For quick tasks that take under a few seconds — just do them directly
- When you need the result immediately before responding — use direct tool calls instead
- When the task requires back-and-forth with the user

**Writing good prompts for workers:**
- Be explicit and self-contained — the worker has no conversation history
- Specify the expected output format
- Include all relevant context (file paths, constraints, goals)
- Tell the worker what to do if it gets stuck

**Examples:**
```
spawn_task prompt="Explore the src/aether directory and list all Python files with their purpose. Return a structured summary." agent_type="explore"

spawn_task prompt="Analyze the test coverage in app/tests/. List which modules have tests and which don't. Return a gap analysis." agent_type="general"

spawn_task prompt="Break down the task of adding OAuth2 support to the FastAPI app into concrete subtasks with dependencies." agent_type="planner"
```

---

### `check_task`

Check the status of a background task spawned with `spawn_task`. Returns status and result when complete.

**Parameters:**
- `task_id` (required) — The `task_id` returned by `spawn_task`

**Possible statuses:**
- `running` — Still in progress. Check again later.
- `done` — Completed successfully. Result is included.
- `error` — Failed. Last output is included for debugging.
- `not_found` — Invalid task_id or task expired.

**When to use:**
- After `spawn_task`, to poll for completion
- When the user asks "is that background task done yet?"
- Before using the result of a spawned task in a follow-up action

**Polling strategy:**
- Don't poll immediately — give the worker a moment to start
- For quick tasks, check once after a short pause
- For long tasks, inform the user it's running and check when they ask
- If still running, tell the user and offer to check again later

**Examples:**
```
check_task task_id="abc123def456"
```

**Typical workflow:**
```
1. spawn_task → get task_id "abc123"
2. Tell user: "I've started that in the background (task abc123). I'll check back when it's done."
3. check_task task_id="abc123" → "running"
4. [user continues chatting]
5. check_task task_id="abc123" → "done" + result
6. Share result with user
```
