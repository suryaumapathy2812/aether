# Plan: Consolidate Google Plugins into Unified Google Workspace Plugin

## Context

Aether currently has 4 separate Google plugins (`gmail`, `google-calendar`, `google-drive`, `google-contacts`), each with its own OAuth config, SKILL.md, webhook handler, and token rotation. This creates unnecessary fragmentation since all Google services share the same OAuth provider and token endpoint. The user wants to:

1. **Consolidate** the 4 existing Google plugins into a single `google-workspace` plugin
2. **Extend** with new Google services: Sheets, Docs, Slides, Tasks, Forms, Keep, Meet
3. **Keep** separate SKILL.md files per service for focused API documentation
4. **Use** combined OAuth scopes (one consent, one token for all services)

## Current State

```
assets/plugins/builtin/
├── gmail/            plugin.yaml + SKILL.md (oauth2, gmail scopes, pubsub webhook)
├── google-calendar/  plugin.yaml + SKILL.md (oauth2, calendar scopes, pubsub webhook)
├── google-drive/     plugin.yaml + SKILL.md (oauth2, drive scope)
├── google-contacts/  plugin.yaml + SKILL.md (oauth2, contacts scope)
```

**Internal code touchpoints:**
- `internal/plugins/token_rotation.go` — registers rotator for each of the 4 plugin names
- `internal/plugins/webhooks_gmail.go` — handler keyed to "gmail"
- `internal/plugins/webhooks_calendar.go` — handler keyed to "google-calendar"
- `internal/tools/builtin/execute_tool.go` — `credentialEnvMapping` maps 4 plugin names → 4 env vars
- `internal/llm/context_builder.go` — `envVarNameForPlugin()` maps 4 names → 4 env vars
- `internal/db/store.go` — `PluginRecord` per plugin name in SQLite

## Target State

```
assets/plugins/builtin/google-workspace/
├── plugin.yaml                    # Combined OAuth, all scopes
├── SKILL.md                       # Master overview: auth + API quick-reference
└── services/
    ├── gmail.md                   # Gmail API (existing content, adapted)
    ├── calendar.md                # Calendar API (existing content, adapted)
    ├── drive.md                   # Drive API (existing content, adapted)
    ├── contacts.md                # People/Contacts API (existing content, adapted)
    ├── sheets.md                  # Sheets API (NEW)
    ├── docs.md                    # Docs API (NEW)
    ├── slides.md                  # Slides API (NEW)
    ├── tasks.md                   # Tasks API (NEW)
    ├── forms.md                   # Forms API (NEW)
    ├── keep.md                    # Keep API (NEW)
    └── meet.md                    # Meet API (NEW)
```

**Single DB record:** `name: "google-workspace"`, one `access_token`, one config set
**Single env var:** `$GOOGLE_WORKSPACE_ACCESS_TOKEN` injected for all Google API calls

---

## Implementation Steps

### Phase 1: Create the Consolidated Plugin

#### Step 1.1 — Create `plugin.yaml`

Create `assets/plugins/builtin/google-workspace/plugin.yaml`:

```yaml
name: google-workspace
display_name: Google Workspace
description: Google Workspace services — Gmail, Calendar, Drive, Contacts, Sheets, Docs, Slides, Tasks, Forms, Keep, and Meet
version: 1.0.0
plugin_type: sensor
auth:
  type: oauth2
  provider: google
  token_url: https://oauth2.googleapis.com/token
  use_basic_auth: false
  auto_refresh: true
  refresh_interval: 3000
  scopes:
    # Gmail
    - https://www.googleapis.com/auth/gmail.readonly
    - https://www.googleapis.com/auth/gmail.send
    - https://www.googleapis.com/auth/gmail.modify
    # Calendar
    - https://www.googleapis.com/auth/calendar
    - https://www.googleapis.com/auth/calendar.events
    # Drive
    - https://www.googleapis.com/auth/drive
    # Contacts/People
    - https://www.googleapis.com/auth/contacts.readonly
    - https://www.googleapis.com/auth/contacts
    # Sheets
    - https://www.googleapis.com/auth/spreadsheets
    # Docs
    - https://www.googleapis.com/auth/documents
    # Slides
    - https://www.googleapis.com/auth/presentations
    # Tasks
    - https://www.googleapis.com/auth/tasks
    # Forms
    - https://www.googleapis.com/auth/forms.body
    - https://www.googleapis.com/auth/forms.responses.readonly
    # Keep
    - https://www.googleapis.com/auth/keep
    # Meet
    - https://www.googleapis.com/auth/meetings.space.created
    - https://www.googleapis.com/auth/meetings.space.readonly
  config_fields:
    - key: client_id
      label: Client ID
      type: text
      required: true
    - key: client_secret
      label: Client Secret
      type: password
      required: true
    - key: project_id
      label: Project ID
      type: text
    - key: pubsub_topic
      label: Gmail Pub/Sub Topic
      type: text
      description: Required for Gmail push notifications
api:
  base_url: https://www.googleapis.com
webhook:
  protocol: pubsub
  event_field: eventType
  supported_events:
    - messageReceived
    - sync
```

#### Step 1.2 — Create Master SKILL.md

Create `assets/plugins/builtin/google-workspace/SKILL.md` — master overview covering:
- Authentication header pattern (`Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN`)
- Service quick-reference table (service → base URL → key endpoints)
- Instructions on when/how to use the execute tool with `credentials=["google-workspace"]`
- Reference to per-service skills for detailed API docs

#### Step 1.3 — Migrate Existing SKILL.md Files

Move existing service SKILL.md content into `services/` subdirectory. Each file keeps the same API documentation but updates the env var reference from `$GMAIL_ACCESS_TOKEN` to `$GOOGLE_WORKSPACE_ACCESS_TOKEN`.

#### Step 1.4 — Create New Service SKILL.md Files

Create SKILL.md files for new services. Each follows the same pattern (auth header, base URL, key endpoints with curl examples):

| Service | Base URL | Key Operations |
|---------|----------|----------------|
| Sheets | `sheets.googleapis.com/v4/spreadsheets` | create, get, values.get, values.update, values.append, batchUpdate |
| Docs | `docs.googleapis.com/v1/documents` | create, get, batchUpdate (insert text, replace text) |
| Slides | `slides.googleapis.com/v1/presentations` | create, get, batchUpdate (create slides, insert text) |
| Tasks | `tasks.googleapis.com/v1/users/@tasklists` | list tasklists, list tasks, insert, update, delete |
| Forms | `forms.googleapis.com/v1/forms` | create, get, setpublishsettings, responses |
| Keep | `keep.googleapis.com/v1/notes` | create, get, list, update, delete |
| Meet | `meet.googleapis.com/v2/spaces` | create, get, endActiveConference |

#### Step 1.5 — Remove Old Plugin Directories

Delete:
- `assets/plugins/builtin/gmail/`
- `assets/plugins/builtin/google-calendar/`
- `assets/plugins/builtin/google-drive/`
- `assets/plugins/builtin/google-contacts/`

---

### Phase 2: Update Internal Code

#### Step 2.1 — Update `credentialEnvMapping` in `execute_tool.go`

Remove the 4 individual Google entries, add one consolidated entry:

```go
var credentialEnvMapping = map[string]string{
    "google-workspace": "GOOGLE_WORKSPACE_ACCESS_TOKEN",
    "spotify":          "SPOTIFY_ACCESS_TOKEN",
    "weather":          "WEATHER_API_KEY",
    "brave-search":     "BRAVE_SEARCH_API_KEY",
    "wolfram":          "WOLFRAM_APP_ID",
}
```

#### Step 2.2 — Update `envVarNameForPlugin` in `context_builder.go`

Same change — remove 4 Google entries, add one consolidated entry.

#### Step 2.3 — Update Token Rotation in `token_rotation.go`

Replace 4 separate rotators with one:

```go
func RegisterDefaultTokenRotators(registry *CronRegistry) {
    if registry == nil { return }
    registry.RegisterTokenRotator("google-workspace", oauthTokenRotator("https://oauth2.googleapis.com/token", false))
    registry.RegisterTokenRotator("spotify", oauthTokenRotator("https://accounts.spotify.com/api/token", true))
}
```

#### Step 2.4 — Update Watch Renewal in `token_rotation.go`

Change watch renewer registration from "gmail" to "google-workspace":

```go
func RegisterDefaultWatchRenewers(registry *CronRegistry) {
    if registry == nil { return }
    registry.RegisterWatchRenewer("google-workspace", gmailWatchRenewer())
}
```

#### Step 2.5 — Consolidate Webhook Handlers

Create `internal/plugins/webhooks_google.go` that replaces both gmail and calendar webhook handlers. The consolidated handler inspects the payload to determine the service type:

- If payload has `notification.notification.emailAddress` or `notification.historyId` → Gmail logic
- If payload has `eventType` or calendar-specific fields → Calendar logic

Register as:
```go
func init() {
    DefaultWebhookRegistry().Register("google-workspace", newGoogleWorkspaceWebhookHandler())
}
```

#### Step 2.6 — Delete Old Webhook Files

- Delete `internal/plugins/webhooks_gmail.go`
- Delete `internal/plugins/webhooks_calendar.go`

#### Step 2.7 — Update System Prompt (context_builder.go)

Add a service listing section after the plugin line so the LLM knows what APIs are available:

```
Google Workspace services (all use $GOOGLE_WORKSPACE_ACCESS_TOKEN):
- Gmail: https://gmail.googleapis.com/gmail/v1 — email send, read, search, labels
- Calendar: https://www.googleapis.com/calendar/v3 — events, calendars, free/busy
- Drive: https://www.googleapis.com/drive/v3 — files, folders, permissions
- Contacts: https://people.googleapis.com/v1 — contacts, directory
- Sheets: https://sheets.googleapis.com/v4/spreadsheets — read/write spreadsheets
- Docs: https://docs.googleapis.com/v1/documents — create/edit documents
- Slides: https://slides.googleapis.com/v1/presentations — create/edit presentations
- Tasks: https://tasks.googleapis.com/v1 — task lists and tasks
- Forms: https://forms.googleapis.com/v1 — create/manage forms
- Keep: https://keep.googleapis.com/v1 — notes management
- Meet: https://meet.googleapis.com/v2 — meeting spaces
```

---

### Phase 3: Update Cron Jobs

#### Step 3.1 — Reschedule Token Rotation Jobs

Existing cron jobs with `{"plugin":"gmail"}` need to be rescheduled with `{"plugin":"google-workspace"}`:

```bash
# Cancel old jobs:
go run ./cmd/cron cancel --job-id <gmail-rotate-job-id>
go run ./cmd/cron cancel --job-id <calendar-rotate-job-id>
go run ./cmd/cron cancel --job-id <drive-rotate-job-id>
go run ./cmd/cron cancel --job-id <contacts-rotate-job-id>

# Schedule one consolidated job:
go run ./cmd/cron schedule --module plugins --type rotate_token \
  --run-at 2026-03-26T00:00:00Z --interval-s 3600 \
  --payload '{"plugin":"google-workspace"}'
```

---

### Phase 4: Verification

#### Step 4.1 — Plugin Discovery

```bash
go run ./cmd/plugins discover
go run ./cmd/plugins list
go run ./cmd/plugins read google-workspace
```

#### Step 4.2 — Skill Discovery

```bash
go run ./cmd/skills discover
go run ./cmd/skills list
go run ./cmd/skills search sheets
```

#### Step 4.3 — Run Tests

```bash
go test ./internal/plugins/... -v
go test ./internal/tools/... -v
go test ./internal/llm/... -v
```

#### Step 4.4 — End-to-End Tool Execution

Configure the consolidated plugin with OAuth credentials, then test a Google API call through the execute tool.

---

## Files Changed Summary

| File | Action |
|------|--------|
| `assets/plugins/builtin/google-workspace/plugin.yaml` | CREATE |
| `assets/plugins/builtin/google-workspace/SKILL.md` | CREATE |
| `assets/plugins/builtin/google-workspace/services/gmail.md` | CREATE (from existing) |
| `assets/plugins/builtin/google-workspace/services/calendar.md` | CREATE (from existing) |
| `assets/plugins/builtin/google-workspace/services/drive.md` | CREATE (from existing) |
| `assets/plugins/builtin/google-workspace/services/contacts.md` | CREATE (from existing) |
| `assets/plugins/builtin/google-workspace/services/sheets.md` | CREATE (new) |
| `assets/plugins/builtin/google-workspace/services/docs.md` | CREATE (new) |
| `assets/plugins/builtin/google-workspace/services/slides.md` | CREATE (new) |
| `assets/plugins/builtin/google-workspace/services/tasks.md` | CREATE (new) |
| `assets/plugins/builtin/google-workspace/services/forms.md` | CREATE (new) |
| `assets/plugins/builtin/google-workspace/services/keep.md` | CREATE (new) |
| `assets/plugins/builtin/google-workspace/services/meet.md` | CREATE (new) |
| `assets/plugins/builtin/gmail/` | DELETE |
| `assets/plugins/builtin/google-calendar/` | DELETE |
| `assets/plugins/builtin/google-drive/` | DELETE |
| `assets/plugins/builtin/google-contacts/` | DELETE |
| `internal/tools/builtin/execute_tool.go` | MODIFY |
| `internal/llm/context_builder.go` | MODIFY |
| `internal/plugins/token_rotation.go` | MODIFY |
| `internal/plugins/webhooks_gmail.go` | DELETE |
| `internal/plugins/webhooks_calendar.go` | DELETE |
| `internal/plugins/webhooks_google.go` | CREATE |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing installations | Write migration script or startup check that consolidates old plugin DB records |
| Webhook URL change breaks existing subscriptions | Update webhook URLs in Google Cloud Console to `/internal/hooks/google-workspace` |
| Unverified OAuth app hits scope limits (~25 scopes) | Document that users must verify their OAuth app or manually select scopes |
| Per-service SKILL.md overwhelms context | Per-service docs are in `services/` subdirectory — NOT auto-loaded. Agent loads on demand via `read_skill`. Master SKILL.md stays concise. |
| Single token for all services is less secure | Acceptable trade-off for personal agent use. Enterprise would use service accounts per service. |

## Design Decisions

1. **Webhook URLs:** Clean break. Only `/internal/hooks/google-workspace` is supported. Update webhook URLs in Google Cloud Console after deployment.
2. **gws skills:** Not importing recipes or persona skills. Only the 11 per-service SKILL.md files (4 migrated + 7 new).
