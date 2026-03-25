---
name: tasks
description: Google Tasks API — task lists, tasks, subtasks, and completion
integration: google-workspace
---
# Google Tasks API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://tasks.googleapis.com/v1`
- Token auto-refreshes on 401 response

## List Task Lists

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://tasks.googleapis.com/v1/users/@me/lists"
```

Returns `items[]` with `id` and `title` for each task list.

## Create Task List

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "My Task List"}' \
  "https://tasks.googleapis.com/v1/users/@me/lists"
```

## List Tasks in a Task List

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks"
```

### Query Parameters
- `maxResults` — max tasks to return (default 20)
- `showCompleted` — include completed tasks (default false)
- `showDeleted` — include deleted tasks (default false)
- `showHidden` — include hidden tasks (default false)
- `dueMin` — lower bound for due date (RFC 3339)
- `dueMax` — upper bound for due date (RFC 3339)

## Get Task

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks/{TASK_ID}"
```

## Create Task

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Buy groceries",
    "notes": "Milk, eggs, bread",
    "due": "2025-01-25T00:00:00.000Z"
  }' \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks"
```

## Create Subtask

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Subtask item"}' \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks?parent={PARENT_TASK_ID}"
```

## Update Task

```bash
curl -s -X PATCH \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated title", "status": "completed"}' \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks/{TASK_ID}"
```

### Task Fields
- `title` — task name
- `notes` — description/notes
- `status` — `"needsAction"` or `"completed"`
- `due` — due date (RFC 3339)
- `deleted` — boolean, soft delete

## Complete a Task

```bash
curl -s -X PATCH \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}' \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks/{TASK_ID}"
```

## Delete Task

```bash
curl -s -X DELETE \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks/{TASK_ID}"
```

## Move Task

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://tasks.googleapis.com/v1/lists/{TASKLIST_ID}/tasks/{TASK_ID}/move?parent={PARENT_ID}&previous={PREVIOUS_TASK_ID}"
```

## Rate Limits
- 50,000 queries per day
- 500 queries per 100 seconds per user

## Error Handling
- **400 Bad Request**: Invalid task data
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Task list or task ID doesn't exist
- **429 Rate Limited**: Wait and retry
