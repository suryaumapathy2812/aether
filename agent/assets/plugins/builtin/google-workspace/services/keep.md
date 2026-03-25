# Google Keep API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://keep.googleapis.com/v1`
- Token auto-refreshes on 401 response

## List Notes

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://keep.googleapis.com/v1/notes?filter=TRASHED=false"
```

### Query Parameters
- `filter` — filter expression (e.g., `TRASHED=false`)
- `pageSize` — max results (default 20, max 100)

## Create Note

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Shopping List",
    "body": {
      "text": {
        "text": "Milk\nEggs\nBread"
      }
    }
  }' \
  "https://keep.googleapis.com/v1/notes"
```

### Create Note with List Items

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Todo List",
    "listContent": [
      {"listItem": {"text": "Task 1", "checked": false}},
      {"listItem": {"text": "Task 2", "checked": false}}
    ]
  }' \
  "https://keep.googleapis.com/v1/notes"
```

## Get Note

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://keep.googleapis.com/v1/notes/{NOTE_NAME}"
```

- `{NOTE_NAME}` format: `notes/{NOTE_ID}`

## Update Note

```bash
curl -s -X PATCH \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "updateMask": "body.text.text",
    "body": {
      "text": {
        "text": "Updated note content"
      }
    }
  }' \
  "https://keep.googleapis.com/v1/notes/{NOTE_NAME}"
```

### Update Fields
Use `updateMask` to specify which fields to update:
- `body.text.text` — note body text
- `title` — note title
- `listContent` — list items

## Delete (Trash) Note

```bash
curl -s -X DELETE \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://keep.googleapis.com/v1/notes/{NOTE_NAME}"
```

## Add Label to Note

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://keep.googleapis.com/v1/notes/{NOTE_NAME}/labels" \
  -H "Content-Type: application/json" \
  -d '{"label": {"name": "labels/{LABEL_ID}"}}'
```

## Rate Limits
- 300 requests per minute per project

## Error Handling
- **400 Bad Request**: Invalid note structure
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Note name doesn't exist
- **429 Rate Limited**: Wait and retry
