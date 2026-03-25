---
name: docs
description: Google Docs API — create, read, and batch update documents
integration: google-workspace
---
# Google Docs API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://docs.googleapis.com/v1/documents`
- Token auto-refreshes on 401 response

## Create Document

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "My Document"}' \
  "https://docs.googleapis.com/v1/documents"
```

## Get Document

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://docs.googleapis.com/v1/documents/{DOCUMENT_ID}"
```

## Batch Update (Insert Text, Replace Text, etc.)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "insertText": {
          "location": {"index": 1},
          "text": "Hello, World!\n"
        }
      }
    ]
  }' \
  "https://docs.googleapis.com/v1/documents/{DOCUMENT_ID}:batchUpdate"
```

### Common Request Types

**Insert Text at End:**
```json
{
  "insertText": {
    "endOfSegmentLocation": {"segmentId": ""},
    "text": "New paragraph text\n"
  }
}
```

**Replace All Text:**
```json
{
  "replaceAllText": {
    "containsText": {"text": "OLD_TEXT", "matchCase": true},
    "replaceText": "NEW_TEXT"
  }
}
```

**Insert Page Break:**
```json
{
  "insertPageBreak": {
    "location": {"index": 5}
  }
}
```

**Create Named Range:**
```json
{
  "createNamedRange": {
    "name": "myRange",
    "range": {"startIndex": 1, "endIndex": 10}
  }
}
```

### Getting Document Structure

The response contains `body.content[]` — each element has:
- `startIndex` / `endIndex` — position in the document
- `paragraph` — contains `elements[]` with text runs
- `table` — table structure
- `sectionBreak` — page breaks

### Finding Text Positions

To insert at a specific location, first read the document, then search `body.content` for the target text's `startIndex`.

## Rate Limits
- 300 requests per minute per project

## Error Handling
- **400 Bad Request**: Invalid index or request structure
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Document ID doesn't exist
- **429 Rate Limited**: Wait and retry
