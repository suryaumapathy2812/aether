---
name: meet
description: Google Meet API — meeting spaces, conference records, recordings, transcripts
integration: google-workspace
---
# Google Meet API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://meet.googleapis.com/v2`
- Token auto-refreshes on 401 response

## Create Space (Meeting Room)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "accessType": "OPEN",
      "entryPointAccess": "ALL"
    }
  }' \
  "https://meet.googleapis.com/v2/spaces"
```

### accessType Options
- `OPEN` — anyone with the link can join
- `TRUSTED` — only organization members
- `RESTRICTED` — only invited participants

### Response
Returns a Space with `meetingUri` (the join link) and `meetingCode`.

## Get Space

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/spaces/{SPACE_NAME}"
```

- `{SPACE_NAME}` format: `spaces/{SPACE_ID}`

## Get Space by Meeting Code

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/spaces/{SPACE_ID}"
```

## End Active Conference

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' \
  "https://meet.googleapis.com/v2/spaces/{SPACE_ID}:endActiveConference"
```

## List Conference Records

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/conferenceRecords"
```

### Query Parameters
- `filter` — filter expression (e.g., `space.name = "spaces/{SPACE_ID}"`)
- `pageSize` — max results (default 25, max 100)

## Get Conference Record

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/conferenceRecords/{RECORD_NAME}"
```

## List Participants in Conference

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/conferenceRecords/{RECORD_NAME}/participants"
```

## Get Recordings

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/conferenceRecords/{RECORD_NAME}/recordings"
```

## Get Transcripts

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://meet.googleapis.com/v2/conferenceRecords/{RECORD_NAME}/transcripts"
```

## Rate Limits
- 120 queries per minute

## Error Handling
- **400 Bad Request**: Invalid space configuration
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions (requires Workspace account)
- **404 Not Found**: Space or record doesn't exist
- **429 Rate Limited**: Wait and retry
- **Note**: Google Meet API requires a Google Workspace account (not available with personal Gmail)
