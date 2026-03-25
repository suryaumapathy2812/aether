# Google Calendar API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://www.googleapis.com/calendar/v3`
- Token auto-refreshes on 401 response

## List Calendars
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/calendar/v3/users/me/calendarList"
```

## List Events
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/calendar/v3/calendars/primary/events?singleEvents=true&orderBy=startTime&timeMin=$(date -u +%Y-%m-%dT%H:%M:%SZ)&maxResults=10"
```

### Query Parameters
- `singleEvents=true` — expand recurring events
- `orderBy=startTime` — sort by start time (requires singleEvents=true)
- `timeMin` — RFC3339 start boundary (e.g., `2025-01-15T00:00:00Z`)
- `timeMax` — RFC3339 end boundary
- `q` — free text search
- `maxResults` — max events to return (default 250)

## Get Event
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/calendar/v3/calendars/primary/events/{EVENT_ID}"
```

## Create Event
```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Meeting with Team",
    "description": "Weekly sync",
    "location": "Conference Room A",
    "start": {"dateTime": "2025-01-20T10:00:00-05:00"},
    "end": {"dateTime": "2025-01-20T11:00:00-05:00"},
    "attendees": [{"email": "alice@example.com"}]
  }' \
  "https://www.googleapis.com/calendar/v3/calendars/primary/events"
```

### All-Day Event
Use `date` instead of `dateTime`:
```json
{"start": {"date": "2025-01-20"}, "end": {"date": "2025-01-21"}}
```

## Update Event
```bash
curl -s -X PATCH \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"summary": "Updated Title"}' \
  "https://www.googleapis.com/calendar/v3/calendars/primary/events/{EVENT_ID}"
```

## Delete Event
```bash
curl -s -X DELETE \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/calendar/v3/calendars/primary/events/{EVENT_ID}"
```

## Quick Add (Natural Language)
```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/calendar/v3/calendars/primary/events/quick?text=Meeting+with+Alice+tomorrow+at+3pm"
```

## Free/Busy Query
```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "timeMin": "2025-01-20T00:00:00Z",
    "timeMax": "2025-01-21T00:00:00Z",
    "items": [{"id": "primary"}]
  }' \
  "https://www.googleapis.com/calendar/v3/freeBusy"
```

## Rate Limits
- 1,000,000 queries/day
- 500 queries per 100 seconds per user

## Error Handling
- **401**: Token expired — auto-refreshes, retry
- **403**: Insufficient permissions or quota exceeded
- **404**: Event or calendar not found
- **409**: Concurrent modification conflict — re-fetch and retry
- **429**: Rate limited — wait and retry
