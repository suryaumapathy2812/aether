# Google Calendar Plugin

View, search, and create calendar events.

## Core Workflow

Calendar requests follow one of two patterns:

```
Viewing schedule                    Creating events
────────────────                    ───────────────
upcoming_events / search_events     Confirm details with user
        ↓                                   ↓
get_event (if more detail needed)   create_event
```

## Decision Rules

**Which tool to use:**
- "What's today?" / "What's on my calendar?" → `upcoming_events` with `days=1`
- "What's this week?" → `upcoming_events` with `days=7`
- "What's coming up?" → `upcoming_events` with `days=14`
- "When is my dentist appointment?" → `search_events` with `query="dentist"`
- "Who's in the 3pm meeting?" → find the event first, then `get_event` for full details (attendees, conferencing links, description)

**Creating events:**
- Confirm title, date, time, and duration with the user before calling `create_event`. If the user gives a vague time ("tomorrow afternoon"), ask for the exact time.
- Default to 1 hour duration if the user doesn't specify an end time.
- The `create_event` tool uses `summary` for the event title (not `title`).
- Share the calendar link after creating so the user can verify.

**Timezones:**
- Default to Asia/Kolkata (IST, UTC+5:30) unless the user specifies otherwise.
- Display times in the user's local timezone, not UTC. If the user mentions a different city or timezone, adjust accordingly.

**Formatting responses:**
- Keep event summaries concise: "Team standup at 10:00 AM - 10:30 AM"
- Group multi-day schedules by day.
- Mention location and conferencing links when present and relevant.
- If no events found: "Your calendar is clear for that period."

## Pagination

The Calendar API returns up to 250 events per request (max 2500). For busy calendars, the response includes `nextPageToken` for follow-up calls.

## Rate Limits

| Quota | Limit |
|---|---|
| Queries per minute per user | 600 |
| Event creation per day | 10,000 |

Rate limits are generous — safe to make parallel calls for multiple date ranges.

## Example Workflows

**User: "What's on my calendar today?"**
```
1. upcoming_events days=1
2. "You have 3 events today:
   - Team standup at 10:00 AM - 10:30 AM
   - Lunch with Priya at 12:30 PM (Cafe Coffee Day, Koramangala)
   - Sprint review at 4:00 PM - 5:00 PM (Google Meet link)"
```

**User: "Schedule a meeting with Priya on Friday at 3pm for an hour"**
```
1. Confirm: "I'll create 'Meeting with Priya' on Friday Mar 20 at 3:00 PM - 4:00 PM IST. Go ahead?"
2. create_event summary="Meeting with Priya" start_time="2026-03-20T15:00:00" end_time="2026-03-20T16:00:00"
3. "Done! Here's the link: [link]"
```

**User: "When is my next dentist appointment?"**
```
1. search_events query="dentist" days_ahead=90
2. "Your next dentist appointment is on April 5 at 11:00 AM at Smile Dental Clinic."
```
