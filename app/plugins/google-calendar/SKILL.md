# Google Calendar Plugin

You have access to Google Calendar tools for viewing and creating events.

## Tools Available

- `upcoming_events` — Get upcoming events for the next N days. Returns event titles, times, and locations.
- `search_events` — Search calendar events by keyword (matches title, description, location).
- `create_event` — Create a new calendar event with title, start/end time, optional description and location.
- `get_event` — Get full details of a specific event by its ID.

## Guidelines

- **When the user asks about their schedule**, use `upcoming_events` first to see what's coming up.
- **When creating events**, confirm the details with the user before calling `create_event`.
- **Use ISO 8601 format** for dates/times (e.g. 2026-02-20T14:00:00).
- **Default timezone is Asia/Kolkata** — adjust if the user mentions a different timezone.
- **Keep event summaries concise** — mention the title, time, and location.
- **For "what's on my calendar today"**, use `upcoming_events` with `days=1`.
