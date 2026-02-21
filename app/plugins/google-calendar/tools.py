"""Google Calendar tools for viewing and creating events.

Uses the Google Calendar API v3. Each plugin manages its own OAuth tokens
independently. Credentials arrive via ``self._context`` at call time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult
from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

logger = logging.getLogger(__name__)

CALENDAR_API = "https://www.googleapis.com/calendar/v3"


class _CalendarTool(AetherTool):
    """Base for Calendar tools — provides token extraction from runtime context."""

    def _get_token(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("access_token") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}


def _format_event(event: dict) -> str:
    """Format a calendar event into a readable string."""
    summary = event.get("summary", "(No title)")
    start = event.get("start", {})
    end = event.get("end", {})

    # All-day events use "date", timed events use "dateTime"
    start_str = start.get("dateTime", start.get("date", ""))
    end_str = end.get("dateTime", end.get("date", ""))

    # Clean up ISO format for readability
    if "T" in start_str:
        try:
            dt = datetime.fromisoformat(start_str)
            start_str = dt.strftime("%b %d, %I:%M %p")
            dt_end = datetime.fromisoformat(end_str)
            end_str = dt_end.strftime("%I:%M %p")
        except (ValueError, TypeError):
            pass
    else:
        # All-day event
        try:
            dt = datetime.strptime(start_str, "%Y-%m-%d")
            start_str = dt.strftime("%b %d")
            end_str = "all day"
        except (ValueError, TypeError):
            pass

    location = event.get("location", "")
    event_id = event.get("id", "")

    line = f"**{summary}** — {start_str}"
    if end_str and end_str != start_str:
        line += f" to {end_str}"
    if location:
        line += f"\n   Location: {location}"
    return line


class UpcomingEventsTool(_CalendarTool):
    """Get upcoming calendar events."""

    name = "upcoming_events"
    description = "Get your upcoming calendar events"
    status_text = "Checking your calendar..."
    parameters = [
        ToolParam(
            name="days",
            type="integer",
            description="Number of days ahead to look (default 7)",
            required=False,
            default=7,
        ),
        ToolParam(
            name="max_results",
            type="integer",
            description="Max events to return (default 10)",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, days: int = 7, max_results: int = 10, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail(
                "Google Calendar not connected — missing access token."
            )

        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days)).isoformat()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{CALENDAR_API}/calendars/primary/events",
                    headers=self._auth_headers(),
                    params={
                        "timeMin": time_min,
                        "timeMax": time_max,
                        "maxResults": max_results,
                        "singleEvents": "true",
                        "orderBy": "startTime",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            events = data.get("items", [])
            if not events:
                return ToolResult.success(f"No events in the next {days} days.")

            output = f"**Upcoming events (next {days} days):**\n"
            for i, event in enumerate(events, 1):
                output += f"\n{i}. {_format_event(event)}"

            return ToolResult.success(output, count=len(events))

        except Exception as e:
            logger.error(f"Error fetching calendar events: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class SearchEventsTool(_CalendarTool):
    """Search calendar events by keyword."""

    name = "search_events"
    description = "Search your calendar events by keyword"
    status_text = "Searching calendar..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="Search keyword (matches event title, description, location)",
            required=True,
        ),
        ToolParam(
            name="max_results",
            type="integer",
            description="Max events to return (default 10)",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, query: str, max_results: int = 10, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail(
                "Google Calendar not connected — missing access token."
            )

        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{CALENDAR_API}/calendars/primary/events",
                    headers=self._auth_headers(),
                    params={
                        "q": query,
                        "timeMin": now.isoformat(),
                        "maxResults": max_results,
                        "singleEvents": "true",
                        "orderBy": "startTime",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            events = data.get("items", [])
            if not events:
                return ToolResult.success(f"No events found matching '{query}'.")

            output = f"**Events matching '{query}':**\n"
            for i, event in enumerate(events, 1):
                output += f"\n{i}. {_format_event(event)}"

            return ToolResult.success(output, count=len(events))

        except Exception as e:
            logger.error(f"Error searching calendar: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class CreateEventTool(_CalendarTool):
    """Create a new calendar event."""

    name = "create_event"
    description = "Create a new event on your Google Calendar"
    status_text = "Creating event..."
    parameters = [
        ToolParam(
            name="summary",
            type="string",
            description="Event title",
            required=True,
        ),
        ToolParam(
            name="start_time",
            type="string",
            description="Start time in ISO 8601 format (e.g. 2026-02-20T14:00:00)",
            required=True,
        ),
        ToolParam(
            name="end_time",
            type="string",
            description="End time in ISO 8601 format (e.g. 2026-02-20T15:00:00)",
            required=True,
        ),
        ToolParam(
            name="description",
            type="string",
            description="Event description (optional)",
            required=False,
            default="",
        ),
        ToolParam(
            name="location",
            type="string",
            description="Event location (optional)",
            required=False,
            default="",
        ),
    ]

    async def execute(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        **_,
    ) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail(
                "Google Calendar not connected — missing access token."
            )

        event_body: dict = {
            "summary": summary,
            "start": {"dateTime": start_time, "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": end_time, "timeZone": "Asia/Kolkata"},
        }
        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{CALENDAR_API}/calendars/primary/events",
                    headers={
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    },
                    json=event_body,
                )
                resp.raise_for_status()
                created = resp.json()

            link = created.get("htmlLink", "")
            return ToolResult.success(
                f"Event created: **{summary}**\n"
                f"Start: {start_time}\n"
                f"End: {end_time}\n"
                f"{f'Link: {link}' if link else ''}",
                event_id=created.get("id", ""),
            )

        except Exception as e:
            logger.error(f"Error creating event: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class GetEventTool(_CalendarTool):
    """Get details of a specific calendar event."""

    name = "get_event"
    description = "Get details of a specific calendar event by ID"
    status_text = "Fetching event details..."
    parameters = [
        ToolParam(
            name="event_id",
            type="string",
            description="The event ID",
            required=True,
        ),
    ]

    async def execute(self, event_id: str, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail(
                "Google Calendar not connected — missing access token."
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{CALENDAR_API}/calendars/primary/events/{event_id}",
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                event = resp.json()

            output = _format_event(event)
            desc = event.get("description", "")
            if desc:
                output += f"\n   Description: {desc}"
            attendees = event.get("attendees", [])
            if attendees:
                names = [a.get("email", "") for a in attendees[:5]]
                output += f"\n   Attendees: {', '.join(names)}"
            link = event.get("htmlLink", "")
            if link:
                output += f"\n   Link: {link}"

            return ToolResult.success(output)

        except Exception as e:
            logger.error(f"Error fetching event: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RefreshGoogleCalendarTokenTool(RefreshOAuthTokenTool):
    """Refresh the Google Calendar OAuth access token before it expires.

    Called automatically by the cron system every 50 minutes.
    Can also be called manually if Calendar tools start returning auth errors.
    """

    name = "refresh_google_calendar_token"
    plugin_name = "google-calendar"
    description = (
        "Refresh the Google Calendar OAuth access token. "
        "Call this when Calendar tools return authentication errors, "
        "or when instructed by the system to prevent token expiry."
    )
