"""Google Calendar tools for viewing and creating events.

Uses the Google Calendar API v3. Each plugin manages its own OAuth tokens
independently. Credentials arrive via ``self._context`` at call time.
"""

from __future__ import annotations

import logging
import os as _os
from datetime import datetime, timedelta, timezone

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult
from aether.tools.base_watch_tool import BaseWatchTool
from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

logger = logging.getLogger(__name__)

CALENDAR_API = "https://www.googleapis.com/calendar/v3"

_ORCHESTRATOR_URL_W = _os.getenv("ORCHESTRATOR_URL", "")
_AGENT_USER_ID_W = _os.getenv("AETHER_USER_ID", "")
# PUBLIC_HOOK_URL: the public HTTPS base URL of the orchestrator, used as the
# Google Calendar push webhook address. Google requires HTTPS.
# Example: https://api.yourdomain.com
# Falls back to ORCHESTRATOR_URL if not set (will fail for HTTP-only setups).
_PUBLIC_HOOK_URL = _os.getenv("PUBLIC_HOOK_URL", "") or _ORCHESTRATOR_URL_W


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


# ── Watch tools ────────────────────────────────────────────────────────────────


class SetupCalendarWatchTool(BaseWatchTool):
    """Register a Google Calendar push notification channel.

    Called automatically by the cron system ~30 s after OAuth connect.
    Uses the Calendar Events.watch() API to push change notifications to
    POST /api/hooks/http/google-calendar/{user_id} on the orchestrator.

    The watch expires in up to 7 days; renewal is scheduled automatically.
    """

    name = "setup_calendar_watch"
    plugin_name = "google-calendar"
    description = (
        "Register Google Calendar push notifications so the agent receives "
        "real-time alerts when calendar events change. "
        "Called automatically by the system after connecting Google Calendar."
    )
    status_text = "Setting up Google Calendar push notifications..."

    async def execute(self, **_) -> ToolResult:  # type: ignore[override]
        token = self._get_token()
        if not token:
            return ToolResult.fail(
                "Cannot set up Calendar watch: no access token available. "
                "Ensure Google Calendar is connected and the token is valid."
            )

        if not _PUBLIC_HOOK_URL or not _AGENT_USER_ID_W:
            return ToolResult.fail(
                "Cannot set up Calendar watch: PUBLIC_HOOK_URL (or ORCHESTRATOR_URL) "
                "and AETHER_USER_ID must be configured."
            )

        if not _PUBLIC_HOOK_URL.startswith("https://"):
            return ToolResult.fail(
                f"Cannot set up Calendar watch: Google requires an HTTPS webhook URL. "
                f"Set PUBLIC_HOOK_URL to your public HTTPS orchestrator URL "
                f"(current value starts with '{_PUBLIC_HOOK_URL[:10]}...')."
            )

        # The push endpoint the orchestrator exposes for HTTP-push plugins
        hook_url = (
            f"{_PUBLIC_HOOK_URL}/api/hooks/http/google-calendar/{_AGENT_USER_ID_W}"
        )

        import uuid as _uuid

        channel_id = str(_uuid.uuid4())

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{CALENDAR_API}/calendars/primary/events/watch",
                    headers=self._auth_headers(),
                    json={
                        "id": channel_id,
                        "type": "web_hook",
                        "address": hook_url,
                    },
                )
                if resp.status_code == 403:
                    return ToolResult.fail(
                        "Calendar watch registration failed: permission denied. "
                        "Ensure the Google Calendar API is enabled and the OAuth "
                        "scope includes calendar.readonly."
                    )
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as e:
            logger.error("Calendar watch setup failed: %s — %s", e, e.response.text)
            return ToolResult.fail(
                f"Calendar watch setup failed (HTTP {e.response.status_code}): {e.response.text}"
            )
        except Exception as e:
            logger.error("Calendar watch setup error: %s", e)
            return ToolResult.fail(f"Calendar watch setup error: {e}")

        resource_id = data.get("resourceId", "")
        expiration_ms = int(data.get("expiration", 0))

        # Persist watch registration to orchestrator DB
        await self._register_watch(
            watch_id=channel_id,
            resource_id=resource_id,
            protocol="http",
            expires_at=expiration_ms,
        )

        # Schedule renewal 1 day before expiry
        await self._schedule_renewal(
            renew_tool="renew_calendar_watch",
            expires_at_ms=expiration_ms,
            renew_interval_s=518400,
        )

        logger.info(
            "Calendar watch registered: channelId=%s resourceId=%s expiration=%s",
            channel_id,
            resource_id,
            expiration_ms,
        )
        return ToolResult.success(
            "Google Calendar push notifications enabled. "
            "Calendar changes will be delivered in real time. "
            "Watch expires in ~7 days and will be renewed automatically.",
            channel_id=channel_id,
            resource_id=resource_id,
        )


class RenewCalendarWatchTool(BaseWatchTool):
    """Renew the Google Calendar push channel before it expires.

    Called automatically by the cron system 1 day before the channel expires.
    Google Calendar requires stopping the old channel and creating a new one.
    """

    name = "renew_calendar_watch"
    plugin_name = "google-calendar"
    description = (
        "Renew the Google Calendar push notification channel before it expires. "
        "Called automatically by the system. Do not call manually."
    )
    status_text = "Renewing Google Calendar push notifications..."

    async def execute(self, **_) -> ToolResult:  # type: ignore[override]
        # Renewal = create a new channel (Google Calendar doesn't support in-place renewal)
        setup = SetupCalendarWatchTool()
        object.__setattr__(setup, "_context", getattr(self, "_context", None))
        return await setup.execute()


class HandleCalendarEventTool(BaseWatchTool):
    """Process an inbound Google Calendar HTTP push notification.

    Called by the agent when a webhook event arrives from Google Calendar.
    The orchestrator HTTP adapter enriches the payload with x-goog-* headers.

    x-goog-resource-state values:
      "sync"       — watch just created, no real change (already filtered by orchestrator)
      "exists"     — an event was created or updated
      "not_exists" — an event was deleted

    This tool fetches the changed events from the Calendar API and returns
    a summary for the LLM to decide what to do (notify, speak, ignore).
    """

    name = "handle_calendar_event"
    plugin_name = "google-calendar"
    description = (
        "Process an inbound Google Calendar push notification. "
        "Fetches recently changed events and returns their details "
        "so you can decide whether to notify the user or take action."
    )
    status_text = "Fetching updated calendar events..."
    parameters = [
        ToolParam(
            name="payload",
            type="object",
            description="Raw webhook payload from the Google Calendar HTTP push notification",
            required=True,
        ),
    ]

    async def execute(self, payload: dict, **_) -> ToolResult:  # type: ignore[override]
        token = self._get_token()
        if not token:
            return ToolResult.fail("Cannot handle Calendar event: no access token.")

        resource_state = payload.get("x-goog-resource-state", "exists")

        if resource_state == "not_exists":
            return ToolResult.success(
                "A Google Calendar event was deleted. "
                "Use upcoming_events to check your current schedule."
            )

        # Fetch events updated in the last 5 minutes to catch the change
        now = datetime.now(timezone.utc)
        updated_min = (now - timedelta(minutes=5)).isoformat()

        headers = self._auth_headers()

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    f"{CALENDAR_API}/calendars/primary/events",
                    headers=headers,
                    params={
                        "updatedMin": updated_min,
                        "orderBy": "updated",
                        "singleEvents": "true",
                        "maxResults": "5",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as e:
            logger.error("Calendar event handling failed: %s", e)
            return ToolResult.fail(
                f"Failed to fetch Calendar events (HTTP {e.response.status_code})"
            )
        except Exception as e:
            logger.error("Calendar event handling error: %s", e)
            return ToolResult.fail(f"Calendar event handling error: {e}")

        items = data.get("items", [])
        if not items:
            return ToolResult.success(
                "Google Calendar notification received — no recently updated events found."
            )

        summaries: list[str] = []
        for event in items:
            title = event.get("summary", "(No title)")
            start = event.get("start", {})
            start_str = start.get("dateTime", start.get("date", ""))
            status = event.get("status", "confirmed")
            summaries.append(f"{title} @ {start_str} [{status}]")

        count = len(items)
        summary_text = "\n".join(summaries)
        return ToolResult.success(
            f"{count} calendar event{'s' if count > 1 else ''} recently updated:\n{summary_text}\n\n"
            f"Use get_event or upcoming_events for full details.",
            event_count=count,
            events=summaries,
        )
