"""World time built-in tool.

Returns the current date and time for any timezone or city.
Uses Python's stdlib zoneinfo module — no network call, no external dependency.

Built-in tool — always available, no plugin enable/disable needed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)


class WorldTimeTool(AetherTool):
    """Get the current date and time in any timezone or city."""

    name = "world_time"
    description = (
        "Get the current date and time in any timezone or city. "
        "Examples: 'What time is it in San Francisco?', 'current time in Tokyo', 'time in IST'."
    )
    status_text = "Checking the time..."
    parameters = [
        ToolParam(
            name="timezone",
            type="string",
            description=(
                "Timezone name or city. Accepts IANA timezone names (e.g. 'Asia/Kolkata', "
                "'America/New_York', 'Europe/London'), common abbreviations (e.g. 'IST', 'PST', 'UTC'), "
                "or city names (e.g. 'Mumbai', 'Tokyo', 'London')."
            ),
            required=True,
        ),
    ]

    # Common abbreviation → IANA timezone mapping
    TZ_ALIASES: dict[str, str] = {
        "IST": "Asia/Kolkata",
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles",
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "CST": "America/Chicago",
        "CDT": "America/Chicago",
        "MST": "America/Denver",
        "MDT": "America/Denver",
        "GMT": "Europe/London",
        "BST": "Europe/London",
        "CET": "Europe/Paris",
        "CEST": "Europe/Paris",
        "JST": "Asia/Tokyo",
        "SGT": "Asia/Singapore",
        "AEST": "Australia/Sydney",
        "AEDT": "Australia/Sydney",
        "UAE": "Asia/Dubai",
        "GST": "Asia/Dubai",
        "PKT": "Asia/Karachi",
        "BDT": "Asia/Dhaka",
        "NPT": "Asia/Kathmandu",
        "HKT": "Asia/Hong_Kong",
        "KST": "Asia/Seoul",
        "CST_CHINA": "Asia/Shanghai",
        "MSK": "Europe/Moscow",
        "UTC": "UTC",
    }

    # City name → IANA timezone mapping
    CITY_TZ: dict[str, str] = {
        "mumbai": "Asia/Kolkata",
        "delhi": "Asia/Kolkata",
        "new delhi": "Asia/Kolkata",
        "bangalore": "Asia/Kolkata",
        "bengaluru": "Asia/Kolkata",
        "chennai": "Asia/Kolkata",
        "hyderabad": "Asia/Kolkata",
        "kolkata": "Asia/Kolkata",
        "pune": "Asia/Kolkata",
        "ahmedabad": "Asia/Kolkata",
        "india": "Asia/Kolkata",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "amsterdam": "Europe/Amsterdam",
        "rome": "Europe/Rome",
        "madrid": "Europe/Madrid",
        "new york": "America/New_York",
        "nyc": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "la": "America/Los_Angeles",
        "san francisco": "America/Los_Angeles",
        "sf": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "toronto": "America/Toronto",
        "vancouver": "America/Vancouver",
        "mexico city": "America/Mexico_City",
        "sao paulo": "America/Sao_Paulo",
        "buenos aires": "America/Argentina/Buenos_Aires",
        "tokyo": "Asia/Tokyo",
        "osaka": "Asia/Tokyo",
        "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai",
        "hong kong": "Asia/Hong_Kong",
        "singapore": "Asia/Singapore",
        "seoul": "Asia/Seoul",
        "dubai": "Asia/Dubai",
        "abu dhabi": "Asia/Dubai",
        "riyadh": "Asia/Riyadh",
        "istanbul": "Europe/Istanbul",
        "moscow": "Europe/Moscow",
        "sydney": "Australia/Sydney",
        "melbourne": "Australia/Melbourne",
        "auckland": "Pacific/Auckland",
        "karachi": "Asia/Karachi",
        "dhaka": "Asia/Dhaka",
        "kathmandu": "Asia/Kathmandu",
        "colombo": "Asia/Colombo",
        "nairobi": "Africa/Nairobi",
        "cairo": "Africa/Cairo",
        "lagos": "Africa/Lagos",
        "johannesburg": "Africa/Johannesburg",
    }

    DAY_NAMES = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    async def execute(self, timezone: str, **_) -> ToolResult:
        tz_input = timezone.strip()
        tz_upper = tz_input.upper()
        tz_lower = tz_input.lower()

        # Resolve alias or city name → IANA timezone
        resolved_tz = (
            self.TZ_ALIASES.get(tz_upper)
            or self.CITY_TZ.get(tz_lower)
            or tz_input  # assume it's already a valid IANA name
        )

        try:
            tz = ZoneInfo(resolved_tz)
        except ZoneInfoNotFoundError:
            return ToolResult.fail(
                f"Timezone '{timezone}' not recognised. "
                "Try an IANA timezone name like 'Asia/Kolkata', 'America/New_York', "
                "or 'Europe/London', or a city name like 'Mumbai', 'Tokyo', 'London'."
            )
        except Exception as e:
            logger.error("ZoneInfo error for '%s': %s", resolved_tz, e)
            return ToolResult.fail(f"Could not load timezone '{timezone}': {e}")

        now = datetime.now(tz)

        dt_clean = now.strftime("%Y-%m-%d %H:%M:%S")
        day_name = self.DAY_NAMES[now.weekday()]
        utc_offset = now.strftime("%z")  # e.g. "+0530"
        # Format as "+05:30"
        if utc_offset and len(utc_offset) == 5:
            utc_offset = f"{utc_offset[:3]}:{utc_offset[3:]}"

        # Use the tzname from the datetime (e.g. "IST", "EST", "UTC")
        tz_label = now.strftime("%Z") or resolved_tz

        output = f"**Current time in {tz_input}** ({tz_label})\n"
        output += f"🕐 **{dt_clean}** — {day_name}"
        if utc_offset:
            output += f"\nUTC offset: {utc_offset}"

        return ToolResult.success(
            output,
            timezone=resolved_tz,
            datetime=now.isoformat(),
        )
