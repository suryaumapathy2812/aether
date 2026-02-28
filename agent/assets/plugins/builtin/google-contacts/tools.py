"""Google Contacts tools for searching and looking up contacts.

Uses the Google People API v1. Each plugin manages its own OAuth tokens
independently. Credentials arrive via ``self._context`` at call time.
"""

from __future__ import annotations

import logging

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult
from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

logger = logging.getLogger(__name__)

PEOPLE_API = "https://people.googleapis.com/v1"


class _ContactsTool(AetherTool):
    """Base for Contacts tools — provides token extraction from runtime context."""

    def _get_token(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("access_token") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}


def _format_contact(person: dict) -> str:
    """Format a contact person into a readable string."""
    names = person.get("names", [])
    name = names[0].get("displayName", "Unknown") if names else "Unknown"

    emails = person.get("emailAddresses", [])
    email_str = emails[0].get("value", "") if emails else ""

    phones = person.get("phoneNumbers", [])
    phone_str = phones[0].get("value", "") if phones else ""

    orgs = person.get("organizations", [])
    org_str = ""
    if orgs:
        org_name = orgs[0].get("name", "")
        org_title = orgs[0].get("title", "")
        if org_title and org_name:
            org_str = f"{org_title} at {org_name}"
        elif org_name:
            org_str = org_name

    line = f"**{name}**"
    if email_str:
        line += f"\n   Email: {email_str}"
    if phone_str:
        line += f"\n   Phone: {phone_str}"
    if org_str:
        line += f"\n   {org_str}"

    return line


class SearchContactsTool(_ContactsTool):
    """Search contacts by name, email, or phone."""

    name = "search_contacts"
    description = "Search your Google Contacts by name, email, or phone number"
    status_text = "Searching contacts..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="Search query (name, email, or phone)",
            required=True,
        ),
        ToolParam(
            name="max_results",
            type="integer",
            description="Max contacts to return (default 10)",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, query: str, max_results: int = 10, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail(
                "Google Contacts not connected — missing access token."
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{PEOPLE_API}/people:searchContacts",
                    headers=self._auth_headers(),
                    params={
                        "query": query,
                        "readMask": "names,emailAddresses,phoneNumbers,organizations",
                        "pageSize": max_results,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            if not results:
                return ToolResult.success(f"No contacts found matching '{query}'.")

            output = f"**Contacts matching '{query}':**\n"
            for i, result in enumerate(results, 1):
                person = result.get("person", {})
                output += f"\n{i}. {_format_contact(person)}"

            return ToolResult.success(output, count=len(results))

        except Exception as e:
            logger.error(f"Error searching contacts: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class GetContactTool(_ContactsTool):
    """Get detailed info about a specific contact."""

    name = "get_contact"
    description = "Get detailed information about a specific contact by resource name"
    status_text = "Looking up contact..."
    parameters = [
        ToolParam(
            name="resource_name",
            type="string",
            description="Contact resource name (e.g. people/c1234567890)",
            required=True,
        ),
    ]

    async def execute(self, resource_name: str, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail(
                "Google Contacts not connected — missing access token."
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{PEOPLE_API}/{resource_name}",
                    headers=self._auth_headers(),
                    params={
                        "personFields": "names,emailAddresses,phoneNumbers,organizations,addresses,birthdays,biographies",
                    },
                )
                resp.raise_for_status()
                person = resp.json()

            output = _format_contact(person)

            # Add extra details
            addresses = person.get("addresses", [])
            if addresses:
                addr = addresses[0].get("formattedValue", "")
                if addr:
                    output += f"\n   Address: {addr}"

            birthdays = person.get("birthdays", [])
            if birthdays:
                bday = birthdays[0].get("date", {})
                if bday:
                    month = bday.get("month", "")
                    day = bday.get("day", "")
                    year = bday.get("year", "")
                    if month and day:
                        output += f"\n   Birthday: {month}/{day}"
                        if year:
                            output += f"/{year}"

            bios = person.get("biographies", [])
            if bios:
                bio = bios[0].get("value", "")
                if bio:
                    output += f"\n   Notes: {bio[:200]}"

            return ToolResult.success(output)

        except Exception as e:
            logger.error(f"Error fetching contact: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RefreshGoogleContactsTokenTool(RefreshOAuthTokenTool):
    """Refresh the Google Contacts OAuth access token before it expires.

    Called automatically by the cron system every 50 minutes.
    Can also be called manually if Contacts tools start returning auth errors.
    """

    name = "refresh_google_contacts_token"
    plugin_name = "google-contacts"
    description = (
        "Refresh the Google Contacts OAuth access token. "
        "Call this when Contacts tools return authentication errors, "
        "or when instructed by the system to prevent token expiry."
    )
