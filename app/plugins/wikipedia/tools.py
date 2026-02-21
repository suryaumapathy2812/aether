"""Wikipedia & Knowledge tools — zero/minimal auth knowledge layer.

Provides:
- Wikipedia search and full article retrieval (no auth)
- Wolfram Alpha computational queries (api_key optional)
- Live currency conversion (api_key optional)
- World time zone lookup (no auth)

Auth: mixed — Wikipedia and World Time require no auth.
Wolfram Alpha and Currency use optional api_key fields from plugin config.
All keys read from ``self._context`` at runtime.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1"
WOLFRAM_API = "https://api.wolframalpha.com/v1/result"
EXCHANGERATE_API = "https://v6.exchangerate-api.com/v6"
WORLDTIME_API = "https://worldtimeapi.org/api/timezone"


class _WikipediaTool(AetherTool):
    """Base for Wikipedia tools — no auth needed."""

    pass


class _KnowledgeTool(AetherTool):
    """Base for knowledge tools with optional API keys."""

    def _get_wolfram_key(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("wolfram_app_id") if ctx else None

    def _get_exchangerate_key(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("exchangerate_api_key") if ctx else None


class WikipediaSearchTool(_WikipediaTool):
    """Search Wikipedia and return a concise summary of the top result."""

    name = "wikipedia_search"
    description = (
        "Search Wikipedia for any topic and return a concise summary. "
        "Use for factual questions about people, places, events, concepts, history, science, etc."
    )
    status_text = "Searching Wikipedia..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The topic or question to search Wikipedia for",
            required=True,
        ),
    ]

    async def execute(self, query: str, **_) -> ToolResult:
        try:
            encoded = urllib.parse.quote(query)

            async with httpx.AsyncClient() as client:
                # First: search for the best matching article title
                search_resp = await client.get(
                    f"{WIKIPEDIA_API}/page/summary/{encoded}",
                    headers={"User-Agent": "Aether/1.0 (personal AI agent)"},
                    follow_redirects=True,
                    timeout=10,
                )

                if search_resp.status_code == 404:
                    # Try search endpoint to find the right title
                    search_resp2 = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "format": "json",
                            "srlimit": 1,
                        },
                        headers={"User-Agent": "Aether/1.0"},
                        timeout=10,
                    )
                    search_resp2.raise_for_status()
                    search_data = search_resp2.json()
                    hits = search_data.get("query", {}).get("search", [])
                    if not hits:
                        return ToolResult.success(
                            f"No Wikipedia article found for: {query}"
                        )

                    title = hits[0]["title"]
                    encoded_title = urllib.parse.quote(title)
                    search_resp = await client.get(
                        f"{WIKIPEDIA_API}/page/summary/{encoded_title}",
                        headers={"User-Agent": "Aether/1.0"},
                        follow_redirects=True,
                        timeout=10,
                    )

                search_resp.raise_for_status()
                data = search_resp.json()

            title = data.get("title", query)
            extract = data.get("extract", "")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            description = data.get("description", "")

            if not extract:
                return ToolResult.success(f"No summary available for: {query}")

            output = f"**{title}**"
            if description:
                output += f" — *{description}*"
            output += f"\n\n{extract}"
            if page_url:
                output += f"\n\n[Read more on Wikipedia]({page_url})"

            return ToolResult.success(output, title=title, url=page_url)

        except httpx.HTTPStatusError as e:
            logger.error(f"Wikipedia HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Wikipedia error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}", exc_info=True)
            return ToolResult.fail(f"Wikipedia search failed: {e}")


class WikipediaGetArticleTool(_WikipediaTool):
    """Get the full text of a Wikipedia article by exact title."""

    name = "wikipedia_get_article"
    description = (
        "Get the full text content of a Wikipedia article by its exact title. "
        "Use when you need more detail than the summary provides."
    )
    status_text = "Fetching Wikipedia article..."
    parameters = [
        ToolParam(
            name="title",
            type="string",
            description="The exact Wikipedia article title (e.g. 'Indian Space Research Organisation')",
            required=True,
        ),
    ]

    async def execute(self, title: str, **_) -> ToolResult:
        try:
            encoded = urllib.parse.quote(title)

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{WIKIPEDIA_API}/page/summary/{encoded}",
                    headers={"User-Agent": "Aether/1.0"},
                    follow_redirects=True,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

            article_title = data.get("title", title)
            extract = data.get("extract", "")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            description = data.get("description", "")

            if not extract:
                return ToolResult.success(f"No content available for: {title}")

            output = f"**{article_title}**"
            if description:
                output += f" — *{description}*"
            output += f"\n\n{extract}"
            if page_url:
                output += f"\n\n[Wikipedia: {article_title}]({page_url})"

            return ToolResult.success(output, title=article_title, url=page_url)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return ToolResult.fail(
                    f"No Wikipedia article found with title: '{title}'. Try wikipedia_search instead."
                )
            logger.error(f"Wikipedia get article HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Wikipedia error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Wikipedia get article error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to fetch article: {e}")


class WolframQueryTool(_KnowledgeTool):
    """Query Wolfram Alpha for mathematical, scientific, and computational answers."""

    name = "wolfram_query"
    description = (
        "Query Wolfram Alpha for mathematical computations, unit conversions, scientific facts, "
        "and complex calculations. Examples: '15% of 3200', 'integral of x^2', "
        "'distance from Mumbai to Delhi', 'calories in 100g of rice'."
    )
    status_text = "Computing with Wolfram Alpha..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The mathematical, scientific, or computational query",
            required=True,
        ),
    ]

    async def execute(self, query: str, **_) -> ToolResult:
        app_id = self._get_wolfram_key()
        if not app_id:
            return ToolResult.fail(
                "Wolfram Alpha is not configured — no App ID set. "
                "Add your Wolfram Alpha App ID in the Wikipedia plugin settings."
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    WOLFRAM_API,
                    params={"input": query, "appid": app_id},
                    timeout=15,
                )

                if resp.status_code == 501:
                    return ToolResult.fail(
                        f"Wolfram Alpha couldn't compute an answer for: '{query}'. "
                        "Try rephrasing the query more precisely."
                    )

                resp.raise_for_status()
                answer = resp.text.strip()

            if not answer:
                return ToolResult.fail(f"Wolfram Alpha returned no answer for: {query}")

            return ToolResult.success(
                f"**Wolfram Alpha:** {query}\n\n**Answer:** {answer}",
                query=query,
                answer=answer,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Wolfram Alpha HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Wolfram Alpha error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Wolfram Alpha error: {e}", exc_info=True)
            return ToolResult.fail(f"Wolfram Alpha query failed: {e}")


class CurrencyConvertTool(_KnowledgeTool):
    """Convert between currencies using live exchange rates."""

    name = "currency_convert"
    description = (
        "Convert an amount between any two currencies using live exchange rates. "
        "Examples: 'convert 100 USD to INR', '500 EUR to GBP', '1000 INR to USD'."
    )
    status_text = "Fetching live exchange rates..."
    parameters = [
        ToolParam(
            name="amount",
            type="number",
            description="The amount to convert",
            required=True,
        ),
        ToolParam(
            name="from_currency",
            type="string",
            description="The source currency code (e.g. 'USD', 'EUR', 'INR', 'GBP')",
            required=True,
        ),
        ToolParam(
            name="to_currency",
            type="string",
            description="The target currency code (e.g. 'INR', 'USD', 'EUR')",
            required=True,
        ),
    ]

    async def execute(
        self, amount: float, from_currency: str, to_currency: str, **_
    ) -> ToolResult:
        api_key = self._get_exchangerate_key()

        from_currency = from_currency.upper().strip()
        to_currency = to_currency.upper().strip()

        try:
            if api_key:
                # Use ExchangeRate-API with key for live rates
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{EXCHANGERATE_API}/{api_key}/pair/{from_currency}/{to_currency}/{amount}",
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                if data.get("result") != "success":
                    error = data.get("error-type", "unknown error")
                    return ToolResult.fail(f"Currency conversion failed: {error}")

                converted = data.get("conversion_result", 0)
                rate = data.get("conversion_rate", 0)
                last_updated = data.get("time_last_update_utc", "")

                output = (
                    f"**{amount:,.2f} {from_currency}** = **{converted:,.2f} {to_currency}**\n"
                    f"Exchange rate: 1 {from_currency} = {rate} {to_currency}"
                )
                if last_updated:
                    output += f"\n*Rates updated: {last_updated}*"

                return ToolResult.success(
                    output,
                    amount=amount,
                    from_currency=from_currency,
                    to_currency=to_currency,
                    converted=converted,
                    rate=rate,
                )
            else:
                # Fallback: use open.er-api.com (free, no key, less reliable)
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"https://open.er-api.com/v6/latest/{from_currency}",
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                rates = data.get("rates", {})
                if to_currency not in rates:
                    return ToolResult.fail(
                        f"Currency '{to_currency}' not found. Check the currency code."
                    )

                rate = rates[to_currency]
                converted = amount * rate

                output = (
                    f"**{amount:,.2f} {from_currency}** = **{converted:,.2f} {to_currency}**\n"
                    f"Exchange rate: 1 {from_currency} = {rate:.4f} {to_currency}\n"
                    f"*Note: Add an ExchangeRate-API key in plugin settings for more reliable rates.*"
                )

                return ToolResult.success(
                    output,
                    amount=amount,
                    from_currency=from_currency,
                    to_currency=to_currency,
                    converted=converted,
                    rate=rate,
                )

        except httpx.HTTPStatusError as e:
            logger.error(f"Currency convert HTTP error: {e.response.status_code}")
            return ToolResult.fail(
                f"Currency conversion error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Currency convert error: {e}", exc_info=True)
            return ToolResult.fail(f"Currency conversion failed: {e}")


class WorldTimeTool(AetherTool):
    """Get the current time in any timezone or city."""

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
                "'America/New_York', 'Europe/London') or common abbreviations (e.g. 'IST', 'PST', 'UTC')."
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
        "MST": "America/Denver",
        "GMT": "Europe/London",
        "BST": "Europe/London",
        "CET": "Europe/Paris",
        "JST": "Asia/Tokyo",
        "SGT": "Asia/Singapore",
        "AEST": "Australia/Sydney",
        "UAE": "Asia/Dubai",
        "GST": "Asia/Dubai",
        "PKT": "Asia/Karachi",
        "BDT": "Asia/Dhaka",
        "NPT": "Asia/Kathmandu",
    }

    # City → IANA timezone mapping for common cities
    CITY_TZ: dict[str, str] = {
        "mumbai": "Asia/Kolkata",
        "delhi": "Asia/Kolkata",
        "bangalore": "Asia/Kolkata",
        "bengaluru": "Asia/Kolkata",
        "chennai": "Asia/Kolkata",
        "hyderabad": "Asia/Kolkata",
        "kolkata": "Asia/Kolkata",
        "india": "Asia/Kolkata",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "new york": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "san francisco": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "tokyo": "Asia/Tokyo",
        "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai",
        "singapore": "Asia/Singapore",
        "dubai": "Asia/Dubai",
        "sydney": "Australia/Sydney",
        "melbourne": "Australia/Melbourne",
        "toronto": "America/Toronto",
        "moscow": "Europe/Moscow",
        "karachi": "Asia/Karachi",
        "dhaka": "Asia/Dhaka",
        "kathmandu": "Asia/Kathmandu",
    }

    async def execute(self, timezone: str, **_) -> ToolResult:
        # Resolve aliases and city names
        tz_input = timezone.strip()
        tz_upper = tz_input.upper()
        tz_lower = tz_input.lower()

        resolved_tz = (
            self.TZ_ALIASES.get(tz_upper)
            or self.CITY_TZ.get(tz_lower)
            or tz_input  # assume it's already an IANA name
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{WORLDTIME_API}/{urllib.parse.quote(resolved_tz)}",
                    timeout=10,
                )

                if resp.status_code == 404:
                    # Try listing available timezones for a hint
                    return ToolResult.fail(
                        f"Timezone '{timezone}' not recognised. "
                        "Try using an IANA timezone name like 'Asia/Kolkata', 'America/New_York', or 'Europe/London'."
                    )

                resp.raise_for_status()
                data = resp.json()

            datetime_str = data.get("datetime", "")
            utc_offset = data.get("utc_offset", "")
            day_of_week = data.get("day_of_week", "")
            abbreviation = data.get("abbreviation", "")

            # Parse datetime for clean display
            if datetime_str:
                # Format: 2026-02-21T14:30:00.123456+05:30
                dt_clean = datetime_str[:19].replace("T", " ")
                day_names = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                day_name = (
                    day_names[day_of_week]
                    if isinstance(day_of_week, int) and 0 <= day_of_week <= 6
                    else ""
                )

                output = (
                    f"**Current time in {timezone}** ({abbreviation or resolved_tz})\n"
                )
                output += f"🕐 **{dt_clean}**"
                if day_name:
                    output += f" — {day_name}"
                if utc_offset:
                    output += f"\nUTC offset: {utc_offset}"
            else:
                output = (
                    f"Time data received for {timezone} but could not parse datetime."
                )

            return ToolResult.success(
                output, timezone=resolved_tz, datetime=datetime_str
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"World Time HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"World Time API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"World Time error: {e}", exc_info=True)
            return ToolResult.fail(f"Time lookup failed: {e}")
