"""Brave Search tools for real-time web search.

Provides tools to search the live web, news, images, and local places
using the Brave Search API.

Auth: api_key — credentials stored in plugin config as ``api_key``.
All tools read the key from ``self._context`` at runtime.
"""

from __future__ import annotations

import logging

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

BRAVE_BASE = "https://api.search.brave.com/res/v1"


class _BraveTool(AetherTool):
    """Base for Brave Search tools — provides API key extraction."""

    def _get_api_key(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("api_key") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._get_api_key() or "",
        }


class WebSearchTool(_BraveTool):
    """Search the live web using Brave Search."""

    name = "web_search"
    description = "Search the live web for any query. Returns titles, URLs, and snippets from real-time web results."
    status_text = "Searching the web..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The search query",
            required=True,
        ),
        ToolParam(
            name="count",
            type="integer",
            description="Number of results to return (1–20, default 10)",
            required=False,
            default=10,
        ),
        ToolParam(
            name="country",
            type="string",
            description="Country code for localised results (e.g. 'IN' for India, 'US' for USA). Default: IN",
            required=False,
            default="IN",
        ),
    ]

    async def execute(
        self, query: str, count: int = 10, country: str = "IN", **_
    ) -> ToolResult:
        if not self._get_api_key():
            return ToolResult.fail("Brave Search not connected — missing API key.")

        count = min(max(count, 1), 20)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BRAVE_BASE}/web/search",
                    headers=self._auth_headers(),
                    params={"q": query, "count": count, "country": country},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

            web = data.get("web", {})
            results = web.get("results", [])

            if not results:
                return ToolResult.success(f"No web results found for: {query}")

            lines = [f"**Web results for:** {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                desc = r.get("description", "")
                lines.append(f"**{i}. {title}**")
                if desc:
                    lines.append(f"   {desc}")
                lines.append(f"   {url}\n")

            return ToolResult.success(
                "\n".join(lines), result_count=len(results), query=query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Brave Search HTTP error: {e.response.status_code}")
            return ToolResult.fail(
                f"Brave Search error: {e.response.status_code} — {e.response.text[:200]}"
            )
        except Exception as e:
            logger.error(f"Brave Search error: {e}", exc_info=True)
            return ToolResult.fail(f"Search failed: {e}")


class NewsSearchTool(_BraveTool):
    """Search for recent news articles using Brave Search."""

    name = "news_search"
    description = "Search for recent news articles on any topic. Returns headlines, sources, and publication times."
    status_text = "Searching the news..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The news search query",
            required=True,
        ),
        ToolParam(
            name="count",
            type="integer",
            description="Number of news articles to return (1–20, default 10)",
            required=False,
            default=10,
        ),
        ToolParam(
            name="country",
            type="string",
            description="Country code for localised news (e.g. 'IN', 'US'). Default: IN",
            required=False,
            default="IN",
        ),
    ]

    async def execute(
        self, query: str, count: int = 10, country: str = "IN", **_
    ) -> ToolResult:
        if not self._get_api_key():
            return ToolResult.fail("Brave Search not connected — missing API key.")

        count = min(max(count, 1), 20)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BRAVE_BASE}/news/search",
                    headers=self._auth_headers(),
                    params={"q": query, "count": count, "country": country},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])

            if not results:
                return ToolResult.success(f"No news found for: {query}")

            lines = [f"**News results for:** {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                desc = r.get("description", "")
                source = r.get("meta_url", {}).get("hostname", "")
                age = r.get("age", "")

                lines.append(f"**{i}. {title}**")
                if source:
                    lines.append(f"   Source: {source}" + (f" · {age}" if age else ""))
                if desc:
                    lines.append(f"   {desc}")
                lines.append(f"   {url}\n")

            return ToolResult.success(
                "\n".join(lines), result_count=len(results), query=query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Brave News HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Brave News error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Brave News error: {e}", exc_info=True)
            return ToolResult.fail(f"News search failed: {e}")


class LlmContextSearchTool(_BraveTool):
    """Search the web and return LLM-optimised context snippets for grounding answers."""

    name = "llm_context_search"
    description = (
        "Search the web and return structured, LLM-optimised context snippets. "
        "Use this when you need factual grounding for a complex question — it returns "
        "clean, cited text passages ideal for synthesising an answer."
    )
    status_text = "Gathering context from the web..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The question or topic to research",
            required=True,
        ),
    ]

    async def execute(self, query: str, **_) -> ToolResult:
        if not self._get_api_key():
            return ToolResult.fail("Brave Search not connected — missing API key.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BRAVE_BASE}/web/search",
                    headers={**self._auth_headers(), "X-Respond-With": "llm_context"},
                    params={"q": query, "count": 5},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

            # Try llm_context first, fall back to web results
            llm_ctx = data.get("llm_context", {})
            snippets = llm_ctx.get("snippets", [])

            if snippets:
                lines = [f"**Context for:** {query}\n"]
                for s in snippets:
                    text = s.get("text", "")
                    url = s.get("url", "")
                    title = s.get("title", "")
                    if text:
                        lines.append(f"**{title}** ({url})")
                        lines.append(f"{text}\n")
                return ToolResult.success(
                    "\n".join(lines), query=query, source="llm_context"
                )

            # Fallback: use web results
            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                return ToolResult.success(f"No context found for: {query}")

            lines = [f"**Context for:** {query}\n"]
            for r in web_results[:5]:
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("description", "")
                if desc:
                    lines.append(f"**{title}** ({url})")
                    lines.append(f"{desc}\n")

            return ToolResult.success(
                "\n".join(lines), query=query, source="web_fallback"
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Brave LLM context HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Brave Search error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Brave LLM context error: {e}", exc_info=True)
            return ToolResult.fail(f"Context search failed: {e}")


class ImageSearchTool(_BraveTool):
    """Search for images using Brave Search."""

    name = "image_search"
    description = (
        "Search for images on any topic. Returns image titles, URLs, and source pages."
    )
    status_text = "Searching for images..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The image search query",
            required=True,
        ),
        ToolParam(
            name="count",
            type="integer",
            description="Number of image results to return (1–20, default 5)",
            required=False,
            default=5,
        ),
    ]

    async def execute(self, query: str, count: int = 5, **_) -> ToolResult:
        if not self._get_api_key():
            return ToolResult.fail("Brave Search not connected — missing API key.")

        count = min(max(count, 1), 20)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BRAVE_BASE}/images/search",
                    headers=self._auth_headers(),
                    params={"q": query, "count": count},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])

            if not results:
                return ToolResult.success(f"No images found for: {query}")

            lines = [f"**Image results for:** {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                source = r.get("source", "")
                img_url = r.get("properties", {}).get("url", "")

                lines.append(f"**{i}. {title}**")
                if source:
                    lines.append(f"   Source: {source}")
                if img_url:
                    lines.append(f"   Image: {img_url}")
                if url:
                    lines.append(f"   Page: {url}\n")

            return ToolResult.success(
                "\n".join(lines), result_count=len(results), query=query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Brave Image Search HTTP error: {e.response.status_code}")
            return ToolResult.fail(
                f"Brave Image Search error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Brave Image Search error: {e}", exc_info=True)
            return ToolResult.fail(f"Image search failed: {e}")


class LocalSearchTool(_BraveTool):
    """Search for local places, businesses, and services near a location."""

    name = "local_search"
    description = (
        "Search for local places, restaurants, businesses, or services near a location. "
        "Returns place names, addresses, ratings, and hours. "
        "Use for queries like 'restaurants near Koramangala' or 'ATMs in Bandra'."
    )
    status_text = "Searching nearby places..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="What to search for (e.g. 'biryani restaurants', 'ATM', 'pharmacy')",
            required=True,
        ),
        ToolParam(
            name="location",
            type="string",
            description="The location to search near (e.g. 'Koramangala, Bangalore', 'Bandra, Mumbai')",
            required=True,
        ),
        ToolParam(
            name="count",
            type="integer",
            description="Number of results to return (1–20, default 5)",
            required=False,
            default=5,
        ),
    ]

    async def execute(
        self, query: str, location: str, count: int = 5, **_
    ) -> ToolResult:
        if not self._get_api_key():
            return ToolResult.fail("Brave Search not connected — missing API key.")

        count = min(max(count, 1), 20)
        combined_query = f"{query} near {location}"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BRAVE_BASE}/web/search",
                    headers=self._auth_headers(),
                    params={
                        "q": combined_query,
                        "count": count,
                        "result_filter": "locations",
                        "country": "IN",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

            # Try locations results first
            locations = data.get("locations", {}).get("results", [])

            if locations:
                lines = [f"**Places matching '{query}' near {location}:**\n"]
                for i, place in enumerate(locations[:count], 1):
                    name = place.get("title", "Unknown")
                    address = place.get("address", {})
                    addr_str = ", ".join(
                        filter(
                            None,
                            [
                                address.get("streetAddress", ""),
                                address.get("addressLocality", ""),
                                address.get("addressRegion", ""),
                            ],
                        )
                    )
                    rating = place.get("rating", {})
                    rating_val = rating.get("ratingValue", "")
                    rating_count = rating.get("ratingCount", "")
                    phone = place.get("phone", "")
                    hours = place.get("openingHours", [])

                    lines.append(f"**{i}. {name}**")
                    if addr_str:
                        lines.append(f"   📍 {addr_str}")
                    if rating_val:
                        lines.append(
                            f"   ⭐ {rating_val}"
                            + (f" ({rating_count} reviews)" if rating_count else "")
                        )
                    if phone:
                        lines.append(f"   📞 {phone}")
                    if hours:
                        lines.append(
                            f"   🕐 {hours[0] if isinstance(hours, list) else hours}"
                        )
                    lines.append("")

                return ToolResult.success(
                    "\n".join(lines), result_count=len(locations), query=combined_query
                )

            # Fallback: use web results
            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                return ToolResult.success(
                    f"No local results found for '{query}' near {location}."
                )

            lines = [f"**Web results for '{query}' near {location}:**\n"]
            for i, r in enumerate(web_results[:count], 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                desc = r.get("description", "")
                lines.append(f"**{i}. {title}**")
                if desc:
                    lines.append(f"   {desc}")
                lines.append(f"   {url}\n")

            return ToolResult.success(
                "\n".join(lines), result_count=len(web_results), query=combined_query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Brave Local Search HTTP error: {e.response.status_code}")
            return ToolResult.fail(
                f"Brave Local Search error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Brave Local Search error: {e}", exc_info=True)
            return ToolResult.fail(f"Local search failed: {e}")
