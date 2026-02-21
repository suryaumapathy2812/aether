"""Wikipedia tools — zero-auth encyclopaedic knowledge.

Provides search and full article retrieval from Wikipedia's REST API.
No API key required — completely free and open.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1"


class WikipediaSearchTool(AetherTool):
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
                # Try direct summary lookup first
                resp = await client.get(
                    f"{WIKIPEDIA_API}/page/summary/{encoded}",
                    headers={"User-Agent": "Aether/1.0 (personal AI agent)"},
                    follow_redirects=True,
                    timeout=10,
                )

                if resp.status_code == 404:
                    # Fall back to MediaWiki search to find the right title
                    search_resp = await client.get(
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
                    search_resp.raise_for_status()
                    hits = search_resp.json().get("query", {}).get("search", [])
                    if not hits:
                        return ToolResult.success(
                            f"No Wikipedia article found for: {query}"
                        )

                    title = hits[0]["title"]
                    resp = await client.get(
                        f"{WIKIPEDIA_API}/page/summary/{urllib.parse.quote(title)}",
                        headers={"User-Agent": "Aether/1.0"},
                        follow_redirects=True,
                        timeout=10,
                    )

                resp.raise_for_status()
                data = resp.json()

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


class WikipediaGetArticleTool(AetherTool):
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
                    f"No Wikipedia article found with title: '{title}'. "
                    "Try wikipedia_search instead."
                )
            logger.error(f"Wikipedia get article HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Wikipedia error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Wikipedia get article error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to fetch article: {e}")
