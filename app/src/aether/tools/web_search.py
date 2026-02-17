"""Web Search tool — search the web using a simple API.

Uses DuckDuckGo instant answer API (no API key needed) as the default.
Can be swapped to any search provider later.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.duckduckgo.com/"
MAX_RESULTS = 5


class WebSearchTool(AetherTool):
    name = "web_search"
    description = "Search the web for information. Returns a summary of top results. Use this when you need current information or facts you're unsure about."
    status_text = "Searching the web..."
    parameters = [
        ToolParam(name="query", type="string", description="The search query"),
    ]

    async def execute(self, query: str) -> ToolResult:
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            })
            url = f"{SEARCH_URL}?{params}"

            req = urllib.request.Request(url, headers={"User-Agent": "Aether/0.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []

            # Abstract (main answer)
            if data.get("Abstract"):
                results.append(f"**{data.get('Heading', 'Result')}**\n{data['Abstract']}")
                if data.get("AbstractURL"):
                    results.append(f"Source: {data['AbstractURL']}")

            # Related topics
            for topic in data.get("RelatedTopics", [])[:MAX_RESULTS]:
                if isinstance(topic, dict) and topic.get("Text"):
                    text = topic["Text"][:200]
                    url = topic.get("FirstURL", "")
                    results.append(f"- {text}")
                    if url:
                        results.append(f"  {url}")

            if not results:
                # Fallback: try the Answer field
                if data.get("Answer"):
                    return ToolResult.success(data["Answer"], query=query)
                return ToolResult.success(
                    f"No results found for: {query}. Try a different search query.",
                    query=query,
                )

            return ToolResult.success("\n".join(results), query=query)

        except Exception as e:
            logger.error(f"Web search failed: {e}", exc_info=True)
            return ToolResult.fail(f"Search failed: {e}")
