"""RSS Feeds and Hacker News tools.

Provides tools to subscribe to and read any RSS/Atom feed, plus dedicated
Hacker News tools using the free HN Firebase and Algolia APIs.

Auth: none — all APIs are public and free.
Feed subscriptions are stored in Aether's memory store via save_memory.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from html.parser import HTMLParser

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

HN_FIREBASE_BASE = "https://hacker-news.firebaseio.com/v0"
HN_ALGOLIA_BASE = "https://hn.algolia.com/api/v1"

# Simple HTML tag stripper
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text.strip()


def _truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


class FetchFeedTool(AetherTool):
    """Fetch and parse an RSS or Atom feed URL."""

    name = "fetch_feed"
    description = (
        "Fetch and parse any RSS or Atom feed URL. Returns recent items with titles, "
        "dates, summaries, and links. Use for tech blogs, news sites, newsletters, podcasts, etc."
    )
    status_text = "Fetching feed..."
    parameters = [
        ToolParam(
            name="url",
            type="string",
            description="The RSS or Atom feed URL to fetch",
            required=True,
        ),
        ToolParam(
            name="max_items",
            type="integer",
            description="Maximum number of items to return (1–20, default 10)",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, url: str, max_items: int = 10, **_) -> ToolResult:
        max_items = min(max(max_items, 1), 20)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Aether/1.0 RSS Reader"},
                    follow_redirects=True,
                    timeout=15,
                )
                resp.raise_for_status()
                content = resp.text

            items = _parse_feed(content, max_items)

            if not items:
                return ToolResult.success(f"No items found in feed: {url}")

            feed_title = _extract_feed_title(content)
            lines = [f"**{feed_title or url}** — {len(items)} recent items\n"]

            for i, item in enumerate(items, 1):
                title = item.get("title", "No title")
                link = item.get("link", "")
                summary = item.get("summary", "")
                pub_date = item.get("pub_date", "")

                lines.append(f"**{i}. {title}**")
                if pub_date:
                    lines.append(f"   📅 {pub_date}")
                if summary:
                    lines.append(f"   {_truncate(_strip_html(summary))}")
                if link:
                    lines.append(f"   🔗 {link}")
                lines.append("")

            return ToolResult.success(
                "\n".join(lines), item_count=len(items), feed_url=url
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Feed fetch HTTP error: {e.response.status_code} for {url}")
            return ToolResult.fail(
                f"Could not fetch feed ({e.response.status_code}): {url}"
            )
        except Exception as e:
            logger.error(f"Feed fetch error for {url}: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to fetch feed: {e}")


class GetItemContentTool(AetherTool):
    """Fetch and extract the full text content of an article URL."""

    name = "get_item_content"
    description = (
        "Fetch and extract the readable text content from any article URL. "
        "Use to read the full text of a feed item or web article."
    )
    status_text = "Fetching article content..."
    parameters = [
        ToolParam(
            name="url",
            type="string",
            description="The article URL to fetch and extract text from",
            required=True,
        ),
    ]

    async def execute(self, url: str, **_) -> ToolResult:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; Aether/1.0)",
                        "Accept": "text/html,application/xhtml+xml",
                    },
                    follow_redirects=True,
                    timeout=20,
                )
                resp.raise_for_status()
                html = resp.text

            # Extract title
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL
            )
            title = _strip_html(title_match.group(1)) if title_match else ""

            # Extract meta description
            desc_match = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
                html,
                re.IGNORECASE,
            )
            description = desc_match.group(1) if desc_match else ""

            # Extract article/main content (best-effort)
            # Try <article>, then <main>, then <div class="content">, then body
            content = ""
            for pattern in [
                r"<article[^>]*>(.*?)</article>",
                r"<main[^>]*>(.*?)</main>",
                r'<div[^>]+class=["\'][^"\']*(?:content|article|post|entry)[^"\']*["\'][^>]*>(.*?)</div>',
            ]:
                match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                if match:
                    content = _strip_html(match.group(1))
                    # Clean up excessive whitespace
                    content = re.sub(r"\n{3,}", "\n\n", content)
                    content = re.sub(r" {2,}", " ", content)
                    content = content.strip()
                    if len(content) > 200:
                        break

            if not content and description:
                content = description

            if not content:
                return ToolResult.fail(
                    f"Could not extract readable content from: {url}"
                )

            output = ""
            if title:
                output += f"**{title}**\n\n"
            output += _truncate(content, 3000)
            output += f"\n\n🔗 [Read full article]({url})"

            return ToolResult.success(output, url=url, title=title)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Article fetch HTTP error: {e.response.status_code} for {url}"
            )
            return ToolResult.fail(
                f"Could not fetch article ({e.response.status_code}): {url}"
            )
        except Exception as e:
            logger.error(f"Article fetch error for {url}: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to fetch article: {e}")


class ListSubscribedFeedsTool(AetherTool):
    """List all RSS feeds the user has subscribed to."""

    name = "list_subscribed_feeds"
    description = (
        "List all RSS/Atom feeds the user has saved. "
        "Returns feed names and URLs from the subscription list."
    )
    status_text = "Loading subscribed feeds..."
    parameters = []

    async def execute(self, **_) -> ToolResult:
        ctx = getattr(self, "_context", {}) or {}
        feeds_json = ctx.get("subscribed_feeds", "[]")

        try:
            feeds = (
                json.loads(feeds_json) if isinstance(feeds_json, str) else feeds_json
            )
        except (json.JSONDecodeError, TypeError):
            feeds = []

        if not feeds:
            return ToolResult.success(
                "No feeds subscribed yet. Use `add_feed` to add RSS/Atom feed URLs."
            )

        lines = [f"**Subscribed feeds ({len(feeds)}):**\n"]
        for i, feed in enumerate(feeds, 1):
            name = feed.get("name", feed.get("url", "Unknown"))
            url = feed.get("url", "")
            lines.append(f"**{i}. {name}**")
            if url:
                lines.append(f"   {url}")
            lines.append("")

        return ToolResult.success("\n".join(lines), feed_count=len(feeds))


class AddFeedTool(AetherTool):
    """Add a new RSS/Atom feed to the subscription list."""

    name = "add_feed"
    description = (
        "Add a new RSS or Atom feed URL to the subscription list. "
        "The feed will be saved and can be fetched later with fetch_feed or list_subscribed_feeds."
    )
    status_text = "Adding feed..."
    parameters = [
        ToolParam(
            name="url",
            type="string",
            description="The RSS or Atom feed URL to subscribe to",
            required=True,
        ),
        ToolParam(
            name="name",
            type="string",
            description="A friendly name for this feed (e.g. 'Hacker News', 'The Pragmatic Engineer')",
            required=False,
            default="",
        ),
    ]

    async def execute(self, url: str, name: str = "", **_) -> ToolResult:
        # Validate the URL is reachable and is a feed
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Aether/1.0 RSS Reader"},
                    follow_redirects=True,
                    timeout=10,
                )
                resp.raise_for_status()
                content = resp.text

            # Try to detect feed title if no name given
            if not name:
                name = _extract_feed_title(content) or url

        except Exception as e:
            return ToolResult.fail(
                f"Could not reach feed URL: {url}\nError: {e}\n"
                "Please check the URL is a valid RSS or Atom feed."
            )

        # Return the feed info — the agent should use save_memory to persist it
        feed_entry = {"name": name, "url": url}

        return ToolResult.success(
            f"✅ Feed ready to save: **{name}**\n{url}\n\n"
            f"Use `save_memory` to persist this feed to your subscription list:\n"
            f'`save_memory key="rss_feed_{urllib.parse.quote(url[:30])}" value=\'{json.dumps(feed_entry)}\' category="feeds"`',
            feed_name=name,
            feed_url=url,
        )


class GetHackerNewsTopTool(AetherTool):
    """Get the top stories from Hacker News."""

    name = "get_hacker_news_top"
    description = (
        "Get the current top stories from Hacker News. "
        "Returns titles, scores, comment counts, and links."
    )
    status_text = "Fetching Hacker News top stories..."
    parameters = [
        ToolParam(
            name="count",
            type="integer",
            description="Number of top stories to return (1–30, default 15)",
            required=False,
            default=15,
        ),
        ToolParam(
            name="story_type",
            type="string",
            description="Type of stories: 'top' (default), 'new', 'best', 'ask', 'show'",
            required=False,
            default="top",
        ),
    ]

    async def execute(
        self, count: int = 15, story_type: str = "top", **_
    ) -> ToolResult:
        count = min(max(count, 1), 30)
        valid_types = {"top", "new", "best", "ask", "show"}
        if story_type not in valid_types:
            story_type = "top"

        endpoint_map = {
            "top": "topstories",
            "new": "newstories",
            "best": "beststories",
            "ask": "askstories",
            "show": "showstories",
        }
        endpoint = endpoint_map[story_type]

        try:
            async with httpx.AsyncClient() as client:
                # Get story IDs
                ids_resp = await client.get(
                    f"{HN_FIREBASE_BASE}/{endpoint}.json",
                    timeout=10,
                )
                ids_resp.raise_for_status()
                story_ids = ids_resp.json()[:count]

                # Fetch story details in parallel
                import asyncio

                tasks = [
                    client.get(f"{HN_FIREBASE_BASE}/item/{sid}.json", timeout=10)
                    for sid in story_ids
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

            stories = []
            for resp in responses:
                if isinstance(resp, Exception):
                    continue
                try:
                    story = resp.json()
                    if story and story.get("type") in ("story", "ask", "show"):
                        stories.append(story)
                except Exception:
                    continue

            if not stories:
                return ToolResult.success("No Hacker News stories found.")

            type_label = story_type.capitalize()
            lines = [f"**Hacker News — {type_label} Stories**\n"]

            for i, story in enumerate(stories, 1):
                title = story.get("title", "No title")
                url = story.get("url", "")
                score = story.get("score", 0)
                comments = story.get("descendants", 0)
                story_id = story.get("id", "")
                hn_url = f"https://news.ycombinator.com/item?id={story_id}"

                lines.append(f"**{i}. {title}**")
                lines.append(f"   ⬆️ {score} points · 💬 {comments} comments")
                if url:
                    lines.append(f"   🔗 {url}")
                lines.append(f"   HN: {hn_url}\n")

            return ToolResult.success(
                "\n".join(lines), story_count=len(stories), story_type=story_type
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HN Firebase HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Hacker News API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"HN top stories error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to fetch Hacker News stories: {e}")


class SearchHackerNewsTool(AetherTool):
    """Search Hacker News stories and comments using Algolia."""

    name = "search_hn"
    description = (
        "Search Hacker News for stories, Ask HN posts, and discussions using full-text search. "
        "Use for finding HN discussions about a specific topic, library, company, or technology."
    )
    status_text = "Searching Hacker News..."
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
            name="search_type",
            type="string",
            description="'stories' (default) to search story titles, or 'all' to include comments",
            required=False,
            default="stories",
        ),
    ]

    async def execute(
        self, query: str, count: int = 10, search_type: str = "stories", **_
    ) -> ToolResult:
        count = min(max(count, 1), 20)
        tags = "story" if search_type == "stories" else "(story,comment)"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{HN_ALGOLIA_BASE}/search_by_date",
                    params={
                        "query": query,
                        "tags": tags,
                        "hitsPerPage": count,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

            hits = data.get("hits", [])

            if not hits:
                return ToolResult.success(f"No Hacker News results found for: {query}")

            lines = [f"**Hacker News search:** {query}\n"]

            for i, hit in enumerate(hits, 1):
                title = hit.get("title") or hit.get("story_title") or "No title"
                url = hit.get("url", "")
                points = hit.get("points", 0)
                num_comments = hit.get("num_comments", 0)
                created_at = hit.get("created_at", "")[:10]  # date only
                object_id = hit.get("objectID", "")
                hn_url = f"https://news.ycombinator.com/item?id={object_id}"

                # For comments, show the comment text
                comment_text = hit.get("comment_text", "")

                lines.append(f"**{i}. {title}**")
                if created_at:
                    lines.append(
                        f"   📅 {created_at}"
                        + (f" · ⬆️ {points}" if points else "")
                        + (f" · 💬 {num_comments}" if num_comments else "")
                    )
                if comment_text:
                    lines.append(f"   💬 {_truncate(_strip_html(comment_text), 200)}")
                if url:
                    lines.append(f"   🔗 {url}")
                lines.append(f"   HN: {hn_url}\n")

            return ToolResult.success(
                "\n".join(lines), result_count=len(hits), query=query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HN Algolia HTTP error: {e.response.status_code}")
            return ToolResult.fail(
                f"Hacker News search error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"HN search error: {e}", exc_info=True)
            return ToolResult.fail(f"Hacker News search failed: {e}")


# ── Feed parsing helpers ──────────────────────────────────────────────────────


def _extract_feed_title(xml: str) -> str:
    """Extract the feed/channel title from RSS or Atom XML."""
    # Atom: <title>...</title> (first occurrence after <feed>)
    # RSS: <channel><title>...</title>
    match = re.search(
        r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
        xml,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return _strip_html(match.group(1)).strip()
    return ""


def _parse_feed(xml: str, max_items: int) -> list[dict]:
    """Parse RSS 2.0 or Atom 1.0 feed XML into a list of item dicts."""
    items = []

    # Detect feed type
    is_atom = "<feed" in xml.lower() and 'xmlns="http://www.w3.org/2005/Atom"' in xml

    if is_atom:
        # Atom feed: items are <entry> elements
        entries = re.findall(
            r"<entry[^>]*>(.*?)</entry>", xml, re.IGNORECASE | re.DOTALL
        )
        for entry in entries[:max_items]:
            item = _parse_atom_entry(entry)
            if item:
                items.append(item)
    else:
        # RSS feed: items are <item> elements
        rss_items = re.findall(
            r"<item[^>]*>(.*?)</item>", xml, re.IGNORECASE | re.DOTALL
        )
        for rss_item in rss_items[:max_items]:
            item = _parse_rss_item(rss_item)
            if item:
                items.append(item)

    return items


def _extract_tag(xml: str, tag: str) -> str:
    """Extract the text content of the first occurrence of a tag."""
    match = re.search(
        rf"<{tag}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>",
        xml,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_attr(xml: str, tag: str, attr: str) -> str:
    """Extract an attribute value from a self-closing or opening tag."""
    match = re.search(rf'<{tag}[^>]+{attr}=["\']([^"\']+)["\']', xml, re.IGNORECASE)
    return match.group(1) if match else ""


def _parse_rss_item(xml: str) -> dict | None:
    """Parse a single RSS <item> block."""
    title = _strip_html(_extract_tag(xml, "title"))
    link = _extract_tag(xml, "link") or _extract_attr(xml, "link", "href")
    summary = _extract_tag(xml, "description") or _extract_tag(xml, "summary")
    pub_date = _extract_tag(xml, "pubDate") or _extract_tag(xml, "dc:date")

    if not title and not link:
        return None

    return {
        "title": title,
        "link": link.strip(),
        "summary": summary,
        "pub_date": pub_date[:16] if pub_date else "",  # trim to date+time
    }


def _parse_atom_entry(xml: str) -> dict | None:
    """Parse a single Atom <entry> block."""
    title = _strip_html(_extract_tag(xml, "title"))
    # Atom links: <link href="..." rel="alternate"/>
    link_match = re.search(r'<link[^>]+href=["\']([^"\']+)["\']', xml, re.IGNORECASE)
    link = link_match.group(1) if link_match else ""
    summary = _extract_tag(xml, "summary") or _extract_tag(xml, "content")
    pub_date = _extract_tag(xml, "published") or _extract_tag(xml, "updated")

    if not title and not link:
        return None

    return {
        "title": title,
        "link": link.strip(),
        "summary": summary,
        "pub_date": pub_date[:16] if pub_date else "",
    }
