# RSS Feeds & Hacker News Plugin

Read RSS/Atom feeds from any blog or news site, and browse or search Hacker News.

## Core Workflow

This plugin serves two distinct use cases:

```
RSS Feeds                              Hacker News
─────────                              ───────────
fetch_feed (read any feed URL)         get_hacker_news_top (front page)
get_item_content (full article)        search_hacker_news (find discussions)
list_subscribed_feeds (saved feeds)    get_item_content (read linked article)
add_feed (save a new subscription)
```

The common pattern: get headlines → offer to read the full article with `get_item_content`.

## Decision Rules

**Which tool:**
- "What's on Hacker News?" / "Top HN stories" → `get_hacker_news_top`
- "What does HN say about X?" / "Find HN discussions on X" → `search_hacker_news`
- "Read this feed: [URL]" / "What's new on [blog]?" → `fetch_feed`
- "Read this article" / "Summarize this post" → `get_item_content`
- "What feeds do I follow?" → `list_subscribed_feeds`
- "Subscribe to this feed" → `add_feed`

**Presenting HN stories:**
- Don't list all stories verbatim. Summarize the top 5-7 with title, score, and comment count.
- Lead with the most interesting or highest-scored story.
- Offer to fetch the full article for any story the user wants to read.

**Presenting feed items:**
- Summarize the most recent 3-5 items conversationally.
- Highlight the most interesting headline.
- Offer to read the full article for any item.

**Feed URLs for common sources:**
- Hacker News: `https://news.ycombinator.com/rss`
- The Pragmatic Engineer: `https://newsletter.pragmaticengineer.com/feed`
- TechCrunch: `https://techcrunch.com/feed/`
- The Verge: `https://www.theverge.com/rss/index.xml`

## Example Workflows

**User: "What's on Hacker News today?"**
```
1. get_hacker_news_top count=15
2. "Top stories on HN right now:
   1. [Title] (342 points, 87 comments)
   2. [Title] (256 points, 45 comments)
   ...
   Want me to read any of these articles?"
```

**User: "What's new on the Pragmatic Engineer?"**
```
1. fetch_feed url="https://newsletter.pragmaticengineer.com/feed" max_items=5
2. "Latest from The Pragmatic Engineer:
   - [Title] (published March 14) — [brief summary]
   - [Title] (published March 10) — [brief summary]
   Want me to read the full article?"
```

**User: "Summarize this article: https://..."**
```
1. get_item_content url="https://..."
2. Provide a 3-5 sentence summary of the key points.
```
