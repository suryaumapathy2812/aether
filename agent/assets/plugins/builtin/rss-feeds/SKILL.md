# RSS Feeds & Hacker News Plugin

You have access to RSS/Atom feed reading and Hacker News tools. Use these to fetch content from any blog, news site, or newsletter, and to browse and search Hacker News stories and discussions.

---

## Tools Available

### `fetch_feed`
Fetch and parse any RSS or Atom feed URL.

**Parameters:**
- `url` (required) — The RSS or Atom feed URL
- `max_items` (optional, default 10) — Number of items to return (1–20)

**Returns:** Recent items with titles, publication dates, summaries, and links.

**Use when:** The user asks to read a specific blog, newsletter, or news feed by URL.

---

### `get_item_content`
Fetch and extract the full readable text from any article URL.

**Parameters:**
- `url` (required) — The article URL to read

**Returns:** Article title and extracted readable text content.

**Use when:** The user wants to read the full content of a specific article, or you need to summarise an article from a feed item.

---

### `list_subscribed_feeds`
List all RSS feeds the user has saved.

**Parameters:** None

**Returns:** List of saved feed names and URLs.

**Use when:** The user asks "what feeds do I follow?" or "show me my subscriptions".

---

### `add_feed`
Add a new RSS/Atom feed to the subscription list.

**Parameters:**
- `url` (required) — The RSS or Atom feed URL
- `name` (optional) — A friendly name for the feed

**Returns:** Confirmation with instructions to save via `save_memory`.

**Use when:** The user asks to subscribe to or save a new feed.

---

### `get_hacker_news_top`
Get the current top stories from Hacker News.

**Parameters:**
- `count` (optional, default 15) — Number of stories (1–30)
- `story_type` (optional, default `top`) — `top`, `new`, `best`, `ask`, or `show`

**Returns:** Story titles, scores, comment counts, article URLs, and HN discussion links.

**Use when:** The user asks about Hacker News, "what's trending in tech", or "what's on HN today".

---

### `search_hn`
Search Hacker News stories and discussions using full-text search.

**Parameters:**
- `query` (required) — The search query
- `count` (optional, default 10) — Number of results (1–20)
- `search_type` (optional, default `stories`) — `stories` or `all` (includes comments)

**Returns:** Matching stories with scores, comment counts, dates, and links.

**Use when:** The user asks "what does HN say about X?" or wants to find past HN discussions on a topic.

---

## Decision Rules

**Which tool to use:**
- "What's on Hacker News?" / "Top HN stories" → `get_hacker_news_top`
- "What does HN say about X?" / "Find HN discussions on X" → `search_hn query="X"`
- "Read this feed: [URL]" / "What's new on [blog]?" → `fetch_feed url="..."`
- "Read this article: [URL]" / "Summarise this post" → `get_item_content url="..."`
- "What feeds do I follow?" → `list_subscribed_feeds`
- "Subscribe to / save this feed" → `add_feed`

**HN story types:**
- `top` — Current front page (default, most popular)
- `new` — Newest submissions (unfiltered)
- `best` — Highest-ranked stories of recent days
- `ask` — Ask HN posts (community questions)
- `show` — Show HN posts (project showcases)

**Presenting HN stories:**
- Lead with the most interesting/highest-scored story
- For each story: title, score, comment count, and the article URL
- If the user wants to discuss a story, offer to fetch the full article with `get_item_content`
- Don't list all 15 stories verbatim — summarise the top 5–7 and offer to show more

**Presenting feed items:**
- Summarise the most recent 3–5 items conversationally
- For newsletters/blogs: highlight the most interesting headline
- Offer to read the full article for any item the user wants more detail on

**Feed URLs for common sources:**
- Hacker News RSS: `https://news.ycombinator.com/rss`
- The Pragmatic Engineer: `https://newsletter.pragmaticengineer.com/feed`
- Paul Graham essays: `http://www.paulgraham.com/rss.html`
- TechCrunch: `https://techcrunch.com/feed/`
- The Verge: `https://www.theverge.com/rss/index.xml`
- Wired: `https://www.wired.com/feed/rss`

---

## Example Workflows

**"What's on Hacker News today?"**
```
1. get_hacker_news_top count=15 story_type="top"
2. Summarise the top 5–7 stories: "Top story: [title] (342 points, 87 comments). Also trending: ..."
3. Offer: "Want me to read any of these articles?"
```

**"What does HN say about Anthropic?"**
```
1. search_hn query="Anthropic" count=10
2. Summarise the most relevant discussions with dates and scores
```

**"What's new on the Pragmatic Engineer?"**
```
1. fetch_feed url="https://newsletter.pragmaticengineer.com/feed" max_items=5
2. "Latest from The Pragmatic Engineer: [title] (published [date]) — [summary]"
3. Offer to read the full article
```

**"Summarise this article: https://..."**
```
1. get_item_content url="https://..."
2. Provide a 3–5 sentence summary of the key points
```

**"Subscribe me to TechCrunch"**
```
1. add_feed url="https://techcrunch.com/feed/" name="TechCrunch"
2. Follow the save_memory instruction from the tool response to persist the subscription
3. "Done! I've saved TechCrunch to your feed list. Say 'what's new on TechCrunch' anytime to check it."
```

**"Show me Ask HN posts"**
```
1. get_hacker_news_top story_type="ask" count=10
2. Present the top Ask HN discussions with scores and comment counts
```
