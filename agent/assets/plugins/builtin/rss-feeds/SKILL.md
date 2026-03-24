# RSS Feeds & Hacker News API

## Authentication
- **Env var**: None required (RSS feeds and Hacker News API are free)
- **Credentials**: No credentials needed
- **RSS Base URL**: Direct feed URLs
- **Hacker News Base URL**: `https://hacker-news.firebaseio.com/v0`

## Fetch Any RSS/Atom Feed

```bash
# Fetch and parse an RSS feed
curl -s "https://news.ycombinator.com/rss"

# Fetch a specific blog feed
curl -s "https://newsletter.pragmaticengineer.com/feed"

# TechCrunch RSS
curl -s "https://techcrunch.com/feed/"

# The Verge RSS
curl -s "https://www.theverge.com/rss/index.xml"
```

### Parsing RSS XML
Each `<item>` contains:
- `<title>` — article headline
- `<link>` — article URL
- `<pubDate>` — publication date
- `<description>` — summary/snippet
- `<creator>` or `<author>` — author name

### Parsing Atom XML
Each `<entry>` contains:
- `<title>` — article headline
- `<link href="..."/>` — article URL
- `<published>` or `<updated>` — timestamp
- `<summary>` or `<content>` — body text
- `<author><name>` — author name

## Hacker News — Top Stories

```bash
# Get top story IDs
curl -s "https://hacker-news.firebaseio.com/v0/topstories.json"

# Get details for a specific story
curl -s "https://hacker-news.firebaseio.com/v0/item/{STORY_ID}.json"
```

### Story Item Fields
- `id` — item ID
- `title` — story title
- `url` — external link (if any)
- `score` — upvote count
- `by` — author username
- `time` — Unix timestamp
- `descendants` — comment count
- `kids` — array of comment IDs

### Fetch Top N Stories
```bash
# Get top 10 story IDs, then fetch each
STORY_IDS=$(curl -s "https://hacker-news.firebaseio.com/v0/topstories.json" | python3 -c "import sys,json; print(' '.join(str(i) for i in json.load(sys.stdin)[:10]))")

for id in $STORY_IDS; do
  curl -s "https://hacker-news.firebaseio.com/v0/item/${id}.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('title','')} ({d.get('score',0)} pts, {d.get('descendants',0)} comments)\")"
done
```

## Hacker News — Search

Use the Algolia HN Search API:

```bash
curl -s "https://hn.algolia.com/api/v1/search?query=react+server+components&tags=story&hitsPerPage=10"
```

### Search Parameters
- `query` — search terms
- `tags=story` — stories only (use `comment` for comments, `poll` for polls)
- `hitsPerPage` — results per page (default 20)
- `page` — page number for pagination
- `numericFilters` — e.g., `points>100`, `created_at_i>1700000000`

### Search by Date (Recent)
```bash
curl -s "https://hn.algolia.com/api/v1/search_by_date?query=AI&tags=story&hitsPerPage=10"
```

## Hacker News — Comments

```bash
# Get a story's comments (kids array contains comment IDs)
curl -s "https://hacker-news.firebaseio.com/v0/item/{STORY_ID}.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for kid in d.get('kids', [])[:5]:
    print(kid)
"

# Fetch a specific comment
curl -s "https://hacker-news.firebaseio.com/v0/item/{COMMENT_ID}.json"
```

## Hacker News — Best/New/Ask/Show

```bash
# Best stories
curl -s "https://hacker-news.firebaseio.com/v0/beststories.json"

# Newest stories
curl -s "https://hacker-news.firebaseio.com/v0/newstories.json"

# Ask HN
curl -s "https://hacker-news.firebaseio.com/v0/askstories.json"

# Show HN
curl -s "https://hacker-news.firebaseio.com/v0/showstories.json"
```

## Read Full Article Content

To extract article text from a URL (for summarization):

```bash
# Use a readability extractor or fetch raw HTML
curl -sL "https://example.com/article" | python3 -c "
import sys
from html.parser import HTMLParser
# Basic text extraction — for production, use a proper readability library
print(sys.stdin.read()[:5000])
"
```

## Error Handling
- **Feed URL returns HTML**: The URL may not be an RSS feed — look for `/feed`, `/rss`, or `/atom.xml` paths
- **HN API 404**: Story/comment ID doesn't exist
- **Algolia rate limit**: ~10,000 requests/day, be conservative
- **Feed parse errors**: Some feeds have malformed XML — handle gracefully, skip bad items
- **Timeout on slow feeds**: Set `curl --max-time 10` for unreliable sources
- No authentication errors possible — all APIs are free
