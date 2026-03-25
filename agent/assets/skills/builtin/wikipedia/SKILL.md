---
name: wikipedia
description: Wikipedia API — article search, summaries, content, and multi-language support (free, no auth)
integration: none
---
# Wikipedia API

## Authentication
- **Env var**: None required (Wikipedia API is free)
- **Credentials**: No credentials needed
- **Base URL**: `https://en.wikipedia.org/api/rest_v1` (REST API)
- **Alternative**: `https://en.wikipedia.org/w/api.php` (Action API for search)

## Search Articles

Use the Action API for search (the REST API doesn't have a search endpoint):

```bash
curl -s "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=Sundar+Pichai&srlimit=5&format=json"
```

### Response Fields
- `query.search[].title` — article title
- `query.search[].snippet` — highlighted snippet (contains HTML `<span>` tags)
- `query.search[].pageid` — page ID

## Get Article Summary

The REST API provides a clean summary for any article by title:

```bash
curl -s "https://en.wikipedia.org/api/rest_v1/page/summary/Sundar_Pichai"
```

### Response Fields
- `title` — article title
- `extract` — plain-text summary (first few paragraphs)
- `description` — short description (e.g., "CEO of Google and Alphabet")
- `content_urls.desktop.page` — full article URL
- `thumbnail.source` — thumbnail image URL (if available)

### Handling Redirects
If the title has spaces, replace them with underscores:
```bash
curl -s "https://en.wikipedia.org/api/rest_v1/page/summary/Mughal_Empire"
```

The response `type` field indicates:
- `standard` — normal article
- `disambiguation` — disambiguation page (multiple topics match)
- `no-exists` — page not found

## Get Full Article Content

For the full HTML or wikitext content:

```bash
# HTML content
curl -s "https://en.wikipedia.org/api/rest_v1/page/html/Sundar_Pichai"

# Mobile-optimized HTML
curl -s "https://en.wikipedia.org/api/rest_v1/page/mobile-html/Sundar_Pichai"
```

## Get Article Links and Categories

```bash
# Links from the article
curl -s "https://en.wikipedia.org/w/api.php?action=query&titles=Sundar_Pichai&prop=links&pllimit=50&format=json"

# Categories of the article
curl -s "https://en.wikipedia.org/w/api.php?action=query&titles=Sundar_Pichai&prop=categories&cllimit=50&format=json"
```

## Get Related Articles

```bash
curl -s "https://en.wikipedia.org/api/rest_v1/page/related/Sundar_Pichai"
```

Returns up to 20 related articles with summaries.

## Multi-Language Support

Replace `en` with the language code:
```bash
# French Wikipedia
curl -s "https://fr.wikipedia.org/api/rest_v1/page/summary/Paris"

# Hindi Wikipedia
curl -s "https://hi.wikipedia.org/api/rest_v1/page/summary/दिल्ली"
```

## Error Handling
- **404 Not Found**: Article doesn't exist — try searching first with the Action API
- **400 Bad Request**: Malformed title — ensure URL encoding for special characters
- **Disambiguation pages**: Check `type: "disambiguation"` in summary response — refine the search term
- **Rate limits**: Wikipedia allows ~200 requests/second per IP. Be respectful; cache results when possible.
- No authentication errors possible — API is fully open
