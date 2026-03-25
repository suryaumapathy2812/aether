---
name: brave-search
description: Brave Search API — web, news, image, and video search
integration: brave-search
---
# Brave Search API

## Authentication
- **Env var**: `$BRAVE_SEARCH_API_KEY`
- **Credentials**: Pass `credentials=["brave-search"]` to the execute tool
- **Base URL**: `https://api.search.brave.com/res/v1`
- **Header**: `X-Subscription-Token: $BRAVE_SEARCH_API_KEY`
- Get your API key at [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com)

## Web Search

```bash
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/web/search?q=iPhone+16+price+India&count=10&country=IN"
```

### Query Parameters
- `q` — search query (required)
- `count` — number of results (1-20, default 10)
- `offset` — pagination offset (for getting more results)
- `country` — country code for localized results (e.g., `IN`, `US`, `GB`)
- `search_lang` — language code (e.g., `en`, `hi`, `fr`)
- `safesearch` — `off`, `moderate`, `strict`
- `freshness` — `pd` (past day), `pw` (past week), `pm` (past month), `py` (past year)

### Response Fields
- `web.results[].title` — page title
- `web.results[].url` — page URL
- `web.results[].description` — snippet text
- `web.results[].age` — relative age (e.g., "2 days ago")
- `web.results[].language` — detected language

## News Search

```bash
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/news/search?q=India+budget+2026&count=10&country=IN&freshness=pw"
```

### News-Specific Response Fields
- `results[].title` — headline
- `results[].url` — article URL
- `results[].description` — summary
- `results[].age` — time since publication
- `results[].source` — news source name
- `results[].thumbnail.src` — thumbnail image URL
- `results[].breaking` — whether it's a breaking story

## Image Search

```bash
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/images/search?q=Golden+Gate+Bridge&count=5"
```

## Video Search

```bash
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/videos/search?q=React+tutorial+2026&count=5"
```

## Suggest (Autocomplete)

```bash
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/suggest?q=how+to+learn"
```

## Spell Check

```bash
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/spellcheck?q=recieve+a+package"
```

## Pagination

For comprehensive results beyond the first page:
```bash
# First page
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/web/search?q=machine+learning&count=20"

# Second page
curl -s -H "X-Subscription-Token: $BRAVE_SEARCH_API_KEY" \
  "https://api.search.brave.com/res/v1/web/search?q=machine+learning&count=20&offset=20"
```

## Rate Limits
| Plan | Limit |
|------|-------|
| Free | 2,000 queries/month, 1 req/sec |
| Basic | 20,000 queries/month |
| Pro | 100,000+ queries/month |

## Error Handling
- **401 Unauthorized**: Invalid or missing API key — check `$BRAVE_SEARCH_API_KEY`
- **403 Forbidden**: API key doesn't have access to this endpoint
- **422 Unprocessable Entity**: Invalid query parameters
- **429 Rate Limited**: Exceeded rate limit — wait and retry, respect `Retry-After` header
- **Empty results**: Rephrase the query with different keywords or broaden the search
- On free tier, limit to 1 search per user request to conserve quota
