# Brave Search Plugin

Real-time web search for current events, live information, news, and anything beyond training knowledge.

## Core Workflow

Pick the right search tool based on what the user needs:

```
Factual question / current info  →  brave_web_search
Breaking news / recent events    →  news_search
Complex research / synthesis     →  llm_context_search
```

## Decision Rules

**Which tool to use:**
- "What is X?" / "How does X work?" / "Latest on X" / prices / specs → `brave_web_search`
- "What's in the news about X?" / "Recent news on X" / breaking events → `news_search`
- Complex research where you need to synthesize from multiple sources → `llm_context_search` (returns clean, cited text passages optimized for building an answer)

**When to search at all:**
- Anything with "current", "today", "latest", "now", "recent", "price of"
- Questions about events after your training cutoff
- Specific product specs, documentation, or API details that may have changed

**Country code:**
- Default is `IN` for India-localized results.
- Use `US` or other codes only if the user explicitly asks about results from another region.

**Presenting results:**
- Don't dump all URLs. Synthesize the key information from snippets and cite the source.
- For news: lead with the most important headline, then summarize 2-3 others.
- For web search: extract the answer from snippets, include the URL for the most relevant result.
- If no results, try rephrasing with different keywords.

## Pagination

Brave Search supports an `offset` parameter. For comprehensive research:
1. First call with `count=20`
2. Follow-up with `offset=20` if the user needs more

## Rate Limits

| Plan | Limit |
|---|---|
| Free tier | 2,000 queries/month (1 query/second) |
| Basic | 20,000 queries/month |
| Pro | 100,000+ queries/month |

On the free tier, be conservative — one search per request is usually enough. On paid tiers, parallel calls (web + news simultaneously) are fine.

## Example Workflows

**User: "What's the latest news on the Indian budget?"**
```
1. news_search query="India budget 2026" country="IN"
2. "Here's the latest:
   - [Headline 1] — key figures and dates
   - [Headline 2] — summary
   - [Headline 3] — summary
   [source link]"
```

**User: "What's the price of iPhone 16 in India?"**
```
1. brave_web_search query="iPhone 16 price India 2026" country="IN"
2. "The iPhone 16 starts at ₹79,900 in India (source: Apple India). [link]"
```

**User: "Explain how React Server Components work"**
```
1. llm_context_search query="React Server Components how they work"
2. Synthesize a clear explanation from the retrieved context passages, citing sources.
```
