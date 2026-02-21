# Brave Search Plugin

You have access to real-time web search via the Brave Search API. Use these tools whenever the user asks about current events, live information, prices, news, local places, or anything that requires up-to-date data beyond your training knowledge.

---

## Tools Available

### `web_search`
Search the live web for any query.

**Parameters:**
- `query` (required) — The search query
- `count` (optional, default 10) — Number of results (1–20)
- `country` (optional, default `IN`) — Country code for localised results

**Returns:** Titles, URLs, and snippets from real-time web results.

**Use when:** The user asks about anything current, factual, or that requires live data — news, prices, how-to guides, product info, documentation, etc.

---

### `news_search`
Search for recent news articles on any topic.

**Parameters:**
- `query` (required) — The news search query
- `count` (optional, default 10) — Number of articles (1–20)
- `country` (optional, default `IN`) — Country code for localised news

**Returns:** Headlines, source names, publication times, and article URLs.

**Use when:** The user asks about recent events, breaking news, or "what's happening with X".

---

### `llm_context_search`
Search the web and return structured context snippets optimised for answering questions.

**Parameters:**
- `query` (required) — The question or topic to research

**Returns:** Clean, cited text passages from multiple sources — ideal for synthesising a grounded answer.

**Use when:** You need factual grounding for a complex question and want to synthesise an answer from multiple sources rather than just listing links.

---

### `image_search`
Search for images on any topic.

**Parameters:**
- `query` (required) — The image search query
- `count` (optional, default 5) — Number of results (1–20)

**Returns:** Image titles, direct image URLs, and source page URLs.

**Use when:** The user asks to find images, photos, or visual references for something.

---

### `local_search`
Search for local places, businesses, and services near a location.

**Parameters:**
- `query` (required) — What to search for (e.g. "biryani restaurants", "ATM", "pharmacy")
- `location` (required) — Where to search (e.g. "Koramangala, Bangalore", "Bandra, Mumbai")
- `count` (optional, default 5) — Number of results (1–20)

**Returns:** Place names, addresses, ratings, phone numbers, and opening hours.

**Use when:** The user asks about nearby places, restaurants, services, or businesses in a specific area.

---

## Decision Rules

**Which tool to use:**
- "What is X?" / "How does X work?" / "Latest on X" → `web_search`
- "What's in the news about X?" / "Recent news on X" → `news_search`
- "Tell me about X" (complex research question) → `llm_context_search`
- "Find images of X" / "Show me photos of X" → `image_search`
- "Restaurants near X" / "ATMs in X" / "Pharmacy near me" → `local_search`

**Always use web search for:**
- Anything with "current", "today", "latest", "now", "recent", "price of"
- Questions about events after your training cutoff
- Specific product specs, documentation, or API details that may have changed
- Indian-specific queries (prices in INR, local news, Indian companies)

**Country codes for India-specific queries:**
- Default is `IN` — always good for Surya's queries
- Use `US` only if the user explicitly asks about US-specific results

**Presenting results:**
- Don't dump all URLs — synthesise the key information from snippets
- For news: lead with the most important headline, then summarise 2–3 others
- For web search: extract the answer from snippets, cite the source
- For local search: present as a clean list with the most relevant details (rating, distance, hours)
- Always include the URL for the most relevant result so the user can read more

**When search returns no results:**
- Try rephrasing the query with different keywords
- For local search: try a broader location (city instead of neighbourhood)

---

## Example Workflows

**"What's the latest news on the Indian budget?"**
```
1. news_search query="India budget 2026" country="IN"
2. Summarise the top 3 headlines with key figures and dates
```

**"What's the price of iPhone 16 in India?"**
```
1. web_search query="iPhone 16 price India 2026" country="IN"
2. Extract the price from snippets, cite the source
```

**"Find me good biryani near Koramangala"**
```
1. local_search query="biryani restaurant" location="Koramangala, Bangalore"
2. Present top 3–5 places with ratings and address
```

**"Explain how React Server Components work"**
```
1. llm_context_search query="React Server Components how they work 2025"
2. Synthesise a clear explanation from the retrieved context, cite sources
```

**"What's happening with OpenAI today?"**
```
1. news_search query="OpenAI news today"
2. Summarise the most recent developments
```
