# Wikipedia Plugin

Search Wikipedia and read articles for encyclopedic knowledge on any topic.

## Core Workflow

```
"Who is X?" / "What is X?"  →  wikipedia_search
Need the full article        →  wikipedia_get_article (requires exact title)
```

`wikipedia_search` returns a concise summary with the article title and link. Use `wikipedia_get_article` only when you need more depth than the search summary provides — it requires the exact article title (which you get from `wikipedia_search`).

## Decision Rules

**When to use Wikipedia:**
- Established facts, historical events, well-known people/places, scientific concepts, organizations
- Anything encyclopedic where a stable, well-sourced answer exists

**When NOT to use Wikipedia:**
- Current events or recent news → use web search instead
- Topics too new or niche for Wikipedia to cover
- Calculations or conversions → use Wolfram Alpha

**Presenting results:**
- Paraphrase and highlight key facts naturally. Don't just paste the raw extract.
- Keep it conversational: "Sundar Pichai is the CEO of Google and Alphabet. Born in Chennai, he joined Google in 2004 and became CEO in 2015."
- Include the article link if the user might want to read more.

## Example Workflows

**User: "Who is Sundar Pichai?"**
```
1. wikipedia_search query="Sundar Pichai"
2. "Sundar Pichai is the CEO of Google and Alphabet. Born in Chennai, India, he joined Google in 2004, led the development of Chrome, and became CEO in 2015."
```

**User: "Tell me about the history of the Mughal Empire"**
```
1. wikipedia_search query="Mughal Empire"
2. Summarize the key points from the extract
3. If the user wants more depth: wikipedia_get_article title="Mughal Empire"
```

**User: "What's the capital of Nagaland?"**
```
1. wikipedia_search query="Nagaland"
2. "The capital of Nagaland is Kohima."
```
