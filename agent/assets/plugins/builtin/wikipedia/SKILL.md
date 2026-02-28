# Wikipedia & Knowledge Plugin

You have access to Wikipedia, Wolfram Alpha, live currency conversion, and world time lookups. Use these for instant factual answers, computations, and reference lookups — no web search needed for these categories.

---

## Tools Available

### `wikipedia_search`
Search Wikipedia and return a concise summary of the top matching article.

**Parameters:**
- `query` (required) — The topic or question to search Wikipedia for

**Returns:** Article title, description, and a concise summary extract with a link to the full article.

**Use when:** The user asks about a person, place, event, concept, organisation, historical fact, scientific topic, or anything encyclopaedic.

---

### `wikipedia_get_article`
Get the full text of a Wikipedia article by its exact title.

**Parameters:**
- `title` (required) — The exact Wikipedia article title

**Returns:** Full article summary with description and link.

**Use when:** You already know the exact Wikipedia article title and need more detail than `wikipedia_search` provides.

---

### `wolfram_query`
Query Wolfram Alpha for mathematical, scientific, and computational answers.

**Parameters:**
- `query` (required) — The mathematical, scientific, or computational query

**Returns:** A direct computed answer from Wolfram Alpha.

**Use when:** The user asks for a calculation, unit conversion, mathematical result, scientific constant, nutritional data, or any query that requires computation rather than lookup.

*Note: Requires Wolfram Alpha App ID to be configured in plugin settings.*

---

### `currency_convert`
Convert between currencies using live exchange rates.

**Parameters:**
- `amount` (required) — The amount to convert
- `from_currency` (required) — Source currency code (e.g. `USD`, `EUR`, `INR`)
- `to_currency` (required) — Target currency code (e.g. `INR`, `USD`, `GBP`)

**Returns:** Converted amount, exchange rate, and last update time.

**Use when:** The user asks to convert money between currencies, or asks "how much is X dollars in rupees?".

---

### `world_time`
Get the current date and time in any timezone or city.

**Parameters:**
- `timezone` (required) — Timezone name, city, or abbreviation (e.g. `Asia/Kolkata`, `IST`, `New York`, `Tokyo`)

**Returns:** Current date and time with UTC offset.

**Use when:** The user asks what time it is in another city or timezone, or needs to know the current time for scheduling across timezones.

---

## Decision Rules

**Which tool to use:**
- "Who is X?" / "What is X?" / "Tell me about X" → `wikipedia_search`
- "What is 15% of 3200?" / "Convert 98.6°F to Celsius" / "Integral of x²" → `wolfram_query`
- "How much is $500 in rupees?" / "Convert 1000 EUR to INR" → `currency_convert`
- "What time is it in San Francisco?" / "Current time in Tokyo" → `world_time`

**Wikipedia vs web search:**
- Use `wikipedia_search` for established facts, historical events, well-known people/places
- Use `web_search` (Brave) for current events, recent news, or topics too new for Wikipedia

**Wolfram Alpha vs calculator:**
- Use `wolfram_query` for anything mathematical, scientific, or computational
- It handles: arithmetic, algebra, calculus, unit conversions, nutritional data, physics constants, statistics, date calculations
- Examples: "15% tip on ₹2,400", "distance from Earth to Moon in km", "calories in 2 cups of rice"

**Currency conversion:**
- Always use `currency_convert` for money questions — never estimate exchange rates from memory
- Common codes: INR (Indian Rupee), USD (US Dollar), EUR (Euro), GBP (British Pound), AED (UAE Dirham), SGD (Singapore Dollar), JPY (Japanese Yen)

**World time:**
- Supports IANA timezone names (`Asia/Kolkata`), common abbreviations (`IST`, `PST`, `GMT`), and city names (`Mumbai`, `London`, `New York`)
- For IST (India Standard Time): use `IST` or `Asia/Kolkata`

**Presenting results:**
- Wikipedia: give the summary naturally, don't just paste the extract — paraphrase and highlight key facts
- Wolfram: state the answer directly: "15% of ₹3,200 is ₹480"
- Currency: be precise: "₹500 = $5.98 at today's rate (1 USD = ₹83.62)"
- Time: be conversational: "It's 9:30 AM on Saturday in San Francisco right now"

---

## Example Workflows

**"Who is Sundar Pichai?"**
```
1. wikipedia_search query="Sundar Pichai"
2. Summarise: CEO of Google/Alphabet, born in Chennai, joined Google in 2004, became CEO in 2015...
```

**"What's 18% GST on ₹15,000?"**
```
1. wolfram_query query="18% of 15000"
2. "18% GST on ₹15,000 is ₹2,700. Total with GST: ₹17,700."
```

**"How much is $1,000 in rupees today?"**
```
1. currency_convert amount=1000 from_currency="USD" to_currency="INR"
2. "$1,000 = ₹83,620 at today's rate (1 USD = ₹83.62)"
```

**"What time is it in New York right now?"**
```
1. world_time timezone="New York"
2. "It's 11:45 PM on Friday in New York (EST, UTC-5)."
```

**"What's the capital of Nagaland?"**
```
1. wikipedia_search query="Nagaland capital"
2. "The capital of Nagaland is Kohima."
```

**"Convert 37°C to Fahrenheit"**
```
1. wolfram_query query="37 degrees Celsius to Fahrenheit"
2. "37°C = 98.6°F"
```
