# Weather Plugin

You have access to weather tools for checking current conditions and multi-day forecasts for any city.

---

## Tools Available

### `current_weather`
Get current weather conditions for a city.

**Parameters:**
- `city` (required) — City name (e.g. "Chennai", "New York", "London, UK")

**Returns:** Temperature (°C), feels-like temperature, humidity, wind speed and direction, weather condition (sunny/cloudy/rainy/etc.), and visibility.

**Use when:** The user asks about the weather right now, today's conditions, or "is it raining?".

---

### `forecast`
Get a multi-day weather forecast for a city.

**Parameters:**
- `city` (required) — City name
- `days` (optional, default `3`) — Number of days to forecast (1–7)

**Returns:** Day-by-day forecast with high/low temperatures, conditions, precipitation chance, and wind.

**Use when:** The user asks about upcoming weather, planning for a trip, or "will it rain this week?".

---

## Decision Rules

**Choosing the right tool:**
- "What's the weather like?" → `current_weather`
- "What's the weather today?" → `current_weather`
- "Will it rain tomorrow?" → `forecast days=2`
- "What's the weather this week?" → `forecast days=7`
- "Should I bring an umbrella for my trip on Friday?" → `forecast` covering that date

**Location handling:**
- If the user doesn't specify a city, check memory for their known location first (`search_memory query="user location city"`)
- If no location is in memory, ask: "Which city would you like the weather for?"
- For ambiguous city names, add the country: "Paris, France" vs "Paris, Texas"

**Presenting weather:**
- Be conversational: "It's 32°C and sunny in Chennai right now, feels like 36°C with high humidity"
- Don't dump raw numbers — translate them: "wind at 25 km/h" → "breezy"
- **Always highlight notable conditions:** heavy rain, extreme heat/cold, storms, poor visibility
- For forecasts, lead with the most important information: "Rain expected Thursday and Friday, clear for the weekend"

**Practical advice:**
- Offer relevant suggestions when conditions are notable:
  - Heavy rain → "You might want to carry an umbrella"
  - Extreme heat → "Stay hydrated and avoid going out midday"
  - Storm warning → "It might be best to stay indoors"
- For travel planning, summarize the forecast for the trip dates specifically

**Units:**
- Use Celsius (°C) by default
- Switch to Fahrenheit if the user is in the US or explicitly asks for it

---

## Example Workflows

**"What's the weather in Chennai?"**
```
1. current_weather city="Chennai"
2. "It's 34°C and partly cloudy in Chennai right now. Feels like 38°C with 78% humidity and a light breeze."
```

**"Will it rain this week?"**
```
1. [check memory for user's city if not specified]
2. forecast city="Chennai" days=7
3. "Rain is expected Wednesday through Friday. The weekend looks clear with temperatures around 30°C."
```

**"I'm traveling to London next week — what's the weather like?"**
```
1. forecast city="London, UK" days=7
2. "London next week: mostly cloudy with temperatures between 8–14°C. Rain expected Tuesday and Wednesday. Pack a light jacket and umbrella."
```
