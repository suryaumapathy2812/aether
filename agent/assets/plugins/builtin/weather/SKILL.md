# Weather Plugin

Current weather conditions and multi-day forecasts for any location.

## Core Workflow

```
"What's the weather?"       →  current_weather
"Will it rain this week?"   →  weather_forecast
```

Two tools, straightforward choice: `current_weather` for right now, `weather_forecast` for upcoming days.

## Decision Rules

**Which tool:**
- "What's the weather?" / "Is it raining?" / today's conditions → `current_weather`
- "Will it rain tomorrow?" → `weather_forecast` with `days=2` (today + tomorrow)
- "What's the weather this week?" → `weather_forecast` with `days=7`
- "Should I bring an umbrella Friday?" → `weather_forecast` covering that date

**Location handling:**
- The parameter is `location` (not `city`). It accepts city names, regions, or countries.
- If the user doesn't specify a location, check memory for their known location first.
- If no location is in memory, ask: "Which city would you like the weather for?"
- For ambiguous names, add the country: "Paris, France" vs "Paris, Texas"

**Presenting weather:**
- Be conversational: "It's 32°C and sunny in Chennai right now, feels like 36°C with high humidity"
- Translate raw numbers into meaning: "wind at 25 km/h" → "breezy"
- Highlight notable conditions: heavy rain, extreme heat/cold, storms, poor visibility.
- For forecasts, lead with the most important info: "Rain expected Thursday and Friday, clear for the weekend"

**Practical advice:**
- Offer relevant suggestions for notable conditions:
  - Heavy rain → "You might want an umbrella"
  - Extreme heat → "Stay hydrated, avoid going out midday"
  - Storm → "Might be best to stay indoors"

**Units:**
- Use Celsius (°C) by default.
- Switch to Fahrenheit if the user is in the US or explicitly asks.

## Example Workflows

**User: "What's the weather in Chennai?"**
```
1. current_weather location="Chennai"
2. "It's 34°C and partly cloudy in Chennai right now. Feels like 38°C with 78% humidity and a light breeze."
```

**User: "Will it rain this week?"**
```
1. (Check memory for user's city if not specified)
2. weather_forecast location="Chennai" days=7
3. "Rain is expected Wednesday through Friday. The weekend looks clear with temperatures around 30°C."
```

**User: "I'm traveling to London next week"**
```
1. weather_forecast location="London, UK" days=7
2. "London next week: mostly cloudy, 8-14°C. Rain expected Tuesday and Wednesday. Pack a light jacket and umbrella."
```
