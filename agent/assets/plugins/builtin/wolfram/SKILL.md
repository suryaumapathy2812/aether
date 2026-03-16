# Wolfram Alpha & Currency Plugin

Precise mathematical computations, scientific queries, and live currency conversion.

## Core Workflow

```
Math / science / units / data  →  wolfram_query
Currency conversion            →  currency_convert
```

## Decision Rules

**When to use `wolfram_query`:**
- Arithmetic, algebra, calculus, statistics
- Unit conversions (km to miles, kg to lbs, °C to °F)
- Scientific constants, nutritional data, physical properties
- Date calculations, distance between cities
- Cryptocurrency prices (Wolfram has live crypto data)

**When to use `currency_convert`:**
- Any fiat currency conversion (USD→INR, EUR→GBP, etc.)
- The `from` and `to` parameters use standard currency codes: USD, EUR, INR, GBP, AED, SGD, JPY, etc.

**When NOT to use these tools:**
- Simple mental math (2+2, 10% of 50) — just answer directly, no need for an API call.
- General knowledge questions → use Wikipedia instead.
- Live stock prices or current news → use web search instead.

**When Wolfram Alpha is not configured:**
- The tool will tell you if the App ID is missing. Suggest: "Add your Wolfram Alpha App ID in the plugin settings."
- For simple arithmetic and common unit conversions, calculate it yourself as a fallback.

**Presenting results:**
- Be natural: "15% of 3,200 is 480" — not "Wolfram Alpha result: 480"
- For currency: "100 USD is about ₹8,342 right now" — include the rate if helpful.
- Round to sensible decimal places for the context.

## Example Workflows

**User: "What's 18% GST on ₹15,000?"**
```
1. wolfram_query query="18% of 15000"
2. "18% GST on ₹15,000 is ₹2,700. Total with GST: ₹17,700."
```

**User: "How much is $1,000 in rupees?"**
```
1. currency_convert amount="1000" from="USD" to="INR"
2. "$1,000 = ₹83,620 at today's rate (1 USD = ₹83.62)"
```

**User: "Convert 37°C to Fahrenheit"**
```
1. wolfram_query query="37 degrees Celsius to Fahrenheit"
2. "37°C = 98.6°F"
```
