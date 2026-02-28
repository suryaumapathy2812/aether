# Wolfram Alpha & Currency Plugin

You have access to Wolfram Alpha for mathematical and scientific computations, and live currency conversion. Use these for precise calculations, unit conversions, and real-time exchange rates.

---

## Tools Available

### `wolfram_query`
Query Wolfram Alpha for mathematical, scientific, and computational answers.

**Parameters:**
- `query` (required) — The mathematical, scientific, or computational query

**Returns:** A direct computed answer from Wolfram Alpha.

**Use when:** The user asks for a calculation, unit conversion, mathematical result, scientific constant, nutritional data, or any query that requires computation rather than lookup.

*Note: Requires a Wolfram Alpha App ID configured in plugin settings. If not configured, the tool will tell you.*

**Examples:**
- `"15% of 3200"` → `480`
- `"integral of x^2"` → `x^3/3 + constant`
- `"distance from Mumbai to Delhi"` → `1153 km`
- `"calories in 100g of rice"` → `130 kcal`
- `"speed of light in km/h"` → `1.079 × 10^9 km/h`
- `"boiling point of water in Fahrenheit"` → `212 °F`

---

### `currency_convert`
Convert between currencies using live exchange rates.

**Parameters:**
- `amount` (required) — The amount to convert
- `from_currency` (required) — Source currency code (e.g. `USD`, `EUR`, `INR`, `GBP`)
- `to_currency` (required) — Target currency code (e.g. `INR`, `USD`, `GBP`)

**Returns:** Converted amount, exchange rate, and last update time.

**Use when:** The user asks to convert money between currencies, asks what something costs in another currency, or needs a live exchange rate.

*Note: Works without an API key using a free fallback. Add an ExchangeRate-API key in plugin settings for more reliable rates.*

**Examples:**
- `"convert 100 USD to INR"` → `"100.00 USD = 8,342.50 INR"`
- `"500 EUR to GBP"` → `"500.00 EUR = 426.15 GBP"`
- `"what is 1 BTC worth in USD"` → use `wolfram_query` instead for crypto

---

## Decision Rules

**Choosing the right tool:**
- Arithmetic, algebra, calculus, statistics → `wolfram_query`
- Unit conversions (km to miles, kg to lbs, °C to °F) → `wolfram_query`
- Scientific constants, nutritional data, physical properties → `wolfram_query`
- Currency conversion (USD→INR, EUR→GBP) → `currency_convert`
- Cryptocurrency prices → `wolfram_query` (Wolfram has live crypto data)

**When Wolfram Alpha is not configured:**
- Tell the user: "Wolfram Alpha isn't configured yet — add your App ID in the Wolfram Alpha & Currency plugin settings."
- For simple arithmetic, calculate it yourself instead of failing
- For unit conversions, use your own knowledge for common ones (km↔miles, °C↔°F)

**Presenting results:**
- Be natural: "15% of 3,200 is 480" not "Wolfram Alpha result: 480"
- For currency: "100 USD is about ₹8,342 right now" not a raw data dump
- Round to a sensible number of decimal places for the context
- Mention if rates are from the free fallback (less reliable) vs. live API

**When NOT to use these tools:**
- Simple mental math (2+2, 10% of 50) — just answer directly
- General knowledge questions → use `wikipedia_search` instead
- Current news or prices (stocks, crypto live prices) → use `web_search` instead
