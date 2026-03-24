# Wolfram Alpha & Currency API

## Authentication

### Wolfram Alpha (Short Answers API)
- **Env var**: `$WOLFRAM_APP_ID`
- **Credentials**: Pass `credentials=["wolfram"]` to the execute tool
- **Base URL**: `https://api.wolframalpha.com/v1`
- Get your App ID at [developer.wolframalpha.com](https://developer.wolframalpha.com)

### ExchangeRate-API (Currency Conversion)
- **Env var**: `$EXCHANGERATE_API_KEY` (optional — free tier works without key)
- **Base URL**: `https://v6.exchangerate-api.com/v6`

## Wolfram Alpha — Short Answers

Returns a plain-text answer for math, science, unit conversions, and factual queries:

```bash
curl -s "https://api.wolframalpha.com/v1/result?i=18%25+of+15000&appid=$WOLFRAM_APP_ID"
```

### Query Examples
```bash
# Arithmetic
curl -s "https://api.wolframalpha.com/v1/result?i=2%5E10&appid=$WOLFRAM_APP_ID"

# Unit conversion
curl -s "https://api.wolframalpha.com/v1/result?i=37+celsius+to+fahrenheit&appid=$WOLFRAM_APP_ID"

# Date calculation
curl -s "https://api.wolframalpha.com/v1/result?i=days+until+December+25&appid=$WOLFRAM_APP_ID"

# Scientific constants
curl -s "https://api.wolframalpha.com/v1/result?i=speed+of+light&appid=$WOLFRAM_APP_ID"

# Distance between cities
curl -s "https://api.wolframalpha.com/v1/result?i=distance+from+Chennai+to+London&appid=$WOLFRAM_APP_ID"
```

### URL-Encode Your Queries
- Spaces → `+` or `%20`
- Percent → `%25` (e.g., `18%` → `18%25`)
- Special chars must be encoded

## Wolfram Alpha — Full Results (XML/JSON)

For multi-pod results with more detail:

```bash
curl -s "https://api.wolframalpha.com/v2/query?input=population+of+India&format=plaintext&output=JSON&appid=$WOLFRAM_APP_ID"
```

### Response Structure
- `queryresult.success` — whether the query was understood
- `queryresult.pods[].title` — pod title (e.g., "Result", "Plot", "Statistics")
- `queryresult.pods[].subpods[].plaintext` — the answer text

## Currency Conversion (ExchangeRate-API)

### Get Latest Rates for a Base Currency
```bash
curl -s "https://v6.exchangerate-api.com/v6/$EXCHANGERATE_API_KEY/latest/USD"
```

### Convert Specific Amount
```bash
# Pair conversion
curl -s "https://v6.exchangerate-api.com/v6/$EXCHANGERATE_API_KEY/pair/USD/INR/1000"
```

Response includes `conversion_rate` and `conversion_result`.

### Free Tier (No API Key)
```bash
curl -s "https://open.er-api.com/v6/latest/USD"
```

The open endpoint has no auth but may have lower rate limits.

### Supported Currency Codes
Standard ISO 4217 codes: `USD`, `EUR`, `INR`, `GBP`, `AED`, `SGD`, `JPY`, `CAD`, `AUD`, `CHF`, `CNY`, etc.

## Error Handling
- **Wolfram 403**: Invalid or missing App ID — check `$WOLFRAM_APP_ID`
- **Wolfram 501**: Query not understood — rephrase the query or try simpler terms
- **ExchangeRate 401**: Invalid API key — check `$EXCHANGERATE_API_KEY`
- **ExchangeRate 404**: Unsupported currency code — verify the ISO code
- **Rate limits**: Wolfram free tier allows ~2,000 queries/month; ExchangeRate free tier allows 1,500 requests/month
- If Wolfram is unavailable, fall back to direct calculation for simple arithmetic
