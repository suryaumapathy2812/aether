# Weather API

## Authentication
- **Env var**: None required (uses Open-Meteo free API)
- **Credentials**: No credentials needed
- **Base URL**: `https://api.open-meteo.com/v1`

Open-Meteo is a free, open-source weather API. No API key or authentication required.

## Geocoding — Resolve City Name to Coordinates

The weather endpoints require latitude/longitude. First, resolve a city name:

```bash
curl -s "https://geocoding-api.open-meteo.com/v1/search?name=Chennai&count=1"
```

Response contains `results[0].latitude` and `results[0].longitude`.

## Current Weather

```bash
curl -s "https://api.open-meteo.com/v1/forecast?latitude=13.08&longitude=80.27&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&timezone=auto"
```

### Key Response Fields
- `current.temperature_2m` — temperature in °C
- `current.apparent_temperature` — feels-like temperature
- `current.relative_humidity_2m` — humidity percentage
- `current.weather_code` — WMO weather code (see below)
- `current.wind_speed_10m` — wind speed in km/h

### WMO Weather Codes
| Code | Condition |
|------|-----------|
| 0 | Clear sky |
| 1-3 | Mainly clear, partly cloudy, overcast |
| 45, 48 | Fog |
| 51-55 | Drizzle |
| 61-65 | Rain |
| 71-75 | Snow |
| 80-82 | Rain showers |
| 95-99 | Thunderstorm |

## Weather Forecast (Multi-Day)

```bash
curl -s "https://api.open-meteo.com/v1/forecast?latitude=13.08&longitude=80.27&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=auto&forecast_days=7"
```

### Query Parameters
- `forecast_days` — number of days (1-16, default 7)
- `daily=temperature_2m_max,temperature_2m_min` — daily high/low
- `daily=precipitation_sum` — total precipitation in mm
- `daily=weather_code` — dominant weather condition
- `timezone=auto` — use local timezone of the coordinates

### Combining Current + Forecast

```bash
curl -s "https://api.open-meteo.com/v1/forecast?latitude=13.08&longitude=80.27&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=5"
```

## Hourly Forecast

```bash
curl -s "https://api.open-meteo.com/v1/forecast?latitude=13.08&longitude=80.27&hourly=temperature_2m,precipitation_probability,weather_code&timezone=auto&forecast_days=2"
```

## Units

Default units: temperature in °C, wind in km/h, precipitation in mm.

To use Fahrenheit and mph:
```bash
curl -s "https://api.open-meteo.com/v1/forecast?latitude=40.71&longitude=-74.01&current=temperature_2m&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
```

## Error Handling
- **400 Bad Request**: Invalid coordinates or parameters — check latitude/longitude values
- **No auth errors**: API is free, no 401/403 possible
- **Rate limits**: Open-Meteo allows 10,000 requests/day on the free tier
- If geocoding returns no results, the city name is misspelled or ambiguous — try adding country code
