"""Weather tools using Open-Meteo API (free, no API key required).

Provides current weather and multi-day forecasts for any location.
Uses geocoding to resolve city names to coordinates.
Zero-auth plugin — no OAuth or API key needed.
"""

from __future__ import annotations

import logging

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes → human-readable descriptions
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _geocode(location: str) -> dict | None:
    """Resolve a location name to lat/lon coordinates."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                GEOCODE_URL,
                params={"name": location, "count": 1, "language": "en"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        r = results[0]
        return {
            "name": r.get("name", location),
            "country": r.get("country", ""),
            "lat": r["latitude"],
            "lon": r["longitude"],
            "timezone": r.get("timezone", "auto"),
        }
    except Exception as e:
        logger.error(f"Geocoding failed for '{location}': {e}")
        return None


class CurrentWeatherTool(AetherTool):
    """Get current weather for a location."""

    name = "current_weather"
    description = "Get the current weather for any city or location"
    status_text = "Checking the weather..."
    parameters = [
        ToolParam(
            name="location",
            type="string",
            description="City name (e.g. 'Chennai', 'New York', 'London')",
            required=True,
        ),
    ]

    async def execute(self, location: str, **_) -> ToolResult:
        geo = await _geocode(location)
        if not geo:
            return ToolResult.fail(f"Could not find location: {location}")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    WEATHER_URL,
                    params={
                        "latitude": geo["lat"],
                        "longitude": geo["lon"],
                        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_direction_10m",
                        "timezone": geo["timezone"],
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            current = data.get("current", {})
            temp = current.get("temperature_2m", "?")
            feels_like = current.get("apparent_temperature", "?")
            humidity = current.get("relative_humidity_2m", "?")
            wind_speed = current.get("wind_speed_10m", "?")
            weather_code = current.get("weather_code", 0)
            condition = WMO_CODES.get(weather_code, "Unknown")

            city = geo["name"]
            country = geo["country"]
            label = f"{city}, {country}" if country else city

            output = f"**Weather in {label}:**\n"
            output += f"**{condition}** | {temp}°C (feels like {feels_like}°C)\n"
            output += f"Humidity: {humidity}% | Wind: {wind_speed} km/h"

            return ToolResult.success(
                output,
                temperature=temp,
                condition=condition,
                location=label,
            )

        except Exception as e:
            logger.error(f"Error fetching weather: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ForecastTool(AetherTool):
    """Get a multi-day weather forecast."""

    name = "forecast"
    description = "Get a multi-day weather forecast for any city or location"
    status_text = "Checking the forecast..."
    parameters = [
        ToolParam(
            name="location",
            type="string",
            description="City name (e.g. 'Chennai', 'New York', 'London')",
            required=True,
        ),
        ToolParam(
            name="days",
            type="integer",
            description="Number of days to forecast (1-7, default 3)",
            required=False,
            default=3,
        ),
    ]

    async def execute(self, location: str, days: int = 3, **_) -> ToolResult:
        geo = await _geocode(location)
        if not geo:
            return ToolResult.fail(f"Could not find location: {location}")

        days = min(max(days, 1), 7)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    WEATHER_URL,
                    params={
                        "latitude": geo["lat"],
                        "longitude": geo["lon"],
                        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                        "timezone": geo["timezone"],
                        "forecast_days": days,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            codes = daily.get("weather_code", [])
            highs = daily.get("temperature_2m_max", [])
            lows = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            winds = daily.get("wind_speed_10m_max", [])

            if not dates:
                return ToolResult.fail("No forecast data available.")

            city = geo["name"]
            country = geo["country"]
            label = f"{city}, {country}" if country else city

            output = f"**{days}-day forecast for {label}:**\n"
            for i in range(min(days, len(dates))):
                condition = WMO_CODES.get(codes[i] if i < len(codes) else 0, "Unknown")
                high = highs[i] if i < len(highs) else "?"
                low = lows[i] if i < len(lows) else "?"
                rain = precip[i] if i < len(precip) else 0
                wind = winds[i] if i < len(winds) else "?"

                output += f"\n**{dates[i]}** — {condition}\n"
                output += f"   High: {high}°C | Low: {low}°C"
                if rain and rain > 0:
                    output += f" | Rain: {rain}mm"
                output += f" | Wind: {wind} km/h"

            return ToolResult.success(output, days=days, location=label)

        except Exception as e:
            logger.error(f"Error fetching forecast: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")
