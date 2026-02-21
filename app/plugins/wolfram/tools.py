"""Wolfram Alpha and currency conversion tools.

Provides:
- Wolfram Alpha computational queries (requires App ID)
- Live currency conversion (uses ExchangeRate-API with key, or free fallback)

Auth: api_key — both keys are optional. Tools degrade gracefully when keys
are absent. All keys read from ``self._context`` at runtime.
"""

from __future__ import annotations

import logging

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

WOLFRAM_API = "https://api.wolframalpha.com/v1/result"
EXCHANGERATE_API = "https://v6.exchangerate-api.com/v6"


class _WolframBaseTool(AetherTool):
    """Base for Wolfram plugin tools — provides key extraction."""

    def _get_wolfram_key(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("wolfram_app_id") if ctx else None

    def _get_exchangerate_key(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("exchangerate_api_key") if ctx else None


class WolframQueryTool(_WolframBaseTool):
    """Query Wolfram Alpha for mathematical, scientific, and computational answers."""

    name = "wolfram_query"
    description = (
        "Query Wolfram Alpha for mathematical computations, unit conversions, scientific facts, "
        "and complex calculations. Examples: '15% of 3200', 'integral of x^2', "
        "'distance from Mumbai to Delhi', 'calories in 100g of rice'."
    )
    status_text = "Computing with Wolfram Alpha..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="The mathematical, scientific, or computational query",
            required=True,
        ),
    ]

    async def execute(self, query: str, **_) -> ToolResult:
        app_id = self._get_wolfram_key()
        if not app_id:
            return ToolResult.fail(
                "Wolfram Alpha is not configured — no App ID set. "
                "Add your Wolfram Alpha App ID in the Wolfram Alpha & Currency plugin settings."
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    WOLFRAM_API,
                    params={"input": query, "appid": app_id},
                    timeout=15,
                )

                if resp.status_code == 501:
                    return ToolResult.fail(
                        f"Wolfram Alpha couldn't compute an answer for: '{query}'. "
                        "Try rephrasing the query more precisely."
                    )

                resp.raise_for_status()
                answer = resp.text.strip()

            if not answer:
                return ToolResult.fail(f"Wolfram Alpha returned no answer for: {query}")

            return ToolResult.success(
                f"**Wolfram Alpha:** {query}\n\n**Answer:** {answer}",
                query=query,
                answer=answer,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Wolfram Alpha HTTP error: {e.response.status_code}")
            return ToolResult.fail(f"Wolfram Alpha error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Wolfram Alpha error: {e}", exc_info=True)
            return ToolResult.fail(f"Wolfram Alpha query failed: {e}")


class CurrencyConvertTool(_WolframBaseTool):
    """Convert between currencies using live exchange rates."""

    name = "currency_convert"
    description = (
        "Convert an amount between any two currencies using live exchange rates. "
        "Examples: 'convert 100 USD to INR', '500 EUR to GBP', '1000 INR to USD'."
    )
    status_text = "Fetching live exchange rates..."
    parameters = [
        ToolParam(
            name="amount",
            type="number",
            description="The amount to convert",
            required=True,
        ),
        ToolParam(
            name="from_currency",
            type="string",
            description="The source currency code (e.g. 'USD', 'EUR', 'INR', 'GBP')",
            required=True,
        ),
        ToolParam(
            name="to_currency",
            type="string",
            description="The target currency code (e.g. 'INR', 'USD', 'EUR')",
            required=True,
        ),
    ]

    async def execute(
        self, amount: float, from_currency: str, to_currency: str, **_
    ) -> ToolResult:
        from_currency = from_currency.upper().strip()
        to_currency = to_currency.upper().strip()
        api_key = self._get_exchangerate_key()

        try:
            if api_key:
                # ExchangeRate-API with key — live rates, reliable
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{EXCHANGERATE_API}/{api_key}/pair/{from_currency}/{to_currency}/{amount}",
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                if data.get("result") != "success":
                    error = data.get("error-type", "unknown error")
                    return ToolResult.fail(f"Currency conversion failed: {error}")

                converted = data.get("conversion_result", 0)
                rate = data.get("conversion_rate", 0)
                last_updated = data.get("time_last_update_utc", "")

                output = (
                    f"**{amount:,.2f} {from_currency}** = **{converted:,.2f} {to_currency}**\n"
                    f"Exchange rate: 1 {from_currency} = {rate} {to_currency}"
                )
                if last_updated:
                    output += f"\n*Rates updated: {last_updated}*"

            else:
                # Free fallback — open.er-api.com, no key needed
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"https://open.er-api.com/v6/latest/{from_currency}",
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                rates = data.get("rates", {})
                if to_currency not in rates:
                    return ToolResult.fail(
                        f"Currency '{to_currency}' not found. Check the currency code."
                    )

                rate = rates[to_currency]
                converted = amount * rate

                output = (
                    f"**{amount:,.2f} {from_currency}** = **{converted:,.2f} {to_currency}**\n"
                    f"Exchange rate: 1 {from_currency} = {rate:.4f} {to_currency}\n"
                    f"*Note: Add an ExchangeRate-API key in plugin settings for more reliable rates.*"
                )

            return ToolResult.success(
                output,
                amount=amount,
                from_currency=from_currency,
                to_currency=to_currency,
                converted=converted,
                rate=rate,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Currency convert HTTP error: {e.response.status_code}")
            return ToolResult.fail(
                f"Currency conversion error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Currency convert error: {e}", exc_info=True)
            return ToolResult.fail(f"Currency conversion failed: {e}")
