"""Vobiz Telephony Tools.

Tools for making phone calls via Vobiz API.
Requires plugin configuration: auth_id, auth_token, from_number.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

VOBIZ_API_BASE = "https://api.vobiz.ai/api/v1"


@dataclass
class OutboundCallResult:
    """Result of an outbound call initiation."""

    success: bool
    call_uuid: str | None = None
    error: str | None = None


async def make_vobiz_call(
    auth_id: str,
    auth_token: str,
    from_number: str,
    to_number: str,
    answer_url: str,
    caller_name: str = "Aether",
) -> OutboundCallResult:
    """Initiate an outbound phone call via Vobiz API.

    Args:
        auth_id: Vobiz Auth ID
        auth_token: Vobiz Auth Token
        from_number: Caller ID (Vobiz phone number)
        to_number: Destination phone number in E.164 format
        answer_url: URL that Vobiz will call when the call is answered
        caller_name: Name to display as caller

    Returns:
        OutboundCallResult with success status and call_uuid
    """
    # Normalize phone numbers (remove + prefix for Vobiz API)
    to_clean = to_number.lstrip("+")
    from_clean = from_number.lstrip("+")

    url = f"{VOBIZ_API_BASE}/Account/{auth_id}/Call/"
    headers = {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
        "Content-Type": "application/json",
    }
    payload = {
        "from": from_clean,
        "to": to_clean,
        "answer_url": answer_url,
        "answer_method": "POST",
        "caller_name": caller_name,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code == 201:
            data = response.json()
            call_uuid = data.get("call_uuid")
            logger.info(
                "Vobiz call initiated: call_uuid=%s to=%s", call_uuid, to_number
            )
            return OutboundCallResult(success=True, call_uuid=call_uuid)

        error_text = response.text
        logger.error(
            "Vobiz outbound call failed: status=%d body=%s",
            response.status_code,
            error_text,
        )
        return OutboundCallResult(
            success=False,
            error=f"Vobiz API error: {response.status_code}",
        )

    except httpx.TimeoutException:
        logger.error("Vobiz API timeout")
        return OutboundCallResult(success=False, error="Vobiz API timeout")
    except Exception as e:
        logger.error("Vobiz outbound call error: %s", e, exc_info=True)
        return OutboundCallResult(success=False, error=str(e))


class MakePhoneCallTool(AetherTool):
    """Make an outbound phone call to the user or a specified number."""

    name = "make_phone_call"
    description = (
        "Make a phone call to the user or a specified phone number. "
        "Use this when you need to speak with the user over the phone, "
        "or when the user asks you to call someone."
    )
    status_text = "Initiating phone call..."
    parameters = [
        ToolParam(
            name="to_number",
            type="string",
            description=(
                "Phone number to call in E.164 format (e.g., +919123456789). "
                "If not provided, calls the user's configured phone number."
            ),
            required=False,
        ),
        ToolParam(
            name="greeting",
            type="string",
            description="Optional greeting to speak when the call connects.",
            required=False,
        ),
        ToolParam(
            name="reason",
            type="string",
            description="Brief reason for the call (for logging/context).",
            required=False,
        ),
    ]

    async def execute(
        self,
        to_number: str | None = None,
        greeting: str | None = None,
        reason: str | None = None,
        **_,
    ) -> ToolResult:
        """Execute the phone call tool."""
        # Get credentials from plugin context
        context = getattr(self, "_context", {})
        auth_id = context.get("auth_id", "")
        auth_token = context.get("auth_token", "")
        from_number = context.get("from_number", "")
        user_phone = context.get("user_phone_number", "")
        base_url = context.get("base_url", "")

        if not auth_id or not auth_token:
            return ToolResult.fail(
                "Vobiz plugin not configured. Please set up your Vobiz credentials "
                "(auth_id, auth_token, from_number) in the plugin settings."
            )

        if not from_number:
            return ToolResult.fail(
                "Vobiz phone number not configured. Please set your Vobiz phone number "
                "(from_number) in the plugin settings."
            )

        # Determine destination number
        dest = to_number or user_phone
        if not dest:
            return ToolResult.fail(
                "No phone number to call. Please provide a phone number or "
                "configure your phone number in the plugin settings."
            )

        if not base_url:
            return ToolResult.fail(
                "Server base URL not configured. Cannot initiate call."
            )

        # Build answer URL with optional greeting
        answer_url = f"{base_url.rstrip('/')}/plugins/vobiz/answer"
        if greeting:
            answer_url += f"?greeting={urllib.parse.quote(greeting)}"

        # Make the call
        result = await make_vobiz_call(
            auth_id=auth_id,
            auth_token=auth_token,
            from_number=from_number,
            to_number=dest,
            answer_url=answer_url,
        )

        if result.success:
            msg = f"Phone call initiated successfully. Call ID: {result.call_uuid}"
            if reason:
                msg += f" Reason: {reason}"
            return ToolResult.success(msg, call_uuid=result.call_uuid)

        return ToolResult.fail(f"Failed to initiate call: {result.error}")


class GetUserPhoneNumberTool(AetherTool):
    """Get the user's configured phone number."""

    name = "get_user_phone_number"
    description = (
        "Get the user's configured phone number. "
        "Use this to check if the user has a phone number configured before making a call."
    )
    status_text = "Checking phone number..."
    parameters = []

    async def execute(self, **_) -> ToolResult:
        """Execute the tool."""
        context = getattr(self, "_context", {})
        user_phone = context.get("user_phone_number", "")

        if user_phone:
            return ToolResult.success(
                f"Your phone number is configured: {user_phone}",
                phone_number=user_phone,
            )

        return ToolResult.success(
            "No phone number configured. You can provide a phone number when making a call, "
            "or configure your phone number in the Vobiz plugin settings."
        )
