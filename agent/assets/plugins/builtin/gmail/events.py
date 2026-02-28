"""Gmail event normalizer.

Converts Gmail push notifications into unified PluginEvent format.
Fetches message details from Gmail API and enriches with context.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx

from aether.plugins.event import PluginEvent

logger = logging.getLogger(__name__)

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

# Gmail label category mappings
LABEL_CATEGORIES = {
    "CATEGORY_PROMOTIONS": "marketing",
    "CATEGORY_SOCIAL": "social",
    "CATEGORY_PERSONAL": "personal",
    "CATEGORY_UPDATES": "notification",
}


async def normalize_gmail_event(payload: dict, access_token: str) -> PluginEvent:
    """
    Convert a Gmail push notification into a unified PluginEvent.

    Args:
        payload: Gmail push notification payload with message ID and history ID
        access_token: OAuth2 access token for Gmail API

    Returns:
        PluginEvent with normalized data and metadata
    """
    try:
        # Extract IDs from payload
        message_id = payload.get("message", {}).get("id")
        history_id = payload.get("message", {}).get("historyId")

        if not message_id:
            logger.error("No message ID in Gmail payload")
            return _create_error_event("Invalid payload: missing message ID")

        # Fetch full message from Gmail API
        message_data = await _fetch_message(message_id, access_token)

        if not message_data:
            return _create_error_event(f"Could not fetch message {message_id}")

        # Extract headers
        headers_list = message_data.get("payload", {}).get("headers", [])
        subject = _get_header(headers_list, "Subject") or "(no subject)"
        sender_raw = _get_header(headers_list, "From") or "Unknown sender"
        date = _get_header(headers_list, "Date") or ""

        # Parse sender name and email
        sender_dict = _parse_sender(sender_raw)

        # Extract body preview
        body_preview = _extract_body_preview(message_data.get("payload", {}), max_chars=200)

        # Determine urgency based on labels
        labels = message_data.get("labelIds", [])
        urgency = _determine_urgency(labels, sender_dict.get("email"))

        # Determine category based on Gmail labels
        category = _determine_category(labels)

        # Create summary
        sender_name = sender_dict.get("name") or sender_dict.get("email")
        summary = f"Email from {sender_name}: {subject}"

        # Determine if action is required
        requires_action = urgency in ["high", "medium"]

        return PluginEvent(
            plugin="gmail",
            event_type="message.new",
            source_id=message_id,
            summary=summary,
            content=body_preview,
            sender=sender_dict,
            urgency=urgency,
            category=category,
            requires_action=requires_action,
            available_actions=["read_gmail", "send_reply", "archive_email"],
            metadata={
                "subject": subject,
                "date": date,
                "thread_id": message_data.get("threadId"),
                "labels": labels,
                "history_id": history_id,
            },
        )

    except Exception as e:
        logger.error(f"Error normalizing Gmail event: {e}", exc_info=True)
        return _create_error_event(f"Event processing error: {e}")


async def _fetch_message(message_id: str, access_token: str) -> Optional[dict]:
    """Fetch message details from Gmail API."""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient() as client:
            url = f"{GMAIL_API_BASE}/messages/{message_id}"
            response = await client.get(
                url,
                headers=headers,
                params={"format": "full"},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch message {message_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching message: {e}")
        return None


def _get_header(headers_list: list, header_name: str) -> Optional[str]:
    """Extract a header value by name."""
    for header in headers_list:
        if header.get("name") == header_name:
            return header.get("value")
    return None


def _parse_sender(sender_raw: str) -> dict:
    """Parse sender email and name from raw string.

    Handles formats like:
    - "John Doe <john@example.com>"
    - "john@example.com"
    - "John Doe <john@example.com> (Personal)"
    """
    # Try to extract name and email
    if "<" in sender_raw and ">" in sender_raw:
        name_part = sender_raw[: sender_raw.index("<")].strip()
        email_part = sender_raw[sender_raw.index("<") + 1 : sender_raw.index(">")].strip()
        return {
            "name": name_part or email_part,
            "email": email_part,
        }
    else:
        # Just an email
        return {
            "email": sender_raw.strip(),
        }


def _extract_body_preview(payload: dict, max_chars: int = 200) -> str:
    """Extract a preview of the email body."""
    body = ""

    if "parts" in payload:
        # Multipart email - look for text/plain part
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    try:
                        body = base64.urlsafe_b64decode(data).decode("utf-8")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to decode body part: {e}")
    else:
        # Simple email
        data = payload.get("body", {}).get("data", "")
        if data:
            try:
                body = base64.urlsafe_b64decode(data).decode("utf-8")
            except Exception as e:
                logger.debug(f"Failed to decode body: {e}")

    # Return truncated preview
    if body:
        return body[:max_chars].split("\n")[0] + ("..." if len(body) > max_chars else "")
    return ""


def _determine_urgency(labels: list, sender_email: Optional[str]) -> str:
    """Determine urgency based on labels and sender.

    High: IMPORTANT label or starred
    Medium: Known contact (contact/frequent)
    Low: Everything else
    """
    if "IMPORTANT" in labels or "STARRED" in labels:
        return "high"

    # Could check contacts here in the future
    if sender_email and _is_known_contact(sender_email):
        return "medium"

    return "low"


def _is_known_contact(email: str) -> bool:
    """Check if sender is a known contact.

    Placeholder for future integration with Google Contacts API.
    """
    # TODO: Integrate with Google Contacts API
    return False


def _determine_category(labels: list) -> str:
    """Determine category from Gmail labels."""
    for label in labels:
        if label in LABEL_CATEGORIES:
            return LABEL_CATEGORIES[label]

    # Default categories based on label presence
    if "DRAFT" in labels:
        return "notification"
    if any("work" in label.lower() for label in labels):
        return "work"
    if any("personal" in label.lower() for label in labels):
        return "personal"

    return "general"


def _create_error_event(error_msg: str) -> PluginEvent:
    """Create an error event."""
    return PluginEvent(
        plugin="gmail",
        event_type="message.new",
        source_id="error",
        summary=f"Gmail error: {error_msg}",
        content=error_msg,
        urgency="low",
        category="notification",
    )
