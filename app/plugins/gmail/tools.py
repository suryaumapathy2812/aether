"""Gmail tools for reading and managing emails.

Provides tools to interact with Gmail API:
- List unread emails
- Read specific email by message ID
- Send replies to email threads
- Archive emails

All tools receive credentials at call time via ``self._context``
(set by ``safe_execute`` from the plugin context store).
No ``__init__`` args required — the loader can instantiate with ``cls()``.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


class _GmailTool(AetherTool):
    """Base for Gmail tools — provides token extraction from runtime context."""

    def _get_token(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("access_token") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}


class ListUnreadTool(_GmailTool):
    """List unread emails in the inbox."""

    name = "list_unread"
    description = "List unread emails in your Gmail inbox"
    status_text = "Checking inbox..."
    parameters = [
        ToolParam(
            name="max_results",
            type="integer",
            description="Max emails to return",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, max_results: int = 10, **_) -> ToolResult:
        """Fetch unread emails from Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            async with httpx.AsyncClient() as client:
                url = f"{GMAIL_API_BASE}/messages"
                params = {"q": "is:unread", "maxResults": max_results}
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                messages_list = response.json()

            if "messages" not in messages_list or not messages_list["messages"]:
                return ToolResult.success("No unread emails.")

            # Fetch details for each message
            message_details = []
            async with httpx.AsyncClient() as client:
                for msg in messages_list["messages"]:
                    msg_id = msg["id"]
                    detail_url = f"{GMAIL_API_BASE}/messages/{msg_id}"
                    detail_response = await client.get(
                        detail_url,
                        headers=headers,
                        params={
                            "format": "metadata",
                            "metadataHeaders": ["Subject", "From", "Date"],
                        },
                    )
                    detail_response.raise_for_status()
                    msg_detail = detail_response.json()

                    headers_list = msg_detail.get("payload", {}).get("headers", [])
                    subject = next(
                        (h["value"] for h in headers_list if h["name"] == "Subject"),
                        "No subject",
                    )
                    sender = next(
                        (h["value"] for h in headers_list if h["name"] == "From"),
                        "Unknown",
                    )
                    date = next(
                        (h["value"] for h in headers_list if h["name"] == "Date"), ""
                    )

                    message_details.append(
                        {"id": msg_id, "subject": subject, "from": sender, "date": date}
                    )

            output = "**Unread Emails:**\n"
            for i, msg in enumerate(message_details, 1):
                output += f"\n{i}. **{msg['subject']}**\n"
                output += f"   From: {msg['from']}\n"
                output += f"   Date: {msg['date']}\n"
                output += f"   ID: {msg['id']}\n"

            return ToolResult.success(output, count=len(message_details))

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to fetch emails: {e}")
        except Exception as e:
            logger.error(f"Error listing unread emails: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ReadEmailTool(_GmailTool):
    """Read a specific email by message ID."""

    name = "read_gmail"
    description = "Read a specific email by its message ID"
    status_text = "Reading email..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, **_) -> ToolResult:
        """Fetch full email content from Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            async with httpx.AsyncClient() as client:
                url = f"{GMAIL_API_BASE}/messages/{message_id}"
                response = await client.get(
                    url, headers=headers, params={"format": "full"}
                )
                response.raise_for_status()
                message = response.json()

            # Extract headers
            headers_list = message.get("payload", {}).get("headers", [])
            subject = next(
                (h["value"] for h in headers_list if h["name"] == "Subject"),
                "No subject",
            )
            sender = next(
                (h["value"] for h in headers_list if h["name"] == "From"), "Unknown"
            )
            to = next(
                (h["value"] for h in headers_list if h["name"] == "To"), "Unknown"
            )
            date = next((h["value"] for h in headers_list if h["name"] == "Date"), "")

            # Extract body
            body = self._extract_body(message.get("payload", {}))

            output = f"**From:** {sender}\n"
            output += f"**To:** {to}\n"
            output += f"**Date:** {date}\n"
            output += f"**Subject:** {subject}\n"
            output += f"\n---\n\n{body}"

            return ToolResult.success(output, message_id=message_id)

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to read email: {e}")
        except Exception as e:
            logger.error(f"Error reading email: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")

    @staticmethod
    def _extract_body(payload: dict) -> str:
        """Extract email body from payload."""
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8")
        else:
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8")

        return "(No body content)"


class SendReplyTool(_GmailTool):
    """Send a reply to an email thread."""

    name = "send_reply"
    description = "Send a reply to an email thread"
    status_text = "Sending reply..."
    parameters = [
        ToolParam(
            name="thread_id",
            type="string",
            description="Gmail thread ID",
            required=True,
        ),
        ToolParam(
            name="to",
            type="string",
            description="Recipient email address",
            required=True,
        ),
        ToolParam(
            name="body",
            type="string",
            description="Email body text",
            required=True,
        ),
        ToolParam(
            name="subject",
            type="string",
            description="Optional subject override",
            required=False,
        ),
    ]

    async def execute(
        self,
        thread_id: str,
        to: str,
        body: str,
        subject: Optional[str] = None,
        **_,
    ) -> ToolResult:
        """Send a reply via Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            email_message = self._create_email(
                to=to, subject=subject or "Re: (no subject)", body=body
            )

            async with httpx.AsyncClient() as client:
                url = f"{GMAIL_API_BASE}/messages/send"
                payload = {"raw": email_message, "threadId": thread_id}
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

            return ToolResult.success(
                f"Reply sent successfully to {to}",
                message_id=result.get("id"),
                thread_id=thread_id,
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to send reply: {e}")
        except Exception as e:
            logger.error(f"Error sending reply: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")

    @staticmethod
    def _create_email(to: str, subject: str, body: str) -> str:
        """Create a base64-encoded email message."""
        message = f"To: {to}\nSubject: {subject}\n\n{body}"
        return base64.urlsafe_b64encode(message.encode()).decode()


class ArchiveEmailTool(_GmailTool):
    """Archive an email (remove from inbox)."""

    name = "archive_email"
    description = "Archive an email (remove from inbox)"
    status_text = "Archiving..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID to archive",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, **_) -> ToolResult:
        """Archive an email via Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            async with httpx.AsyncClient() as client:
                url = f"{GMAIL_API_BASE}/messages/{message_id}/modify"
                payload = {"removeLabelIds": ["INBOX"]}
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

            return ToolResult.success(f"Email {message_id} archived successfully")

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to archive email: {e}")
        except Exception as e:
            logger.error(f"Error archiving email: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")
