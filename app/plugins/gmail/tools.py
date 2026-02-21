"""Gmail tools for reading and managing emails.

Provides tools to interact with Gmail API:
- List unread emails
- Read specific email by message ID
- Send a new email
- Send replies to email threads
- Create email drafts
- Archive emails
- Search emails using Gmail query syntax
- Reply to all recipients of an email thread
- Get all messages in a thread
- Move email to trash
- Mark email as read/unread
- List, add, and remove Gmail labels

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
from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

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


class SendEmailTool(_GmailTool):
    """Send a new email (not a reply to an existing thread)."""

    name = "send_email"
    description = "Send a new email to one or more recipients"
    status_text = "Sending email..."
    parameters = [
        ToolParam(
            name="to",
            type="string",
            description="Recipient email address(es), comma-separated for multiple",
            required=True,
        ),
        ToolParam(
            name="subject",
            type="string",
            description="Email subject line",
            required=True,
        ),
        ToolParam(
            name="body",
            type="string",
            description="Email body text",
            required=True,
        ),
        ToolParam(
            name="cc",
            type="string",
            description="CC recipients, comma-separated (optional)",
            required=False,
        ),
        ToolParam(
            name="bcc",
            type="string",
            description="BCC recipients, comma-separated (optional)",
            required=False,
        ),
    ]

    async def execute(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        **_,
    ) -> ToolResult:
        """Send a new email via Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            raw_message = self._build_raw_email(to, subject, body, cc, bcc)

            async with httpx.AsyncClient() as client:
                url = f"{GMAIL_API_BASE}/messages/send"
                payload = {"raw": raw_message}
                response = await client.post(
                    url, headers=self._auth_headers(), json=payload
                )
                response.raise_for_status()
                result = response.json()

            return ToolResult.success(
                f"Email sent successfully to {to}",
                message_id=result.get("id"),
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to send email: {e}")
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")

    @staticmethod
    def _build_raw_email(
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
    ) -> str:
        """Build a base64url-encoded RFC 2822 email message."""
        lines = [f"To: {to}", f"Subject: {subject}"]
        if cc:
            lines.append(f"Cc: {cc}")
        if bcc:
            lines.append(f"Bcc: {bcc}")
        lines.append("Content-Type: text/plain; charset=utf-8")
        lines.append("")
        lines.append(body)
        raw = "\r\n".join(lines)
        return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


class CreateDraftTool(_GmailTool):
    """Create a draft email in Gmail."""

    name = "create_draft"
    description = "Create a draft email in Gmail (saved but not sent)"
    status_text = "Creating draft..."
    parameters = [
        ToolParam(
            name="to",
            type="string",
            description="Recipient email address(es), comma-separated for multiple",
            required=True,
        ),
        ToolParam(
            name="subject",
            type="string",
            description="Email subject line",
            required=True,
        ),
        ToolParam(
            name="body",
            type="string",
            description="Email body text",
            required=True,
        ),
        ToolParam(
            name="cc",
            type="string",
            description="CC recipients, comma-separated (optional)",
            required=False,
        ),
        ToolParam(
            name="bcc",
            type="string",
            description="BCC recipients, comma-separated (optional)",
            required=False,
        ),
    ]

    async def execute(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        **_,
    ) -> ToolResult:
        """Create a draft via Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            raw_message = SendEmailTool._build_raw_email(to, subject, body, cc, bcc)

            async with httpx.AsyncClient() as client:
                url = f"{GMAIL_API_BASE}/drafts"
                payload = {"message": {"raw": raw_message}}
                response = await client.post(
                    url, headers=self._auth_headers(), json=payload
                )
                response.raise_for_status()
                result = response.json()

            draft_id = result.get("id", "")
            return ToolResult.success(
                f"Draft created successfully (to: {to}, subject: {subject})\nDraft ID: {draft_id}",
                draft_id=draft_id,
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to create draft: {e}")
        except Exception as e:
            logger.error(f"Error creating draft: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


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


# ── New Gmail Tools (v0.3.0) ──────────────────────────────────


class SearchEmailTool(_GmailTool):
    """Search Gmail using Gmail query syntax."""

    name = "search_email"
    description = (
        "Search Gmail using Gmail query syntax "
        "(e.g. 'from:boss@company.com', 'subject:invoice', "
        "'has:attachment', 'is:unread after:2024/01/01')"
    )
    status_text = "Searching emails..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="Gmail search query string",
            required=True,
        ),
        ToolParam(
            name="max_results",
            type="integer",
            description="Max emails to return (default 10)",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, query: str, max_results: int = 10, **_) -> ToolResult:
        """Search Gmail using the provided query string."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            async with httpx.AsyncClient() as client:
                # Step 1: Get matching message IDs
                resp = await client.get(
                    f"{GMAIL_API_BASE}/messages",
                    headers=headers,
                    params={"q": query, "maxResults": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            messages = data.get("messages", [])
            if not messages:
                return ToolResult.success(f"No emails found matching '{query}'.")

            # Step 2: Fetch metadata for each message
            results = []
            async with httpx.AsyncClient() as client:
                for msg in messages:
                    msg_id = msg["id"]
                    detail_resp = await client.get(
                        f"{GMAIL_API_BASE}/messages/{msg_id}",
                        headers=headers,
                        params={
                            "format": "metadata",
                            "metadataHeaders": ["Subject", "From", "Date"],
                        },
                    )
                    detail_resp.raise_for_status()
                    detail = detail_resp.json()

                    hdrs = detail.get("payload", {}).get("headers", [])
                    subject = next(
                        (h["value"] for h in hdrs if h["name"] == "Subject"),
                        "No subject",
                    )
                    sender = next(
                        (h["value"] for h in hdrs if h["name"] == "From"),
                        "Unknown",
                    )
                    date = next((h["value"] for h in hdrs if h["name"] == "Date"), "")
                    results.append(
                        {"id": msg_id, "subject": subject, "from": sender, "date": date}
                    )

            output = f"**Search results for '{query}':**\n"
            for i, msg in enumerate(results, 1):
                output += f"\n{i}. **{msg['subject']}**\n"
                output += f"   From: {msg['from']}\n"
                output += f"   Date: {msg['date']}\n"
                output += f"   ID: {msg['id']}\n"

            return ToolResult.success(output, count=len(results))

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to search emails: {e}")
        except Exception as e:
            logger.error(f"Error searching emails: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ReplyAllTool(_GmailTool):
    """Reply to all recipients of an email thread (To + CC)."""

    name = "reply_all"
    description = "Reply to all recipients of an email (sender + all To/CC recipients)"
    status_text = "Sending reply-all..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID to reply to",
            required=True,
        ),
        ToolParam(
            name="body",
            type="string",
            description="Reply body text",
            required=True,
        ),
        ToolParam(
            name="subject",
            type="string",
            description="Subject override (optional — defaults to Re: [original subject])",
            required=False,
        ),
    ]

    async def execute(
        self,
        message_id: str,
        body: str,
        subject: Optional[str] = None,
        **_,
    ) -> ToolResult:
        """Fetch original message, build reply-all, and send."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()
            my_email = (getattr(self, "_context", {}) or {}).get("account_email", "")

            # Step 1: Fetch original message metadata
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{GMAIL_API_BASE}/messages/{message_id}",
                    headers=headers,
                    params={
                        "format": "metadata",
                        "metadataHeaders": [
                            "From",
                            "To",
                            "Cc",
                            "Subject",
                            "Message-ID",
                        ],
                    },
                )
                resp.raise_for_status()
                original = resp.json()

            thread_id = original.get("threadId", "")
            orig_hdrs = original.get("payload", {}).get("headers", [])

            orig_from = next((h["value"] for h in orig_hdrs if h["name"] == "From"), "")
            orig_to = next((h["value"] for h in orig_hdrs if h["name"] == "To"), "")
            orig_cc = next((h["value"] for h in orig_hdrs if h["name"] == "Cc"), "")
            orig_subject = next(
                (h["value"] for h in orig_hdrs if h["name"] == "Subject"),
                "Re: (no subject)",
            )
            orig_message_id = next(
                (h["value"] for h in orig_hdrs if h["name"] == "Message-ID"), ""
            )

            # Build reply subject
            reply_subject = subject or (
                orig_subject
                if orig_subject.lower().startswith("re:")
                else f"Re: {orig_subject}"
            )

            # Collect all CC addresses (original To + Cc), excluding own email
            def _split_addresses(addr_str: str) -> list[str]:
                """Split a comma-separated address string into individual addresses."""
                if not addr_str:
                    return []
                return [a.strip() for a in addr_str.split(",") if a.strip()]

            all_cc_raw = _split_addresses(orig_to) + _split_addresses(orig_cc)
            # Filter out the user's own email address
            cc_addresses = [
                addr for addr in all_cc_raw if my_email.lower() not in addr.lower()
            ]

            # Build RFC 2822 message
            lines = [
                f"To: {orig_from}",
                f"Subject: {reply_subject}",
            ]
            if cc_addresses:
                lines.append(f"Cc: {', '.join(cc_addresses)}")
            if orig_message_id:
                lines.append(f"In-Reply-To: {orig_message_id}")
                lines.append(f"References: {orig_message_id}")
            lines.append("Content-Type: text/plain; charset=utf-8")
            lines.append("")
            lines.append(body)

            raw = "\r\n".join(lines)
            raw_encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")

            # Step 2: Send the reply-all
            async with httpx.AsyncClient() as client:
                send_resp = await client.post(
                    f"{GMAIL_API_BASE}/messages/send",
                    headers=headers,
                    json={"raw": raw_encoded, "threadId": thread_id},
                )
                send_resp.raise_for_status()
                result = send_resp.json()

            all_recipients = [orig_from] + cc_addresses
            return ToolResult.success(
                f"Reply-all sent successfully.\n"
                f"Recipients: {', '.join(all_recipients)}",
                message_id=result.get("id"),
                thread_id=thread_id,
                recipients=all_recipients,
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to send reply-all: {e}")
        except Exception as e:
            logger.error(f"Error sending reply-all: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class GetThreadTool(_GmailTool):
    """Get all messages in an email thread."""

    name = "get_thread"
    description = "Get all messages in an email thread"
    status_text = "Loading thread..."
    parameters = [
        ToolParam(
            name="thread_id",
            type="string",
            description="Gmail thread ID",
            required=True,
        ),
    ]

    async def execute(self, thread_id: str, **_) -> ToolResult:
        """Fetch all messages in a thread from Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{GMAIL_API_BASE}/threads/{thread_id}",
                    headers=headers,
                    params={"format": "full"},
                )
                resp.raise_for_status()
                thread = resp.json()

            messages = thread.get("messages", [])
            if not messages:
                return ToolResult.success("Thread is empty.")

            output = f"**Thread ({len(messages)} messages):**\n"
            for i, msg in enumerate(messages, 1):
                msg_hdrs = msg.get("payload", {}).get("headers", [])
                sender = next(
                    (h["value"] for h in msg_hdrs if h["name"] == "From"), "Unknown"
                )
                date = next((h["value"] for h in msg_hdrs if h["name"] == "Date"), "")
                # Use snippet for a brief preview (first 200 chars)
                snippet = msg.get("snippet", "")[:200]

                output += f"\n**Message {i}**\n"
                output += f"   From: {sender}\n"
                output += f"   Date: {date}\n"
                output += f"   ID: {msg.get('id', '')}\n"
                if snippet:
                    output += f"   Preview: {snippet}...\n"

            return ToolResult.success(output, message_count=len(messages))

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to get thread: {e}")
        except Exception as e:
            logger.error(f"Error getting thread: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class TrashEmailTool(_GmailTool):
    """Move an email to trash (reversible, safer than permanent delete)."""

    name = "trash_email"
    description = "Move an email to trash (can be recovered from Trash within 30 days)"
    status_text = "Moving to trash..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID to trash",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, **_) -> ToolResult:
        """Move an email to trash via Gmail API."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{GMAIL_API_BASE}/messages/{message_id}/trash",
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()

            return ToolResult.success(
                f"Email {message_id} moved to trash. "
                "It can be recovered from Trash within 30 days.",
                message_id=message_id,
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to trash email: {e}")
        except Exception as e:
            logger.error(f"Error trashing email: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class MarkReadTool(_GmailTool):
    """Mark an email as read."""

    name = "mark_read"
    description = "Mark an email as read"
    status_text = "Marking as read..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID to mark as read",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, **_) -> ToolResult:
        """Remove the UNREAD label from an email."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{GMAIL_API_BASE}/messages/{message_id}/modify",
                    headers=self._auth_headers(),
                    json={"removeLabelIds": ["UNREAD"]},
                )
                resp.raise_for_status()

            return ToolResult.success(
                f"Email {message_id} marked as read.", message_id=message_id
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to mark email as read: {e}")
        except Exception as e:
            logger.error(f"Error marking email as read: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class MarkUnreadTool(_GmailTool):
    """Mark an email as unread."""

    name = "mark_unread"
    description = "Mark an email as unread"
    status_text = "Marking as unread..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID to mark as unread",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, **_) -> ToolResult:
        """Add the UNREAD label to an email."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{GMAIL_API_BASE}/messages/{message_id}/modify",
                    headers=self._auth_headers(),
                    json={"addLabelIds": ["UNREAD"]},
                )
                resp.raise_for_status()

            return ToolResult.success(
                f"Email {message_id} marked as unread.", message_id=message_id
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to mark email as unread: {e}")
        except Exception as e:
            logger.error(f"Error marking email as unread: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ListLabelsTool(_GmailTool):
    """List all Gmail labels (folders)."""

    name = "list_labels"
    description = "List all Gmail labels (folders)"
    status_text = "Fetching labels..."
    parameters = []  # No parameters required

    async def execute(self, **_) -> ToolResult:
        """Fetch all user-defined Gmail labels."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{GMAIL_API_BASE}/labels",
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            labels = data.get("labels", [])
            # Filter out system category labels (CATEGORY_*) — they're not user-facing
            user_labels = [
                lbl for lbl in labels if not lbl.get("name", "").startswith("CATEGORY_")
            ]

            if not user_labels:
                return ToolResult.success("No labels found.")

            output = "**Gmail Labels:**\n"
            for lbl in user_labels:
                output += f"\n- **{lbl.get('name', '')}** (ID: `{lbl.get('id', '')}`)"

            return ToolResult.success(output, count=len(user_labels))

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to list labels: {e}")
        except Exception as e:
            logger.error(f"Error listing labels: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class AddLabelTool(_GmailTool):
    """Add a label to an email by label name."""

    name = "add_label"
    description = "Add a label to an email"
    status_text = "Adding label..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID",
            required=True,
        ),
        ToolParam(
            name="label_name",
            type="string",
            description="Label name to add (e.g. 'Work', 'Important')",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, label_name: str, **_) -> ToolResult:
        """Look up label ID by name, then add it to the message."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Step 1: Fetch all labels to find the matching ID
            async with httpx.AsyncClient() as client:
                labels_resp = await client.get(
                    f"{GMAIL_API_BASE}/labels", headers=headers
                )
                labels_resp.raise_for_status()
                labels_data = labels_resp.json()

            labels = labels_data.get("labels", [])
            label_id = next(
                (
                    lbl["id"]
                    for lbl in labels
                    if lbl.get("name", "").lower() == label_name.lower()
                ),
                None,
            )

            if not label_id:
                return ToolResult.fail(
                    f"Label '{label_name}' not found. "
                    "Use list_labels to see available labels."
                )

            # Step 2: Apply the label to the message
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{GMAIL_API_BASE}/messages/{message_id}/modify",
                    headers=headers,
                    json={"addLabelIds": [label_id]},
                )
                resp.raise_for_status()

            return ToolResult.success(
                f"Label '{label_name}' added to email {message_id}.",
                message_id=message_id,
                label_name=label_name,
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to add label: {e}")
        except Exception as e:
            logger.error(f"Error adding label: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RemoveLabelTool(_GmailTool):
    """Remove a label from an email by label name."""

    name = "remove_label"
    description = "Remove a label from an email"
    status_text = "Removing label..."
    parameters = [
        ToolParam(
            name="message_id",
            type="string",
            description="Gmail message ID",
            required=True,
        ),
        ToolParam(
            name="label_name",
            type="string",
            description="Label name to remove (e.g. 'Work', 'Important')",
            required=True,
        ),
    ]

    async def execute(self, message_id: str, label_name: str, **_) -> ToolResult:
        """Look up label ID by name, then remove it from the message."""
        if not self._get_token():
            return ToolResult.fail("Gmail not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Step 1: Fetch all labels to find the matching ID
            async with httpx.AsyncClient() as client:
                labels_resp = await client.get(
                    f"{GMAIL_API_BASE}/labels", headers=headers
                )
                labels_resp.raise_for_status()
                labels_data = labels_resp.json()

            labels = labels_data.get("labels", [])
            label_id = next(
                (
                    lbl["id"]
                    for lbl in labels
                    if lbl.get("name", "").lower() == label_name.lower()
                ),
                None,
            )

            if not label_id:
                return ToolResult.fail(
                    f"Label '{label_name}' not found. "
                    "Use list_labels to see available labels."
                )

            # Step 2: Remove the label from the message
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{GMAIL_API_BASE}/messages/{message_id}/modify",
                    headers=headers,
                    json={"removeLabelIds": [label_id]},
                )
                resp.raise_for_status()

            return ToolResult.success(
                f"Label '{label_name}' removed from email {message_id}.",
                message_id=message_id,
                label_name=label_name,
            )

        except httpx.HTTPError as e:
            logger.error(f"Gmail API error: {e}", exc_info=True)
            return ToolResult.fail(f"Failed to remove label: {e}")
        except Exception as e:
            logger.error(f"Error removing label: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RefreshGmailTokenTool(RefreshOAuthTokenTool):
    """Refresh the Gmail OAuth access token before it expires.

    Called automatically by the cron system every 50 minutes.
    Can also be called manually if Gmail tools start returning auth errors.
    """

    name = "refresh_gmail_token"
    plugin_name = "gmail"
    description = (
        "Refresh the Gmail OAuth access token. "
        "Call this when Gmail tools return authentication errors, "
        "or when instructed by the system to prevent token expiry."
    )
