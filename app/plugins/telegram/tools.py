"""
Telegram plugin tools.

Provides tools to send messages, photos, and handle incoming Telegram
updates via the Bot API.

All tools receive credentials at call time via self._context (set by
safe_execute from the plugin context store). No __init__ args required —
the loader instantiates with cls().

Credentials in context:
  bot_token       — Telegram Bot API token from @BotFather
  allowed_chat_ids — Optional comma-separated list of permitted chat IDs

Telegram Bot API base: https://api.telegram.org/bot{token}/{method}
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class _TelegramTool(AetherTool):
    """Base for Telegram tools — provides token and API call helpers."""

    def _token(self) -> str | None:
        ctx = getattr(self, "_context", None) or {}
        return ctx.get("bot_token") or None

    def _api_url(self, method: str) -> str:
        return TELEGRAM_API.format(token=self._token(), method=method)

    def _allowed_chat_ids(self) -> set[str]:
        """Return the set of allowed chat IDs, or empty set (= allow all)."""
        ctx = getattr(self, "_context", None) or {}
        raw = ctx.get("allowed_chat_ids", "").strip()
        if not raw:
            return set()
        return {cid.strip() for cid in raw.split(",") if cid.strip()}

    def _is_chat_allowed(self, chat_id: str | int) -> bool:
        allowed = self._allowed_chat_ids()
        if not allowed:
            return True  # No restriction configured
        return str(chat_id) in allowed

    async def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to Telegram Bot API. Raises httpx.HTTPError on failure."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self._api_url(method), json=payload)
            resp.raise_for_status()
            return resp.json()


class SendMessageTool(_TelegramTool):
    """Send a text message to a Telegram chat."""

    name = "telegram_send_message"
    description = (
        "Send a text message to a Telegram chat. "
        "Use this to notify the user, reply to their message, or send any text. "
        "Supports Markdown formatting."
    )
    status_text = "Sending Telegram message..."
    parameters = [
        ToolParam(
            name="chat_id",
            type="string",
            description=(
                "Telegram chat ID to send the message to. "
                "Use the chat_id from an incoming message, or the user's known chat ID."
            ),
            required=True,
        ),
        ToolParam(
            name="text",
            type="string",
            description="Message text. Supports Markdown (bold, italic, code, links).",
            required=True,
        ),
        ToolParam(
            name="parse_mode",
            type="string",
            description="Text formatting: Markdown or HTML. Defaults to Markdown.",
            required=False,
            default="Markdown",
            enum=["Markdown", "HTML", "MarkdownV2"],
        ),
        ToolParam(
            name="reply_to_message_id",
            type="integer",
            description="Optional. Reply to a specific message by its ID.",
            required=False,
        ),
    ]

    async def execute(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "Markdown",
        reply_to_message_id: int | None = None,
        **_,
    ) -> ToolResult:
        if not self._token():
            return ToolResult.fail("Telegram not connected — missing bot token.")

        if not self._is_chat_allowed(chat_id):
            return ToolResult.fail(
                f"Chat ID {chat_id} is not in the allowed list. "
                "Update allowed_chat_ids in plugin settings to permit this chat."
            )

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id

        try:
            result = await self._post("sendMessage", payload)
            msg_id = result.get("result", {}).get("message_id", "")
            return ToolResult.success(
                f"Message sent to chat {chat_id} (message_id={msg_id}).",
                chat_id=chat_id,
                message_id=msg_id,
            )
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            logger.error("Telegram sendMessage failed: %s — %s", e, body)
            return ToolResult.fail(
                f"Failed to send Telegram message (HTTP {e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error("Telegram sendMessage error: %s", e, exc_info=True)
            return ToolResult.fail(f"Telegram error: {e}")


class SendPhotoTool(_TelegramTool):
    """Send a photo to a Telegram chat via URL."""

    name = "telegram_send_photo"
    description = (
        "Send a photo to a Telegram chat using a public image URL. "
        "Optionally include a caption."
    )
    status_text = "Sending photo..."
    parameters = [
        ToolParam(
            name="chat_id",
            type="string",
            description="Telegram chat ID to send the photo to.",
            required=True,
        ),
        ToolParam(
            name="photo_url",
            type="string",
            description="Public URL of the image to send.",
            required=True,
        ),
        ToolParam(
            name="caption",
            type="string",
            description="Optional caption for the photo. Supports Markdown.",
            required=False,
        ),
    ]

    async def execute(
        self,
        chat_id: str,
        photo_url: str,
        caption: str | None = None,
        **_,
    ) -> ToolResult:
        if not self._token():
            return ToolResult.fail("Telegram not connected — missing bot token.")

        if not self._is_chat_allowed(chat_id):
            return ToolResult.fail(f"Chat ID {chat_id} is not in the allowed list.")

        payload: dict[str, Any] = {"chat_id": chat_id, "photo": photo_url}
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "Markdown"

        try:
            result = await self._post("sendPhoto", payload)
            msg_id = result.get("result", {}).get("message_id", "")
            return ToolResult.success(
                f"Photo sent to chat {chat_id} (message_id={msg_id}).",
                chat_id=chat_id,
                message_id=msg_id,
            )
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            logger.error("Telegram sendPhoto failed: %s — %s", e, body)
            return ToolResult.fail(
                f"Failed to send photo (HTTP {e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error("Telegram sendPhoto error: %s", e, exc_info=True)
            return ToolResult.fail(f"Telegram error: {e}")


class GetChatInfoTool(_TelegramTool):
    """Get information about a Telegram chat or user."""

    name = "telegram_get_chat"
    description = (
        "Get information about a Telegram chat or user: "
        "name, username, type (private/group/supergroup/channel), and member count."
    )
    status_text = "Getting chat info..."
    parameters = [
        ToolParam(
            name="chat_id",
            type="string",
            description="Telegram chat ID or @username.",
            required=True,
        ),
    ]

    async def execute(self, chat_id: str, **_) -> ToolResult:
        if not self._token():
            return ToolResult.fail("Telegram not connected — missing bot token.")

        try:
            result = await self._post("getChat", {"chat_id": chat_id})
            chat = result.get("result", {})

            chat_type = chat.get("type", "unknown")
            title = chat.get("title") or chat.get("first_name", "")
            username = chat.get("username", "")
            member_count = chat.get("member_count")

            lines = [
                f"Chat ID: {chat.get('id', chat_id)}",
                f"Type: {chat_type}",
                f"Name: {title}",
            ]
            if username:
                lines.append(f"Username: @{username}")
            if member_count is not None:
                lines.append(f"Members: {member_count}")

            return ToolResult.success("\n".join(lines), chat=chat)

        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            logger.error("Telegram getChat failed: %s — %s", e, body)
            return ToolResult.fail(
                f"Failed to get chat info (HTTP {e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error("Telegram getChat error: %s", e, exc_info=True)
            return ToolResult.fail(f"Telegram error: {e}")


class HandleTelegramEventTool(_TelegramTool):
    """
    Process an incoming Telegram webhook update.

    Called by the agent when a message arrives from Telegram.
    Extracts the message content, sender info, and chat ID, then
    returns a structured summary for the LLM to decide what to do.

    The LLM typically responds by calling telegram_send_message with
    the same chat_id to reply to the user.
    """

    name = "handle_telegram_event"
    description = (
        "Process an incoming Telegram message or update. "
        "Extracts the sender, chat ID, and message text. "
        "After calling this, use telegram_send_message to reply."
    )
    status_text = "Processing Telegram message..."
    parameters = [
        ToolParam(
            name="payload",
            type="object",
            description="Raw Telegram webhook update payload.",
            required=True,
        ),
    ]

    async def execute(self, payload: dict, **_) -> ToolResult:  # type: ignore[override]
        if not self._token():
            return ToolResult.fail("Telegram not connected — missing bot token.")

        # Telegram sends Update objects. We handle message and edited_message.
        update_id = payload.get("update_id", "")
        message = (
            payload.get("message")
            or payload.get("edited_message")
            or payload.get("channel_post")
        )

        if not message:
            # Could be a callback_query, inline_query, etc. — not handled yet.
            update_type = next((k for k in payload if k != "update_id"), "unknown")
            return ToolResult.success(
                f"Telegram update received (type={update_type}, id={update_id}). "
                "No message to process — this update type is not yet handled."
            )

        # Extract fields
        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))
        chat_type = chat.get("type", "private")  # private, group, supergroup, channel

        sender = message.get("from", {})
        sender_id = str(sender.get("id", ""))
        first_name = sender.get("first_name", "")
        last_name = sender.get("last_name", "")
        username = sender.get("username", "")
        sender_name = f"{first_name} {last_name}".strip() or username or sender_id

        text = message.get("text", "")
        caption = message.get("caption", "")  # For photo/video messages
        message_id = message.get("message_id", "")
        date = message.get("date", "")

        # Check if this chat is allowed
        if not self._is_chat_allowed(chat_id):
            logger.warning(
                "Telegram: message from disallowed chat_id=%s (sender=%s)",
                chat_id,
                sender_name,
            )
            return ToolResult.success(
                f"Message from chat {chat_id} ignored — not in allowed_chat_ids list."
            )

        # Handle non-text messages
        content = text or caption
        if not content:
            # Photo, sticker, voice, etc. without caption
            has_photo = bool(message.get("photo"))
            has_voice = bool(message.get("voice"))
            has_document = bool(message.get("document"))
            has_sticker = bool(message.get("sticker"))

            media_type = (
                "photo"
                if has_photo
                else "voice message"
                if has_voice
                else "document"
                if has_document
                else "sticker"
                if has_sticker
                else "media"
            )
            return ToolResult.success(
                f"Telegram {media_type} received from {sender_name} "
                f"(chat_id={chat_id}, message_id={message_id}). "
                f"No text content to process. "
                f"You can acknowledge with telegram_send_message to chat_id={chat_id}.",
                chat_id=chat_id,
                message_id=message_id,
                sender_name=sender_name,
                sender_id=sender_id,
                media_type=media_type,
            )

        # Build summary for LLM
        chat_label = (
            f"group '{chat.get('title', chat_id)}'"
            if chat_type in ("group", "supergroup")
            else f"channel '{chat.get('title', chat_id)}'"
            if chat_type == "channel"
            else f"private chat"
        )

        summary = (
            f"Telegram message from {sender_name} "
            f"(chat_id={chat_id}, {chat_label}, message_id={message_id}):\n\n"
            f"{content}\n\n"
            f"To reply, call telegram_send_message with chat_id={chat_id}."
        )

        return ToolResult.success(
            summary,
            chat_id=chat_id,
            message_id=message_id,
            sender_name=sender_name,
            sender_id=sender_id,
            text=content,
            chat_type=chat_type,
        )
