"""Telegram Plugin Routes.

FastAPI routes for the Telegram Bot webhook.
Registered when the plugin is loaded with plugin_type: telephony.

Incoming Telegram updates arrive at POST /plugins/telegram/webhook.
The endpoint validates the secret token, then dispatches the update
to agent_core.run_session() as a background task — Telegram requires
a fast HTTP 200 response (within 5 seconds).

The agent session calls handle_telegram_event with the raw payload,
then decides whether to reply via telegram_send_message.

Security:
  - X-Telegram-Bot-Api-Secret-Token header validated against secret_token
    config field (if configured). Requests without a valid token are
    rejected with 403.
  - allowed_chat_ids (in plugin context) restricts which chats can
    reach the agent — enforced inside HandleTelegramEventTool.

Config fields (injected via set_config after startup):
  bot_token       — Telegram Bot API token
  secret_token    — Optional webhook secret (set when registering webhook)
  allowed_chat_ids — Optional comma-separated list of permitted chat IDs
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from aether.agent import AgentCore

logger = logging.getLogger(__name__)

# Module-level state (set by _init_telegram in main.py after startup)
_agent_core: AgentCore | None = None
_bot_token: str | None = None
_secret_token: str | None = None


def set_agent(agent: "AgentCore") -> None:
    """Inject the AgentCore instance."""
    global _agent_core
    _agent_core = agent


def set_config(config: dict) -> None:
    """Inject plugin config (bot_token, secret_token, allowed_chat_ids)."""
    global _bot_token, _secret_token
    _bot_token = config.get("bot_token") or None
    _secret_token = config.get("secret_token") or None


def get_config_token() -> str | None:
    return _bot_token


def create_router() -> APIRouter:
    """Create the FastAPI router for Telegram plugin endpoints."""
    router = APIRouter(prefix="/plugins/telegram", tags=["telegram"])

    @router.post("/webhook")
    async def telegram_webhook(request: Request) -> JSONResponse:
        """Receive incoming Telegram updates.

        Telegram calls this URL for every update (message, edited_message,
        callback_query, etc.).  We validate the secret token, then fire off
        a background agent session and immediately return 200 OK.

        Telegram will retry delivery if it doesn't receive a 200 within 5s,
        so we must not block here.
        """
        # ── Secret token validation ──────────────────────────────────────
        if _secret_token:
            incoming = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if incoming != _secret_token:
                logger.warning(
                    "Telegram webhook: invalid secret token (got=%r)", incoming[:8]
                )
                return JSONResponse(
                    {"ok": False, "error": "Forbidden"}, status_code=403
                )

        # ── Parse body ───────────────────────────────────────────────────
        try:
            body = await request.json()
        except Exception as e:
            logger.warning("Telegram webhook: failed to parse JSON body: %s", e)
            return JSONResponse({"ok": False, "error": "Bad request"}, status_code=400)

        if not _agent_core:
            logger.warning(
                "Telegram webhook: agent_core not configured — update dropped "
                "(update_id=%s)",
                body.get("update_id", "?"),
            )
            # Still return 200 so Telegram doesn't keep retrying
            return JSONResponse({"ok": True, "warning": "agent not ready"})

        # ── Dispatch to agent ────────────────────────────────────────────
        update_id = body.get("update_id", "unknown")

        # Derive a stable session ID from the chat ID so the agent has
        # per-chat conversation history.  Fall back to a shared session
        # for non-message updates (callback_query, etc.).
        chat_id = _extract_chat_id(body)
        session_id = f"telegram-{chat_id}" if chat_id else "telegram-webhook"

        payload_json = json.dumps(body, ensure_ascii=False)
        instruction = (
            "A Telegram message has arrived. "
            "Call the `handle_telegram_event` tool now with the following payload "
            "and decide what action to take (reply to the user, or ignore).\n\n"
            f"Payload:\n{payload_json}"
        )

        logger.info(
            "Telegram update received (update_id=%s, chat_id=%s) → session %s",
            update_id,
            chat_id or "?",
            session_id,
        )

        asyncio.create_task(
            _agent_core.run_session(
                session_id=session_id,
                user_message=instruction,
                background=False,
            )
        )

        return JSONResponse({"ok": True})

    @router.get("/status")
    async def telegram_status() -> JSONResponse:
        """Get the status of the Telegram plugin."""
        return JSONResponse(
            {
                "configured": bool(_bot_token),
                "secret_token_set": bool(_secret_token),
                "agent_ready": _agent_core is not None,
            }
        )

    return router


def _extract_chat_id(update: dict) -> str | None:
    """Extract the chat ID from a Telegram Update object.

    Handles message, edited_message, channel_post, callback_query.
    Returns None if the update type doesn't carry a chat ID.
    """
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        msg = update.get(key)
        if msg and isinstance(msg, dict):
            chat = msg.get("chat", {})
            cid = chat.get("id")
            if cid is not None:
                return str(cid)

    # callback_query has a message nested inside
    cq = update.get("callback_query")
    if cq and isinstance(cq, dict):
        msg = cq.get("message", {})
        chat = msg.get("chat", {})
        cid = chat.get("id")
        if cid is not None:
            return str(cid)

    return None
