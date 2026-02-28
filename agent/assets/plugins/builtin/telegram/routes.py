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
import pathlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from aether.agent import AgentCore

logger = logging.getLogger(__name__)

# ── Module-level state (set by _init_telegram in main.py after startup) ──────
_agent_core: AgentCore | None = None
_bot_token: str | None = None
_secret_token: str | None = None

# ── Per-chat sequentialization ────────────────────────────────────────────────
# Prevents concurrent LLM calls for the same chat when messages arrive rapidly.
_chat_locks: dict[str, asyncio.Lock] = {}
_chat_locks_mutex = asyncio.Lock()
_chat_lock_last_used: dict[str, float] = {}


async def _get_chat_lock(chat_id: str) -> asyncio.Lock:
    """Return (or create) the asyncio.Lock for a given chat_id."""
    async with _chat_locks_mutex:
        _chat_lock_last_used[chat_id] = time.monotonic()
        if chat_id not in _chat_locks:
            _chat_locks[chat_id] = asyncio.Lock()
        return _chat_locks[chat_id]


async def _cleanup_stale_locks() -> None:
    """Periodically remove locks for chats inactive for more than 1 hour.

    Runs every 30 minutes as a background task.
    """
    while True:
        await asyncio.sleep(30 * 60)  # 30 minutes
        try:
            cutoff = time.monotonic() - 3600  # 1 hour
            async with _chat_locks_mutex:
                stale = [
                    cid for cid, last in _chat_lock_last_used.items() if last < cutoff
                ]
                for cid in stale:
                    # Only remove if the lock is not currently held
                    lock = _chat_locks.get(cid)
                    if lock is not None and not lock.locked():
                        del _chat_locks[cid]
                        del _chat_lock_last_used[cid]
            if stale:
                logger.debug("Telegram: cleaned up %d stale chat locks", len(stale))
        except Exception as exc:
            logger.debug("Telegram: lock cleanup error (non-fatal): %s", exc)


# ── Update deduplication ──────────────────────────────────────────────────────


class _UpdateDedupeCache:
    """Rolling TTL cache of seen update_ids. Max 2000 entries, 5-minute TTL."""

    def __init__(self, max_size: int = 2000, ttl_seconds: float = 300):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def seen(self, key: str) -> bool:
        """Return True if key was already seen; record it if not."""
        now = time.monotonic()
        self._evict(now)
        if key in self._cache:
            return True
        self._cache[key] = now
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return False

    def _evict(self, now: float) -> None:
        """Remove entries older than TTL."""
        cutoff = now - self._ttl
        while self._cache:
            oldest_key, oldest_time = next(iter(self._cache.items()))
            if oldest_time < cutoff:
                self._cache.popitem(last=False)
            else:
                break


_update_dedupe = _UpdateDedupeCache()

# ── Persisted update_id offset (Layer 2 deduplication) ───────────────────────
_OFFSET_FILE = pathlib.Path("/tmp/aether_telegram_update_offset.json")
_last_offset: int = 0  # loaded at set_config time


def _load_offset() -> int:
    """Load the last-processed update_id from disk."""
    try:
        return json.loads(_OFFSET_FILE.read_text()).get("offset", 0)
    except Exception:
        return 0


def _save_offset(update_id: int) -> None:
    """Atomically persist the last-processed update_id to disk."""
    try:
        tmp = _OFFSET_FILE.with_suffix(f".{update_id}.tmp")
        tmp.write_text(json.dumps({"offset": update_id}))
        tmp.rename(_OFFSET_FILE)
    except Exception:
        pass  # Non-fatal


# ── Config / agent injection ──────────────────────────────────────────────────


def set_agent(agent: "AgentCore") -> None:
    """Inject the AgentCore instance."""
    global _agent_core
    _agent_core = agent
    # Start the stale-lock cleanup background task
    asyncio.create_task(_cleanup_stale_locks())


def set_config(config: dict) -> None:
    """Inject plugin config (bot_token, secret_token, allowed_chat_ids)."""
    global _bot_token, _secret_token, _last_offset
    _bot_token = config.get("bot_token") or None
    _secret_token = config.get("secret_token") or None
    _last_offset = _load_offset()


def get_config_token() -> str | None:
    return _bot_token


# ── Background dispatch with per-chat lock ────────────────────────────────────


async def _dispatch(session_id: str, instruction: str, chat_id: str) -> None:
    """Acquire the per-chat lock, then run the agent session.

    Serialises concurrent messages from the same chat so the LLM never
    processes two messages simultaneously for the same conversation.
    """
    lock = await _get_chat_lock(chat_id)
    async with lock:
        try:
            await _agent_core.run_session(  # type: ignore[union-attr]
                session_id=session_id,
                user_message=instruction,
                background=False,
            )
        except Exception as exc:
            logger.error(
                "Telegram: run_session error (session=%s): %s",
                session_id,
                exc,
                exc_info=True,
            )


# ── Router ────────────────────────────────────────────────────────────────────


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

        # ── Deduplication ────────────────────────────────────────────────
        global _last_offset
        update_id_int: int = body.get("update_id", 0)

        # Layer 2: skip if below (or equal to) the persisted offset
        if update_id_int > 0 and update_id_int <= _last_offset:
            logger.debug(
                "Telegram: skipping already-processed update_id=%d", update_id_int
            )
            return JSONResponse({"ok": True})

        # Layer 1: skip if seen in this session's in-memory cache
        dedupe_key = str(update_id_int) if update_id_int else f"body:{hash(str(body))}"
        if _update_dedupe.seen(dedupe_key):
            logger.debug("Telegram: duplicate update_id=%s dropped", dedupe_key)
            return JSONResponse({"ok": True})

        # Persist the new offset after accepting the update
        if update_id_int > 0:
            _save_offset(update_id_int)
            _last_offset = update_id_int

        # ── Derive session / chat IDs ────────────────────────────────────
        update_id = body.get("update_id", "unknown")
        chat_id = _extract_chat_id(body)
        session_id = f"telegram-{chat_id}" if chat_id else "telegram-webhook"
        dispatch_chat_id = chat_id or "default"

        # ── Build instruction for the agent ─────────────────────────────
        text_content = _extract_message_text(body)

        if text_content and text_content.strip().startswith("/start"):
            instruction = (
                "The user just started a conversation with you on Telegram by "
                "sending /start. Greet them warmly, introduce yourself briefly, "
                "and let them know they can message you here just like a normal "
                f"conversation. Their chat_id is {chat_id}. "
                "Reply using telegram_send_message."
            )
        elif text_content and text_content.strip().startswith("/help"):
            instruction = (
                "The user sent /help on Telegram. Briefly explain what you can "
                f"do and how to use you. Their chat_id is {chat_id}. "
                "Reply using telegram_send_message."
            )
        else:
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

        # ── Dispatch with per-chat serialisation ─────────────────────────
        asyncio.create_task(_dispatch(session_id, instruction, dispatch_chat_id))

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


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _extract_message_text(update: dict) -> str | None:
    """Extract the text or caption from a Telegram Update object.

    Checks message, edited_message, and channel_post fields.
    Returns None if no text content is found.
    """
    for key in ("message", "edited_message", "channel_post"):
        msg = update.get(key)
        if msg and isinstance(msg, dict):
            return msg.get("text") or msg.get("caption")
    return None
