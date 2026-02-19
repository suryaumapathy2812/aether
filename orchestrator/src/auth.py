"""
Session validation for the orchestrator.

Auth is handled by better-auth in the dashboard (Next.js).
The orchestrator validates sessions by reading the shared `session` table in Postgres.

Two auth methods are supported:
  1. Session token (dashboard) — `Authorization: Bearer <session-token>`
     Validated against the `session` table (created by better-auth).
  2. Device token (iOS) — `Authorization: Bearer <device-token>`
     Validated against the `devices` table (created by the pairing flow).

The orchestrator no longer issues tokens or manages passwords.
"""

from __future__ import annotations

import logging
from fastapi import Request, WebSocket, HTTPException

from .db import get_pool

log = logging.getLogger("orchestrator.auth")


async def get_user_id(request: Request) -> str:
    """
    Extract and validate the user ID from the request.

    Checks Authorization header for a Bearer token, then validates it
    against the session table (better-auth) or devices table (iOS pairing).

    Raises HTTPException(401) if no valid session is found.
    """
    tokens = _extract_tokens_from_request(request)
    if not tokens:
        raise HTTPException(401, "Missing authorization")

    for token in tokens:
        user_id = await _validate_token(token)
        if user_id:
            return user_id

    raise HTTPException(401, "Invalid or expired session")


async def get_user_id_from_ws(ws: WebSocket) -> str | None:
    """
    Extract and validate the user ID from a WebSocket connection.

    Checks query param `token` (for backward compat) and Authorization header.
    Returns None if no valid session is found (caller should close the WS).
    """
    # Try query param first (WS connections can't easily send cookies)
    token = ws.query_params.get("token")

    # Fall back to Authorization header
    if not token:
        auth_header = ws.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        return None

    return await _validate_token(token)


def _extract_tokens_from_request(request: Request) -> list[str]:
    """Extract candidate tokens from Authorization header, cookies, or query.

    Priority:
      1. Authorization: Bearer <token> header
      2. __Secure-better-auth session cookie
      3. better-auth session cookie
      4. ?token= query param (OAuth redirect flows)
    """
    tokens: list[str] = []

    # 1. Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        tokens.append(auth_header[7:])

    # 2/3. better-auth session cookies (sent by browser on same-origin requests)
    # The cookie value is URL-encoded: token.signature
    for cookie_name in (
        "__Secure-better-auth.session_token",
        "better-auth.session_token",
    ):
        cookie_token = request.cookies.get(cookie_name)
        if not cookie_token:
            continue
        token_part = cookie_token.split(".")[0] if "." in cookie_token else cookie_token
        tokens.append(token_part)

    # 4. Fallback: query param (used by OAuth redirect flows)
    query_token = request.query_params.get("token")
    if query_token:
        tokens.append(query_token)

    # Deduplicate while preserving priority order.
    return list(dict.fromkeys(tokens))


async def _validate_token(token: str) -> str | None:
    """
    Validate a token against the session table or devices table.

    Returns the user_id if valid, None otherwise.
    """
    pool = await get_pool()

    # 1. Check better-auth session table
    row = await pool.fetchrow(
        "SELECT user_id FROM session WHERE token = $1 AND expires_at > now()",
        token,
    )
    if row:
        return row["user_id"]

    # 2. Check device tokens (iOS pairing flow)
    row = await pool.fetchrow(
        "SELECT user_id FROM devices WHERE token = $1",
        token,
    )
    if row:
        # Update last_seen for the device
        await pool.execute(
            "UPDATE devices SET last_seen = now() WHERE token = $1", token
        )
        return row["user_id"]

    log.debug(f"Token validation failed (not found in session or devices)")
    return None
