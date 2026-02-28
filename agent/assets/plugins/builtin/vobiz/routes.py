"""Vobiz Plugin Routes.

FastAPI routes for Vobiz telephony webhooks.
These routes are registered when the plugin is loaded and configured.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:
    from aether.voice.telephony import TelephonyTransport

logger = logging.getLogger(__name__)

# Module-level transport instance (set by plugin loader)
_telephony_transport: TelephonyTransport | None = None
_plugin_config: dict = {}


def set_transport(transport: "TelephonyTransport | None") -> None:
    """Set the telephony transport instance."""
    global _telephony_transport
    _telephony_transport = transport


def set_config(config: dict) -> None:
    """Set the plugin configuration."""
    global _plugin_config
    _plugin_config = config


def get_config() -> dict:
    """Get the plugin configuration."""
    return _plugin_config


def create_router() -> APIRouter:
    """Create the FastAPI router for Vobiz plugin endpoints."""
    router = APIRouter(prefix="/plugins/vobiz", tags=["vobiz"])

    def _build_ws_url(request: Request) -> str:
        """Build the WebSocket URL for VoBiz <Stream> XML.

        If public_base_url and user_id are available in the plugin config,
        route through the orchestrator proxy (/api/plugins/vobiz/ws?uid=...).
        Otherwise fall back to a direct connection (dev mode).
        """
        config = get_config()
        public_base = config.get("public_base_url", "").rstrip("/")
        uid = config.get("user_id", "")

        if public_base and uid:
            # Route through orchestrator proxy
            ws_scheme = "wss" if public_base.startswith("https") else "ws"
            host = public_base.split("://", 1)[-1]
            return f"{ws_scheme}://{host}/api/plugins/vobiz/ws?uid={uid}"

        # Fallback: direct connection (dev / local mode)
        host = request.headers.get("host", "localhost:8000")
        scheme = "wss" if request.url.scheme == "https" else "ws"
        return f"{scheme}://{host}/plugins/vobiz/ws"

    @router.websocket("/ws")
    async def vobiz_ws(ws: WebSocket) -> None:
        """WebSocket endpoint for Vobiz media streams."""
        if not _telephony_transport:
            await ws.close(code=1008, reason="Vobiz plugin not configured")
            return

        await _telephony_transport.handle_call(ws)

    @router.post("/webhook")
    async def vobiz_webhook(request: Request) -> Response:
        """Webhook for inbound calls from Vobiz.

        Vobiz calls this URL when someone calls your Vobiz number.
        Returns XML that instructs Vobiz to connect the call to our WebSocket.
        """
        if not _telephony_transport:
            return Response(
                content='<?xml version="1.0"?><Response><Hangup/></Response>',
                media_type="application/xml",
            )

        ws_url = _build_ws_url(request)

        # Vobiz XML Stream format
        stream_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-l16;rate=16000">{ws_url}</Stream>
</Response>"""
        return Response(content=stream_xml, media_type="application/xml")

    @router.post("/answer")
    async def vobiz_answer(request: Request) -> Response:
        """Webhook called by Vobiz when an outbound call is answered.

        This is the answer_url for outbound calls initiated by make_phone_call tool.
        """
        if not _telephony_transport:
            return Response(
                content='<?xml version="1.0"?><Response><Hangup/></Response>',
                media_type="application/xml",
            )

        # Parse query params for optional greeting
        greeting = request.query_params.get("greeting", "")

        ws_url = _build_ws_url(request)

        # Build XML response with optional greeting
        if greeting:
            stream_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak>{greeting}</Speak>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-l16;rate=16000">{ws_url}</Stream>
</Response>"""
        else:
            stream_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-l16;rate=16000">{ws_url}</Stream>
</Response>"""
        return Response(content=stream_xml, media_type="application/xml")

    @router.get("/status")
    async def vobiz_status() -> JSONResponse:
        """Get the status of the Vobiz plugin."""
        config = get_config()
        return JSONResponse(
            {
                "configured": bool(config.get("auth_id") and config.get("auth_token")),
                "from_number": config.get("from_number", ""),
                "user_phone_number": config.get("user_phone_number", ""),
                "transport_active": _telephony_transport is not None,
            }
        )

    return router
