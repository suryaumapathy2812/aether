"""
Tests for modality adapters and transport pairing.

Tests:
- ModalityAdapter ABC contract
- TextAdapter text pipeline
- VoiceAdapter event handling
- SystemAdapter no-op behavior
- TransportPairing rules
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aether.modality.base import ModalityAdapter
from aether.modality.system_adapter import SystemAdapter
from aether.modality.text_adapter import TextAdapter
from aether.transport.pairing import TransportPairing


# ─── ModalityAdapter ABC ─────────────────────────────────────────


class TestModalityAdapterABC:
    """Test that ModalityAdapter enforces the interface."""

    def test_cannot_instantiate_abc(self):
        """ModalityAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ModalityAdapter()  # type: ignore[abstract]

    def test_concrete_adapter_must_implement_methods(self):
        """A concrete adapter must implement all abstract methods."""

        class IncompleteAdapter(ModalityAdapter):
            @property
            def modality(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]


# ─── SystemAdapter ───────────────────────────────────────────────


class TestSystemAdapter:
    """Test SystemAdapter no-op behavior."""

    def test_modality_is_system(self):
        adapter = SystemAdapter()
        assert adapter.modality == "system"

    @pytest.mark.asyncio
    async def test_handle_input_is_noop(self):
        """SystemAdapter.handle_input yields nothing."""
        adapter = SystemAdapter()
        msg = MagicMock()
        results = []
        async for resp in adapter.handle_input(msg, {}):
            results.append(resp)
        assert results == []

    @pytest.mark.asyncio
    async def test_handle_output_is_noop(self):
        """SystemAdapter.handle_output yields nothing."""
        adapter = SystemAdapter()
        results = []
        async for resp in adapter.handle_output("text_chunk", {"text": "hi"}, {}):
            results.append(resp)
        assert results == []


# ─── TextAdapter ─────────────────────────────────────────────────


class TestTextAdapter:
    """Test TextAdapter text pipeline."""

    def _make_adapter(self):
        """Create a TextAdapter with mocked dependencies."""
        llm_core = AsyncMock()
        context_builder = AsyncMock()
        memory_store = AsyncMock()
        memory_store.search = AsyncMock(return_value=[])

        return TextAdapter(
            llm_core=llm_core,
            context_builder=context_builder,
            memory_store=memory_store,
        )

    def test_modality_is_text(self):
        adapter = self._make_adapter()
        assert adapter.modality == "text"

    @pytest.mark.asyncio
    async def test_handle_output_text_chunk(self):
        """handle_output converts text_chunk to CoreMsg."""
        adapter = self._make_adapter()
        session_state = {"user_id": "u1", "session_id": "s1"}

        results = []
        async for resp in adapter.handle_output(
            "text_chunk", {"text": "Hello"}, session_state
        ):
            results.append(resp)

        assert len(results) == 1
        assert results[0].content.text == "Hello"

    @pytest.mark.asyncio
    async def test_handle_output_status(self):
        """handle_output converts status to CoreMsg."""
        adapter = self._make_adapter()
        session_state = {"user_id": "u1", "session_id": "s1"}

        results = []
        async for resp in adapter.handle_output(
            "status", {"text": "thinking..."}, session_state
        ):
            results.append(resp)

        assert len(results) == 1
        assert results[0].content.text == "thinking..."

    @pytest.mark.asyncio
    async def test_handle_output_stream_end(self):
        """handle_output converts stream_end to CoreMsg event."""
        adapter = self._make_adapter()
        session_state = {"user_id": "u1", "session_id": "s1"}

        results = []
        async for resp in adapter.handle_output("stream_end", {}, session_state):
            results.append(resp)

        assert len(results) == 1
        assert results[0].content.event_type == "stream_end"


# ─── TransportPairing ───────────────────────────────────────────


class TestTransportPairing:
    """Test transport → adapter pairing rules."""

    def _make_pairing(self):
        """Create a TransportPairing with mock adapters."""
        text = MagicMock(spec=ModalityAdapter)
        text.modality = "text"
        voice = MagicMock(spec=ModalityAdapter)
        voice.modality = "voice"
        system = MagicMock(spec=ModalityAdapter)
        system.modality = "system"
        return (
            TransportPairing(
                text_adapter=text,
                voice_adapter=voice,
                system_adapter=system,
            ),
            text,
            voice,
            system,
        )

    def test_http_maps_to_text(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("http") is text

    def test_http_text_maps_to_text(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("http", "text") is text

    def test_websocket_text_maps_to_text(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("websocket", "text") is text

    def test_websocket_voice_maps_to_voice(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("websocket", "voice") is voice

    def test_webrtc_maps_to_voice(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("webrtc") is voice

    def test_webrtc_voice_maps_to_voice(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("webrtc", "voice") is voice

    def test_internal_maps_to_system(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("internal") is system

    def test_internal_system_maps_to_system(self):
        pairing, text, voice, system = self._make_pairing()
        assert pairing.get_adapter("internal", "system") is system

    def test_unknown_transport_raises(self):
        pairing, _, _, _ = self._make_pairing()
        with pytest.raises(ValueError, match="No pairing rule"):
            pairing.get_adapter("bluetooth")

    def test_get_adapter_type_returns_string(self):
        pairing, _, _, _ = self._make_pairing()
        assert pairing.get_adapter_type("http") == "text"
        assert pairing.get_adapter_type("webrtc") == "voice"
        assert pairing.get_adapter_type("internal") == "system"

    def test_get_adapter_type_unknown_raises(self):
        pairing, _, _, _ = self._make_pairing()
        with pytest.raises(ValueError):
            pairing.get_adapter_type("bluetooth")
