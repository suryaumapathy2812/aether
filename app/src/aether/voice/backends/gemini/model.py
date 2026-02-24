"""Gemini implementation of the realtime model factory."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types as genai_types

from aether.voice.backends.gemini.session import GeminiRealtimeSession
from aether.voice.realtime import RealtimeModel, RealtimeModelConfig, RealtimeSession

logger = logging.getLogger(__name__)


class GeminiRealtimeModel(RealtimeModel):
    """Creates Gemini Live sessions for voice/text realtime interactions."""

    def __init__(self, config: RealtimeModelConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.api_key)

    async def create_session(
        self,
        *,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        resumption_token: str | None = None,
    ) -> RealtimeSession:
        effective_instructions = instructions or self._config.instructions
        live_config = self._build_live_config(
            instructions=effective_instructions,
            tools=tools or [],
            resumption_token=resumption_token,
        )

        session = GeminiRealtimeSession(
            client=self._client,
            model=self._config.model,
            live_config=live_config,
            input_sample_rate=self._config.input_sample_rate,
            output_sample_rate=self._config.output_sample_rate,
            requested_resumption=bool(resumption_token),
        )
        await session.connect()
        return session

    async def probe_live_model_support(self) -> tuple[bool, str, list[str]]:
        """Validate that configured model supports Gemini Live bidiGenerateContent."""

        def _fetch_supported() -> list[str]:
            supported: list[str] = []
            for item in self._client.models.list():
                actions = set(getattr(item, "supported_actions", []) or [])
                if "bidiGenerateContent" in actions:
                    name = str(getattr(item, "name", ""))
                    if name:
                        supported.append(name)
            return supported

        try:
            supported = await asyncio.to_thread(_fetch_supported)
        except Exception as exc:
            return False, f"Failed listing Gemini models: {exc}", []

        configured = self._config.model.strip()
        configured_full = (
            configured if configured.startswith("models/") else f"models/{configured}"
        )
        supported_set = set(supported)
        if configured in supported_set or configured_full in supported_set:
            return True, "ok", supported

        preview = ", ".join(sorted(supported)[:5]) or "none"
        return (
            False,
            f"Configured realtime model '{configured}' is not bidi-capable for this key. "
            f"Supported examples: {preview}",
            supported,
        )

    async def close(self) -> None:
        return None

    @property
    def config(self) -> RealtimeModelConfig:
        return self._config

    def _build_live_config(
        self,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
        resumption_token: str | None,
    ) -> genai_types.LiveConnectConfig:
        kwargs: dict[str, Any] = {
            "response_modalities": self._config.response_modalities,
            "input_audio_transcription": genai_types.AudioTranscriptionConfig(),
            "output_audio_transcription": genai_types.AudioTranscriptionConfig(),
        }

        if instructions:
            kwargs["system_instruction"] = genai_types.Content(
                parts=[genai_types.Part(text=instructions)]
            )

        kwargs["speech_config"] = genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=self._config.voice
                )
            )
        )

        gemini_tools = self._build_tools(tools)
        if gemini_tools:
            kwargs["tools"] = gemini_tools

        if self._config.enable_session_resumption:
            if resumption_token:
                kwargs["session_resumption"] = genai_types.SessionResumptionConfig(
                    handle=resumption_token
                )
            else:
                # Always request resumption so we get a handle for future reconnects
                kwargs["session_resumption"] = genai_types.SessionResumptionConfig()

        return genai_types.LiveConnectConfig(**kwargs)

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[genai_types.Tool]:
        if not tools:
            return []

        declarations: list[genai_types.FunctionDeclaration] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=name,
                    description=str(tool.get("description", "")),
                    parameters=tool.get("parameters", {"type": "object"}),
                )
            )

        if not declarations:
            return []
        return [genai_types.Tool(function_declarations=declarations)]
