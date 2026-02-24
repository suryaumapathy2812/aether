"""Factory for constructing voice realtime model backends."""

from __future__ import annotations

import aether.core.config as config_module

from aether.voice.realtime import RealtimeModel, RealtimeModelConfig


def create_realtime_model(model_config: RealtimeModelConfig) -> RealtimeModel:
    """Create a realtime model based on configured voice backend."""

    backend = config_module.config.voice_backend.backend.lower()

    if backend == "gemini":
        from aether.voice.backends.gemini.model import GeminiRealtimeModel

        return GeminiRealtimeModel(model_config)

    raise ValueError(f"Unsupported realtime voice backend: {backend}")
