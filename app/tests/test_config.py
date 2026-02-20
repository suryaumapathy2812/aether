"""Tests for the config system."""

import os
import pytest
from aether.core.config import STTConfig, LLMConfig, TTSConfig, AetherConfig


def test_stt_defaults():
    cfg = STTConfig()
    assert cfg.provider == "deepgram"
    assert cfg.model == "nova-3"
    assert cfg.sample_rate == 16000
    assert cfg.reconnect_attempts == 5


def test_llm_defaults():
    cfg = LLMConfig()
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"
    assert cfg.max_tokens == 500
    assert cfg.temperature == 0.7


def test_tts_defaults():
    cfg = TTSConfig()
    assert cfg.provider == "openai"
    assert cfg.voice == "nova"
    assert cfg.timeout == 15.0


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("AETHER_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("AETHER_LLM_MAX_TOKENS", "200")
    cfg = LLMConfig.from_env()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.max_tokens == 200


def test_llm_openrouter_prefers_openrouter_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-legacy")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-router-key")
    cfg = LLMConfig.from_env()
    assert cfg.api_key == "sk-or-v1-router-key"


def test_llm_openrouter_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-fallback")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.api_key == "sk-or-v1-fallback"


def test_config_frozen():
    cfg = STTConfig()
    with pytest.raises(Exception):
        cfg.model = "something-else"  # type: ignore


def test_aether_config_composition():
    cfg = AetherConfig()
    assert cfg.stt.provider == "deepgram"
    assert cfg.llm.provider == "openai"
    assert cfg.tts.provider == "openai"
