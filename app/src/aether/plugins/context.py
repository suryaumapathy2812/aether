"""
Plugin Context Store — runtime credentials for plugin tools.

Holds decrypted config values (access tokens, API keys, etc.) fetched
from the orchestrator at startup.  The LLM processor queries this store
before dispatching a plugin tool so the tool receives its credentials
via ``self._context``.

Design:
  - One store per agent process (module-level singleton).
  - Populated once after orchestrator registration, refreshed on /config/reload.
  - Tool → plugin mapping built during plugin loading so the dispatcher
    knows which config to inject for a given tool name.
"""

from __future__ import annotations

import logging

log = logging.getLogger("aether.plugins.context")


class PluginContextStore:
    """In-memory store: plugin_name → {key: value} config dict."""

    def __init__(self):
        self._configs: dict[str, dict[str, str]] = {}

    # ── Write ──

    def set(self, plugin_name: str, config: dict[str, str]) -> None:
        """Store (or replace) the config for a plugin."""
        self._configs[plugin_name] = config
        log.info(
            "Plugin context: %s (%d keys)",
            plugin_name,
            len(config),
        )

    def clear(self, plugin_name: str) -> None:
        """Remove stored config for a plugin."""
        self._configs.pop(plugin_name, None)

    # ── Read ──

    def get(self, plugin_name: str) -> dict[str, str]:
        """Return the config dict for a plugin (empty dict if not found)."""
        return self._configs.get(plugin_name, {})

    def has(self, plugin_name: str) -> bool:
        return plugin_name in self._configs

    # ── Debug ──

    def loaded_plugins(self) -> list[str]:
        """Names of plugins that have config loaded."""
        return list(self._configs.keys())

    def __repr__(self) -> str:
        return f"<PluginContextStore plugins={self.loaded_plugins()}>"
