"""
Plugin discovery and loading.

Scans plugin directories for plugin.yaml manifests.
Dynamically imports tool classes and returns them for registration.
Follows the same progressive-loading pattern as SkillLoader.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.tools.base import AetherTool

log = logging.getLogger("aether.plugins")

# Simple YAML parser for plugin.yaml (avoids pyyaml dependency)
# Handles: scalars, lists (- item), nested dicts (key:), inline lists [a, b]
_KEY_RE = re.compile(r"^(\s*)(\w[\w_]*):\s*(.*)")
_LIST_ITEM_RE = re.compile(r"^(\s*)-\s+(.*)")


def _parse_simple_yaml(text: str) -> dict:
    """Parse a subset of YAML sufficient for plugin manifests.

    Handles scalars, nested dicts, lists (``- item``), and lists of
    key-value items (``- class: Foo``).  Enough for ``plugin.yaml``.
    """
    lines = text.splitlines()
    result: dict = {}
    stack: list[tuple[int, dict | list]] = [(0, result)]

    def _next_content_line(idx: int) -> str | None:
        """Return the next non-empty, non-comment line after *idx*."""
        for j in range(idx + 1, len(lines)):
            s = lines[j].rstrip()
            if s and not s.lstrip().startswith("#"):
                return s
        return None

    for i, raw_line in enumerate(lines):
        stripped = raw_line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue

        # List item  (e.g. "    - class: Foo" or "    - https://...")
        m = _LIST_ITEM_RE.match(stripped)
        if m:
            indent = len(m.group(1))
            value = m.group(2).strip()

            # Walk up the stack to find the owning list
            while (
                len(stack) > 1
                and stack[-1][0] >= indent
                and not isinstance(stack[-1][1], list)
            ):
                stack.pop()

            parent = stack[-1][1]
            if isinstance(parent, list):
                km = _KEY_RE.match("  " * (indent // 2) + value)
                if km:
                    parent.append({km.group(2): _parse_value(km.group(3))})
                else:
                    parent.append(_parse_value(value))
            continue

        # Key-value  (e.g. "name: gmail" or "tools:")
        m = _KEY_RE.match(stripped)
        if m:
            indent = len(m.group(1))
            key = m.group(2)
            raw_value = m.group(3).strip()

            # Pop stack to correct nesting level
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()

            parent = stack[-1][1]
            if not isinstance(parent, dict):
                continue

            if raw_value == "" or raw_value == "|":
                # Peek ahead: if the next content line is a list item,
                # create a list; otherwise create a nested dict.
                nxt = _next_content_line(i)
                if nxt and _LIST_ITEM_RE.match(nxt):
                    child: dict | list = []
                else:
                    child = {}
                parent[key] = child
                stack.append((indent + 2, child))
            elif raw_value.startswith("[") and raw_value.endswith("]"):
                items = [
                    _parse_value(v.strip())
                    for v in raw_value[1:-1].split(",")
                    if v.strip()
                ]
                parent[key] = items
            else:
                parent[key] = _parse_value(raw_value)

    return result


def _parse_value(v: str):
    """Parse a scalar YAML value."""
    if not v:
        return ""
    # Strip quotes
    if (v.startswith('"') and v.endswith('"')) or (
        v.startswith("'") and v.endswith("'")
    ):
        return v[1:-1]
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


@dataclass
class PluginManifest:
    """Parsed plugin.yaml."""

    name: str
    display_name: str = ""
    description: str = ""
    version: str = "0.1.0"
    plugin_type: str = "sensor"  # sensor | telephony | notification

    auth: dict = field(default_factory=dict)
    webhook: dict = field(default_factory=dict)
    tools: list[dict] = field(default_factory=list)

    # Runtime
    location: str = ""  # Path to plugin directory


@dataclass
class LoadedPlugin:
    """A discovered and loaded plugin."""

    manifest: PluginManifest
    tools: list[AetherTool] = field(default_factory=list)
    skill_content: str | None = None
    # For telephony plugins: FastAPI router factory
    router_factory: Any = None


class PluginLoader:
    """
    Discover plugins from filesystem, load manifests, import tools.

    Usage:
        loader = PluginLoader("/app/plugins")
        plugins = loader.discover()
        for plugin in plugins:
            for tool in plugin.tools:
                tool_registry.register(tool)
    """

    def __init__(self, plugins_dir: str | Path):
        self.plugins_dir = Path(plugins_dir)
        self._plugins: dict[str, LoadedPlugin] = {}

    def discover(self) -> list[LoadedPlugin]:
        """Scan for plugin.yaml files and load plugins."""
        if not self.plugins_dir.exists():
            log.debug(f"Plugins dir not found: {self.plugins_dir}")
            return []

        for manifest_path in self.plugins_dir.glob("*/plugin.yaml"):
            plugin_dir = manifest_path.parent
            try:
                plugin = self._load_plugin(plugin_dir, manifest_path)
                self._plugins[plugin.manifest.name] = plugin
                log.info(
                    f"Plugin: {plugin.manifest.name} "
                    f"(tools={[t.name for t in plugin.tools]})"
                )
            except Exception as e:
                log.warning(f"Failed to load plugin from {plugin_dir}: {e}")

        return list(self._plugins.values())

    def get(self, name: str) -> LoadedPlugin | None:
        return self._plugins.get(name)

    def all(self) -> list[LoadedPlugin]:
        return list(self._plugins.values())

    def all_tools(self) -> list[AetherTool]:
        """Return all tools from all discovered plugins."""
        tools = []
        for plugin in self._plugins.values():
            tools.extend(plugin.tools)
        return tools

    def _load_plugin(self, plugin_dir: Path, manifest_path: Path) -> LoadedPlugin:
        """Load a single plugin from its directory."""
        # Parse manifest
        raw = manifest_path.read_text(encoding="utf-8")
        data = _parse_simple_yaml(raw)

        manifest = PluginManifest(
            name=data.get("name", plugin_dir.name),
            display_name=data.get("display_name", data.get("name", plugin_dir.name)),
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            plugin_type=data.get("plugin_type", "sensor"),
            auth=data.get("auth", {}),
            webhook=data.get("webhook", {}),
            tools=data.get("tools", []),
            location=str(plugin_dir),
        )

        # Load tools from tools.py
        tools = self._load_tools(plugin_dir, manifest)

        # Load skill content from SKILL.md (optional)
        skill_path = plugin_dir / "SKILL.md"
        skill_content = (
            skill_path.read_text(encoding="utf-8") if skill_path.exists() else None
        )

        # Load router factory for telephony plugins
        router_factory = None
        if manifest.plugin_type == "telephony":
            router_factory = self._load_router_factory(plugin_dir, manifest)

        return LoadedPlugin(
            manifest=manifest,
            tools=tools,
            skill_content=skill_content,
            router_factory=router_factory,
        )

    def _load_tools(
        self, plugin_dir: Path, manifest: PluginManifest
    ) -> list[AetherTool]:
        """Dynamically import tool classes from the plugin's tools.py."""
        tools_path = plugin_dir / "tools.py"
        if not tools_path.exists():
            return []

        # Import the module dynamically
        module_name = f"aether_plugin_{manifest.name}_tools"
        spec = importlib.util.spec_from_file_location(module_name, tools_path)
        if not spec or not spec.loader:
            log.warning(f"Cannot create module spec for {tools_path}")
            return []

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            log.warning(f"Failed to import {tools_path}: {e}")
            return []

        # Find tool classes specified in manifest
        tools = []
        for tool_spec in manifest.tools:
            class_name = (
                tool_spec.get("class", "")
                if isinstance(tool_spec, dict)
                else str(tool_spec)
            )
            cls = getattr(module, class_name, None)
            if cls and isinstance(cls, type) and issubclass(cls, AetherTool):
                try:
                    tools.append(cls())
                except Exception as e:
                    log.warning(f"Failed to instantiate {class_name}: {e}")
            else:
                log.warning(f"Tool class not found: {class_name} in {tools_path}")

        return tools

    def _load_router_factory(
        self, plugin_dir: Path, manifest: PluginManifest
    ) -> Any | None:
        """Load the router factory from routes.py for telephony plugins."""
        routes_path = plugin_dir / "routes.py"
        if not routes_path.exists():
            log.debug(f"No routes.py for plugin {manifest.name}")
            return None

        # Import the module dynamically
        module_name = f"aether_plugin_{manifest.name}_routes"
        spec = importlib.util.spec_from_file_location(module_name, routes_path)
        if not spec or not spec.loader:
            log.warning(f"Cannot create module spec for {routes_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            log.warning(f"Failed to import {routes_path}: {e}")
            return None

        # Get the create_router function
        factory = getattr(module, "create_router", None)
        if factory and callable(factory):
            log.info(f"Loaded router factory for plugin {manifest.name}")
            return factory

        log.warning(f"No create_router function in {routes_path}")
        return None
