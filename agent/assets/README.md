# Agent Assets

This directory is the public home for movable runtime assets used by the Go agent.

Current layout:

- `state.db` - SQLite runtime state (skills + plugins tables)
- `skills/builtin/` - checked-in built-in skills (`SKILL.md` trees)
- `skills/user/` - locally created user skills
- `skills/external/` - externally installed skills
- `plugins/builtin/` - checked-in built-in plugins (`plugin.yaml` trees)
- `plugins/user/` - local user plugin manifests
- `plugins/external/` - externally installed plugins

The `cmd/skills` CLI defaults to this assets root and loads skills from these paths.
The `cmd/plugins` CLI defaults to this assets root and loads plugins from these paths.
