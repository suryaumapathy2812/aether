# Agent Assets

This directory is the public home for movable runtime assets used by the Go agent.

Current layout:

- `state.db` - SQLite runtime state (skills + integrations tables)
- `skills/builtin/` - checked-in built-in skills (`SKILL.md` trees)
- `skills/user/` - locally created user skills
- `skills/external/` - externally installed skills
- `integrations/builtin/` - checked-in built-in integrations (`integration.yaml` trees)
- `integrations/user/` - local user integration manifests
- `integrations/external/` - externally installed integrations

The `cmd/skills` CLI defaults to this assets root and loads skills from these paths.
The `cmd/integrations` CLI defaults to this assets root and loads integrations from these paths.
