---
name: file-management
description: Create, organize, and manage files and directories. Use when the user asks to create project structures, organize files, or manage their workspace.
---

## File Management Skill

When the user asks you to manage files or create project structures:

1. **Understand the request** — Ask clarifying questions if the structure isn't clear.
2. **Plan first** — Use `list_directory` to see what exists before making changes.
3. **Create directories first** — Use `run_command` with `mkdir -p` for nested directories.
4. **Write files** — Use `write_file` to create each file with appropriate content.
5. **Verify** — Use `list_directory` to confirm the structure was created correctly.
6. **Summarize** — Tell the user what you created in natural language.

### Common Patterns

For a Python project:
```
project/
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── README.md
├── pyproject.toml
└── .gitignore
```

For a Node.js project:
```
project/
├── src/
│   └── index.ts
├── tests/
├── package.json
├── tsconfig.json
└── .gitignore
```

Always create `.gitignore` with sensible defaults for the project type.
