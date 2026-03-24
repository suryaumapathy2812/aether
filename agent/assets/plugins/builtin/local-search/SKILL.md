# Local Search API

## Authentication
- **Env var**: None required
- **Credentials**: No credentials needed
- **Type**: Local filesystem search

Local search operates on the local filesystem to find files and content within the workspace.

## Search Files by Name

Use `find` or `grep` via shell commands:

```bash
# Find files by name pattern
find /path/to/workspace -name "*.ts" -type f

# Find files modified in the last 7 days
find /path/to/workspace -name "*.py" -mtime -7 -type f

# Case-insensitive name search
find /path/to/workspace -iname "*readme*" -type f
```

## Search File Contents

```bash
# Search for text in files (recursive)
grep -r "function handleRequest" /path/to/workspace --include="*.ts"

# Search with line numbers
grep -rn "TODO" /path/to/workspace --include="*.{js,ts,tsx}"

# Search ignoring case
grep -ri "error" /path/to/workspace --include="*.log"

# Search with context (3 lines before/after)
grep -r -C 3 "export default" /path/to/workspace --include="*.tsx"
```

## Combined Search (Name + Content)

```bash
# Find files matching pattern, then search content
find /path/to/workspace -name "*.ts" -type f -exec grep -l "API_KEY" {} \;

# Search only in specific directories
grep -r "import React" /path/to/workspace/src --include="*.tsx" --include="*.jsx"
```

## List Directory Contents

```bash
# List files recursively with tree-like output
find /path/to/workspace -maxdepth 3 -type f | head -50

# List only directories
find /path/to/workspace -maxdepth 2 -type d

# List files with details (size, modified)
ls -laR /path/to/workspace/src --include="*.ts" 2>/dev/null || find /path/to/workspace/src -name "*.ts" -exec ls -la {} \;
```

## File Stats and Info

```bash
# File size and modification time
stat /path/to/file

# Count lines in a file
wc -l /path/to/file

# Count files matching pattern
find /path/to/workspace -name "*.ts" -type f | wc -l
```

## Error Handling
- **Permission denied**: Some directories may be restricted — use `2>/dev/null` to suppress errors
- **No results**: Broaden the search pattern or check the path is correct
- **Large workspaces**: Use `--max-depth` or `-maxdepth` to limit search scope
- **Binary files**: Add `--binary-files=without-match` to grep to skip binary content
