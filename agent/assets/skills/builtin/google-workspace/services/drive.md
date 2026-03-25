---
name: drive
description: Google Drive — files, folders, search, upload, download via rclone mount
integration: google-workspace
---

# Google Drive

## CRITICAL: Use the FUSE Mount — NOT curl or API calls

Google Drive is mounted at `/workspace/gdrive/`. This is the **only way** to interact with Drive files. The mount is live and works like a local filesystem.

**DO NOT use curl or the Drive API for file operations.** Use the builtin tools below.

## File Operations (Use These)

| Task | Tool | Command |
|------|------|---------|
| List files | `list_directory` | `list_directory /workspace/gdrive/` |
| Read a file | `read_file` | `read_file /workspace/gdrive/report.pdf` |
| Write a file | `write_file` | `write_file /workspace/gdrive/notes.txt "content"` |
| Find files by name | `glob` | `glob /workspace/gdrive/**/*student*` |
| Search file contents | `grep` | `grep /workspace/gdrive/ "search term"` |
| Edit a file | `edit` | `edit /workspace/gdrive/file.txt` |

## Common User Requests → How to Handle

**"Find a file on my Drive"**
→ Use `glob /workspace/gdrive/**/*keyword*`

**"Search my Drive for X"**
→ Use `grep /workspace/gdrive/ "X"` for content search
→ Use `glob /workspace/gdrive/**/*X*` for name search

**"List files in my Drive"**
→ Use `list_directory /workspace/gdrive/`

**"Read/download a file from Drive"**
→ Use `read_file /workspace/gdrive/filename`

**"Save this to Drive"**
→ Use `write_file /workspace/gdrive/filename "content"`
→ Or use `execute` with: `rclone copy /local/path gdrive:/destination/`

**"Check Drive quota/usage"**
→ Use `execute` with: `rclone about gdrive:`

## rclone Commands (Only When Needed)

Use via the `execute` tool only for operations the builtin tools can't do:

```bash
# Upload a local file to Drive
rclone copy /workspace/file.pdf gdrive:/Documents/

# Download from Drive
rclone copy gdrive:/report.pdf /workspace/

# Create folder
rclone mkdir gdrive:/NewFolder

# Delete file
rclone delete gdrive:/old-file.txt

# Quota info
rclone about gdrive:

# Size
rclone size gdrive:/
```

## Raw API (Last Resort Only)

Only use curl for operations that neither the mount nor rclone can handle:
- Sharing files with specific permissions
- Managing comments or revisions
- Creating/managing shared drives

```bash
# Example: share a file
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files/{FILE_ID}/permissions" \
  -X POST -H "Content-Type: application/json" \
  -d '{"role":"reader","type":"user","email":"user@example.com"}'
```

## Notes
- Mount is live — changes sync automatically
- Writes upload in background
- Local storage is minimal — files stream on demand
- If mount is empty or missing, check: `mountpoint /workspace/gdrive`
