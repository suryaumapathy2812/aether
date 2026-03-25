---
name: drive
description: Google Drive — files, folders, search, upload, download via rclone mount and commands
integration: google-workspace
---

# Google Drive

## FUSE Mount (Primary Method)

Google Drive is mounted at `/workspace/gdrive/`. Use your standard file tools on it:

- `list_directory /workspace/gdrive/` — list files and folders
- `read_file /workspace/gdrive/Documents/report.pdf` — read a file
- `write_file /workspace/gdrive/notes.txt "content"` — create/write a file
- `glob /workspace/gdrive/**/*.csv` — find files by pattern
- `grep /workspace/gdrive/ "search term"` — search file contents
- `edit /workspace/gdrive/file.txt` — edit a file

The mount is live — changes on Drive appear automatically, and writes upload in the background.

## rclone Commands (When You Need More Control)

Use via the `execute` tool (no credentials needed — rclone manages auth).

### List Files
```bash
rclone ls gdrive:/                           # all files recursively
rclone lsd gdrive:/                          # directories only
rclone lsf gdrive:/Documents/                # files in a folder (machine-readable)
rclone lsjson gdrive:/Documents/ --files-only  # JSON output
```

### Search
```bash
rclone ls gdrive: --include "*report*"       # name search
rclone lsf gdrive: --include "*.pdf" --recursive  # find all PDFs
```

### Copy Files
```bash
# Upload to Drive
rclone copy /workspace/file.pdf gdrive:/Documents/

# Download from Drive
rclone copy gdrive:/report.pdf /workspace/

# Copy a folder
rclone copy /workspace/project/ gdrive:/Projects/2026/
```

### Create and Delete
```bash
rclone mkdir gdrive:/NewFolder               # create folder
rclone delete gdrive:/old-file.txt           # delete file
rclone purge gdrive:/OldFolder/              # delete folder + contents
```

### Move and Sync
```bash
rclone move /workspace/upload.pdf gdrive:/   # move (delete local after copy)
rclone sync /workspace/backup/ gdrive:/backup/  # make Drive match local
```

### Info
```bash
rclone size gdrive:/                         # total size and file count
rclone about gdrive:                         # quota info
rclone stat gdrive:/file.txt                 # single file details
```

### Advanced
```bash
# Copy with progress
rclone copy /workspace/large.zip gdrive:/ --progress

# Exclude patterns
rclone copy /workspace/ gdrive:/backup/ --exclude "*.tmp" --exclude ".git/**"

# Dry run (see what would happen)
rclone copy /workspace/ gdrive:/backup/ --dry-run

# Limit bandwidth (bytes/sec)
rclone copy /workspace/ gdrive:/ --bwlimit 5M
```

## Raw API (When rclone Can't Handle It)

Use curl for advanced operations that rclone doesn't support — permissions, sharing, comments, revisions:

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files/{FILE_ID}/permissions" \
  -X POST -H "Content-Type: application/json" \
  -d '{"role":"reader","type":"user","email":"user@example.com"}'
```

## Notes
- FUSE mount uses minimal local storage — files stream on demand
- Writes cache locally then upload in background
- If mount is unavailable, all rclone commands still work (they just hit the API directly)
- `--vfs-cache-mode writes` means reads are streamed, writes are cached locally
- Large files (>500MB) — use `rclone copy` instead of reading through the mount
