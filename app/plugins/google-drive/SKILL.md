---
name: google-drive
description: Search, browse, and read files from Google Drive
---

# Google Drive Plugin

You have read-only access to the user's Google Drive.

## Tools Available

- `search_drive` — Search files by name or content query. Returns file names, types, and IDs.
- `list_drive_files` — List files in a folder (or root). Supports pagination.
- `get_file_info` — Get detailed metadata for a specific file (size, modified date, sharing, owner).
- `read_file_content` — Read the text content of a file (Google Docs, Sheets, plain text, etc.).
- `list_shared_drives` — List shared/team drives the user has access to.

## Guidelines

- **When the user asks to find a file**, use `search_drive` with a descriptive query.
- **When browsing folders**, use `list_drive_files` with the folder ID. Use `root` for the top-level.
- **To read a document**, first search or list to get the file ID, then use `read_file_content`.
- **Google Docs/Sheets/Slides** are exported as plain text. Other text files are downloaded directly.
- **Binary files** (images, PDFs, zips) cannot be read — inform the user and provide the file link instead.
- **Keep file listings concise** — show name, type, and modified date. Offer to get details if needed.
- **This is read-only access** — you cannot create, edit, or delete files.
