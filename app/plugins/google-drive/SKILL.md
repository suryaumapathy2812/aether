---
name: google-drive
description: Search, browse, read, and create files in Google Drive
---

# Google Drive Plugin

You have access to the user's Google Drive for reading and creating files.

## Tools Available

### Reading & Browsing
- `search_drive` — Search files by name or content query. Returns file names, types, and IDs.
- `list_drive_files` — List files in a folder (or root). Supports pagination.
- `get_file_info` — Get detailed metadata for a specific file (size, modified date, sharing, owner).
- `read_file_content` — Read the text content of a file (Google Docs, Sheets, plain text, etc.).
- `list_shared_drives` — List shared/team drives the user has access to.

### Creating Files
- `create_document` — Create a new Google Doc with a title and optional text content.
- `create_spreadsheet` — Create a new Google Sheet with a title and optional sheet/tab names.
- `create_presentation` — Create a new Google Slides presentation with a title.

## Guidelines

- **When the user asks to find a file**, use `search_drive` with a descriptive query.
- **When browsing folders**, use `list_drive_files` with the folder ID. Use `root` for the top-level.
- **To read a document**, first search or list to get the file ID, then use `read_file_content`.
- **Google Docs/Sheets/Slides** are exported as plain text when reading. Other text files are downloaded directly.
- **Binary files** (images, PDFs, zips) cannot be read — inform the user and provide the file link instead.
- **When creating files**, always provide the link to the created file so the user can open it.
- **Keep file listings concise** — show name, type, and modified date. Offer to get details if needed.
