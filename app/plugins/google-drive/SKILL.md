# Google Drive Plugin

You have access to Google Drive tools for searching, browsing, reading, and creating files in the user's Google Drive.

---

## Tools Available

### `search_drive`
Search for files by name or content query.

**Parameters:**
- `query` (required) — Search term (matches file names and content)
- `limit` (optional, default `10`) — Max results to return

**Returns:** List of files with name, MIME type, file ID, modified date, and web view link.

**Use when:** The user asks to find a file, or you need a file ID before reading or getting details.

---

### `list_drive_files`
List files in a specific folder.

**Parameters:**
- `folder_id` (optional, default `root`) — Folder ID to list. Use `root` for the top-level Drive.
- `limit` (optional, default `20`) — Max files to return

**Returns:** Files in the folder with name, type, size, and modified date.

**Use when:** The user wants to browse a folder's contents, or you need to navigate the Drive structure.

---

### `get_file_info`
Get detailed metadata for a specific file.

**Parameters:**
- `file_id` (required) — The file ID from search or list results

**Returns:** Full metadata: size, created/modified dates, owner, sharing settings, MIME type, and web link.

**Use when:** The user wants details about a specific file (who owns it, when it was last modified, sharing status).

---

### `read_file_content`
Read the text content of a file.

**Parameters:**
- `file_id` (required) — The file ID to read

**Returns:** Extracted text content of the file.

**Supported formats:**
- Google Docs → exported as plain text
- Google Sheets → exported as CSV
- Google Slides → exported as plain text (slide content)
- Plain text files (`.txt`, `.md`, `.py`, `.json`, etc.) → downloaded directly

**Not supported:** Binary files (images, PDFs, ZIPs, executables) — inform the user and provide the file link instead.

**Use when:** The user wants to read, summarize, or analyze the content of a document.

---

### `list_shared_drives`
List shared/team drives the user has access to.

**Parameters:** None

**Returns:** List of shared drives with name and ID.

**Use when:** The user asks about team drives, shared workspaces, or "company drive".

---

### `create_document`
Create a new Google Doc.

**Parameters:**
- `title` (required) — Document title
- `content` (optional) — Initial text content to populate the document

**Returns:** Document ID, title, and a direct link to open it in Google Docs.

**Use when:** The user asks to create a document, write something up, or draft content in Google Docs.

---

### `create_spreadsheet`
Create a new Google Sheet.

**Parameters:**
- `title` (required) — Spreadsheet title
- `sheet_names` (optional) — List of sheet/tab names (default: one sheet named "Sheet1")

**Returns:** Spreadsheet ID, title, and a direct link to open it in Google Sheets.

**Use when:** The user asks to create a spreadsheet, table, or tracker.

---

### `create_presentation`
Create a new Google Slides presentation.

**Parameters:**
- `title` (required) — Presentation title

**Returns:** Presentation ID, title, and a direct link to open it in Google Slides.

**Use when:** The user asks to create a presentation or slide deck.

---

## Decision Rules

**Finding files:**
- Use `search_drive` for finding files by name or content — it's faster than browsing
- Use `list_drive_files` when the user wants to browse a specific folder
- Always get the file ID from search/list before calling `read_file_content` or `get_file_info`

**Reading files:**
- For Google Docs/Sheets/Slides, `read_file_content` exports the content as text — it won't preserve formatting
- For binary files (images, PDFs), tell the user you can't read the content and share the web link instead
- Summarize long documents rather than dumping the full text — ask the user what they want to know

**Creating files:**
- Always share the link to the created file so the user can open it immediately
- For documents with content, use `create_document` with the `content` parameter to pre-populate it
- Don't create files without confirming the title and purpose with the user

**Presenting results:**
- Keep file listings concise: name, type, and last modified date
- For search results, show the most relevant files first and offer to get more details
- Always include the web link when referencing a specific file

**Error handling:**
- If a file isn't found, suggest trying different search terms
- If `read_file_content` fails on a binary file, explain the limitation and provide the file link
- 403 errors usually mean the Drive API isn't enabled or the user needs to reconnect the plugin

---

## Example Workflows

**"Find the Q4 report"**
```
1. search_drive query="Q4 report"
2. "I found 3 files matching 'Q4 report': [list with links]"
3. Ask which one they want, or offer to read the most recent
```

**"Read my project proposal doc"**
```
1. search_drive query="project proposal" → get file_id
2. read_file_content file_id="..."
3. Summarize the content or answer the user's specific question about it
```

**"Create a new doc called 'Meeting Notes - Feb 2026'"**
```
1. create_document title="Meeting Notes - Feb 2026"
2. "Created! Here's the link: [link]"
```

**"What's in my Drive root folder?"**
```
1. list_drive_files folder_id="root"
2. List files with names and types, grouped by type if helpful
```
