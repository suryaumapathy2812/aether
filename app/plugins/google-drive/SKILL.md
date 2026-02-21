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

### `share_file`
Share a file with specific people by email address.

**Parameters:**
- `file_id` (required) — The file ID to share
- `email` (required) — Email address(es), comma-separated for multiple
- `role` (optional, default `viewer`) — `viewer`, `commenter`, or `editor`
- `send_notification` (optional, default `true`) — Whether to email the recipients

**Returns:** Confirmation with list of emails shared, their role, and the file's web link.

**Use when:** The user wants to give someone access to a file. **Always confirm who you're sharing with and what role before calling this tool.**

---

### `list_permissions`
List all people who have access to a file.

**Parameters:**
- `file_id` (required) — The file ID

**Returns:** Each person's name, email, role (viewer/commenter/editor), and permission type.

**Use when:** The user asks who has access to a file, or before calling `update_permission` or `remove_sharing` to confirm the current state.

---

### `update_permission`
Change someone's access level on a file.

**Parameters:**
- `file_id` (required) — The file ID
- `email` (required) — Email address of the person whose access to change
- `role` (required) — New role: `viewer`, `commenter`, or `editor`

**Use when:** The user wants to upgrade or downgrade someone's access. Call `list_permissions` first to confirm the person currently has access.

---

### `remove_sharing`
Revoke someone's access to a file.

**Parameters:**
- `file_id` (required) — The file ID
- `email` (required) — Email address of the person to remove

**Use when:** The user wants to remove someone's access. Call `list_permissions` first to confirm the person currently has access and show the user before removing.

---

### `make_public`
Make a file accessible to anyone with the link.

**Parameters:**
- `file_id` (required) — The file ID
- `role` (optional, default `viewer`) — `viewer` or `commenter`

**Returns:** Confirmation with the public shareable link.

**Use when:** The user explicitly asks to make a file public or share it with "anyone with the link". **This is a significant action — always confirm with the user before calling this tool.** Inform them that anyone with the link will be able to access the file.

---

### `move_file`
Move a file to a different folder.

**Parameters:**
- `file_id` (required) — The file ID to move
- `folder_id` (required) — The destination folder ID

**Use when:** The user wants to reorganize files. Use `search_drive` or `list_drive_files` to find the destination folder ID if needed.

---

### `rename_file`
Rename a file in Google Drive.

**Parameters:**
- `file_id` (required) — The file ID to rename
- `new_name` (required) — The new name for the file

**Returns:** Confirmation showing old name → new name.

**Use when:** The user wants to rename a file. Confirm the new name before calling.

---

### `copy_file`
Make a copy of a file in Google Drive.

**Parameters:**
- `file_id` (required) — The file ID to copy
- `new_name` (optional) — Name for the copy (defaults to "Copy of [original name]")

**Returns:** New file ID, name, and link.

**Use when:** The user wants to duplicate a file, create a template copy, or make a backup.

---

### `create_folder`
Create a new folder in Google Drive.

**Parameters:**
- `name` (required) — Folder name
- `parent_folder_id` (optional) — Parent folder ID (defaults to Drive root)

**Returns:** Folder ID and link.

**Use when:** The user wants to create a new folder for organization. Confirm the name and location before creating.

---

### `update_document`
Append or replace content in an existing Google Doc.

**Parameters:**
- `document_id` (required) — The Google Doc document ID
- `content` (required) — Text content to insert
- `mode` (optional, default `append`) — `append` to add to end, `replace` to overwrite all content

**Returns:** Confirmation with a link to the updated document.

**Use when:** The user wants to add content to an existing doc (append) or rewrite it entirely (replace).

**Important:**
- `append` mode is safe — it only adds content at the end
- `replace` mode **destroys all existing content** — always confirm with the user before using replace mode
- Use `read_file_content` first if you need to see the current content before updating

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

**File deletion:**
- **Cannot delete files** (security policy — deletion is irreversible and not supported).
- If the user asks to delete a file, respond: "I can't delete files for security reasons. I can move it to a different folder, or remove your access to it instead — would either of those work?"

**Sharing files:**
- **Always confirm who you're sharing with and what role before calling `share_file`** — sharing is hard to undo and may expose sensitive data
- Call `list_permissions` before `update_permission` or `remove_sharing` to confirm the current state and show the user
- `make_public` is a significant action — **always confirm with the user before doing it** and explain that anyone with the link will be able to access the file

**Organizing files:**
- Use `search_drive` to find the destination folder ID before calling `move_file`
- `rename_file` is safe and reversible — confirm the new name with the user
- `copy_file` creates a new independent copy — changes to the copy won't affect the original

**Updating documents:**
- `update_document` with `append` mode is safe — it only adds content at the end
- `update_document` with `replace` mode **destroys all existing content** — always confirm with the user before using replace mode
- Use `read_file_content` first if you need to see the current content before updating

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

**"Share the Q4 report with alice@example.com as an editor"**
```
1. search_drive query="Q4 report" → get file_id
2. Confirm: "I'll share [file name] with alice@example.com as an editor. Proceed?"
3. share_file file_id="..." email="alice@example.com" role="editor"
4. "Shared! Alice now has editor access."
```

**"Who has access to this file?"**
```
1. list_permissions file_id="..."
2. List each person with their name, email, and role
```

**"Remove Bob's access to the budget spreadsheet"**
```
1. search_drive query="budget spreadsheet" → get file_id
2. list_permissions file_id="..." → confirm Bob has access, show current state
3. Confirm: "I'll remove bob@example.com's access. Proceed?"
4. remove_sharing file_id="..." email="bob@example.com"
5. "Done. Bob no longer has access."
```

**"Make this doc public"**
```
1. Confirm: "This will make the file accessible to anyone with the link. Are you sure?"
2. make_public file_id="..." role="viewer"
3. "Done! Anyone with this link can now view it: [link]"
```

**"Move the report to the Archive folder"**
```
1. search_drive query="Archive" → find the Archive folder ID
2. Confirm: "I'll move [file name] to the Archive folder. Proceed?"
3. move_file file_id="..." folder_id="..."
4. "Moved successfully."
```

**"Add notes to my meeting doc"**
```
1. search_drive query="meeting notes" → get file_id (document_id)
2. update_document document_id="..." content="\n\n[new notes]" mode="append"
3. "Added your notes to the document. Here's the link: [link]"
```
