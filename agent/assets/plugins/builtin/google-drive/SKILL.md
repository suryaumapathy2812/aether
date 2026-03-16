# Google Drive Plugin

Search, browse, and organize files in the user's Google Drive.

## Core Workflow

Drive operations always start by finding the file ID, then acting on it:

```
Find file ID                    →  Act on it
──────────────────────             ─────────────────────
search_drive (by name/content)     get_file_info (metadata)
list_drive_files (browse folder)   create_folder
list_shared_drives                 
```

## Autonomy Rules

**Never ask the user to provide search terms.** When they say "find the document about X", search for X. When they say "find my files", start with `list_drive_files` at root. When the request is vague, try multiple search strategies:
- Try the exact phrase first, then individual keywords
- If no results, try broader terms or synonyms
- Try at least 2-3 different queries before reporting nothing found

## Decision Rules

**Finding files:**
- Use `search_drive` when the user wants to find a file by name or content — it's faster than browsing.
- Use `list_drive_files` when the user wants to browse a specific folder's contents. Pass `folder_id="root"` for the top-level Drive.
- Get the file ID from search/list results before calling `get_file_info`.

**File details:**
- `get_file_info` returns full metadata: size, created/modified dates, owner, sharing settings, MIME type, and web link.
- Keep file listings concise: name, type, and last modified date. Include the web link when referencing a specific file.

**Creating folders:**
- Confirm the folder name and location with the user before creating.
- `create_folder` defaults to the Drive root if no `parent_id` is specified.

**Shared drives:**
- Use `list_shared_drives` when the user asks about team drives or shared workspaces.

**File deletion:**
- This plugin cannot delete files (by design — deletion is irreversible). If the user asks to delete, explain: "I can't delete files for security reasons, but I can help you find and organize them."

**Presenting results:**
- Show the most relevant files first with name, type, and modified date.
- For search results, offer to get more details on specific files.
- Always include the web link so the user can open files directly.

**Error handling:**
- If a file isn't found, suggest different search terms.
- 403 errors usually mean the Drive API isn't enabled or the user needs to reconnect the plugin.

## Pagination

The Drive API returns up to 100 files per request (max 1000). Responses include `nextPageToken` when more files exist — pass it as `page_token` in follow-up calls.

## Rate Limits

| Quota | Limit |
|---|---|
| Queries per minute per user | 600 |
| Concurrent requests | 25 per user |

Safe to make 20+ parallel file info reads.

## Example Workflows

**User: "Find the Q4 report"**
```
1. search_drive query="Q4 report"
2. "I found 3 files matching 'Q4 report':
   - Q4 Report 2025.docx (modified Jan 15) [link]
   - Q4 Report Draft.docx (modified Dec 20) [link]
   - Q4 Revenue Report.xlsx (modified Jan 10) [link]
   Which one do you need?"
```

**User: "What's in my Drive root folder?"**
```
1. list_drive_files folder_id="root" max_results=20
2. List files grouped by type: "Your Drive root has 12 items:
   Folders: Projects, Archive, Shared
   Docs: Meeting Notes, Project Proposal
   Sheets: Budget 2026, Expense Tracker
   ..."
```

**User: "Who owns the budget spreadsheet?"**
```
1. search_drive query="budget spreadsheet" → get file_id
2. get_file_info file_id="..."
3. "The Budget 2026 spreadsheet is owned by priya@company.com, last modified on March 10. It's shared with 3 people. [link]"
```
