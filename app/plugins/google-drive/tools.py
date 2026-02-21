"""Google Drive tools for searching, browsing, reading, and creating files.

Uses the Google Drive API v3 and Google Docs/Sheets/Slides APIs.
Each plugin manages its own OAuth tokens independently.
Credentials arrive via ``self._context`` at call time.

New tools (v0.3.0):
- Share files with specific people
- List, update, and remove file permissions
- Make files public
- Move, rename, and copy files
- Create folders
- Update (append/replace) content in existing Google Docs
"""

from __future__ import annotations

import logging

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult
from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

logger = logging.getLogger(__name__)

DRIVE_API = "https://www.googleapis.com/drive/v3"

# Google Workspace MIME types that can be exported as plain text
_EXPORT_MIME_MAP: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.drawing": "image/svg+xml",
}

# MIME types we can read directly (text-based)
_TEXT_MIME_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
)

# Standard fields to request for file listings
_FILE_FIELDS = "id,name,mimeType,modifiedTime,size,parents,webViewLink,owners"
_LIST_FIELDS = f"nextPageToken,files({_FILE_FIELDS})"


class _DriveTool(AetherTool):
    """Base for Drive tools — provides token extraction from runtime context."""

    def _get_token(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("access_token") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}


def _format_file(f: dict) -> str:
    """Format a Drive file into a readable string."""
    name = f.get("name", "Untitled")
    mime = f.get("mimeType", "")
    modified = f.get("modifiedTime", "")
    file_id = f.get("id", "")

    # Friendly type label
    type_label = _friendly_type(mime)

    # Trim ISO timestamp to date
    date_str = modified[:10] if modified else ""

    line = f"**{name}** ({type_label})"
    if date_str:
        line += f" — modified {date_str}"
    line += f"\n   ID: `{file_id}`"

    return line


def _friendly_type(mime: str) -> str:
    """Convert MIME type to a short human-readable label."""
    mapping = {
        "application/vnd.google-apps.document": "Google Doc",
        "application/vnd.google-apps.spreadsheet": "Google Sheet",
        "application/vnd.google-apps.presentation": "Google Slides",
        "application/vnd.google-apps.folder": "Folder",
        "application/vnd.google-apps.drawing": "Google Drawing",
        "application/vnd.google-apps.form": "Google Form",
        "application/pdf": "PDF",
        "image/png": "PNG Image",
        "image/jpeg": "JPEG Image",
        "text/plain": "Text File",
        "text/csv": "CSV",
        "application/json": "JSON",
        "application/zip": "ZIP Archive",
    }
    return mapping.get(mime, mime.split("/")[-1] if "/" in mime else mime)


class SearchDriveTool(_DriveTool):
    """Search files in Google Drive by name or content."""

    name = "search_drive"
    description = "Search your Google Drive files by name or content query"
    status_text = "Searching Drive..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="Search query (matches file name and content)",
            required=True,
        ),
        ToolParam(
            name="max_results",
            type="integer",
            description="Max files to return (default 10)",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, query: str, max_results: int = 10, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{DRIVE_API}/files",
                    headers=self._auth_headers(),
                    params={
                        "q": f"fullText contains '{query}' and trashed = false",
                        "fields": _LIST_FIELDS,
                        "pageSize": max_results,
                        "orderBy": "modifiedTime desc",
                        "supportsAllDrives": "true",
                        "includeItemsFromAllDrives": "true",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            files = data.get("files", [])
            if not files:
                return ToolResult.success(f"No files found matching '{query}'.")

            output = f"**Drive files matching '{query}':**\n"
            for i, f in enumerate(files, 1):
                output += f"\n{i}. {_format_file(f)}"

            return ToolResult.success(output, count=len(files))

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error searching Drive: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ListDriveFilesTool(_DriveTool):
    """List files in a Google Drive folder."""

    name = "list_drive_files"
    description = "List files in a Google Drive folder (use 'root' for top-level)"
    status_text = "Listing files..."
    parameters = [
        ToolParam(
            name="folder_id",
            type="string",
            description="Folder ID to list (use 'root' for top-level, default 'root')",
            required=False,
            default="root",
        ),
        ToolParam(
            name="max_results",
            type="integer",
            description="Max files to return (default 20)",
            required=False,
            default=20,
        ),
        ToolParam(
            name="page_token",
            type="string",
            description="Page token for pagination (from previous response)",
            required=False,
            default="",
        ),
    ]

    async def execute(
        self, folder_id: str = "root", max_results: int = 20, page_token: str = "", **_
    ) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            params: dict = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": _LIST_FIELDS,
                "pageSize": max_results,
                "orderBy": "folder,name",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{DRIVE_API}/files",
                    headers=self._auth_headers(),
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

            files = data.get("files", [])
            next_token = data.get("nextPageToken", "")

            if not files:
                return ToolResult.success("This folder is empty.")

            label = "root" if folder_id == "root" else folder_id
            output = f"**Files in folder `{label}`:**\n"
            for i, f in enumerate(files, 1):
                output += f"\n{i}. {_format_file(f)}"

            if next_token:
                output += f"\n\n_More files available — use page_token: `{next_token}`_"

            return ToolResult.success(
                output, count=len(files), next_page_token=next_token
            )

        except Exception as e:
            logger.error(f"Error listing Drive files: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class GetFileInfoTool(_DriveTool):
    """Get detailed metadata for a specific file."""

    name = "get_file_info"
    description = "Get detailed metadata for a specific Google Drive file by ID"
    status_text = "Fetching file info..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID",
            required=True,
        ),
    ]

    async def execute(self, file_id: str, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=self._auth_headers(),
                    params={
                        "fields": "id,name,mimeType,modifiedTime,createdTime,size,owners,shared,sharingUser,webViewLink,description,parents",
                        "supportsAllDrives": "true",
                    },
                )
                resp.raise_for_status()
                f = resp.json()

            name = f.get("name", "Untitled")
            mime = f.get("mimeType", "")
            output = f"**{name}** ({_friendly_type(mime)})\n"
            output += f"   ID: `{f.get('id', '')}`\n"

            if f.get("description"):
                output += f"   Description: {f['description']}\n"

            created = f.get("createdTime", "")
            modified = f.get("modifiedTime", "")
            if created:
                output += f"   Created: {created[:10]}\n"
            if modified:
                output += f"   Modified: {modified[:10]}\n"

            size = f.get("size")
            if size:
                size_int = int(size)
                if size_int < 1024:
                    output += f"   Size: {size_int} B\n"
                elif size_int < 1024 * 1024:
                    output += f"   Size: {size_int / 1024:.1f} KB\n"
                else:
                    output += f"   Size: {size_int / (1024 * 1024):.1f} MB\n"

            owners = f.get("owners", [])
            if owners:
                owner_names = [
                    o.get("displayName", o.get("emailAddress", "")) for o in owners
                ]
                output += f"   Owner: {', '.join(owner_names)}\n"

            if f.get("shared"):
                output += "   Shared: Yes\n"

            link = f.get("webViewLink", "")
            if link:
                output += f"   Link: {link}\n"

            return ToolResult.success(output.rstrip())

        except Exception as e:
            logger.error(f"Error fetching file info: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ReadFileContentTool(_DriveTool):
    """Read the text content of a Google Drive file.

    Google Docs/Sheets/Slides are exported as plain text.
    Regular text files are downloaded directly.
    Binary files cannot be read.
    """

    name = "read_file_content"
    description = (
        "Read the text content of a Google Drive file (Google Docs exported as text)"
    )
    status_text = "Reading file..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to read",
            required=True,
        ),
        ToolParam(
            name="max_length",
            type="integer",
            description="Max characters to return (default 10000, max 50000)",
            required=False,
            default=10000,
        ),
    ]

    async def execute(self, file_id: str, max_length: int = 10000, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        max_length = min(max_length, 50000)

        try:
            # First, get file metadata to determine type
            async with httpx.AsyncClient() as client:
                meta_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=self._auth_headers(),
                    params={
                        "fields": "id,name,mimeType,size,webViewLink",
                        "supportsAllDrives": "true",
                    },
                )
                meta_resp.raise_for_status()
                meta = meta_resp.json()

            mime = meta.get("mimeType", "")
            name = meta.get("name", "Untitled")
            link = meta.get("webViewLink", "")

            # Google Workspace files — export as text
            if mime in _EXPORT_MIME_MAP:
                export_mime = _EXPORT_MIME_MAP[mime]
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{DRIVE_API}/files/{file_id}/export",
                        headers=self._auth_headers(),
                        params={"mimeType": export_mime},
                    )
                    resp.raise_for_status()
                    content = resp.text

            # Regular text-based files — download directly
            elif any(mime.startswith(p) for p in _TEXT_MIME_PREFIXES):
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{DRIVE_API}/files/{file_id}",
                        headers=self._auth_headers(),
                        params={"alt": "media"},
                    )
                    resp.raise_for_status()
                    content = resp.text

            # Binary files — can't read
            else:
                msg = f"Cannot read binary file **{name}** ({_friendly_type(mime)})."
                if link:
                    msg += f"\nView it here: {link}"
                return ToolResult.fail(msg)

            # Truncate if needed
            truncated = False
            if len(content) > max_length:
                content = content[:max_length]
                truncated = True

            output = f"**Content of {name}:**\n\n{content}"
            if truncated:
                output += f"\n\n_...truncated at {max_length} characters_"

            return ToolResult.success(output, file_name=name, truncated=truncated)

        except Exception as e:
            logger.error(f"Error reading file content: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ListSharedDrivesTool(_DriveTool):
    """List shared/team drives the user has access to."""

    name = "list_shared_drives"
    description = "List shared (team) drives you have access to"
    status_text = "Listing shared drives..."
    parameters = [
        ToolParam(
            name="max_results",
            type="integer",
            description="Max drives to return (default 20)",
            required=False,
            default=20,
        ),
    ]

    async def execute(self, max_results: int = 20, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{DRIVE_API}/drives",
                    headers=self._auth_headers(),
                    params={
                        "pageSize": max_results,
                        "fields": "drives(id,name,createdTime)",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            drives = data.get("drives", [])
            if not drives:
                return ToolResult.success("No shared drives found.")

            output = "**Shared Drives:**\n"
            for i, d in enumerate(drives, 1):
                name = d.get("name", "Untitled")
                drive_id = d.get("id", "")
                created = d.get("createdTime", "")[:10]
                output += f"\n{i}. **{name}**"
                if created:
                    output += f" — created {created}"
                output += f"\n   ID: `{drive_id}`"

            return ToolResult.success(output, count=len(drives))

        except Exception as e:
            logger.error(f"Error listing shared drives: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


# ── Google Workspace Creation APIs ────────────────────────────

DOCS_API = "https://docs.googleapis.com/v1"
SHEETS_API = "https://sheets.googleapis.com/v4"
SLIDES_API = "https://slides.googleapis.com/v1"


class CreateDocumentTool(_DriveTool):
    """Create a new Google Doc with optional initial content."""

    name = "create_document"
    description = "Create a new Google Doc with a title and optional text content"
    status_text = "Creating document..."
    parameters = [
        ToolParam(
            name="title",
            type="string",
            description="Document title",
            required=True,
        ),
        ToolParam(
            name="content",
            type="string",
            description="Initial text content to insert into the document (optional)",
            required=False,
            default="",
        ),
    ]

    async def execute(self, title: str, content: str = "", **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()
            headers["Content-Type"] = "application/json"

            # Step 1: Create the document
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{DOCS_API}/documents",
                    headers=headers,
                    json={"title": title},
                )
                resp.raise_for_status()
                doc = resp.json()

            doc_id = doc.get("documentId", "")
            doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

            # Step 2: Insert content if provided
            if content and doc_id:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{DOCS_API}/documents/{doc_id}:batchUpdate",
                        headers=headers,
                        json={
                            "requests": [
                                {
                                    "insertText": {
                                        "location": {"index": 1},
                                        "text": content,
                                    }
                                }
                            ]
                        },
                    )

            output = f"**Document created:** {title}\n"
            output += f"   ID: `{doc_id}`\n"
            output += f"   Link: {doc_url}"

            return ToolResult.success(output, document_id=doc_id, url=doc_url)

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(f"Docs API error {e.response.status_code}: {body}")
            return ToolResult.fail(f"Docs API error ({e.response.status_code}): {body}")
        except Exception as e:
            logger.error(f"Error creating document: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class CreateSpreadsheetTool(_DriveTool):
    """Create a new Google Sheet."""

    name = "create_spreadsheet"
    description = "Create a new Google Sheet with a title and optional sheet names"
    status_text = "Creating spreadsheet..."
    parameters = [
        ToolParam(
            name="title",
            type="string",
            description="Spreadsheet title",
            required=True,
        ),
        ToolParam(
            name="sheet_names",
            type="string",
            description="Comma-separated sheet/tab names (optional, default: 'Sheet1')",
            required=False,
            default="",
        ),
    ]

    async def execute(self, title: str, sheet_names: str = "", **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()
            headers["Content-Type"] = "application/json"

            # Build sheet properties
            sheets = []
            if sheet_names:
                for name in sheet_names.split(","):
                    name = name.strip()
                    if name:
                        sheets.append({"properties": {"title": name}})

            body: dict = {"properties": {"title": title}}
            if sheets:
                body["sheets"] = sheets

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{SHEETS_API}/spreadsheets",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                sheet = resp.json()

            sheet_id = sheet.get("spreadsheetId", "")
            sheet_url = sheet.get(
                "spreadsheetUrl",
                f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit",
            )
            created_sheets = [
                s.get("properties", {}).get("title", "")
                for s in sheet.get("sheets", [])
            ]

            output = f"**Spreadsheet created:** {title}\n"
            output += f"   ID: `{sheet_id}`\n"
            output += f"   Sheets: {', '.join(created_sheets)}\n"
            output += f"   Link: {sheet_url}"

            return ToolResult.success(output, spreadsheet_id=sheet_id, url=sheet_url)

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(f"Sheets API error {e.response.status_code}: {body}")
            return ToolResult.fail(
                f"Sheets API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error creating spreadsheet: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class CreatePresentationTool(_DriveTool):
    """Create a new Google Slides presentation."""

    name = "create_presentation"
    description = "Create a new Google Slides presentation with a title"
    status_text = "Creating presentation..."
    parameters = [
        ToolParam(
            name="title",
            type="string",
            description="Presentation title",
            required=True,
        ),
    ]

    async def execute(self, title: str, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{SLIDES_API}/presentations",
                    headers=headers,
                    json={"title": title},
                )
                resp.raise_for_status()
                pres = resp.json()

            pres_id = pres.get("presentationId", "")
            pres_url = f"https://docs.google.com/presentation/d/{pres_id}/edit"

            output = f"**Presentation created:** {title}\n"
            output += f"   ID: `{pres_id}`\n"
            output += f"   Link: {pres_url}"

            return ToolResult.success(output, presentation_id=pres_id, url=pres_url)

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(f"Slides API error {e.response.status_code}: {body}")
            return ToolResult.fail(
                f"Slides API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error creating presentation: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


# ── New Drive Tools (v0.3.0) ──────────────────────────────────


class ShareFileTool(_DriveTool):
    """Share a file with one or more email addresses."""

    name = "share_file"
    description = "Share a file with specific people by email address"
    status_text = "Sharing file..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to share",
            required=True,
        ),
        ToolParam(
            name="email",
            type="string",
            description="Email address(es) to share with, comma-separated for multiple",
            required=True,
        ),
        ToolParam(
            name="role",
            type="string",
            description="Access role: 'viewer', 'commenter', or 'editor' (default: 'viewer')",
            required=False,
            default="viewer",
        ),
        ToolParam(
            name="send_notification",
            type="boolean",
            description="Send email notification to recipients (default: true)",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        file_id: str,
        email: str,
        role: str = "viewer",
        send_notification: bool = True,
        **_,
    ) -> ToolResult:
        """Create a permission for each email address on the file."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        # Validate role
        valid_roles = {"viewer", "commenter", "editor"}
        if role not in valid_roles:
            return ToolResult.fail(
                f"Invalid role '{role}'. Must be one of: viewer, commenter, editor."
            )

        try:
            headers = self._auth_headers()
            emails = [e.strip() for e in email.split(",") if e.strip()]
            shared = []

            async with httpx.AsyncClient() as client:
                # First get the file's web link to include in the response
                meta_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=headers,
                    params={"fields": "name,webViewLink", "supportsAllDrives": "true"},
                )
                meta_resp.raise_for_status()
                meta = meta_resp.json()
                file_name = meta.get("name", file_id)
                web_link = meta.get("webViewLink", "")

                # Share with each email address
                for addr in emails:
                    perm_resp = await client.post(
                        f"{DRIVE_API}/files/{file_id}/permissions",
                        headers=headers,
                        params={
                            "sendNotificationEmail": str(send_notification).lower(),
                            "supportsAllDrives": "true",
                        },
                        json={
                            "type": "user",
                            "role": role,
                            "emailAddress": addr,
                        },
                    )
                    perm_resp.raise_for_status()
                    shared.append(addr)

            output = f"**{file_name}** shared successfully.\n"
            output += f"   Shared with: {', '.join(shared)}\n"
            output += f"   Role: {role}\n"
            if web_link:
                output += f"   Link: {web_link}"

            return ToolResult.success(
                output, file_id=file_id, shared_with=shared, role=role
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error sharing file: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class ListPermissionsTool(_DriveTool):
    """List all people who have access to a file."""

    name = "list_permissions"
    description = "List all people who have access to a file"
    status_text = "Fetching permissions..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to list permissions for",
            required=True,
        ),
    ]

    async def execute(self, file_id: str, **_) -> ToolResult:
        """Fetch the permissions list for a Drive file."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}/permissions",
                    headers=self._auth_headers(),
                    params={
                        "fields": "permissions(id,emailAddress,role,type,displayName)",
                        "supportsAllDrives": "true",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            permissions = data.get("permissions", [])
            if not permissions:
                return ToolResult.success("No permissions found for this file.")

            output = "**File Permissions:**\n"
            for perm in permissions:
                ptype = perm.get("type", "")
                role = perm.get("role", "")
                display = perm.get("displayName", "")
                email = perm.get("emailAddress", "")
                perm_id = perm.get("id", "")

                if ptype == "anyone":
                    output += f"\n- **Anyone with the link** — {role}"
                else:
                    name_part = f"{display} " if display else ""
                    email_part = f"<{email}>" if email else ""
                    output += (
                        f"\n- **{name_part}{email_part}** — {role} (ID: `{perm_id}`)"
                    )

            return ToolResult.success(output, count=len(permissions))

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error listing permissions: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class UpdatePermissionTool(_DriveTool):
    """Change a person's access level on a file."""

    name = "update_permission"
    description = "Change someone's access level on a file (viewer/commenter/editor)"
    status_text = "Updating permission..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID",
            required=True,
        ),
        ToolParam(
            name="email",
            type="string",
            description="Email address of the person whose access to change",
            required=True,
        ),
        ToolParam(
            name="role",
            type="string",
            description="New access role: 'viewer', 'commenter', or 'editor'",
            required=True,
        ),
    ]

    async def execute(self, file_id: str, email: str, role: str, **_) -> ToolResult:
        """Find the permission ID for the email, then update the role."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        valid_roles = {"viewer", "commenter", "editor"}
        if role not in valid_roles:
            return ToolResult.fail(
                f"Invalid role '{role}'. Must be one of: viewer, commenter, editor."
            )

        try:
            headers = self._auth_headers()

            # Step 1: List permissions to find the permission ID for this email
            async with httpx.AsyncClient() as client:
                list_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}/permissions",
                    headers=headers,
                    params={
                        "fields": "permissions(id,emailAddress,role)",
                        "supportsAllDrives": "true",
                    },
                )
                list_resp.raise_for_status()
                perms_data = list_resp.json()

            permissions = perms_data.get("permissions", [])
            perm_id = next(
                (
                    p["id"]
                    for p in permissions
                    if p.get("emailAddress", "").lower() == email.lower()
                ),
                None,
            )

            if not perm_id:
                return ToolResult.fail(
                    f"'{email}' does not have access to this file. "
                    "Use list_permissions to see who has access."
                )

            # Step 2: Update the permission role
            async with httpx.AsyncClient() as client:
                patch_resp = await client.patch(
                    f"{DRIVE_API}/files/{file_id}/permissions/{perm_id}",
                    headers=headers,
                    params={"supportsAllDrives": "true"},
                    json={"role": role},
                )
                patch_resp.raise_for_status()

            return ToolResult.success(
                f"Updated {email}'s access to '{role}' on file {file_id}.",
                file_id=file_id,
                email=email,
                role=role,
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error updating permission: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RemoveSharingTool(_DriveTool):
    """Revoke a person's access to a file."""

    name = "remove_sharing"
    description = "Revoke someone's access to a file"
    status_text = "Removing access..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID",
            required=True,
        ),
        ToolParam(
            name="email",
            type="string",
            description="Email address of the person to remove",
            required=True,
        ),
    ]

    async def execute(self, file_id: str, email: str, **_) -> ToolResult:
        """Find the permission ID for the email, then delete it."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Step 1: List permissions to find the permission ID for this email
            async with httpx.AsyncClient() as client:
                list_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}/permissions",
                    headers=headers,
                    params={
                        "fields": "permissions(id,emailAddress)",
                        "supportsAllDrives": "true",
                    },
                )
                list_resp.raise_for_status()
                perms_data = list_resp.json()

            permissions = perms_data.get("permissions", [])
            perm_id = next(
                (
                    p["id"]
                    for p in permissions
                    if p.get("emailAddress", "").lower() == email.lower()
                ),
                None,
            )

            if not perm_id:
                return ToolResult.fail(
                    f"'{email}' does not have access to this file. "
                    "Use list_permissions to see who has access."
                )

            # Step 2: Delete the permission
            async with httpx.AsyncClient() as client:
                del_resp = await client.delete(
                    f"{DRIVE_API}/files/{file_id}/permissions/{perm_id}",
                    headers=headers,
                    params={"supportsAllDrives": "true"},
                )
                del_resp.raise_for_status()

            return ToolResult.success(
                f"Removed {email}'s access to file {file_id}.",
                file_id=file_id,
                email=email,
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error removing sharing: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class MakePublicTool(_DriveTool):
    """Make a file accessible to anyone with the link."""

    name = "make_public"
    description = "Make a file accessible to anyone with the link"
    status_text = "Making file public..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to make public",
            required=True,
        ),
        ToolParam(
            name="role",
            type="string",
            description="Access role for public: 'viewer' or 'commenter' (default: 'viewer')",
            required=False,
            default="viewer",
        ),
    ]

    async def execute(self, file_id: str, role: str = "viewer", **_) -> ToolResult:
        """Set an 'anyone' permission on the file."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        valid_roles = {"viewer", "commenter"}
        if role not in valid_roles:
            return ToolResult.fail(
                f"Invalid role '{role}'. For public access, must be 'viewer' or 'commenter'."
            )

        try:
            headers = self._auth_headers()

            async with httpx.AsyncClient() as client:
                # Get file name and link for the response
                meta_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=headers,
                    params={"fields": "name,webViewLink", "supportsAllDrives": "true"},
                )
                meta_resp.raise_for_status()
                meta = meta_resp.json()
                file_name = meta.get("name", file_id)
                web_link = meta.get("webViewLink", "")

                # Create the 'anyone' permission
                perm_resp = await client.post(
                    f"{DRIVE_API}/files/{file_id}/permissions",
                    headers=headers,
                    params={"supportsAllDrives": "true"},
                    json={"type": "anyone", "role": role},
                )
                perm_resp.raise_for_status()

            output = f"**{file_name}** is now public ({role}).\n"
            if web_link:
                output += f"   Anyone with this link can view: {web_link}"

            return ToolResult.success(output, file_id=file_id, role=role, link=web_link)

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error making file public: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class MoveFileTool(_DriveTool):
    """Move a file to a different folder in Google Drive."""

    name = "move_file"
    description = "Move a file to a different folder"
    status_text = "Moving file..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to move",
            required=True,
        ),
        ToolParam(
            name="folder_id",
            type="string",
            description="The destination folder ID",
            required=True,
        ),
    ]

    async def execute(self, file_id: str, folder_id: str, **_) -> ToolResult:
        """Get current parents, then move the file to the new folder."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Step 1: Get current parents
            async with httpx.AsyncClient() as client:
                meta_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=headers,
                    params={"fields": "name,parents", "supportsAllDrives": "true"},
                )
                meta_resp.raise_for_status()
                meta = meta_resp.json()

            file_name = meta.get("name", file_id)
            current_parents = meta.get("parents", [])
            remove_parents = ",".join(current_parents)

            # Step 2: Move to new folder by updating parents
            async with httpx.AsyncClient() as client:
                move_resp = await client.patch(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=headers,
                    params={
                        "addParents": folder_id,
                        "removeParents": remove_parents,
                        "fields": "id,name,parents",
                        "supportsAllDrives": "true",
                    },
                )
                move_resp.raise_for_status()

            return ToolResult.success(
                f"**{file_name}** moved to folder `{folder_id}` successfully.",
                file_id=file_id,
                folder_id=folder_id,
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error moving file: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RenameFileTool(_DriveTool):
    """Rename a file in Google Drive."""

    name = "rename_file"
    description = "Rename a file in Google Drive"
    status_text = "Renaming file..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to rename",
            required=True,
        ),
        ToolParam(
            name="new_name",
            type="string",
            description="The new name for the file",
            required=True,
        ),
    ]

    async def execute(self, file_id: str, new_name: str, **_) -> ToolResult:
        """Patch the file's name metadata."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Get the current name for the confirmation message
            async with httpx.AsyncClient() as client:
                meta_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=headers,
                    params={"fields": "name", "supportsAllDrives": "true"},
                )
                meta_resp.raise_for_status()
                old_name = meta_resp.json().get("name", file_id)

                # Rename the file
                rename_resp = await client.patch(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=headers,
                    params={"fields": "id,name", "supportsAllDrives": "true"},
                    json={"name": new_name},
                )
                rename_resp.raise_for_status()

            return ToolResult.success(
                f"Renamed: **{old_name}** → **{new_name}**",
                file_id=file_id,
                old_name=old_name,
                new_name=new_name,
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error renaming file: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class CopyFileTool(_DriveTool):
    """Duplicate a file in Google Drive."""

    name = "copy_file"
    description = "Make a copy of a file in Google Drive"
    status_text = "Copying file..."
    parameters = [
        ToolParam(
            name="file_id",
            type="string",
            description="The file ID to copy",
            required=True,
        ),
        ToolParam(
            name="new_name",
            type="string",
            description="Name for the copy (optional — defaults to 'Copy of [original name]')",
            required=False,
            default="",
        ),
    ]

    async def execute(self, file_id: str, new_name: str = "", **_) -> ToolResult:
        """Copy a file, optionally with a new name."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Build the copy request body
            body: dict = {}
            if new_name:
                body["name"] = new_name

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{DRIVE_API}/files/{file_id}/copy",
                    headers=headers,
                    params={
                        "fields": "id,name,webViewLink",
                        "supportsAllDrives": "true",
                    },
                    json=body,
                )
                resp.raise_for_status()
                copy = resp.json()

            copy_id = copy.get("id", "")
            copy_name = copy.get("name", "")
            copy_link = copy.get("webViewLink", "")

            output = f"**File copied successfully.**\n"
            output += f"   New file: **{copy_name}**\n"
            output += f"   ID: `{copy_id}`\n"
            if copy_link:
                output += f"   Link: {copy_link}"

            return ToolResult.success(
                output, new_file_id=copy_id, new_name=copy_name, link=copy_link
            )

        except httpx.HTTPStatusError as e:
            body_text = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body_text}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body_text}"
            )
        except Exception as e:
            logger.error(f"Error copying file: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class CreateFolderTool(_DriveTool):
    """Create a new folder in Google Drive."""

    name = "create_folder"
    description = "Create a new folder in Google Drive"
    status_text = "Creating folder..."
    parameters = [
        ToolParam(
            name="name",
            type="string",
            description="Folder name",
            required=True,
        ),
        ToolParam(
            name="parent_folder_id",
            type="string",
            description="Parent folder ID (optional — defaults to Drive root)",
            required=False,
            default="",
        ),
    ]

    async def execute(self, name: str, parent_folder_id: str = "", **_) -> ToolResult:
        """Create a folder using the Drive files.create API."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        try:
            headers = self._auth_headers()

            # Use provided parent or fall back to root
            parent = parent_folder_id.strip() if parent_folder_id else "root"

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{DRIVE_API}/files",
                    headers=headers,
                    params={
                        "fields": "id,name,webViewLink",
                        "supportsAllDrives": "true",
                    },
                    json={
                        "name": name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [parent],
                    },
                )
                resp.raise_for_status()
                folder = resp.json()

            folder_id = folder.get("id", "")
            folder_link = folder.get("webViewLink", "")

            output = f"**Folder created:** {name}\n"
            output += f"   ID: `{folder_id}`\n"
            if folder_link:
                output += f"   Link: {folder_link}"

            return ToolResult.success(
                output, folder_id=folder_id, name=name, link=folder_link
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Drive API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(
                f"Drive API error ({e.response.status_code}): {body}"
            )
        except Exception as e:
            logger.error(f"Error creating folder: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class UpdateDocumentTool(_DriveTool):
    """Append or replace content in an existing Google Doc."""

    name = "update_document"
    description = "Append or replace content in an existing Google Doc"
    status_text = "Updating document..."
    parameters = [
        ToolParam(
            name="document_id",
            type="string",
            description="The Google Doc document ID",
            required=True,
        ),
        ToolParam(
            name="content",
            type="string",
            description="Text content to insert",
            required=True,
        ),
        ToolParam(
            name="mode",
            type="string",
            description="'append' to add to end, 'replace' to overwrite all content (default: 'append')",
            required=False,
            default="append",
        ),
    ]

    async def execute(
        self, document_id: str, content: str, mode: str = "append", **_
    ) -> ToolResult:
        """Use the Docs API batchUpdate to append or replace document content."""
        if not self._get_token():
            return ToolResult.fail("Google Drive not connected — missing access token.")

        valid_modes = {"append", "replace"}
        if mode not in valid_modes:
            return ToolResult.fail(
                f"Invalid mode '{mode}'. Must be 'append' or 'replace'."
            )

        try:
            headers = self._auth_headers()
            headers["Content-Type"] = "application/json"

            # Step 1: Get the document to find the end index
            async with httpx.AsyncClient() as client:
                doc_resp = await client.get(
                    f"{DOCS_API}/documents/{document_id}",
                    headers=headers,
                    params={"fields": "body.content,title"},
                )
                doc_resp.raise_for_status()
                doc = doc_resp.json()

            doc_title = doc.get("title", document_id)
            body_content = doc.get("body", {}).get("content", [])

            # Find the last endIndex in the document body
            end_index = 1  # Minimum safe index
            for element in body_content:
                ei = element.get("endIndex")
                if ei is not None:
                    end_index = ei

            # Build the batchUpdate requests
            requests: list[dict] = []

            if mode == "replace":
                # Delete all existing content (if doc has more than just the final newline)
                if end_index > 2:
                    requests.append(
                        {
                            "deleteContentRange": {
                                "range": {
                                    "startIndex": 1,
                                    "endIndex": end_index - 1,
                                }
                            }
                        }
                    )
                # After deletion, insert at index 1
                requests.append(
                    {
                        "insertText": {
                            "location": {"index": 1},
                            "text": content,
                        }
                    }
                )
            else:
                # Append: insert before the final newline (endIndex - 1)
                insert_index = max(1, end_index - 1)
                requests.append(
                    {
                        "insertText": {
                            "location": {"index": insert_index},
                            "text": content,
                        }
                    }
                )

            # Step 2: Apply the batchUpdate
            async with httpx.AsyncClient() as client:
                update_resp = await client.post(
                    f"{DOCS_API}/documents/{document_id}:batchUpdate",
                    headers=headers,
                    json={"requests": requests},
                )
                update_resp.raise_for_status()

            doc_url = f"https://docs.google.com/document/d/{document_id}/edit"
            action = "appended to" if mode == "append" else "replaced content in"
            output = f"**{doc_title}** — content {action} successfully.\n"
            output += f"   Link: {doc_url}"

            return ToolResult.success(
                output, document_id=document_id, mode=mode, url=doc_url
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                f"Docs API error {e.response.status_code}: {body}", exc_info=True
            )
            return ToolResult.fail(f"Docs API error ({e.response.status_code}): {body}")
        except Exception as e:
            logger.error(f"Error updating document: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RefreshGoogleDriveTokenTool(RefreshOAuthTokenTool):
    """Refresh the Google Drive OAuth access token before it expires.

    Called automatically by the cron system every 50 minutes.
    Can also be called manually if Drive tools start returning auth errors.
    """

    name = "refresh_google_drive_token"
    plugin_name = "google-drive"
    description = (
        "Refresh the Google Drive OAuth access token. "
        "Call this when Drive tools return authentication errors, "
        "or when instructed by the system to prevent token expiry."
    )
