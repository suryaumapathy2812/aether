---
name: drive
description: Google Drive API — files, folders, search, upload, download, permissions
integration: google-workspace
---
# Google Drive API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://www.googleapis.com/drive/v3`
- Token auto-refreshes on 401 response

## Search Files

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q=name+contains+'Q4+report'&fields=files(id,name,mimeType,modifiedTime,webViewLink)&pageSize=20"
```

### Search Query Syntax
- `name contains 'budget'` — name contains text
- `fullText contains 'quarterly'` — content search
- `mimeType='application/vnd.google-apps.document'` — Google Docs only
- `mimeType='application/vnd.google-apps.spreadsheet'` — Sheets only
- `mimeType='application/vnd.google-apps.folder'` — Folders only
- `'folderId' in parents` — files inside a specific folder
- `trashed=false` — exclude trashed files (recommended)
- Combine with `and`/`or`: `name contains 'report' and mimeType='application/vnd.google-apps.document'`

## List Files in a Folder

```bash
# Root folder
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q='root'+in+parents+and+trashed=false&fields=files(id,name,mimeType,modifiedTime,webViewLink)&pageSize=50"

# Specific folder
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q='FOLDER_ID'+in+parents+and+trashed=false&fields=files(id,name,mimeType,modifiedTime,webViewLink)&pageSize=50"
```

## Get File Metadata

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files/{FILE_ID}?fields=id,name,mimeType,size,createdTime,modifiedTime,owners,sharingUser,webViewLink,webContentLink,parents,permissions"
```

### Key Response Fields
- `name` — file name
- `mimeType` — file type (e.g., `application/vnd.google-apps.document`)
- `size` — file size in bytes (not set for Google Docs/Sheets)
- `owners[].emailAddress` — file owner
- `webViewLink` — link to open in browser
- `modifiedTime` — last modified timestamp

## Download File Content

```bash
# For Google Docs (export as PDF or other format)
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files/{FILE_ID}/export?mimeType=application/pdf" -o document.pdf

# For binary files (images, PDFs, etc.)
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files/{FILE_ID}?alt=media" -o downloaded_file
```

## Create Folder

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New Folder",
    "mimeType": "application/vnd.google-apps.folder",
    "parents": ["root"]
  }' \
  "https://www.googleapis.com/drive/v3/files"
```

## Upload File

```bash
# 1. Get upload URL
UPLOAD_URL="https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

# 2. Upload with metadata
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -F 'metadata={"name":"uploaded.txt","parents":["FOLDER_ID"]};type=application/json' \
  -F 'file=@./local-file.txt;type=text/plain' \
  "$UPLOAD_URL"
```

## List Shared Drives

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/drives?fields=drives(id,name)"
```

## Pagination

Responses include `nextPageToken` when more results exist:
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q=...&pageSize=100&pageToken=NEXT_PAGE_TOKEN"
```

Max `pageSize` is 100. Up to 1,000 files per query with pagination.

## Rate Limits
- 12,000 queries per minute
- 25 concurrent requests per user

## Error Handling
- **401 Unauthorized**: Token expired — system auto-refreshes, retry the command
- **403 Forbidden**: Drive API not enabled, or insufficient permissions/scopes
- **404 Not Found**: File ID doesn't exist or was deleted
- **429 Rate Limited**: Wait and retry
- If search returns no results, try broader terms or `fullText contains` for content search
