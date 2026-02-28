package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

const driveAPI = "https://www.googleapis.com/drive/v3"

type SearchDriveTool struct{}
type ListDriveFilesTool struct{}
type GetFileInfoTool struct{}
type ReadFileContentTool struct{}
type ListSharedDrivesTool struct{}
type CreateDocumentTool struct{}
type CreateSpreadsheetTool struct{}
type CreatePresentationTool struct{}
type ShareFileTool struct{}
type ListPermissionsTool struct{}
type UpdatePermissionTool struct{}
type RemoveSharingTool struct{}
type MakePublicTool struct{}
type MoveFileTool struct{}
type RenameFileTool struct{}
type CopyFileTool struct{}
type CreateFolderTool struct{}
type UpdateDocumentTool struct{}
type RefreshGoogleDriveTokenTool struct{}

func (t *SearchDriveTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_drive", Description: "Search files in Google Drive.", StatusText: "Searching Drive...", Parameters: []tools.Param{{Name: "query", Type: "string", Required: true}, {Name: "max_results", Type: "integer", Required: false, Default: 10}}}
}

func (t *SearchDriveTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	q, _ := call.Args["query"].(string)
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 10
	}
	v := url.Values{}
	v.Set("q", fmt.Sprintf("fullText contains '%s' and trashed = false", strings.ReplaceAll(q, "'", "\\'")))
	v.Set("fields", "nextPageToken,files(id,name,mimeType,modifiedTime,size,parents,webViewLink,owners)")
	v.Set("pageSize", fmt.Sprintf("%d", max))
	v.Set("orderBy", "modifiedTime desc")
	v.Set("supportsAllDrives", "true")
	v.Set("includeItemsFromAllDrives", "true")
	obj, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["files"])
	return tools.Success(string(b), map[string]any{"query": q})
}

func (t *ListDriveFilesTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_drive_files", Description: "List files in a Google Drive folder.", StatusText: "Listing files...", Parameters: []tools.Param{{Name: "folder_id", Type: "string", Required: false, Default: "root"}, {Name: "max_results", Type: "integer", Required: false, Default: 20}, {Name: "page_token", Type: "string", Required: false, Default: ""}}}
}

func (t *ListDriveFilesTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	folderID, _ := call.Args["folder_id"].(string)
	if folderID == "" {
		folderID = "root"
	}
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 20
	}
	pageToken, _ := call.Args["page_token"].(string)
	v := url.Values{}
	v.Set("q", fmt.Sprintf("'%s' in parents and trashed = false", folderID))
	v.Set("fields", "nextPageToken,files(id,name,mimeType,modifiedTime,size,parents,webViewLink,owners)")
	v.Set("pageSize", fmt.Sprintf("%d", max))
	v.Set("orderBy", "folder,name")
	v.Set("supportsAllDrives", "true")
	v.Set("includeItemsFromAllDrives", "true")
	if strings.TrimSpace(pageToken) != "" {
		v.Set("pageToken", pageToken)
	}
	obj, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj)
	return tools.Success(string(b), map[string]any{"folder_id": folderID})
}

func (t *GetFileInfoTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_file_info", Description: "Get Google Drive file metadata.", StatusText: "Loading file info...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}}}
}

func (t *GetFileInfoTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	v := url.Values{}
	v.Set("fields", "id,name,mimeType,modifiedTime,createdTime,size,owners,shared,sharingUser,webViewLink,description,parents")
	v.Set("supportsAllDrives", "true")
	obj, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files/"+url.PathEscape(fileID)+"?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj)
	return tools.Success(string(b), map[string]any{"file_id": fileID})
}

func (t *ReadFileContentTool) Definition() tools.Definition {
	return tools.Definition{Name: "read_file_content", Description: "Read text content from a Google Drive file.", StatusText: "Reading file...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "max_length", Type: "integer", Required: false, Default: 10000}}}
}

func (t *ReadFileContentTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	maxLen, _ := asInt(call.Args["max_length"])
	if maxLen <= 0 {
		maxLen = 10000
	}
	meta, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files/"+url.PathEscape(fileID)+"?fields=id,name,mimeType,webViewLink&supportsAllDrives=true", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	mime, _ := meta["mimeType"].(string)
	contentURL := ""
	if strings.HasPrefix(mime, "application/vnd.google-apps.document") {
		contentURL = driveAPI + "/files/" + url.PathEscape(fileID) + "/export?mimeType=text/plain"
	} else if strings.HasPrefix(mime, "application/vnd.google-apps.spreadsheet") {
		contentURL = driveAPI + "/files/" + url.PathEscape(fileID) + "/export?mimeType=text/csv"
	} else if strings.HasPrefix(mime, "text/") || mime == "application/json" || mime == "application/xml" {
		contentURL = driveAPI + "/files/" + url.PathEscape(fileID) + "?alt=media"
	} else {
		return tools.Fail("Cannot read binary file type: "+mime, nil)
	}
	text, err := driveTextRequest(ctx, call, http.MethodGet, contentURL)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	if len(text) > maxLen {
		text = text[:maxLen] + "\n\n...truncated"
	}
	return tools.Success(text, map[string]any{"file_id": fileID, "mime_type": mime})
}

func (t *ListSharedDrivesTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_shared_drives", Description: "List Google shared drives.", StatusText: "Loading shared drives...", Parameters: []tools.Param{{Name: "max_results", Type: "integer", Required: false, Default: 20}}}
}

func (t *ListSharedDrivesTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 20
	}
	v := url.Values{}
	v.Set("pageSize", fmt.Sprintf("%d", max))
	v.Set("fields", "drives(id,name,createdTime)")
	obj, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/drives?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["drives"])
	return tools.Success(string(b), nil)
}

func (t *CreateDocumentTool) Definition() tools.Definition {
	return tools.Definition{Name: "create_document", Description: "Create a Google Doc.", StatusText: "Creating document...", Parameters: []tools.Param{{Name: "title", Type: "string", Required: true}, {Name: "content", Type: "string", Required: false, Default: ""}}}
}

func (t *CreateDocumentTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	title, _ := call.Args["title"].(string)
	content, _ := call.Args["content"].(string)
	obj, err := googleAPIRequest(ctx, call, http.MethodPost, "https://docs.googleapis.com/v1/documents", map[string]any{"title": title})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	docID, _ := obj["documentId"].(string)
	if strings.TrimSpace(content) != "" && docID != "" {
		_, _ = googleAPIRequest(ctx, call, http.MethodPost, "https://docs.googleapis.com/v1/documents/"+url.PathEscape(docID)+":batchUpdate", map[string]any{"requests": []map[string]any{{"insertText": map[string]any{"location": map[string]any{"index": 1}, "text": content}}}})
	}
	return tools.Success("Document created.", map[string]any{"document_id": docID})
}

func (t *CreateSpreadsheetTool) Definition() tools.Definition {
	return tools.Definition{Name: "create_spreadsheet", Description: "Create a Google Sheet.", StatusText: "Creating spreadsheet...", Parameters: []tools.Param{{Name: "title", Type: "string", Required: true}, {Name: "sheet_names", Type: "string", Required: false, Default: ""}}}
}

func (t *CreateSpreadsheetTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	title, _ := call.Args["title"].(string)
	sheets, _ := call.Args["sheet_names"].(string)
	body := map[string]any{"properties": map[string]any{"title": title}}
	if strings.TrimSpace(sheets) != "" {
		parts := strings.Split(sheets, ",")
		out := make([]map[string]any, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			out = append(out, map[string]any{"properties": map[string]any{"title": p}})
		}
		if len(out) > 0 {
			body["sheets"] = out
		}
	}
	obj, err := googleAPIRequest(ctx, call, http.MethodPost, "https://sheets.googleapis.com/v4/spreadsheets", body)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Spreadsheet created.", map[string]any{"spreadsheet_id": obj["spreadsheetId"], "url": obj["spreadsheetUrl"]})
}

func (t *CreatePresentationTool) Definition() tools.Definition {
	return tools.Definition{Name: "create_presentation", Description: "Create a Google Slides presentation.", StatusText: "Creating presentation...", Parameters: []tools.Param{{Name: "title", Type: "string", Required: true}}}
}

func (t *CreatePresentationTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	title, _ := call.Args["title"].(string)
	obj, err := googleAPIRequest(ctx, call, http.MethodPost, "https://slides.googleapis.com/v1/presentations", map[string]any{"title": title})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Presentation created.", map[string]any{"presentation_id": obj["presentationId"]})
}

func (t *ShareFileTool) Definition() tools.Definition {
	return tools.Definition{Name: "share_file", Description: "Share a Drive file with email addresses.", StatusText: "Sharing file...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "email", Type: "string", Required: true}, {Name: "role", Type: "string", Required: false, Default: "viewer"}, {Name: "send_notification", Type: "boolean", Required: false, Default: true}}}
}

func (t *ShareFileTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	emails, _ := call.Args["email"].(string)
	role, _ := call.Args["role"].(string)
	if role == "" {
		role = "viewer"
	}
	notify, _ := asBool(call.Args["send_notification"])
	for _, email := range strings.Split(emails, ",") {
		email = strings.TrimSpace(email)
		if email == "" {
			continue
		}
		_, err := driveRequest(ctx, call, http.MethodPost, driveAPI+"/files/"+url.PathEscape(fileID)+"/permissions?sendNotificationEmail="+fmt.Sprintf("%t", notify)+"&supportsAllDrives=true", map[string]any{"type": "user", "role": role, "emailAddress": email})
		if err != nil {
			return tools.Fail(err.Error(), nil)
		}
	}
	return tools.Success("File shared.", map[string]any{"file_id": fileID})
}

func (t *ListPermissionsTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_permissions", Description: "List Drive file permissions.", StatusText: "Loading permissions...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}}}
}

func (t *ListPermissionsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	obj, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files/"+url.PathEscape(fileID)+"/permissions?fields=permissions(id,emailAddress,role,type,displayName)&supportsAllDrives=true", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["permissions"])
	return tools.Success(string(b), map[string]any{"file_id": fileID})
}

func (t *UpdatePermissionTool) Definition() tools.Definition {
	return tools.Definition{Name: "update_permission", Description: "Update a user's role on a Drive file.", StatusText: "Updating permission...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "email", Type: "string", Required: true}, {Name: "role", Type: "string", Required: true}}}
}

func (t *UpdatePermissionTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return driveUpdateOrDeletePermission(ctx, call, true)
}

func (t *RemoveSharingTool) Definition() tools.Definition {
	return tools.Definition{Name: "remove_sharing", Description: "Remove a user from Drive file sharing.", StatusText: "Removing sharing...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "email", Type: "string", Required: true}}}
}

func (t *RemoveSharingTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return driveUpdateOrDeletePermission(ctx, call, false)
}

func (t *MakePublicTool) Definition() tools.Definition {
	return tools.Definition{Name: "make_public", Description: "Make a Drive file public by link.", StatusText: "Making file public...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "role", Type: "string", Required: false, Default: "viewer"}}}
}

func (t *MakePublicTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	role, _ := call.Args["role"].(string)
	if role == "" {
		role = "viewer"
	}
	_, err := driveRequest(ctx, call, http.MethodPost, driveAPI+"/files/"+url.PathEscape(fileID)+"/permissions?supportsAllDrives=true", map[string]any{"type": "anyone", "role": role})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("File is now public.", map[string]any{"file_id": fileID, "role": role})
}

func (t *MoveFileTool) Definition() tools.Definition {
	return tools.Definition{Name: "move_file", Description: "Move Drive file to a folder.", StatusText: "Moving file...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "folder_id", Type: "string", Required: true}}}
}

func (t *MoveFileTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	folderID, _ := call.Args["folder_id"].(string)
	meta, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files/"+url.PathEscape(fileID)+"?fields=parents&supportsAllDrives=true", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	parents := []string{}
	if arr, ok := meta["parents"].([]any); ok {
		for _, p := range arr {
			if s, ok := p.(string); ok {
				parents = append(parents, s)
			}
		}
	}
	q := url.Values{}
	q.Set("addParents", folderID)
	q.Set("removeParents", strings.Join(parents, ","))
	q.Set("supportsAllDrives", "true")
	_, err = driveRequest(ctx, call, http.MethodPatch, driveAPI+"/files/"+url.PathEscape(fileID)+"?"+q.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("File moved.", map[string]any{"file_id": fileID, "folder_id": folderID})
}

func (t *RenameFileTool) Definition() tools.Definition {
	return tools.Definition{Name: "rename_file", Description: "Rename a Drive file.", StatusText: "Renaming file...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "new_name", Type: "string", Required: true}}}
}

func (t *RenameFileTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	newName, _ := call.Args["new_name"].(string)
	_, err := driveRequest(ctx, call, http.MethodPatch, driveAPI+"/files/"+url.PathEscape(fileID)+"?fields=id,name&supportsAllDrives=true", map[string]any{"name": newName})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("File renamed.", map[string]any{"file_id": fileID, "new_name": newName})
}

func (t *CopyFileTool) Definition() tools.Definition {
	return tools.Definition{Name: "copy_file", Description: "Copy a Drive file.", StatusText: "Copying file...", Parameters: []tools.Param{{Name: "file_id", Type: "string", Required: true}, {Name: "new_name", Type: "string", Required: false, Default: ""}}}
}

func (t *CopyFileTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	newName, _ := call.Args["new_name"].(string)
	body := map[string]any{}
	if strings.TrimSpace(newName) != "" {
		body["name"] = newName
	}
	obj, err := driveRequest(ctx, call, http.MethodPost, driveAPI+"/files/"+url.PathEscape(fileID)+"/copy?fields=id,name,webViewLink&supportsAllDrives=true", body)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("File copied.", map[string]any{"new_file_id": obj["id"], "name": obj["name"]})
}

func (t *CreateFolderTool) Definition() tools.Definition {
	return tools.Definition{Name: "create_folder", Description: "Create a Google Drive folder.", StatusText: "Creating folder...", Parameters: []tools.Param{{Name: "name", Type: "string", Required: true}, {Name: "parent_folder_id", Type: "string", Required: false, Default: ""}}}
}

func (t *CreateFolderTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	name, _ := call.Args["name"].(string)
	parent, _ := call.Args["parent_folder_id"].(string)
	body := map[string]any{"name": name, "mimeType": "application/vnd.google-apps.folder"}
	if strings.TrimSpace(parent) != "" {
		body["parents"] = []string{parent}
	}
	obj, err := driveRequest(ctx, call, http.MethodPost, driveAPI+"/files?fields=id,name,webViewLink&supportsAllDrives=true", body)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Folder created.", map[string]any{"folder_id": obj["id"], "name": obj["name"]})
}

func (t *UpdateDocumentTool) Definition() tools.Definition {
	return tools.Definition{Name: "update_document", Description: "Append or replace content in a Google Doc.", StatusText: "Updating document...", Parameters: []tools.Param{{Name: "document_id", Type: "string", Required: true}, {Name: "content", Type: "string", Required: true}, {Name: "mode", Type: "string", Required: false, Default: "append", Enum: []string{"append", "replace"}}}}
}

func (t *UpdateDocumentTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	docID, _ := call.Args["document_id"].(string)
	content, _ := call.Args["content"].(string)
	mode, _ := call.Args["mode"].(string)
	if mode == "" {
		mode = "append"
	}
	requests := []map[string]any{}
	if mode == "replace" {
		requests = append(requests, map[string]any{"deleteContentRange": map[string]any{"range": map[string]any{"startIndex": 1, "endIndex": 1 << 30}}})
	}
	requests = append(requests, map[string]any{"insertText": map[string]any{"location": map[string]any{"index": 1}, "text": content}})
	_, err := googleAPIRequest(ctx, call, http.MethodPost, "https://docs.googleapis.com/v1/documents/"+url.PathEscape(docID)+":batchUpdate", map[string]any{"requests": requests})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Document updated.", map[string]any{"document_id": docID, "mode": mode})
}

func (t *RefreshGoogleDriveTokenTool) Definition() tools.Definition {
	return tools.Definition{Name: "refresh_google_drive_token", Description: "Refresh Google Drive OAuth access token.", StatusText: "Refreshing Drive token..."}
}

func (t *RefreshGoogleDriveTokenTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return refreshOAuthAccessToken(ctx, call, "https://oauth2.googleapis.com/token", false)
}

func driveUpdateOrDeletePermission(ctx context.Context, call tools.Call, update bool) tools.Result {
	fileID, _ := call.Args["file_id"].(string)
	email, _ := call.Args["email"].(string)
	perms, err := driveRequest(ctx, call, http.MethodGet, driveAPI+"/files/"+url.PathEscape(fileID)+"/permissions?fields=permissions(id,emailAddress)&supportsAllDrives=true", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	permID := ""
	if arr, ok := perms["permissions"].([]any); ok {
		for _, p := range arr {
			m, _ := p.(map[string]any)
			addr, _ := m["emailAddress"].(string)
			if strings.EqualFold(strings.TrimSpace(addr), strings.TrimSpace(email)) {
				permID, _ = m["id"].(string)
				break
			}
		}
	}
	if permID == "" {
		return tools.Fail("No sharing entry found for email: "+email, nil)
	}
	if update {
		role, _ := call.Args["role"].(string)
		_, err = driveRequest(ctx, call, http.MethodPatch, driveAPI+"/files/"+url.PathEscape(fileID)+"/permissions/"+url.PathEscape(permID)+"?supportsAllDrives=true", map[string]any{"role": role})
		if err != nil {
			return tools.Fail(err.Error(), nil)
		}
		return tools.Success("Permission updated.", map[string]any{"file_id": fileID, "email": email, "role": role})
	}
	_, err = driveRequest(ctx, call, http.MethodDelete, driveAPI+"/files/"+url.PathEscape(fileID)+"/permissions/"+url.PathEscape(permID)+"?supportsAllDrives=true", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Sharing removed.", map[string]any{"file_id": fileID, "email": email})
}

func driveRequest(ctx context.Context, call tools.Call, method, reqURL string, body any) (map[string]any, error) {
	return googleAPIRequest(ctx, call, method, reqURL, body)
}

func driveTextRequest(ctx context.Context, call tools.Call, method, reqURL string) (string, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return "", err
	}
	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return "", fmt.Errorf("Google Drive is not connected: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, method, reqURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("Google Drive API request failed with status %d", resp.StatusCode)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func googleAPIRequest(ctx context.Context, call tools.Call, method, reqURL string, body any) (map[string]any, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return nil, err
	}
	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return nil, fmt.Errorf("Google plugin is not connected: %w", err)
	}
	var reqBody io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		reqBody = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, reqURL, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Google API request failed with status %d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusNoContent {
		return map[string]any{}, nil
	}
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return nil, err
	}
	return obj, nil
}

var (
	_ tools.Tool = (*SearchDriveTool)(nil)
	_ tools.Tool = (*ListDriveFilesTool)(nil)
	_ tools.Tool = (*GetFileInfoTool)(nil)
	_ tools.Tool = (*ReadFileContentTool)(nil)
	_ tools.Tool = (*ListSharedDrivesTool)(nil)
	_ tools.Tool = (*CreateDocumentTool)(nil)
	_ tools.Tool = (*CreateSpreadsheetTool)(nil)
	_ tools.Tool = (*CreatePresentationTool)(nil)
	_ tools.Tool = (*ShareFileTool)(nil)
	_ tools.Tool = (*ListPermissionsTool)(nil)
	_ tools.Tool = (*UpdatePermissionTool)(nil)
	_ tools.Tool = (*RemoveSharingTool)(nil)
	_ tools.Tool = (*MakePublicTool)(nil)
	_ tools.Tool = (*MoveFileTool)(nil)
	_ tools.Tool = (*RenameFileTool)(nil)
	_ tools.Tool = (*CopyFileTool)(nil)
	_ tools.Tool = (*CreateFolderTool)(nil)
	_ tools.Tool = (*UpdateDocumentTool)(nil)
	_ tools.Tool = (*RefreshGoogleDriveTokenTool)(nil)
)
