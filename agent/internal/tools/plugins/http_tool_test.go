package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	coreplugins "github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// ── Helpers ────────────────────────────────────────────

type mockPluginState struct {
	cfg map[string]string
}

func (m *mockPluginState) Ensure(_ context.Context, _ db.PluginRecord) error { return nil }
func (m *mockPluginState) Enabled(_ context.Context) (bool, error)           { return true, nil }
func (m *mockPluginState) SetEnabled(_ context.Context, _ bool) error        { return nil }
func (m *mockPluginState) Config(_ context.Context) (map[string]string, error) {
	if m.cfg == nil {
		return map[string]string{}, nil
	}
	return m.cfg, nil
}
func (m *mockPluginState) SetConfig(_ context.Context, cfg map[string]string) error {
	m.cfg = cfg
	return nil
}
func (m *mockPluginState) EncryptString(s string) (string, error) { return s, nil }
func (m *mockPluginState) DecryptString(s string) (string, error) { return s, nil }

func makeCall(state *mockPluginState, args map[string]any) tools.Call {
	return tools.Call{
		ID:   "call-1",
		Args: args,
		Ctx:  tools.ExecContext{PluginState: state},
	}
}

func makeManifest(auth coreplugins.ManifestAuth, baseURL string) coreplugins.PluginManifest {
	return coreplugins.PluginManifest{
		Name:        "test-plugin",
		DisplayName: "Test Plugin",
		Auth:        auth,
		API:         coreplugins.ManifestAPI{BaseURL: baseURL},
	}
}

// ── Tests ──────────────────────────────────────────────

func TestHTTPTool_Definition(t *testing.T) {
	manifest := makeManifest(coreplugins.ManifestAuth{Type: "none"}, "https://example.com")
	toolDef := coreplugins.ManifestTool{
		Name:        "test_tool",
		Description: "A test tool",
		StatusText:  "Testing...",
		HTTP:        coreplugins.ManifestHTTP{Method: "GET", Path: "/test"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "query", Type: "string", Description: "Search query", Required: true},
			{Name: "limit", Type: "integer", Required: false, Default: 10},
		},
	}

	ht := NewHTTPTool("test-plugin", manifest, toolDef)
	def := ht.Definition()

	if def.Name != "test_tool" {
		t.Errorf("expected name test_tool, got %s", def.Name)
	}
	if def.Description != "A test tool" {
		t.Errorf("expected description 'A test tool', got %s", def.Description)
	}
	if def.StatusText != "Testing..." {
		t.Errorf("expected status text 'Testing...', got %s", def.StatusText)
	}
	if len(def.Parameters) != 2 {
		t.Fatalf("expected 2 params, got %d", len(def.Parameters))
	}
	if def.Parameters[0].Name != "query" || !def.Parameters[0].Required {
		t.Errorf("first param should be query (required)")
	}
	if def.Parameters[1].Default != 10 {
		t.Errorf("second param should default to 10")
	}
}

func TestHTTPTool_GET_NoAuth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/items" {
			t.Errorf("expected /items, got %s", r.URL.Path)
		}
		if r.URL.Query().Get("q") != "test" {
			t.Errorf("expected q=test, got %s", r.URL.Query().Get("q"))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"items": []any{"a", "b", "c"},
		})
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{Type: "none"}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "list_items",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/items"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "query", Type: "string", Required: true, MapTo: "query.q"},
		},
		Response: coreplugins.ManifestResponse{Extract: "items"},
	}

	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"query": "test"}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, `"a"`) {
		t.Errorf("expected extracted items, got: %s", result.Output)
	}
}

func TestHTTPTool_PathTemplating(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages/msg-123" {
			t.Errorf("expected /messages/msg-123, got %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{"id": "msg-123", "body": "hello"})
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{Type: "none"}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "read_msg",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/messages/{{message_id}}"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "message_id", Type: "string", Required: true},
		},
	}

	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"message_id": "msg-123"}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "msg-123") {
		t.Errorf("expected msg-123 in response, got: %s", result.Output)
	}
}

func TestHTTPTool_POST_WithBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["title"] != "Test Event" {
			t.Errorf("expected title 'Test Event', got %v", body["title"])
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "evt-1"})
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{Type: "none"}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "create_item",
		HTTP: coreplugins.ManifestHTTP{Method: "POST", Path: "/items"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "title", Type: "string", Required: true, MapTo: "body.title"},
		},
		Response: coreplugins.ManifestResponse{SuccessMsg: "Created."},
	}

	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"title": "Test Event"}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Created.") {
		t.Errorf("expected success message, got: %s", result.Output)
	}
}

func TestHTTPTool_Auth_APIKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Header.Get("X-Api-Key")
		if key != "secret-key-123" {
			t.Errorf("expected api key header, got: %s", key)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"ok": true})
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{
		Type:      "api_key",
		ConfigKey: "api_key",
	}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "search",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/search"},
	}

	state := &mockPluginState{cfg: map[string]string{"api_key": "secret-key-123"}}
	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(state, map[string]any{}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

func TestHTTPTool_Auth_OAuth2_Bearer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer my-access-token" {
			t.Errorf("expected Bearer token, got: %s", auth)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"messages": []any{}})
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{
		Type:     "oauth2",
		TokenURL: "https://oauth.example.com/token",
	}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "list_items",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/items"},
	}

	state := &mockPluginState{cfg: map[string]string{"access_token": "my-access-token"}}
	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(state, map[string]any{}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

func TestHTTPTool_Auth_OAuth2_AutoRefreshOn401(t *testing.T) {
	var callCount int32

	// Token refresh server
	tokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "new-token",
			"expires_in":   3600,
			"token_type":   "Bearer",
		})
	}))
	defer tokenSrv.Close()

	// API server: first call returns 401, second succeeds
	apiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		if n == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"ok": true})
	}))
	defer apiSrv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{
		Type:        "oauth2",
		TokenURL:    tokenSrv.URL,
		AutoRefresh: true,
	}, apiSrv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "get_data",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/data"},
	}

	state := &mockPluginState{cfg: map[string]string{
		"access_token":  "expired-token",
		"refresh_token": "my-refresh-token",
		"client_id":     "cid",
		"client_secret": "csecret",
	}}

	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(state, map[string]any{}))

	if result.Error {
		t.Fatalf("expected success after token refresh, got error: %s", result.Output)
	}
	if atomic.LoadInt32(&callCount) != 2 {
		t.Errorf("expected 2 API calls (first 401, then retry), got %d", callCount)
	}
}

func TestHTTPTool_Auth_MissingConfig(t *testing.T) {
	manifest := makeManifest(coreplugins.ManifestAuth{
		Type:      "api_key",
		ConfigKey: "api_key",
	}, "https://example.com")
	toolDef := coreplugins.ManifestTool{
		Name: "search",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/search"},
	}

	// Empty config — no api_key
	state := &mockPluginState{cfg: map[string]string{}}
	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(state, map[string]any{}))

	if !result.Error {
		t.Fatal("expected error for missing api_key")
	}
	if !strings.Contains(result.Output, "not connected") {
		t.Errorf("expected 'not connected' message, got: %s", result.Output)
	}
}

func TestHTTPTool_StaticQuery(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("format") != "full" {
			t.Errorf("expected static query format=full, got %s", r.URL.Query().Get("format"))
		}
		if r.URL.Query().Get("q") != "is:unread" {
			t.Errorf("expected static query q=is:unread, got %s", r.URL.Query().Get("q"))
		}
		json.NewEncoder(w).Encode(map[string]any{"items": []any{}})
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{Type: "none"}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "list",
		HTTP: coreplugins.ManifestHTTP{
			Method: "GET",
			Path:   "/items",
			Query:  map[string]string{"format": "full", "q": "is:unread"},
		},
	}

	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

func TestHTTPTool_GoogleCalendarUpcomingEvents_AppliesTimeWindowAndFiltersOutput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("days") != "" {
			t.Errorf("expected days to be omitted from upstream query, got %q", r.URL.Query().Get("days"))
		}
		if r.URL.Query().Get("singleEvents") != "true" {
			t.Errorf("expected singleEvents=true, got %q", r.URL.Query().Get("singleEvents"))
		}
		if r.URL.Query().Get("orderBy") != "startTime" {
			t.Errorf("expected orderBy=startTime, got %q", r.URL.Query().Get("orderBy"))
		}

		minRaw := r.URL.Query().Get("timeMin")
		maxRaw := r.URL.Query().Get("timeMax")
		if strings.TrimSpace(minRaw) == "" || strings.TrimSpace(maxRaw) == "" {
			t.Fatalf("expected timeMin/timeMax in upstream query")
		}

		minTime, err := time.Parse(time.RFC3339, minRaw)
		if err != nil {
			t.Fatalf("invalid timeMin: %v", err)
		}
		insideStart := minTime.Add(2 * time.Hour).Format(time.RFC3339)
		outsideStart := minTime.AddDate(0, 0, -10).Format(time.RFC3339)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"items": []any{
				map[string]any{"summary": "Inside Event", "start": map[string]any{"dateTime": insideStart}},
				map[string]any{"summary": "Outside Event", "start": map[string]any{"dateTime": outsideStart}},
			},
		})
	}))
	defer srv.Close()

	manifest := coreplugins.PluginManifest{
		Name:        "google-calendar",
		DisplayName: "Google Calendar",
		Auth:        coreplugins.ManifestAuth{Type: "none"},
		API:         coreplugins.ManifestAPI{BaseURL: srv.URL},
	}
	toolDef := coreplugins.ManifestTool{
		Name: "upcoming_events",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/calendars/primary/events", Query: map[string]string{"singleEvents": "true", "orderBy": "startTime"}},
		Parameters: []coreplugins.ManifestParam{
			{Name: "days", Type: "integer", Required: false, Default: 7},
			{Name: "max_results", Type: "integer", Required: false, Default: 10, MapTo: "query.maxResults"},
		},
		Response: coreplugins.ManifestResponse{Extract: "items"},
	}

	ht := NewHTTPTool("google-calendar", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"days": 2, "max_results": 10}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Inside Event") {
		t.Fatalf("expected inside event in output, got: %s", result.Output)
	}
	if strings.Contains(result.Output, "Outside Event") {
		t.Fatalf("expected outside event to be filtered out, got: %s", result.Output)
	}
}

func TestHTTPTool_GoogleDriveListDriveFiles_UsesFolderParentQuery(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		if q != "'folder-123' in parents and trashed = false" {
			t.Fatalf("expected parent query filter, got %q", q)
		}
		if got := r.URL.Query().Get("folder_id"); got != "" {
			t.Fatalf("expected folder_id to be removed from query, got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"files": []any{map[string]any{"id": "f1", "name": "Doc"}}})
	}))
	defer srv.Close()

	manifest := coreplugins.PluginManifest{
		Name:        "google-drive",
		DisplayName: "Google Drive",
		Auth:        coreplugins.ManifestAuth{Type: "none"},
		API:         coreplugins.ManifestAPI{BaseURL: srv.URL},
	}
	toolDef := coreplugins.ManifestTool{
		Name: "list_drive_files",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/files"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "folder_id", Type: "string", Required: false, Default: "root"},
		},
		Response: coreplugins.ManifestResponse{Extract: "files"},
	}

	ht := NewHTTPTool("google-drive", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"folder_id": "folder-123"}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Doc") {
		t.Fatalf("expected extracted file, got: %s", result.Output)
	}
}

func TestHTTPTool_GoogleDriveCreateFolder_UsesParentsArrayAndFolderMimeType(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if body["mimeType"] != "application/vnd.google-apps.folder" {
			t.Fatalf("expected folder mimeType, got %v", body["mimeType"])
		}
		parents, ok := body["parents"].([]any)
		if !ok || len(parents) != 1 || parents[0] != "parent-456" {
			t.Fatalf("expected parents [parent-456], got %#v", body["parents"])
		}
		if _, exists := body["parent_id"]; exists {
			t.Fatalf("did not expect parent_id field in request body")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "new-folder"})
	}))
	defer srv.Close()

	manifest := coreplugins.PluginManifest{
		Name:        "google-drive",
		DisplayName: "Google Drive",
		Auth:        coreplugins.ManifestAuth{Type: "none"},
		API:         coreplugins.ManifestAPI{BaseURL: srv.URL},
	}
	toolDef := coreplugins.ManifestTool{
		Name: "create_folder",
		HTTP: coreplugins.ManifestHTTP{Method: "POST", Path: "/files"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "name", Type: "string", Required: true, MapTo: "body.name"},
			{Name: "parent_id", Type: "string", Required: false, Default: "root"},
		},
	}

	ht := NewHTTPTool("google-drive", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"name": "Plans", "parent_id": "parent-456"}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

func TestHTTPTool_SpotifyPlayPause_UsesPauseEndpointForPauseAction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			t.Fatalf("expected PUT, got %s", r.Method)
		}
		if r.URL.Path != "/me/player/pause" {
			t.Fatalf("expected pause endpoint, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	manifest := coreplugins.PluginManifest{
		Name:        "spotify",
		DisplayName: "Spotify",
		Auth:        coreplugins.ManifestAuth{Type: "none"},
		API:         coreplugins.ManifestAPI{BaseURL: srv.URL},
	}
	toolDef := coreplugins.ManifestTool{
		Name: "play_pause",
		HTTP: coreplugins.ManifestHTTP{Method: "PUT", Path: "/me/player/play"},
		Parameters: []coreplugins.ManifestParam{
			{Name: "action", Type: "string", Required: false, Default: "toggle"},
		},
	}

	ht := NewHTTPTool("spotify", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{"action": "pause"}))

	if result.Error {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

func TestHTTPTool_ErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"server error"}`))
	}))
	defer srv.Close()

	manifest := makeManifest(coreplugins.ManifestAuth{Type: "none"}, srv.URL)
	toolDef := coreplugins.ManifestTool{
		Name: "failing_tool",
		HTTP: coreplugins.ManifestHTTP{Method: "GET", Path: "/fail"},
	}

	ht := NewHTTPTool("test", manifest, toolDef)
	result := ht.Execute(context.Background(), makeCall(nil, map[string]any{}))

	if !result.Error {
		t.Fatal("expected error for 500 response")
	}
	if !strings.Contains(result.Output, "500") {
		t.Errorf("expected status 500 in error, got: %s", result.Output)
	}
}

// ── Transform Tests ────────────────────────────────────

func TestTransform_MIMEMessage(t *testing.T) {
	result, err := applyTransform("mime_message", map[string]any{
		"to":      "alice@example.com",
		"subject": "Hello",
		"body":    "Hi Alice!",
		"cc":      "bob@example.com",
	}, nil)
	if err != nil {
		t.Fatalf("transform error: %v", err)
	}
	raw, ok := result["raw"].(string)
	if !ok || raw == "" {
		t.Fatal("expected raw field in result")
	}
}

func TestTransform_MIMEReply(t *testing.T) {
	result, err := applyTransform("mime_reply", map[string]any{
		"to":        "alice@example.com",
		"subject":   "Re: Hello",
		"body":      "Thanks!",
		"thread_id": "thread-123",
	}, nil)
	if err != nil {
		t.Fatalf("transform error: %v", err)
	}
	if result["threadId"] != "thread-123" {
		t.Errorf("expected threadId, got %v", result["threadId"])
	}
}

func TestTransform_MIMEDraft(t *testing.T) {
	result, err := applyTransform("mime_draft", map[string]any{
		"to":      "alice@example.com",
		"subject": "Draft",
		"body":    "Draft body",
	}, nil)
	if err != nil {
		t.Fatalf("transform error: %v", err)
	}
	msg, ok := result["message"].(map[string]any)
	if !ok {
		t.Fatal("expected message wrapper for draft")
	}
	if _, ok := msg["raw"]; !ok {
		t.Fatal("expected raw field inside message")
	}
}

func TestTransform_CalendarEvent(t *testing.T) {
	result, err := applyTransform("calendar_event", map[string]any{
		"summary":    "Team Standup",
		"start_time": "2026-03-16T10:00:00",
		"end_time":   "2026-03-16T10:30:00",
		"location":   "Room 5",
	}, nil)
	if err != nil {
		t.Fatalf("transform error: %v", err)
	}
	if result["summary"] != "Team Standup" {
		t.Errorf("expected summary, got %v", result["summary"])
	}
	start, ok := result["start"].(map[string]any)
	if !ok {
		t.Fatal("expected start object")
	}
	if start["dateTime"] != "2026-03-16T10:00:00" {
		t.Errorf("expected start dateTime, got %v", start["dateTime"])
	}
	if result["location"] != "Room 5" {
		t.Errorf("expected location, got %v", result["location"])
	}
}

func TestTransform_Unknown(t *testing.T) {
	body := map[string]any{"key": "value"}
	result, err := applyTransform("nonexistent", nil, body)
	if err != nil {
		t.Fatalf("unknown transform should not error: %v", err)
	}
	if result["key"] != "value" {
		t.Errorf("unknown transform should pass through body unchanged")
	}
}
