package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/integrations"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type echoTool struct{}

func (t *echoTool) Definition() tools.Definition {
	return tools.Definition{Name: "echo", Description: "echo", Parameters: []tools.Param{{Name: "text", Type: "string", Required: true}}}
}

func (t *echoTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return tools.Success(call.Args["text"].(string), nil)
}

func TestExecuteEndpoint(t *testing.T) {
	r := tools.NewRegistry()
	_ = r.Register(&echoTool{}, "")
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	h := New(Options{Registry: r, Orchestrator: o})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{"name": "echo", "args": map[string]any{"text": "hi"}, "call_id": "c-1"})
	req := httptest.NewRequest(http.MethodPost, "/internal/tools/execute", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	result := resp["result"].(map[string]any)
	if result["Output"].(string) != "hi" {
		t.Fatalf("unexpected output: %#v", result)
	}
}

func TestPluginsStatusEndpoint(t *testing.T) {
	ctx := context.Background()
	assets := t.TempDir()
	builtin := filepath.Join(assets, "integrations", "builtin")
	if err := os.MkdirAll(filepath.Join(builtin, "local-search"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	manifest := "name: local-search\ndisplay_name: Local Search\ndescription: test\nauth:\n  type: api_key\n  config_fields:\n    - key: google_api_key\n      required: true\n"
	if err := os.WriteFile(filepath.Join(builtin, "local-search", "integration.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	store, err := db.Open(filepath.Join(assets, "state.db"), "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	pm := integrations.NewManager(integrations.ManagerOptions{BuiltinDirs: []string{builtin}, StateStore: store})
	if _, err := pm.Discover(ctx); err != nil {
		t.Fatalf("discover: %v", err)
	}

	r := tools.NewRegistry()
	o := tools.NewOrchestrator(r, tools.ExecContext{Store: store, Integrations: pm})
	h := New(Options{Registry: r, Orchestrator: o, Integrations: pm, Store: store})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/internal/integrations/status", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var resp struct {
		Plugins []map[string]any `json:"integrations"`
		Count   int              `json:"count"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Count != 1 {
		t.Fatalf("expected one integration, got %d", resp.Count)
	}
}

func TestPluginConfigMasksSecrets(t *testing.T) {
	ctx := context.Background()
	assets := t.TempDir()
	builtin := filepath.Join(assets, "integrations", "builtin")
	if err := os.MkdirAll(filepath.Join(builtin, "brave-search"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	manifest := "name: brave-search\ndisplay_name: Brave Search\ndescription: test\nauth:\n  type: api_key\n  config_fields:\n    - key: api_key\n      type: password\n      required: true\n"
	if err := os.WriteFile(filepath.Join(builtin, "brave-search", "integration.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	store, err := db.Open(filepath.Join(assets, "state.db"), "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	pm := integrations.NewManager(integrations.ManagerOptions{BuiltinDirs: []string{builtin}, StateStore: store})
	if _, err := pm.Discover(ctx); err != nil {
		t.Fatalf("discover: %v", err)
	}
	if err := store.SetPluginConfig(ctx, "brave-search", map[string]string{"api_key": "secret-123", "country": "IN"}); err != nil {
		t.Fatalf("set config: %v", err)
	}

	h := New(Options{Registry: tools.NewRegistry(), Orchestrator: tools.NewOrchestrator(tools.NewRegistry(), tools.ExecContext{}), Integrations: pm, Store: store})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/agent/v1/integrations/brave-search/config", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var out map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["api_key"] != maskedSecretValue {
		t.Fatalf("expected masked api_key, got %q", out["api_key"])
	}
	if out["country"] != "IN" {
		t.Fatalf("expected country passthrough, got %q", out["country"])
	}
}

func TestPluginOAuthStartRedirect(t *testing.T) {
	ctx := context.Background()
	assets := t.TempDir()
	builtin := filepath.Join(assets, "integrations", "builtin")
	if err := os.MkdirAll(filepath.Join(builtin, "google-workspace"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	manifest := "name: google-workspace\ndisplay_name: Google Workspace\ndescription: test\nauth:\n  type: oauth2\n  provider: google\n  scopes:\n    - https://www.googleapis.com/auth/calendar.readonly\n  config_fields:\n    - key: client_id\n      type: text\n      required: true\n    - key: client_secret\n      type: password\n      required: true\n"
	if err := os.WriteFile(filepath.Join(builtin, "google-workspace", "integration.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	store, err := db.Open(filepath.Join(assets, "state.db"), "12345678901234567890123456789012")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	pm := integrations.NewManager(integrations.ManagerOptions{BuiltinDirs: []string{builtin}, StateStore: store})
	if _, err := pm.Discover(ctx); err != nil {
		t.Fatalf("discover: %v", err)
	}
	if err := store.UpsertPlugin(ctx, db.PluginRecord{Name: "google-workspace", DisplayName: "Google Workspace"}); err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}
	secret, err := store.EncryptString("client-secret")
	if err != nil {
		t.Fatalf("encrypt secret: %v", err)
	}
	if err := store.SetPluginConfig(ctx, "google-workspace", map[string]string{"client_id": "client-id", "client_secret": secret}); err != nil {
		t.Fatalf("set config: %v", err)
	}

	r := tools.NewRegistry()
	h := New(Options{Registry: r, Orchestrator: tools.NewOrchestrator(r, tools.ExecContext{}), Integrations: pm, Store: store})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/agent/v1/integrations/google-workspace/oauth/start", nil)
	req.Header.Set("X-Forwarded-Host", "app.example.com")
	req.Header.Set("X-Forwarded-Proto", "https")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusFound {
		t.Fatalf("expected 302, got %d body=%s", w.Code, w.Body.String())
	}
	location := w.Header().Get("Location")
	if !strings.HasPrefix(location, "https://accounts.google.com/o/oauth2/v2/auth?") {
		t.Fatalf("unexpected oauth redirect: %s", location)
	}
	if !strings.Contains(location, "redirect_uri=https%3A%2F%2Fapp.example.com%2Fintegrations%2Fgoogle-workspace%2Foauth%2Fcallback") {
		t.Fatalf("redirect missing callback uri: %s", location)
	}
}

func TestPluginOAuthStartUsesProviderEnvCredentials(t *testing.T) {
	ctx := context.Background()
	assets := t.TempDir()
	builtin := filepath.Join(assets, "integrations", "builtin")
	if err := os.MkdirAll(filepath.Join(builtin, "spotify"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	manifest := "name: spotify\ndisplay_name: Spotify\ndescription: test\nauth:\n  type: oauth2\n  provider: spotify\n  scopes:\n    - user-read-email\n"
	if err := os.WriteFile(filepath.Join(builtin, "spotify", "integration.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	t.Setenv("SPOTIFY_CLIENT_ID", "env-client-id")
	t.Setenv("SPOTIFY_CLIENT_SECRET", "env-client-secret")
	store, err := db.Open(filepath.Join(assets, "state.db"), "12345678901234567890123456789012")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	pm := integrations.NewManager(integrations.ManagerOptions{BuiltinDirs: []string{builtin}, StateStore: store})
	if _, err := pm.Discover(ctx); err != nil {
		t.Fatalf("discover: %v", err)
	}

	r := tools.NewRegistry()
	h := New(Options{Registry: r, Orchestrator: tools.NewOrchestrator(r, tools.ExecContext{}), Integrations: pm, Store: store})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/agent/v1/integrations/spotify/oauth/start", nil)
	req.Header.Set("X-Forwarded-Host", "app.example.com")
	req.Header.Set("X-Forwarded-Proto", "https")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusFound {
		t.Fatalf("expected 302, got %d body=%s", w.Code, w.Body.String())
	}
	location := w.Header().Get("Location")
	if !strings.HasPrefix(location, "https://accounts.spotify.com/authorize?") {
		t.Fatalf("unexpected oauth redirect: %s", location)
	}

	rec, err := store.GetPlugin(ctx, "spotify")
	if err != nil {
		t.Fatalf("get plugin: %v", err)
	}
	if strings.TrimSpace(rec.Config["client_id"]) != "env-client-id" {
		t.Fatalf("expected env client_id persisted, got %q", rec.Config["client_id"])
	}
	secret, err := store.DecryptString(rec.Config["client_secret"])
	if err != nil {
		t.Fatalf("decrypt client_secret: %v", err)
	}
	if secret != "env-client-secret" {
		t.Fatalf("expected env client_secret persisted, got %q", secret)
	}
}
