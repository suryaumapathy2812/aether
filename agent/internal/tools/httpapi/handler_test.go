package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/suryaumapathy/core-ai/agent/internal/db"
	"github.com/suryaumapathy/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy/core-ai/agent/internal/tools"
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
	builtin := filepath.Join(assets, "plugins", "builtin")
	if err := os.MkdirAll(filepath.Join(builtin, "local-search"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	manifest := "name: local-search\ndisplay_name: Local Search\ndescription: test\nauth:\n  type: api_key\n  config_fields:\n    - key: google_api_key\n      required: true\n"
	if err := os.WriteFile(filepath.Join(builtin, "local-search", "plugin.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	store, err := db.Open(filepath.Join(assets, "state.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	pm := plugins.NewManager(plugins.ManagerOptions{BuiltinDirs: []string{builtin}, StateStore: store})
	if _, err := pm.Discover(ctx); err != nil {
		t.Fatalf("discover: %v", err)
	}

	r := tools.NewRegistry()
	o := tools.NewOrchestrator(r, tools.ExecContext{Store: store, Plugins: pm})
	h := New(Options{Registry: r, Orchestrator: o, Plugins: pm, Store: store})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/internal/plugins/status", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var resp struct {
		Plugins []map[string]any `json:"plugins"`
		Count   int              `json:"count"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Count != 1 {
		t.Fatalf("expected one plugin, got %d", resp.Count)
	}
}
