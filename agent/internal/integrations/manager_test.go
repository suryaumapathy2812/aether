package integrations

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

func TestManagerDiscoverSearchRemove(t *testing.T) {
	root := t.TempDir()
	builtin := filepath.Join(root, "builtin")
	user := filepath.Join(root, "user")

	mustWritePlugin(t, filepath.Join(builtin, "google-workspace", "integration.yaml"), map[string]any{
		"name":         "google-workspace",
		"display_name": "Google Workspace",
		"description":  "Gmail, Calendar, Drive and more",
		"tools":        []map[string]any{{"class": "SendEmailTool"}},
	})
	mustWritePlugin(t, filepath.Join(user, "weather", "integration.yaml"), map[string]any{
		"name":         "weather",
		"display_name": "Weather",
		"description":  "forecasts and current weather",
	})

	m := NewManager(ManagerOptions{BuiltinDirs: []string{builtin}, UserDir: user})
	if _, err := m.Discover(context.Background()); err != nil {
		t.Fatalf("discover failed: %v", err)
	}

	if _, ok := m.Get("google-workspace"); !ok {
		t.Fatalf("expected google-workspace plugin")
	}
	if len(m.Search("weather")) != 1 {
		t.Fatalf("expected one search match")
	}

	if err := m.Remove("google-workspace"); !errors.Is(err, ErrProtected) {
		t.Fatalf("expected protected error, got %v", err)
	}
	if err := m.Remove("weather"); err != nil {
		t.Fatalf("remove failed: %v", err)
	}
}

func TestManagerDuplicateRejected(t *testing.T) {
	root := t.TempDir()
	builtin := filepath.Join(root, "builtin")
	user := filepath.Join(root, "user")

	mustWritePlugin(t, filepath.Join(builtin, "one", "integration.yaml"), map[string]any{"name": "dup"})
	mustWritePlugin(t, filepath.Join(user, "two", "integration.yaml"), map[string]any{"name": "dup"})

	m := NewManager(ManagerOptions{BuiltinDirs: []string{builtin}, UserDir: user})
	_, err := m.Discover(context.Background())
	if !errors.Is(err, ErrDuplicateName) {
		t.Fatalf("expected duplicate error, got %v", err)
	}
}

func TestInstallFromSource(t *testing.T) {
	root := t.TempDir()
	external := filepath.Join(root, "external")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/owner/repo/main/integration.yaml":
			_, _ = w.Write([]byte("name: remote\ndisplay_name: Remote\ndescription: Installed plugin\n"))
		case "/owner/repo/main/SKILL.md":
			_, _ = w.Write([]byte("# Remote Plugin Skill"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	m := NewManager(ManagerOptions{ExternalDir: external, RawBaseURL: server.URL})
	_, _ = m.Discover(context.Background())

	res, err := m.InstallFromSource(context.Background(), "owner/repo")
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if res.Installed.Name != "remote" {
		t.Fatalf("unexpected install result: %#v", res)
	}
	if _, ok := m.Get("remote"); !ok {
		t.Fatalf("expected installed plugin indexed")
	}
}

func TestDiscoverSyncsPluginsToStateStore(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	builtin := filepath.Join(root, "builtin")
	statePath := filepath.Join(root, "state.db")

	mustWritePlugin(t, filepath.Join(builtin, "weather", "integration.yaml"), map[string]any{
		"name":         "weather",
		"display_name": "Weather",
		"description":  "Current weather and forecast",
		"version":      "0.1.0",
	})

	store, err := db.Open(statePath, "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	m := NewManager(ManagerOptions{BuiltinDirs: []string{builtin}, StateStore: store})
	if _, err := m.Discover(ctx); err != nil {
		t.Fatalf("discover failed: %v", err)
	}

	rec, err := store.GetPlugin(ctx, "weather")
	if err != nil {
		t.Fatalf("expected plugin in state store: %v", err)
	}
	if rec.Enabled {
		t.Fatalf("expected plugin default disabled in db")
	}
	if rec.DisplayName != "Weather" {
		t.Fatalf("unexpected plugin record: %#v", rec)
	}
}

func mustWritePlugin(t *testing.T, path string, manifest map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write failed: %v", err)
	}
}
