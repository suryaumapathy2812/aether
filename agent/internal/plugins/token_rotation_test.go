package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

func TestOAuthTokenRotatorUpdatesConfig(t *testing.T) {
	store := openTokenStore(t)
	defer store.Close()
	ctx := context.Background()
	if err := store.UpsertPlugin(ctx, db.PluginRecord{Name: "gmail", DisplayName: "Gmail", Enabled: true}); err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}
	_ = store.SetPluginEnabled(ctx, "gmail", true)
	ps := NewPluginState(store, "gmail")

	encRefresh, err := ps.EncryptString("refresh-abc")
	if err != nil {
		t.Fatalf("encrypt refresh: %v", err)
	}
	if err := ps.SetConfig(ctx, map[string]string{
		"client_id":     "cid",
		"client_secret": "csec",
		"refresh_token": encRefresh,
	}); err != nil {
		t.Fatalf("set config: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "new-access",
			"expires_in":   3600,
			"token_type":   "Bearer",
		})
	}))
	defer server.Close()

	rotator := oauthTokenRotator(server.URL, false)
	if err := rotator(ctx, ps, map[string]any{}); err != nil {
		t.Fatalf("rotate token: %v", err)
	}

	cfg, err := ps.Config(ctx)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if strings.TrimSpace(cfg["access_token"]) == "" {
		t.Fatalf("expected access_token to be set")
	}
	if cfg["last_refresh_status"] != "ok" {
		t.Fatalf("expected status ok, got %q", cfg["last_refresh_status"])
	}
}

func openTokenStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path, "12345678901234567890123456789012")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return store
}
