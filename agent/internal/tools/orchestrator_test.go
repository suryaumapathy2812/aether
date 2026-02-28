package tools

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/suryaumapathy/core-ai/agent/internal/db"
	"github.com/suryaumapathy/core-ai/agent/internal/plugins"
)

type pluginEchoTool struct{}

func (t *pluginEchoTool) Definition() Definition {
	return Definition{Name: "plugin_echo", Description: "echo", Parameters: []Param{{Name: "v", Type: "string", Required: true}}}
}

func (t *pluginEchoTool) Execute(ctx context.Context, call Call) Result {
	return Success(call.Args["v"].(string), nil)
}

func TestOrchestratorBlocksDisabledPlugin(t *testing.T) {
	store := openToolStore(t)
	defer store.Close()
	ctx := context.Background()
	if err := store.UpsertPlugin(ctx, db.PluginRecord{Name: "brave-search", DisplayName: "Brave", Enabled: false}); err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}
	pm := plugins.NewManager(plugins.ManagerOptions{})
	pm.AttachDirectory(t.TempDir(), plugins.SourceBuiltin)
	r := NewRegistry()
	_ = r.Register(&pluginEchoTool{}, "brave-search")
	o := NewOrchestrator(r, ExecContext{Store: store, Plugins: pm})
	res := o.Execute(ctx, "plugin_echo", map[string]any{"v": "x"}, "c1")
	if !res.Error {
		t.Fatalf("expected disabled plugin block")
	}
}

func openToolStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return store
}
