package tools

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

type echoTool struct{}

func (t *echoTool) Definition() Definition {
	return Definition{Name: "echo", Description: "echo", Parameters: []Param{{Name: "v", Type: "string", Required: true}}}
}

func (t *echoTool) Execute(ctx context.Context, call Call) Result {
	return Success(call.Args["v"].(string), nil)
}

func TestOrchestratorExecutesTool(t *testing.T) {
	store := openToolStore(t)
	defer store.Close()

	r := NewRegistry()
	_ = r.Register(&echoTool{}, "")
	o := NewOrchestrator(r, ExecContext{Store: store})
	res := o.Execute(context.Background(), "echo", map[string]any{"v": "hello"}, "c1")
	if res.Error {
		t.Fatalf("expected success, got error: %s", res.Output)
	}
	if res.Output != "hello" {
		t.Fatalf("expected 'hello', got %q", res.Output)
	}
}

func TestOrchestratorUnknownTool(t *testing.T) {
	store := openToolStore(t)
	defer store.Close()

	r := NewRegistry()
	o := NewOrchestrator(r, ExecContext{Store: store})
	res := o.Execute(context.Background(), "nonexistent", map[string]any{}, "c1")
	if !res.Error {
		t.Fatalf("expected error for unknown tool")
	}
}

func openToolStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path, "")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return store
}
