package tools

import (
	"context"
	"testing"
)

type testTool struct{}

func (t *testTool) Definition() Definition {
	return Definition{
		Name:        "test_tool",
		Description: "test",
		StatusText:  "running",
		Parameters:  []Param{{Name: "name", Type: "string", Required: true}},
	}
}

func (t *testTool) Execute(ctx context.Context, call Call) Result {
	return Success("hello "+call.Args["name"].(string), nil)
}

func TestRegistryDispatch(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(&testTool{}, ""); err != nil {
		t.Fatalf("register: %v", err)
	}
	res := r.Dispatch(context.Background(), "test_tool", map[string]any{"name": "sam"}, ExecContext{})
	if res.Error || res.Output != "hello sam" {
		t.Fatalf("unexpected result: %#v", res)
	}
}

func TestRegistryUnknownTool(t *testing.T) {
	r := NewRegistry()
	res := r.Dispatch(context.Background(), "missing", map[string]any{}, ExecContext{})
	if !res.Error {
		t.Fatalf("expected error for unknown tool")
	}
}
