package plugins

import (
	"context"
	"testing"
)

func TestLocalSearch_EmptyQueryFails(t *testing.T) {
	result := (&LocalSearchTool{}).Execute(context.Background(), makeCall(nil, map[string]any{"query": "   "}))
	if !result.Error {
		t.Fatal("expected empty query to fail")
	}
	if result.Output != "Search query is required" {
		t.Fatalf("expected empty query error, got %q", result.Output)
	}
}
