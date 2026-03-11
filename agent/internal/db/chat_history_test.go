package db

import (
	"context"
	"testing"
)

func TestChatMessageRoundTrip(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()

	err := store.AppendChatMessage(ctx, "u1", "s1", map[string]any{
		"role":    "user",
		"content": "hello",
	})
	if err != nil {
		t.Fatalf("append user message: %v", err)
	}
	err = store.AppendChatMessage(ctx, "u1", "s1", map[string]any{
		"role": "assistant",
		"tool_calls": []map[string]any{{
			"id":   "call_1",
			"type": "function",
			"function": map[string]any{
				"name":      "echo",
				"arguments": "{\"text\":\"hi\"}",
			},
		}},
	})
	if err != nil {
		t.Fatalf("append assistant tool call: %v", err)
	}
	err = store.AppendChatMessage(ctx, "u1", "s1", map[string]any{
		"role":         "tool",
		"tool_call_id": "call_1",
		"content":      "tool:hi",
	})
	if err != nil {
		t.Fatalf("append tool result: %v", err)
	}

	messages, err := store.ListChatMessages(ctx, "u1", "s1", 10)
	if err != nil {
		t.Fatalf("list chat messages: %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}
	if got := messages[0].Content["role"]; got != "user" {
		t.Fatalf("expected first role user, got %v", got)
	}
	if got := messages[1].Content["role"]; got != "assistant" {
		t.Fatalf("expected second role assistant, got %v", got)
	}
	if got := messages[2].Content["role"]; got != "tool" {
		t.Fatalf("expected third role tool, got %v", got)
	}
}
