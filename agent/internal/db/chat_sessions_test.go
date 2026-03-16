package db

import (
	"context"
	"testing"
)

func TestChatSessionCRUD(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()

	// Create
	sess, err := store.CreateChatSession(ctx, "user1", "Hello world")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if sess.ID == "" {
		t.Fatal("expected non-empty ID")
	}
	if sess.Title != "Hello world" {
		t.Fatalf("expected title 'Hello world', got %q", sess.Title)
	}
	if sess.UserID != "user1" {
		t.Fatalf("expected user_id 'user1', got %q", sess.UserID)
	}

	// Get
	got, err := store.GetChatSession(ctx, sess.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if got.ID != sess.ID || got.Title != sess.Title {
		t.Fatalf("get mismatch: %+v", got)
	}

	// List
	_, _ = store.CreateChatSession(ctx, "user1", "Second chat")
	sessions, err := store.ListChatSessions(ctx, "user1", 10)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(sessions) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(sessions))
	}
	// Should be ordered by updated_at DESC — most recent first.
	if sessions[0].Title != "Second chat" {
		t.Fatalf("expected most recent first, got %q", sessions[0].Title)
	}

	// Update title
	if err := store.UpdateChatSessionTitle(ctx, sess.ID, "Renamed chat"); err != nil {
		t.Fatalf("update title: %v", err)
	}
	updated, _ := store.GetChatSession(ctx, sess.ID)
	if updated.Title != "Renamed chat" {
		t.Fatalf("expected 'Renamed chat', got %q", updated.Title)
	}

	// Touch
	if err := store.TouchChatSession(ctx, sess.ID); err != nil {
		t.Fatalf("touch: %v", err)
	}

	// Delete
	if err := store.DeleteChatSession(ctx, sess.ID); err != nil {
		t.Fatalf("delete: %v", err)
	}
	_, err = store.GetChatSession(ctx, sess.ID)
	if err == nil {
		t.Fatal("expected error after delete")
	}

	// Verify messages are also deleted.
	_ = store.AppendChatMessage(ctx, "user1", sessions[1].ID, map[string]any{"role": "user", "content": "test"})
	msgs, _ := store.ListChatMessages(ctx, "user1", sessions[1].ID, 100)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message before delete, got %d", len(msgs))
	}
	_ = store.DeleteChatSession(ctx, sessions[1].ID)
	msgs, _ = store.ListChatMessages(ctx, "user1", sessions[1].ID, 100)
	if len(msgs) != 0 {
		t.Fatalf("expected 0 messages after delete, got %d", len(msgs))
	}
}

func TestChatSessionDefaultTitle(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()

	sess, err := store.CreateChatSession(ctx, "user1", "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if sess.Title != "" {
		t.Fatalf("expected empty title, got %q", sess.Title)
	}
}

func TestChatSessionListEmpty(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()

	sessions, err := store.ListChatSessions(ctx, "nobody", 10)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(sessions) != 0 {
		t.Fatalf("expected 0 sessions, got %d", len(sessions))
	}
}
