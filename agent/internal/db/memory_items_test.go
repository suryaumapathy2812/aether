package db

import (
	"context"
	"testing"
	"time"
)

func TestUpsertMemoryItemPrefersRicherContent(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	if _, err := store.AddMemory(ctx, AddMemoryInput{
		UserID:     "user-1",
		Kind:       "fact",
		Category:   "profile",
		Content:    "User works on Aether",
		SourceType: "conversation",
	}); err != nil {
		t.Fatalf("insert first memory item: %v", err)
	}
	id, err := store.AddMemory(ctx, AddMemoryInput{
		UserID:     "user-1",
		Kind:       "fact",
		Category:   "profile",
		Content:    "User works on the Aether AI agent project",
		SourceType: "conversation",
	})
	if err != nil {
		t.Fatalf("upsert richer memory item: %v", err)
	}
	if id == 0 {
		t.Fatal("expected canonical item id")
	}

	item, err := store.findMemoryItemMergeCandidate(ctx, "user-1", "fact", "User works on the Aether AI agent project")
	if err != nil {
		t.Fatalf("load canonical memory item: %v", err)
	}
	if item == nil {
		t.Fatal("expected merge candidate")
	}
	if item.Content != "User works on the Aether AI agent project" {
		t.Fatalf("expected richer content to win, got %q", item.Content)
	}
	if item.EvidenceCount != 2 {
		t.Fatalf("expected evidence count 2, got %d", item.EvidenceCount)
	}
}

func TestSearchMemoryHybridUsesCanonicalFTS(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	_, err := store.AddMemory(ctx, AddMemoryInput{
		UserID:     "user-1",
		Kind:       "decision",
		Category:   "preference",
		Content:    "User prefers Bun over npm",
		SourceType: "manual",
	})
	if err != nil {
		t.Fatalf("insert canonical decision: %v", err)
	}

	results, err := store.SearchMemory(ctx, MemorySearchQuery{UserID: "user-1", Text: "bun preference", Limit: 5})
	if err != nil {
		t.Fatalf("search hybrid: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected canonical memory result")
	}
	if results[0].Type != "decision" {
		t.Fatalf("expected decision result, got %q", results[0].Type)
	}
	if results[0].Decision != "User prefers Bun over npm" {
		t.Fatalf("unexpected decision result: %q", results[0].Decision)
	}
}

func TestLegacyReadAPIsUseCanonicalMemoryItems(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	_, err := store.AddMemory(ctx, AddMemoryInput{UserID: "user-1", Kind: "fact", Category: "profile", Content: "User prefers concise answers", SourceType: "manual"})
	if err != nil {
		t.Fatalf("insert fact item: %v", err)
	}
	_, err = store.AddMemory(ctx, AddMemoryInput{UserID: "user-1", Kind: "memory", Category: "episodic", Content: "User debugged Caddy routing", SourceType: "manual"})
	if err != nil {
		t.Fatalf("insert memory item: %v", err)
	}
	_, err = store.AddMemory(ctx, AddMemoryInput{UserID: "user-1", Kind: "decision", Category: "preference", Content: "Always use Bun when possible", SourceType: "manual"})
	if err != nil {
		t.Fatalf("insert decision item: %v", err)
	}

	facts, err := store.ListMemoryItems(ctx, MemoryListQuery{UserID: "user-1", Kinds: []string{"fact"}, Status: "active", Limit: 10})
	if err != nil {
		t.Fatalf("list facts: %v", err)
	}
	if len(facts) != 1 || facts[0].Content != "User prefers concise answers" {
		t.Fatalf("unexpected facts: %#v", facts)
	}

	memories, err := store.ListMemoryItems(ctx, MemoryListQuery{UserID: "user-1", Kinds: []string{"memory"}, Status: "active", Limit: 10})
	if err != nil {
		t.Fatalf("list memories: %v", err)
	}
	if len(memories) != 1 || memories[0].Content != "User debugged Caddy routing" {
		t.Fatalf("unexpected memories: %#v", memories)
	}

	decisions, err := store.ListMemoryItems(ctx, MemoryListQuery{UserID: "user-1", Kinds: []string{"decision"}, Status: "active", Limit: 10})
	if err != nil {
		t.Fatalf("list decisions: %v", err)
	}
	if len(decisions) != 1 || decisions[0].Content != "Always use Bun when possible" {
		t.Fatalf("unexpected decisions: %#v", decisions)
	}

	results, err := store.SearchMemory(ctx, MemorySearchQuery{UserID: "user-1", Text: "bun routing concise", Limit: 10})
	if err != nil {
		t.Fatalf("search memory: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected canonical search results")
	}
}

func TestMigrateBackfillsLegacyTablesIntoCanonicalMemoryItems(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()
	if _, err := store.db.ExecContext(ctx, `
		CREATE TABLE facts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			fact TEXT NOT NULL,
			fact_key TEXT NOT NULL,
			source_conversation_id INTEGER,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			UNIQUE(user_id, fact_key)
		)
	`); err != nil {
		t.Fatalf("create legacy facts table: %v", err)
	}

	if _, err := store.db.ExecContext(ctx, `
		INSERT INTO facts(user_id, fact, fact_key, source_conversation_id, created_at, updated_at)
		VALUES('user-1', 'User likes Go', 'user likes go', 0, ?, ?)
	`, formatTS(time.Now().UTC()), formatTS(time.Now().UTC())); err != nil {
		t.Fatalf("insert legacy fact: %v", err)
	}
	if err := store.backfillLegacyMemoryItems(ctx); err != nil {
		t.Fatalf("backfill legacy memory: %v", err)
	}
	facts, err := store.ListMemoryItems(ctx, MemoryListQuery{UserID: "user-1", Kinds: []string{"fact"}, Status: "active", Limit: 10})
	if err != nil {
		t.Fatalf("list canonical facts after backfill: %v", err)
	}
	if len(facts) == 0 || facts[0].Content != "User likes Go" {
		t.Fatalf("expected backfilled fact, got %#v", facts)
	}
}

func TestArchiveStaleMemoryItems(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	observedAt := time.Now().UTC().Add(-60 * 24 * time.Hour)
	_, err := store.AddMemory(ctx, AddMemoryInput{
		UserID:     "user-1",
		Kind:       "memory",
		Category:   "episodic",
		Content:    "User mentioned a one-off setup issue",
		Confidence: 0.4,
		ObservedAt: observedAt,
		SourceType: "conversation",
	})
	if err != nil {
		t.Fatalf("insert stale memory: %v", err)
	}

	archived, err := store.ArchiveStaleMemoryItems(ctx, "user-1", time.Now().UTC().Add(-45*24*time.Hour), true)
	if err != nil {
		t.Fatalf("archive stale items: %v", err)
	}
	if archived == 0 {
		t.Fatal("expected stale item to be archived")
	}

	item, err := store.getMemoryItemByKey(ctx, "user-1", "memory", normalizeMemoryKey("User mentioned a one-off setup issue"))
	if err != nil {
		t.Fatalf("load archived item: %v", err)
	}
	if item.Status != "archived" {
		t.Fatalf("expected archived status, got %q", item.Status)
	}
}
