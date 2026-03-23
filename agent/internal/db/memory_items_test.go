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

	if _, err := store.UpsertMemoryItem(ctx, MemoryItemUpsert{
		UserID:     "user-1",
		Kind:       "fact",
		Category:   "profile",
		Content:    "User works on Aether",
		SourceType: "conversation",
	}); err != nil {
		t.Fatalf("insert first memory item: %v", err)
	}
	id, err := store.UpsertMemoryItem(ctx, MemoryItemUpsert{
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

	_, err := store.UpsertMemoryItem(ctx, MemoryItemUpsert{
		UserID:     "user-1",
		Kind:       "decision",
		Category:   "preference",
		Content:    "User prefers Bun over npm",
		SourceType: "manual",
	})
	if err != nil {
		t.Fatalf("insert canonical decision: %v", err)
	}

	results, err := store.SearchMemoryHybrid(ctx, "user-1", "bun preference", nil, 5)
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

func TestArchiveStaleMemoryItems(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	observedAt := time.Now().UTC().Add(-60 * 24 * time.Hour)
	_, err := store.UpsertMemoryItem(ctx, MemoryItemUpsert{
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
