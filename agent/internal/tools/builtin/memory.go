package builtin

import (
	"context"
	"fmt"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type SaveMemoryTool struct{}
type SearchMemoryTool struct{}

func (t *SaveMemoryTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "save_memory",
		Description: "Save a durable memory about the user for future sessions.",
		StatusText:  "Saving to memory...",
		Parameters: []tools.Param{
			{Name: "content", Type: "string", Description: "The memory content to save.", Required: true},
			{Name: "category", Type: "string", Description: "Memory category", Required: false, Default: "episodic", Enum: []string{"episodic", "behavioral", "emotional", "preference"}},
		},
	}
}

func (t *SaveMemoryTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	content, _ := call.Args["content"].(string)
	content = strings.TrimSpace(content)
	if content == "" {
		return tools.Fail("Cannot save empty memory", nil)
	}
	category, _ := call.Args["category"].(string)
	if strings.TrimSpace(category) == "" {
		category = "episodic"
	}
	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	// Save to SQLite (existing behavior)
	if err := call.Ctx.Store.StoreMemory(ctx, userID, content, category, 0, nil); err != nil {
		return tools.Fail("Failed to save memory: "+err.Error(), nil)
	}
	if strings.EqualFold(category, "preference") {
		_ = call.Ctx.Store.StoreMemoryDecision(ctx, userID, content, "preference", "explicit", 0)
	}

	var embedding []float32
	if call.Ctx.EmbeddingProvider != nil {
		var err error
		embedding, err = call.Ctx.EmbeddingProvider.EmbedSingle(ctx, content)
		if err != nil {
			fmt.Printf("warning: failed to generate embedding for memory: %v\n", err)
		}
	}
	_, _ = call.Ctx.Store.UpsertMemoryItem(ctx, db.MemoryItemUpsert{
		UserID:     userID,
		Kind:       "memory",
		Category:   category,
		Content:    content,
		Confidence: 0.95,
		Importance: 0.75,
		SourceType: "manual",
		Embedding:  embedding,
	})
	if strings.EqualFold(category, "preference") {
		_, _ = call.Ctx.Store.UpsertMemoryItem(ctx, db.MemoryItemUpsert{
			UserID:     userID,
			Kind:       "decision",
			Category:   "preference",
			Content:    content,
			Confidence: 1.0,
			Importance: 0.95,
			SourceType: "manual",
			Embedding:  embedding,
		})
	}

	return tools.Success("Saved to memory.", map[string]any{"category": category})
}

func (t *SearchMemoryTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "search_memory",
		Description: "Search long-term memory for relevant context.",
		StatusText:  "Searching memory...",
		Parameters: []tools.Param{
			{Name: "query", Type: "string", Description: "What to search for", Required: true},
			{Name: "limit", Type: "integer", Description: "Maximum results", Required: false, Default: 5},
		},
	}
}

func (t *SearchMemoryTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	query, _ := call.Args["query"].(string)
	query = strings.TrimSpace(query)
	if query == "" {
		return tools.Fail("Cannot search with empty query", nil)
	}
	limit, _ := call.Args["limit"].(int)
	if limit <= 0 {
		limit = 5
	}
	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	// Track all results with their sources
	type memResult struct {
		content string
		source  string // "token" or "semantic"
		score   float64
	}
	allResults := make([]memResult, 0)

	// 1. Token-based search (SQLite) - existing behavior
	tokenResults, err := call.Ctx.Store.SearchMemory(ctx, userID, query, limit)
	if err == nil {
		for _, r := range tokenResults {
			switch r.Type {
			case "fact":
				allResults = append(allResults, memResult{content: r.Fact, source: "token", score: 0})
			case "memory":
				allResults = append(allResults, memResult{content: r.Memory, source: "token", score: 0})
			case "decision":
				allResults = append(allResults, memResult{content: r.Decision, source: "token", score: 0})
			case "action":
				allResults = append(allResults, memResult{content: fmt.Sprintf("%s: %s", r.ToolName, r.Output), source: "token", score: 0})
			case "session":
				allResults = append(allResults, memResult{content: r.Summary, source: "token", score: 0})
			case "conversation":
				allResults = append(allResults, memResult{content: r.UserMessage, source: "token", score: 0})
			case "entity":
				allResults = append(allResults, memResult{content: fmt.Sprintf("%s: %s", r.EntityName, r.EntitySummary), source: "token", score: 0})
			}
		}
	}

	// 2. Hybrid semantic search from canonical memory items.
	if call.Ctx.EmbeddingProvider != nil {
		queryEmbedding, err := call.Ctx.EmbeddingProvider.EmbedSingle(ctx, query)
		if err == nil && queryEmbedding != nil {
			hybridResults, err := call.Ctx.Store.SearchMemoryHybrid(ctx, userID, query, queryEmbedding, limit)
			if err == nil {
				for _, r := range hybridResults {
					content := memorySearchContent(r)
					if strings.TrimSpace(content) == "" {
						continue
					}
					allResults = append(allResults, memResult{content: content, source: "semantic", score: r.Similarity})
				}
			}
		}
	}

	if len(allResults) == 0 {
		return tools.Success("No relevant memories found.", map[string]any{"count": 0})
	}

	// Format results - show both token and semantic results
	lines := make([]string, 0, len(allResults))
	for i, r := range allResults {
		idx := i + 1
		source := r.source
		if r.source == "semantic" {
			source = fmt.Sprintf("semantic (score: %.2f)", r.score)
		}
		lines = append(lines, fmt.Sprintf("%d. [%s] %s", idx, source, truncateOutput(r.content, 100)))
	}

	sourceInfo := ""
	if len(tokenResults) > 0 && len(allResults) > len(tokenResults) {
		sourceInfo = " (hybrid: token + semantic)"
	}
	return tools.Success("Found relevant memories"+sourceInfo+":\n"+strings.Join(lines, "\n"), map[string]any{"count": len(allResults), "token_results": len(tokenResults)})
}

func truncateOutput(v string, max int) string {
	v = strings.TrimSpace(v)
	if max <= 0 || len(v) <= max {
		return v
	}
	return strings.TrimSpace(v[:max]) + "..."
}

func memorySearchContent(r db.MemorySearchResult) string {
	switch r.Type {
	case "fact":
		return r.Fact
	case "decision":
		return r.Decision
	case "summary":
		return r.Summary
	case "entity", "entity_observation":
		if strings.TrimSpace(r.EntityName) != "" {
			return strings.TrimSpace(r.EntityName + ": " + r.EntitySummary)
		}
		return r.EntitySummary
	case "action":
		return strings.TrimSpace(r.ToolName + ": " + r.Output)
	case "conversation":
		return r.UserMessage
	default:
		return r.Memory
	}
}
