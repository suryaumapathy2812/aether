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
	var embedding []float32
	if call.Ctx.EmbeddingProvider != nil {
		var err error
		embedding, err = call.Ctx.EmbeddingProvider.EmbedSingle(ctx, content)
		if err != nil {
			fmt.Printf("warning: failed to generate embedding for memory: %v\n", err)
		}
	}
	if _, err := call.Ctx.Store.AddMemory(ctx, db.AddMemoryInput{
		UserID:     userID,
		Kind:       "memory",
		Category:   category,
		Content:    content,
		Confidence: 0.95,
		Importance: 0.75,
		SourceType: "manual",
		Embedding:  embedding,
	}); err != nil {
		return tools.Fail("Failed to save memory: "+err.Error(), nil)
	}
	if strings.EqualFold(category, "preference") {
		_, _ = call.Ctx.Store.AddMemory(ctx, db.AddMemoryInput{
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

	var queryEmbedding []float32
	if call.Ctx.EmbeddingProvider != nil {
		queryEmbedding, _ = call.Ctx.EmbeddingProvider.EmbedSingle(ctx, query)
	}

	results, err := call.Ctx.Store.SearchMemory(ctx, db.MemorySearchQuery{UserID: userID, Text: query, QueryEmbedding: queryEmbedding, Limit: limit})
	if err != nil {
		return tools.Fail("Failed to search memory: "+err.Error(), nil)
	}
	if len(results) == 0 {
		return tools.Success("No relevant memories found.", map[string]any{"count": 0})
	}

	lines := make([]string, 0, len(results))
	for i, r := range results {
		content := memorySearchContent(r)
		if strings.TrimSpace(content) == "" {
			continue
		}
		idx := i + 1
		lines = append(lines, fmt.Sprintf("%d. [%s | %.2f] %s", idx, r.Type, r.Similarity, truncateOutput(content, 100)))
	}
	if len(lines) == 0 {
		return tools.Success("No relevant memories found.", map[string]any{"count": 0})
	}
	return tools.Success("Found relevant memories:\n"+strings.Join(lines, "\n"), map[string]any{"count": len(lines)})
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
