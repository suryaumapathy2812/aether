package builtin

import (
	"context"
	"fmt"
	"strings"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
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
	if err := call.Ctx.Store.StoreMemory(ctx, userID, content, category, 0, nil); err != nil {
		return tools.Fail("Failed to save memory: "+err.Error(), nil)
	}
	if strings.EqualFold(category, "preference") {
		_ = call.Ctx.Store.StoreMemoryDecision(ctx, userID, content, "preference", "explicit", 0)
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
	results, err := call.Ctx.Store.SearchMemory(ctx, userID, query, limit)
	if err != nil {
		return tools.Fail("Failed to search memory: "+err.Error(), nil)
	}
	if len(results) == 0 {
		return tools.Success("No relevant memories found.", map[string]any{"count": 0})
	}
	lines := make([]string, 0, len(results))
	for i, r := range results {
		idx := i + 1
		switch r.Type {
		case "fact":
			lines = append(lines, fmt.Sprintf("%d. [fact] %s", idx, r.Fact))
		case "memory":
			lines = append(lines, fmt.Sprintf("%d. [memory/%s] %s", idx, r.Category, r.Memory))
		case "decision":
			lines = append(lines, fmt.Sprintf("%d. [decision/%s] %s", idx, r.Category, r.Decision))
		case "action":
			lines = append(lines, fmt.Sprintf("%d. [action] %s: %s", idx, r.ToolName, truncateOutput(r.Output, 120)))
		case "session":
			lines = append(lines, fmt.Sprintf("%d. [session] %s", idx, truncateOutput(r.Summary, 140)))
		case "conversation":
			lines = append(lines, fmt.Sprintf("%d. [conversation] User: %s", idx, truncateOutput(r.UserMessage, 140)))
		}
	}
	return tools.Success("Found relevant memories:\n"+strings.Join(lines, "\n"), map[string]any{"count": len(results)})
}

func truncateOutput(v string, max int) string {
	v = strings.TrimSpace(v)
	if max <= 0 || len(v) <= max {
		return v
	}
	return strings.TrimSpace(v[:max]) + "..."
}
