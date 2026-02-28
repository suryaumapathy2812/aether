package plugins

import (
	"context"
	"fmt"
	"strings"

	logic "github.com/suryaumapathy2812/core-ai/agent/internal/plugins/logic"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type WebSearchTool struct{}
type NewsSearchTool struct{}
type LLMContextSearchTool struct{}

func (t *WebSearchTool) Definition() tools.Definition {
	return tools.Definition{Name: "web_search", Description: "Search the live web via Brave Search.", StatusText: "Searching web...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Search query", Required: true}, {Name: "count", Type: "integer", Description: "Result count", Required: false, Default: 10}, {Name: "country", Type: "string", Description: "Country code", Required: false, Default: "IN"}}}
}

func (t *WebSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	q, _ := call.Args["query"].(string)
	count, _ := asInt(call.Args["count"])
	country, _ := call.Args["country"].(string)
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	key, err := requireString(cfg, "api_key")
	if err != nil {
		return tools.Fail("Brave Search not connected: missing api_key", nil)
	}
	client := logic.BraveClient{APIKey: key}
	results, err := client.WebSearch(ctx, q, count, country)
	if err != nil {
		return tools.Fail("Web search failed: "+err.Error(), nil)
	}
	if len(results) == 0 {
		return tools.Success("No web results found.", map[string]any{"query": q, "count": 0})
	}
	lines := []string{fmt.Sprintf("Web results for: %s", q)}
	for i, r := range results {
		lines = append(lines, fmt.Sprintf("%d. %s\n%s\n%s", i+1, r.Title, r.Description, r.URL))
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{"query": q, "count": len(results)})
}

func (t *NewsSearchTool) Definition() tools.Definition {
	return tools.Definition{Name: "news_search", Description: "Search news via Brave Search.", StatusText: "Searching news...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Search query", Required: true}, {Name: "count", Type: "integer", Description: "Result count", Required: false, Default: 10}, {Name: "country", Type: "string", Description: "Country code", Required: false, Default: "IN"}}}
}

func (t *NewsSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	q, _ := call.Args["query"].(string)
	count, _ := asInt(call.Args["count"])
	country, _ := call.Args["country"].(string)
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	key, err := requireString(cfg, "api_key")
	if err != nil {
		return tools.Fail("Brave Search not connected: missing api_key", nil)
	}
	client := logic.BraveClient{APIKey: key}
	results, err := client.NewsSearch(ctx, q, count, country)
	if err != nil {
		return tools.Fail("News search failed: "+err.Error(), nil)
	}
	if len(results) == 0 {
		return tools.Success("No news results found.", map[string]any{"query": q, "count": 0})
	}
	lines := []string{fmt.Sprintf("News results for: %s", q)}
	for i, r := range results {
		meta := r.Source
		if r.Age != "" {
			if meta != "" {
				meta += " · "
			}
			meta += r.Age
		}
		lines = append(lines, fmt.Sprintf("%d. %s\n%s\n%s\n%s", i+1, r.Title, meta, r.Description, r.URL))
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{"query": q, "count": len(results)})
}

func (t *LLMContextSearchTool) Definition() tools.Definition {
	return tools.Definition{Name: "llm_context_search", Description: "Search web context snippets via Brave Search for grounded answers.", StatusText: "Gathering context...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Search query", Required: true}}}
}

func (t *LLMContextSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	q, _ := call.Args["query"].(string)
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	key, err := requireString(cfg, "api_key")
	if err != nil {
		return tools.Fail("Brave Search not connected: missing api_key", nil)
	}
	client := logic.BraveClient{APIKey: key}
	results, err := client.LLMContextSearch(ctx, q)
	if err != nil {
		return tools.Fail("Context search failed: "+err.Error(), nil)
	}
	if len(results) == 0 {
		return tools.Success("No context found.", map[string]any{"query": q, "count": 0})
	}
	lines := []string{fmt.Sprintf("Context for: %s", q)}
	for _, r := range results {
		lines = append(lines, fmt.Sprintf("%s\n%s\n%s", r.Title, r.URL, r.Text))
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{"query": q, "count": len(results)})
}

var (
	_ tools.Tool = (*WebSearchTool)(nil)
	_ tools.Tool = (*NewsSearchTool)(nil)
	_ tools.Tool = (*LLMContextSearchTool)(nil)
)
