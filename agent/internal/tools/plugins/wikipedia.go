package plugins

import (
	"context"
	"fmt"

	logic "github.com/suryaumapathy2812/core-ai/agent/internal/plugins/logic"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type WikipediaSearchTool struct{}
type WikipediaGetArticleTool struct{}

func (t *WikipediaSearchTool) Definition() tools.Definition {
	return tools.Definition{Name: "wikipedia_search", Description: "Search Wikipedia and return article summary.", StatusText: "Searching Wikipedia...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Search query", Required: true}}}
}

func (t *WikipediaSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	q, _ := call.Args["query"].(string)
	client := logic.WikipediaClient{}
	s, err := client.SearchSummary(ctx, q)
	if err != nil {
		return tools.Fail("Wikipedia search failed: "+err.Error(), nil)
	}
	out := fmt.Sprintf("%s\n%s\n\n%s\n\n%s", s.Title, s.Description, s.Extract, s.PageURL)
	return tools.Success(out, map[string]any{"title": s.Title, "url": s.PageURL})
}

func (t *WikipediaGetArticleTool) Definition() tools.Definition {
	return tools.Definition{Name: "wikipedia_get_article", Description: "Get a Wikipedia article summary by title.", StatusText: "Reading Wikipedia article...", Parameters: []tools.Param{{Name: "title", Type: "string", Description: "Article title", Required: true}}}
}

func (t *WikipediaGetArticleTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	title, _ := call.Args["title"].(string)
	client := logic.WikipediaClient{}
	s, err := client.GetArticle(ctx, title)
	if err != nil {
		return tools.Fail("Wikipedia article fetch failed: "+err.Error(), nil)
	}
	out := fmt.Sprintf("%s\n%s\n\n%s\n\n%s", s.Title, s.Description, s.Extract, s.PageURL)
	return tools.Success(out, map[string]any{"title": s.Title, "url": s.PageURL})
}

var (
	_ tools.Tool = (*WikipediaSearchTool)(nil)
	_ tools.Tool = (*WikipediaGetArticleTool)(nil)
)
