package plugins

import (
	"context"
	"fmt"
	"sort"
	"strings"

	logic "github.com/suryaumapathy/core-ai/agent/internal/plugins/logic"
	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

type FetchFeedTool struct{}
type GetItemContentTool struct{}
type ListSubscribedFeedsTool struct{}
type AddFeedTool struct{}
type GetHackerNewsTopTool struct{}
type SearchHackerNewsTool struct{}

const subscribedFeedsKey = "subscribed_feeds"

func (t *FetchFeedTool) Definition() tools.Definition {
	return tools.Definition{Name: "fetch_feed", Description: "Fetch and read items from an RSS/Atom feed URL.", StatusText: "Fetching feed...", Parameters: []tools.Param{{Name: "url", Type: "string", Description: "Feed URL", Required: true}, {Name: "max_items", Type: "integer", Description: "Maximum items", Required: false, Default: 10}}}
}

func (t *FetchFeedTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	url, _ := call.Args["url"].(string)
	maxItems, _ := asInt(call.Args["max_items"])
	client := logic.RSSClient{}
	items, title, err := client.FetchFeed(ctx, url, maxItems)
	if err != nil {
		return tools.Fail("Fetch feed failed: "+err.Error(), nil)
	}
	if len(items) == 0 {
		return tools.Success("No feed items found.", map[string]any{"url": url, "count": 0})
	}
	lines := []string{fmt.Sprintf("Feed: %s", title)}
	for i, it := range items {
		lines = append(lines, fmt.Sprintf("%d. %s\n%s\n%s", i+1, it.Title, it.PubDate, it.Link))
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{"url": url, "count": len(items), "title": title})
}

func (t *GetItemContentTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_item_content", Description: "Read article content from a URL.", StatusText: "Reading article...", Parameters: []tools.Param{{Name: "url", Type: "string", Description: "Article URL", Required: true}}}
}

func (t *GetItemContentTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	url, _ := call.Args["url"].(string)
	client := logic.RSSClient{}
	title, content, err := client.GetItemContent(ctx, url)
	if err != nil {
		return tools.Fail("Get item content failed: "+err.Error(), nil)
	}
	return tools.Success(fmt.Sprintf("%s\n\n%s", title, content), map[string]any{"url": url, "title": title})
}

func (t *ListSubscribedFeedsTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_subscribed_feeds", Description: "List saved subscribed RSS feeds for this plugin.", StatusText: "Listing subscribed feeds..."}
}

func (t *ListSubscribedFeedsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	feeds := decodeFeeds(cfg[subscribedFeedsKey])
	if len(feeds) == 0 {
		return tools.Success("No subscribed feeds.", map[string]any{"count": 0})
	}
	return tools.Success(strings.Join(feeds, "\n"), map[string]any{"count": len(feeds), "feeds": feeds})
}

func (t *AddFeedTool) Definition() tools.Definition {
	return tools.Definition{Name: "add_feed", Description: "Add a feed URL to subscribed feeds.", StatusText: "Adding feed...", Parameters: []tools.Param{{Name: "url", Type: "string", Description: "Feed URL", Required: true}}}
}

func (t *AddFeedTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.PluginState == nil {
		return tools.Fail("plugin state is unavailable", nil)
	}
	url, _ := call.Args["url"].(string)
	url = strings.TrimSpace(url)
	if url == "" {
		return tools.Fail("url is required", nil)
	}
	cfg, err := call.Ctx.PluginState.Config(ctx)
	if err != nil {
		return tools.Fail("failed to read plugin config: "+err.Error(), nil)
	}
	if cfg == nil {
		cfg = map[string]string{}
	}
	feeds := decodeFeeds(cfg[subscribedFeedsKey])
	for _, f := range feeds {
		if f == url {
			return tools.Success("Feed already subscribed.", map[string]any{"url": url, "count": len(feeds)})
		}
	}
	feeds = append(feeds, url)
	cfg[subscribedFeedsKey] = encodeFeeds(feeds)
	if err := call.Ctx.PluginState.SetConfig(ctx, cfg); err != nil {
		return tools.Fail("failed to update plugin config: "+err.Error(), nil)
	}
	return tools.Success("Feed added.", map[string]any{"url": url, "count": len(feeds)})
}

func (t *GetHackerNewsTopTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_hacker_news_top", Description: "Get top/new/best/ask/show stories from Hacker News.", StatusText: "Fetching Hacker News stories...", Parameters: []tools.Param{{Name: "story_type", Type: "string", Description: "top|new|best|ask|show", Required: false, Default: "top"}, {Name: "count", Type: "integer", Description: "Number of stories", Required: false, Default: 10}}}
}

func (t *GetHackerNewsTopTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	storyType, _ := call.Args["story_type"].(string)
	count, _ := asInt(call.Args["count"])
	client := logic.RSSClient{}
	stories, err := client.GetHNTop(ctx, storyType, count)
	if err != nil {
		return tools.Fail("Hacker News fetch failed: "+err.Error(), nil)
	}
	if len(stories) == 0 {
		return tools.Success("No stories found.", map[string]any{"count": 0})
	}
	lines := []string{fmt.Sprintf("Hacker News (%s):", storyType)}
	for i, s := range stories {
		lines = append(lines, fmt.Sprintf("%d. %s\nscore=%d comments=%d\n%s", i+1, s.Title, s.Score, s.Comments, s.URL))
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{"count": len(stories), "story_type": storyType})
}

func (t *SearchHackerNewsTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_hacker_news", Description: "Search Hacker News stories/comments.", StatusText: "Searching Hacker News...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Search query", Required: true}, {Name: "count", Type: "integer", Description: "Result count", Required: false, Default: 10}, {Name: "search_type", Type: "string", Description: "stories|all", Required: false, Default: "stories"}}}
}

func (t *SearchHackerNewsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	count, _ := asInt(call.Args["count"])
	searchType, _ := call.Args["search_type"].(string)
	if searchType == "stories" {
		searchType = "story"
	}
	client := logic.RSSClient{}
	results, err := client.SearchHN(ctx, query, count, searchType)
	if err != nil {
		return tools.Fail("Hacker News search failed: "+err.Error(), nil)
	}
	if len(results) == 0 {
		return tools.Success("No Hacker News results found.", map[string]any{"count": 0})
	}
	lines := []string{fmt.Sprintf("Hacker News search for: %s", query)}
	for i, s := range results {
		lines = append(lines, fmt.Sprintf("%d. %s\nscore=%d comments=%d\n%s", i+1, s.Title, s.Score, s.Comments, s.URL))
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{"count": len(results), "query": query})
}

func decodeFeeds(v string) []string {
	if strings.TrimSpace(v) == "" {
		return []string{}
	}
	parts := strings.Split(v, "\n")
	out := make([]string, 0, len(parts))
	seen := map[string]struct{}{}
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if _, ok := seen[p]; ok {
			continue
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func encodeFeeds(feeds []string) string {
	norm := make([]string, 0, len(feeds))
	seen := map[string]struct{}{}
	for _, f := range feeds {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		if _, ok := seen[f]; ok {
			continue
		}
		seen[f] = struct{}{}
		norm = append(norm, f)
	}
	sort.Strings(norm)
	return strings.Join(norm, "\n")
}

var (
	_ tools.Tool = (*FetchFeedTool)(nil)
	_ tools.Tool = (*GetItemContentTool)(nil)
	_ tools.Tool = (*ListSubscribedFeedsTool)(nil)
	_ tools.Tool = (*AddFeedTool)(nil)
	_ tools.Tool = (*GetHackerNewsTopTool)(nil)
	_ tools.Tool = (*SearchHackerNewsTool)(nil)
)
