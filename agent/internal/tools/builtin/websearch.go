package builtin

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// WebSearchTool performs real-time web searches via DuckDuckGo's HTML endpoint.
// Completely free, no API key, no rate limits, no freemium restrictions.
// Always available as a builtin tool.
type WebSearchTool struct{}

const (
	ddgBaseURL      = "https://html.duckduckgo.com/html/"
	ddgTimeout      = 15 * time.Second
	ddgDefaultCount = 10
	ddgUserAgent    = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)

func (t *WebSearchTool) Definition() tools.Definition {
	year := time.Now().Year()
	desc := fmt.Sprintf(
		"Search the web using DuckDuckGo. Performs real-time web searches and returns titles, URLs, and snippets from the most relevant websites. "+
			"Completely free with no rate limits. Use this for current events, recent data, or anything beyond your knowledge cutoff. "+
			"The current year is %d — always use this year when searching for recent information.",
		year,
	)
	return tools.Definition{
		Name:        "web_search",
		Description: desc,
		StatusText:  "Searching the web...",
		Parameters: []tools.Param{
			{Name: "query", Type: "string", Description: "Search query", Required: true},
			{Name: "region", Type: "string", Description: "Region code for localized results (e.g. in-en for India, us-en for US, uk-en for UK). Default: wt-wt (no region)", Required: false, Default: "wt-wt"},
			{Name: "time_range", Type: "string", Description: "Time range filter: d (past day), w (past week), m (past month), y (past year). Default: no filter", Required: false, Enum: []string{"d", "w", "m", "y"}},
		},
	}
}

func (t *WebSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	query = strings.TrimSpace(query)
	if query == "" {
		return tools.Fail("Search query is required", nil)
	}

	region := "wt-wt"
	if v, _ := call.Args["region"].(string); v != "" {
		region = v
	}
	timeRange, _ := call.Args["time_range"].(string)

	// Build POST form data
	form := url.Values{}
	form.Set("q", query)
	form.Set("kl", region)
	if timeRange != "" {
		form.Set("df", timeRange)
	}

	httpCtx, cancel := context.WithTimeout(ctx, ddgTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(httpCtx, http.MethodPost, ddgBaseURL, strings.NewReader(form.Encode()))
	if err != nil {
		return tools.Fail("Failed to create search request: "+err.Error(), nil)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", ddgUserAgent)
	req.Header.Set("Accept", "text/html")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if httpCtx.Err() != nil {
			return tools.Fail("Search request timed out", nil)
		}
		return tools.Fail("Search request failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return tools.Fail(fmt.Sprintf("Search failed with status %d", resp.StatusCode), nil)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if err != nil {
		return tools.Fail("Failed to read search response: "+err.Error(), nil)
	}

	results := parseDDGResults(string(body))
	if len(results) == 0 {
		return tools.Success("No search results found. Try a different query.", map[string]any{"query": query, "count": 0})
	}

	// Format results for LLM consumption
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Web search results for: %s\n\n", query))
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, r.title))
		sb.WriteString(fmt.Sprintf("   %s\n", r.url))
		if r.snippet != "" {
			sb.WriteString(fmt.Sprintf("   %s\n", r.snippet))
		}
		sb.WriteString("\n")
	}

	return tools.Success(sb.String(), map[string]any{"query": query, "count": len(results)})
}

// --- DuckDuckGo HTML parser ---

type ddgResult struct {
	title   string
	url     string
	snippet string
}

// Regex patterns for parsing DDG HTML results.
// NOTE 2026-03-24: DuckDuckGo's HTML markup is fragile and changes over time.
// Keep the block matcher loose and fall back to broader result link parsing if the
// primary block regex stops matching.
// The expected HTML structure is a div.web-result containing:
//   - Title in <a class="result__a" href="...uddg=ENCODED_URL...">TITLE</a>
//   - Snippet in <a class="result__snippet">SNIPPET</a>
var (
	reResultBlock = regexp.MustCompile(`(?s)<div[^>]*\bweb-result\b[^>]*>(.*?)</div>\s*</div>\s*</div>`)
	reTitle       = regexp.MustCompile(`(?s)<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]*>(.*?)</a>`)
	reTitleHref   = regexp.MustCompile(`(?s)<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]+href="([^"]*)"`)
	reSnippet     = regexp.MustCompile(`(?s)<(?:a|div|span)[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</(?:a|div|span)>`)
	reUddg        = regexp.MustCompile(`uddg=([^&]+)`)
	reHTMLTag     = regexp.MustCompile(`<[^>]+>`)
	reHTMLEntity2 = regexp.MustCompile(`&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;`)
)

func parseDDGResults(html string) []ddgResult {
	blocks := reResultBlock.FindAllStringSubmatch(html, -1)
	results := make([]ddgResult, 0, len(blocks))

	for _, block := range blocks {
		content := block[0]

		// Extract title
		titleMatch := reTitle.FindStringSubmatch(content)
		if titleMatch == nil {
			continue
		}
		title := stripHTML(titleMatch[1])
		if title == "" {
			continue
		}

		// Extract URL from the uddg parameter in the href
		resultURL := ""
		hrefMatch := reTitleHref.FindStringSubmatch(content)
		if hrefMatch != nil {
			resultURL = decodeDDGResultURL(hrefMatch[1])
		}
		if resultURL == "" {
			continue
		}

		// Extract snippet
		snippet := ""
		snippetMatch := reSnippet.FindStringSubmatch(content)
		if snippetMatch != nil {
			snippet = stripHTML(snippetMatch[1])
		}

		results = append(results, ddgResult{
			title:   title,
			url:     resultURL,
			snippet: snippet,
		})
	}

	if len(results) > 0 {
		return results
	}

	return parseDDGResultsFallback(html)
}

func parseDDGResultsFallback(html string) []ddgResult {
	matches := reTitle.FindAllStringSubmatchIndex(html, -1)
	results := make([]ddgResult, 0, len(matches))
	seen := map[string]struct{}{}

	for i, match := range matches {
		if len(match) < 4 {
			continue
		}
		segmentStart := match[0]
		segmentEnd := len(html)
		if i+1 < len(matches) && len(matches[i+1]) >= 2 {
			segmentEnd = matches[i+1][0]
		}

		segment := html[segmentStart:segmentEnd]
		title := stripHTML(html[match[2]:match[3]])
		if title == "" {
			continue
		}

		hrefMatch := reTitleHref.FindStringSubmatch(segment)
		if hrefMatch == nil {
			continue
		}
		resultURL := decodeDDGResultURL(hrefMatch[1])
		if resultURL == "" {
			continue
		}
		if _, ok := seen[resultURL]; ok {
			continue
		}
		seen[resultURL] = struct{}{}

		snippet := ""
		if snippetMatch := reSnippet.FindStringSubmatch(segment); snippetMatch != nil {
			snippet = stripHTML(snippetMatch[1])
		}

		results = append(results, ddgResult{
			title:   title,
			url:     resultURL,
			snippet: snippet,
		})
	}

	return results
}

func decodeDDGResultURL(href string) string {
	href = decodeDDGEntities(strings.TrimSpace(href))
	if href == "" {
		return ""
	}
	if uddgMatch := reUddg.FindStringSubmatch(href); uddgMatch != nil {
		decoded, err := url.QueryUnescape(uddgMatch[1])
		if err == nil {
			return decoded
		}
	}
	if decoded, err := url.QueryUnescape(href); err == nil {
		href = decoded
	}
	if strings.HasPrefix(href, "//") {
		return "https:" + href
	}
	if strings.HasPrefix(href, "http://") || strings.HasPrefix(href, "https://") {
		return href
	}
	return ""
}

// stripHTML removes HTML tags and decodes common entities.
func stripHTML(s string) string {
	s = reHTMLTag.ReplaceAllString(s, "")
	s = decodeDDGEntities(s)
	s = strings.TrimSpace(s)
	return s
}

// decodeDDGEntities handles common HTML entities found in DDG results.
func decodeDDGEntities(s string) string {
	replacements := map[string]string{
		"&amp;":    "&",
		"&lt;":     "<",
		"&gt;":     ">",
		"&quot;":   `"`,
		"&apos;":   "'",
		"&#39;":    "'",
		"&#x27;":   "'",
		"&nbsp;":   " ",
		"&ndash;":  "–",
		"&mdash;":  "—",
		"&hellip;": "…",
	}
	for ent, repl := range replacements {
		s = strings.ReplaceAll(s, ent, repl)
	}
	// Strip any remaining entities
	s = reHTMLEntity2.ReplaceAllString(s, "")
	return s
}

// asIntVal converts JSON number types to int.
func asIntVal(v any) (int, error) {
	switch n := v.(type) {
	case int:
		return n, nil
	case int64:
		return int(n), nil
	case float64:
		return int(n), nil
	default:
		return 0, fmt.Errorf("not an integer")
	}
}

var _ tools.Tool = (*WebSearchTool)(nil)
