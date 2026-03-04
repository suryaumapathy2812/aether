package builtin

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// WebFetchTool fetches a URL and converts the content to clean text for LLM consumption.
// Always available as a builtin tool. No API key required.
type WebFetchTool struct{}

const (
	fetchMaxSize        = 5 * 1024 * 1024 // 5MB
	fetchDefaultTimeout = 30 * time.Second
	fetchUserAgent      = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)

func (t *WebFetchTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "web_fetch",
		Description: "Fetch content from a URL and convert it to clean text for reading. Supports HTML pages, plain text, JSON, and other text formats. HTML is automatically converted to readable text with scripts and styles removed.",
		StatusText:  "Fetching web page...",
		Parameters: []tools.Param{
			{Name: "url", Type: "string", Description: "The URL to fetch content from (must start with http:// or https://)", Required: true},
			{Name: "format", Type: "string", Description: "Output format: text (plain text extraction) or markdown (structured with headers/links preserved). Default: markdown", Required: false, Default: "markdown", Enum: []string{"text", "markdown"}},
		},
	}
}

func (t *WebFetchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	rawURL, _ := call.Args["url"].(string)
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return tools.Fail("URL is required", nil)
	}
	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
		return tools.Fail("URL must start with http:// or https://", nil)
	}

	format := "markdown"
	if v, _ := call.Args["format"].(string); v != "" {
		format = v
	}

	httpCtx, cancel := context.WithTimeout(ctx, fetchDefaultTimeout)
	defer cancel()

	// Build request with browser-like headers
	req, err := http.NewRequestWithContext(httpCtx, http.MethodGet, rawURL, nil)
	if err != nil {
		return tools.Fail("Failed to create request: "+err.Error(), nil)
	}
	req.Header.Set("User-Agent", fetchUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if httpCtx.Err() != nil {
			return tools.Fail("Request timed out", nil)
		}
		return tools.Fail("Fetch failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()

	// Retry with honest UA if Cloudflare bot detection blocks us
	if resp.StatusCode == 403 && resp.Header.Get("Cf-Mitigated") == "challenge" {
		resp.Body.Close()
		req2, _ := http.NewRequestWithContext(httpCtx, http.MethodGet, rawURL, nil)
		req2.Header.Set("User-Agent", "aether-agent")
		req2.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
		req2.Header.Set("Accept-Language", "en-US,en;q=0.9")
		resp, err = http.DefaultClient.Do(req2)
		if err != nil {
			return tools.Fail("Fetch failed on retry: "+err.Error(), nil)
		}
		defer resp.Body.Close()
	}

	if resp.StatusCode != http.StatusOK {
		return tools.Fail(fmt.Sprintf("Request failed with status %d", resp.StatusCode), nil)
	}

	// Check content length before reading
	if resp.ContentLength > int64(fetchMaxSize) {
		return tools.Fail("Response too large (exceeds 5MB limit)", nil)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, int64(fetchMaxSize)+1))
	if err != nil {
		return tools.Fail("Failed to read response: "+err.Error(), nil)
	}
	if len(body) > fetchMaxSize {
		return tools.Fail("Response too large (exceeds 5MB limit)", nil)
	}

	contentType := resp.Header.Get("Content-Type")
	content := string(body)

	// If HTML, convert to text
	if strings.Contains(contentType, "text/html") || strings.Contains(contentType, "application/xhtml") {
		if format == "markdown" {
			content = htmlToMarkdown(content)
		} else {
			content = htmlToText(content)
		}
	}

	// Trim excessive whitespace
	content = collapseWhitespace(content)

	if strings.TrimSpace(content) == "" {
		return tools.Success("Page fetched but no readable text content found.", map[string]any{"url": rawURL})
	}

	// Truncate if extremely long (keep first ~100k chars for LLM context)
	const maxChars = 100_000
	if len(content) > maxChars {
		content = content[:maxChars] + "\n\n[Content truncated — showing first 100,000 characters]"
	}

	return tools.Success(content, map[string]any{"url": rawURL, "content_type": contentType})
}

// --- HTML to text conversion (pure Go, no external deps) ---

var (
	reScript   = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	reStyle    = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	reNoscript = regexp.MustCompile(`(?is)<noscript[^>]*>.*?</noscript>`)
	reComment  = regexp.MustCompile(`(?s)<!--.*?-->`)
	reTag      = regexp.MustCompile(`<[^>]+>`)
	reEntity   = regexp.MustCompile(`&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;`)
	reMultiNL  = regexp.MustCompile(`\n{3,}`)
	reMultiSP  = regexp.MustCompile(`[ \t]{2,}`)

	// Markdown conversion patterns
	reHeading   = regexp.MustCompile(`(?i)<h([1-6])[^>]*>(.*?)</h[1-6]>`)
	reLink      = regexp.MustCompile(`(?is)<a[^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>`)
	reBold      = regexp.MustCompile(`(?is)<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>`)
	reItalic    = regexp.MustCompile(`(?is)<(?:i|em)[^>]*>(.*?)</(?:i|em)>`)
	reCode      = regexp.MustCompile(`(?is)<code[^>]*>(.*?)</code>`)
	rePre       = regexp.MustCompile(`(?is)<pre[^>]*>(.*?)</pre>`)
	reLi        = regexp.MustCompile(`(?is)<li[^>]*>(.*?)</li>`)
	reBr        = regexp.MustCompile(`(?i)<br\s*/?>`)
	reParagraph = regexp.MustCompile(`(?i)</p>|</div>|</section>|</article>`)
	reBlockEnd  = regexp.MustCompile(`(?i)</(?:tr|table|blockquote|header|footer|nav|main)>`)
)

// htmlToText strips all HTML and returns plain text.
func htmlToText(html string) string {
	s := html
	s = reScript.ReplaceAllString(s, "")
	s = reStyle.ReplaceAllString(s, "")
	s = reNoscript.ReplaceAllString(s, "")
	s = reComment.ReplaceAllString(s, "")
	s = reBr.ReplaceAllString(s, "\n")
	s = reParagraph.ReplaceAllString(s, "\n\n")
	s = reBlockEnd.ReplaceAllString(s, "\n")
	s = reTag.ReplaceAllString(s, "")
	s = decodeHTMLEntities(s)
	return s
}

// htmlToMarkdown converts HTML to a lightweight markdown representation.
func htmlToMarkdown(html string) string {
	s := html
	// Remove non-content elements
	s = reScript.ReplaceAllString(s, "")
	s = reStyle.ReplaceAllString(s, "")
	s = reNoscript.ReplaceAllString(s, "")
	s = reComment.ReplaceAllString(s, "")

	// Convert pre blocks before other processing
	s = rePre.ReplaceAllStringFunc(s, func(m string) string {
		inner := rePre.FindStringSubmatch(m)
		if len(inner) < 2 {
			return m
		}
		code := reTag.ReplaceAllString(inner[1], "")
		return "\n```\n" + strings.TrimSpace(decodeHTMLEntities(code)) + "\n```\n"
	})

	// Convert headings
	s = reHeading.ReplaceAllStringFunc(s, func(m string) string {
		parts := reHeading.FindStringSubmatch(m)
		if len(parts) < 3 {
			return m
		}
		level := parts[1]
		text := reTag.ReplaceAllString(parts[2], "")
		n := 0
		if len(level) > 0 {
			n = int(level[0] - '0')
		}
		prefix := strings.Repeat("#", n)
		return "\n" + prefix + " " + strings.TrimSpace(decodeHTMLEntities(text)) + "\n"
	})

	// Convert links
	s = reLink.ReplaceAllStringFunc(s, func(m string) string {
		parts := reLink.FindStringSubmatch(m)
		if len(parts) < 3 {
			return m
		}
		href := parts[1]
		text := reTag.ReplaceAllString(parts[2], "")
		text = strings.TrimSpace(decodeHTMLEntities(text))
		if text == "" {
			return href
		}
		return "[" + text + "](" + href + ")"
	})

	// Convert inline formatting
	s = reBold.ReplaceAllString(s, "**$1**")
	s = reItalic.ReplaceAllString(s, "*$1*")
	s = reCode.ReplaceAllString(s, "`$1`")

	// Convert list items
	s = reLi.ReplaceAllStringFunc(s, func(m string) string {
		inner := reLi.FindStringSubmatch(m)
		if len(inner) < 2 {
			return m
		}
		text := reTag.ReplaceAllString(inner[1], "")
		return "\n- " + strings.TrimSpace(decodeHTMLEntities(text))
	})

	// Convert block-level breaks
	s = reBr.ReplaceAllString(s, "\n")
	s = reParagraph.ReplaceAllString(s, "\n\n")
	s = reBlockEnd.ReplaceAllString(s, "\n")

	// Strip remaining tags
	s = reTag.ReplaceAllString(s, "")
	s = decodeHTMLEntities(s)
	return s
}

// decodeHTMLEntities handles common HTML entities.
func decodeHTMLEntities(s string) string {
	entities := map[string]string{
		"&amp;":    "&",
		"&lt;":     "<",
		"&gt;":     ">",
		"&quot;":   `"`,
		"&apos;":   "'",
		"&#39;":    "'",
		"&nbsp;":   " ",
		"&ndash;":  "–",
		"&mdash;":  "—",
		"&laquo;":  "«",
		"&raquo;":  "»",
		"&copy;":   "©",
		"&reg;":    "®",
		"&trade;":  "™",
		"&hellip;": "…",
		"&bull;":   "•",
		"&middot;": "·",
		"&lsquo;":  "'",
		"&rsquo;":  "'",
		"&ldquo;":  "\u201c",
		"&rdquo;":  "\u201d",
	}
	for ent, repl := range entities {
		s = strings.ReplaceAll(s, ent, repl)
	}
	// Strip any remaining entities we don't handle
	s = reEntity.ReplaceAllString(s, "")
	return s
}

// collapseWhitespace normalizes excessive whitespace.
func collapseWhitespace(s string) string {
	s = reMultiSP.ReplaceAllString(s, " ")
	s = reMultiNL.ReplaceAllString(s, "\n\n")
	// Trim leading/trailing whitespace per line
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

var _ tools.Tool = (*WebFetchTool)(nil)
