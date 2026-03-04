package builtin

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// WebSearchTool performs real-time web searches via Exa AI's free MCP endpoint.
// No API key required. Always available as a builtin tool.
type WebSearchTool struct{}

const (
	exaBaseURL        = "https://mcp.exa.ai/mcp"
	exaDefaultResults = 8
	exaTimeout        = 25 * time.Second
)

type exaRPCRequest struct {
	JSONRPC string       `json:"jsonrpc"`
	ID      int          `json:"id"`
	Method  string       `json:"method"`
	Params  exaRPCParams `json:"params"`
}

type exaRPCParams struct {
	Name      string        `json:"name"`
	Arguments exaSearchArgs `json:"arguments"`
}

type exaSearchArgs struct {
	Query                string `json:"query"`
	NumResults           int    `json:"numResults,omitempty"`
	Livecrawl            string `json:"livecrawl,omitempty"`
	Type                 string `json:"type,omitempty"`
	ContextMaxCharacters int    `json:"contextMaxCharacters,omitempty"`
}

type exaRPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	Result  struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"result"`
}

func (t *WebSearchTool) Definition() tools.Definition {
	year := time.Now().Year()
	desc := fmt.Sprintf(
		"Search the web using Exa AI. Performs real-time web searches and returns LLM-optimized context from the most relevant websites. "+
			"Use this for current events, recent data, or anything beyond your knowledge cutoff. "+
			"The current year is %d — always use this year when searching for recent information.",
		year,
	)
	return tools.Definition{
		Name:        "web_search",
		Description: desc,
		StatusText:  "Searching the web...",
		Parameters: []tools.Param{
			{Name: "query", Type: "string", Description: "Search query", Required: true},
			{Name: "num_results", Type: "integer", Description: "Number of results to return (default: 8)", Required: false, Default: exaDefaultResults},
			{Name: "type", Type: "string", Description: "Search type: auto (balanced, default), fast (quick), deep (comprehensive)", Required: false, Default: "auto", Enum: []string{"auto", "fast", "deep"}},
			{Name: "livecrawl", Type: "string", Description: "Live crawl mode: fallback (use if cached unavailable, default), preferred (prioritize live crawling)", Required: false, Default: "fallback", Enum: []string{"fallback", "preferred"}},
		},
	}
}

func (t *WebSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	query = strings.TrimSpace(query)
	if query == "" {
		return tools.Fail("Search query is required", nil)
	}

	numResults := exaDefaultResults
	if n, ok := call.Args["num_results"]; ok {
		if v, err := asIntVal(n); err == nil && v > 0 {
			numResults = v
		}
	}
	searchType := "auto"
	if v, _ := call.Args["type"].(string); v != "" {
		searchType = v
	}
	livecrawl := "fallback"
	if v, _ := call.Args["livecrawl"].(string); v != "" {
		livecrawl = v
	}

	rpcReq := exaRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: exaRPCParams{
			Name: "web_search_exa",
			Arguments: exaSearchArgs{
				Query:      query,
				NumResults: numResults,
				Type:       searchType,
				Livecrawl:  livecrawl,
			},
		},
	}

	body, err := json.Marshal(rpcReq)
	if err != nil {
		return tools.Fail("Failed to build search request: "+err.Error(), nil)
	}

	httpCtx, cancel := context.WithTimeout(ctx, exaTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(httpCtx, http.MethodPost, exaBaseURL, bytes.NewReader(body))
	if err != nil {
		return tools.Fail("Failed to create request: "+err.Error(), nil)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if httpCtx.Err() != nil {
			return tools.Fail("Search request timed out", nil)
		}
		return tools.Fail("Search request failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return tools.Fail(fmt.Sprintf("Search error (%d): %s", resp.StatusCode, string(errBody)), nil)
	}

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 5*1024*1024))
	if err != nil {
		return tools.Fail("Failed to read search response: "+err.Error(), nil)
	}

	// Parse SSE response — look for "data: " lines containing JSON
	text := parseExaSSE(string(respBody))
	if text == "" {
		// Try direct JSON parse (non-SSE response)
		var direct exaRPCResponse
		if err := json.Unmarshal(respBody, &direct); err == nil {
			if len(direct.Result.Content) > 0 {
				text = direct.Result.Content[0].Text
			}
		}
	}

	if text == "" {
		return tools.Success("No search results found. Try a different query.", map[string]any{"query": query, "count": 0})
	}

	return tools.Success(text, map[string]any{"query": query})
}

// parseExaSSE extracts the result text from an SSE-formatted response.
func parseExaSSE(body string) string {
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var rpcResp exaRPCResponse
		if err := json.Unmarshal([]byte(data), &rpcResp); err != nil {
			continue
		}
		if len(rpcResp.Result.Content) > 0 {
			return rpcResp.Result.Content[0].Text
		}
	}
	return ""
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
