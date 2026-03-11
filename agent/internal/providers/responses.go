package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/config"
)

type OpenAIResponsesProvider struct {
	apiKey   string
	baseURL  string
	model    string
	http     *http.Client
	headers  map[string]string
	provider string
}

func NewOpenAIResponsesProvider(cfg config.LLMConfig) *OpenAIResponsesProvider {
	apiKey := strings.TrimSpace(cfg.APIKey)
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/v1"
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = "minimax/minimax-m2.5"
	}

	headers := map[string]string{}
	if site := strings.TrimSpace(cfg.OpenRouterSiteURL); site != "" {
		headers["HTTP-Referer"] = site
	}
	if title := strings.TrimSpace(cfg.OpenRouterAppName); title != "" {
		headers["X-Title"] = title
	}

	provider := "openai"
	if strings.Contains(strings.ToLower(baseURL), "openrouter") {
		provider = "openrouter"
		if !strings.Contains(model, "/") {
			model = "openai/" + model
		}
	}

	return &OpenAIResponsesProvider{
		apiKey:   apiKey,
		baseURL:  baseURL,
		model:    model,
		http:     &http.Client{Timeout: 120 * time.Second},
		headers:  headers,
		provider: provider,
	}
}

func (p *OpenAIResponsesProvider) Name() string {
	return p.provider + "_responses"
}

func (p *OpenAIResponsesProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		SupportsChat:           true,
		SupportsResponses:      true,
		SupportsResponsesWS:    false,
		SupportsRealtime:       false,
		SupportsStructuredJSON: true,
	}
}

func (p *OpenAIResponsesProvider) StreamWithTools(ctx context.Context, opts GenerateOptions) (<-chan LLMStreamEvent, error) {
	items := make([]map[string]any, 0, len(opts.Messages))
	for _, msg := range opts.Messages {
		items = append(items, convertToResponsesInput(msg))
	}
	respOpts := ResponsesOptions{
		Model:       opts.Model,
		Input:       items,
		Tools:       opts.Tools,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
		Store:       false,
	}
	return p.StreamResponses(ctx, respOpts)
}

func (p *OpenAIResponsesProvider) StreamResponses(ctx context.Context, opts ResponsesOptions) (<-chan LLMStreamEvent, error) {
	if strings.TrimSpace(p.apiKey) == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY is not set")
	}
	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = p.model
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1200
	}
	temperature := opts.Temperature
	if temperature == 0 {
		temperature = 0.2
	}

	body := map[string]any{
		"model":       model,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"stream":      true,
		"store":       opts.Store,
	}

	if len(opts.Input) > 0 {
		body["input"] = opts.Input
	}
	if opts.Instructions != "" {
		body["instructions"] = opts.Instructions
	}
	if opts.PreviousResponseID != "" {
		body["previous_response_id"] = opts.PreviousResponseID
	}
	if len(opts.Tools) > 0 {
		body["tools"] = convertToolsToResponses(opts.Tools)
	}

	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/responses", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	for k, v := range p.headers {
		req.Header.Set(k, v)
	}

	resp, err := p.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		defer resp.Body.Close()
		bodyBytes := make([]byte, 1024)
		n, _ := resp.Body.Read(bodyBytes)
		return nil, fmt.Errorf("Responses request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes[:n])))
	}

	out := make(chan LLMStreamEvent, 64)
	go func() {
		defer close(out)
		defer resp.Body.Close()

		pendingCalls := map[string]*responsesFunctionCall{}
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		finishReason := "stop"

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") {
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "[DONE]" {
				break
			}

			var rawEvent map[string]any
			if err := json.Unmarshal([]byte(data), &rawEvent); err != nil {
				continue
			}

			eventType, _ := rawEvent["type"].(string)
			switch eventType {
			case "response.created":
				continue
			case "response.output_text.delta":
				content, _ := rawEvent["delta"].(string)
				if content != "" {
					out <- LLMStreamEvent{Type: EventToken, Content: content}
				}
			case "response.function_call.begin":
				callID, _ := rawEvent["id"].(string)
				name, _ := rawEvent["name"].(string)
				pendingCalls[callID] = &responsesFunctionCall{
					ID:   callID,
					Name: name,
				}
			case "response.function_call.argument.delta":
				callID, _ := rawEvent["id"].(string)
				delta, _ := rawEvent["delta"].(string)
				if pc, ok := pendingCalls[callID]; ok {
					pc.Arguments += delta
				}
			case "response.function_call.done":
				callID, _ := rawEvent["id"].(string)
				if pc, ok := pendingCalls[callID]; ok {
					args := map[string]any{}
					if strings.TrimSpace(pc.Arguments) != "" {
						_ = json.Unmarshal([]byte(pc.Arguments), &args)
					}
					out <- LLMStreamEvent{
						Type:      EventToolCalls,
						ToolCalls: []LLMToolCall{{ID: pc.ID, Name: pc.Name, Arguments: args}},
					}
					delete(pendingCalls, callID)
				}
			case "response.function_call_output":
				callID, _ := rawEvent["call_id"].(string)
				output, _ := rawEvent["output"].(string)
				out <- LLMStreamEvent{Type: EventToolResult, CallID: callID, Content: output}
			case "response.completed":
				finishReason = "stop"
			case "response.incomplete":
				finishReason = "length"
			case "error":
				errMsg, _ := rawEvent["message"].(string)
				if errMsg == "" {
					errMsg = "unknown error"
				}
				out <- LLMStreamEvent{Type: EventError, Err: fmt.Errorf(errMsg)}
				return
			}
		}
		if err := scanner.Err(); err != nil {
			out <- LLMStreamEvent{Type: EventError, Err: err}
			return
		}

		out <- LLMStreamEvent{Type: EventDone, FinishReason: finishReason}
	}()

	return out, nil
}

type responsesFunctionCall struct {
	ID        string
	Name      string
	Arguments string
}

func convertToResponsesInput(msg map[string]any) map[string]any {
	role, _ := msg["role"].(string)
	role = strings.TrimSpace(strings.ToLower(role))

	if toolCalls, ok := msg["tool_calls"].([]any); ok && len(toolCalls) > 0 {
		out := make([]map[string]any, 0, len(toolCalls)+1)
		for _, tcRaw := range toolCalls {
			tc, ok := tcRaw.(map[string]any)
			if !ok {
				continue
			}
			fnRaw, _ := tc["function"].(map[string]any)
			fnName, _ := fnRaw["name"].(string)
			fnArgs, _ := fnRaw["arguments"].(string)
			out = append(out, map[string]any{
				"type":      "function_call",
				"id":        tc["id"],
				"name":      fnName,
				"arguments": fnArgs,
			})
		}
		if content, ok := msg["content"].(string); ok && strings.TrimSpace(content) != "" {
			out = append(out, map[string]any{
				"type":    "message",
				"role":    "assistant",
				"content": content,
			})
		}
		return map[string]any{"items": out}
	}

	if role == "tool" {
		callID, _ := msg["tool_call_id"].(string)
		content, _ := msg["content"].(string)
		return map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  content,
		}
	}

	content := msg["content"]
	if content == nil {
		content = ""
	}
	return map[string]any{
		"type":    "message",
		"role":    role,
		"content": content,
	}
}

func convertToolsToResponses(tools []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		fn, ok := tool["function"].(map[string]any)
		if !ok {
			continue
		}
		out = append(out, map[string]any{
			"type":        "function",
			"name":        fn["name"],
			"description": fn["description"],
			"parameters":  fn["parameters"],
		})
	}
	return out
}

type LegacyCompletionsProvider struct {
	delegate *OpenAILLMProvider
}

func NewLegacyCompletionsProvider(cfg config.LLMConfig) *LegacyCompletionsProvider {
	return &LegacyCompletionsProvider{
		delegate: NewOpenAILLMProvider(cfg),
	}
}

func (p *LegacyCompletionsProvider) Name() string {
	return p.delegate.Name() + "_completions"
}

func (p *LegacyCompletionsProvider) Capabilities() ProviderCapabilities {
	return p.delegate.Capabilities()
}

func (p *LegacyCompletionsProvider) StreamWithTools(ctx context.Context, opts GenerateOptions) (<-chan LLMStreamEvent, error) {
	if len(opts.Messages) == 0 {
		return nil, fmt.Errorf("messages required")
	}

	var prompt string
	lastMsg := opts.Messages[len(opts.Messages)-1]
	if content, ok := lastMsg["content"].(string); ok {
		prompt = content
	} else if parts, ok := lastMsg["content"].([]any); ok {
		var buf strings.Builder
		for _, part := range parts {
			if pm, ok := part.(map[string]any); ok {
				if typ, _ := pm["type"].(string); typ == "text" {
					if txt, _ := pm["text"].(string); txt != "" {
						buf.WriteString(txt)
					}
				}
			}
		}
		prompt = buf.String()
	}

	completionReq := map[string]any{
		"model":       opts.Model,
		"prompt":      prompt,
		"max_tokens":  opts.MaxTokens,
		"temperature": opts.Temperature,
		"stream":      true,
	}

	b, _ := json.Marshal(completionReq)
	ctx2 := context.WithValue(ctx, "legacy_completions", true)
	req, err := http.NewRequestWithContext(ctx2, http.MethodPost, p.delegate.baseURL+"/completions", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.delegate.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, fmt.Errorf("completions request failed with status %d", resp.StatusCode)
	}

	out := make(chan LLMStreamEvent, 64)
	go func() {
		defer close(out)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		finishReason := "stop"

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") {
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "[DONE]" {
				break
			}

			var chunk struct {
				Choices []struct {
					Text         string `json:"text"`
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			if chunk.Choices[0].Text != "" {
				out <- LLMStreamEvent{Type: EventToken, Content: chunk.Choices[0].Text}
			}
			if chunk.Choices[0].FinishReason != "" {
				finishReason = chunk.Choices[0].FinishReason
			}
		}

		out <- LLMStreamEvent{Type: EventDone, FinishReason: finishReason}
	}()

	return out, nil
}

var _ LLMProvider = (*LegacyCompletionsProvider)(nil)
var _ LLMProvider = (*OpenAIResponsesProvider)(nil)
var _ LLMProvider = (*OpenAILLMProvider)(nil)
var _ ResponsesProvider = (*OpenAIResponsesProvider)(nil)
