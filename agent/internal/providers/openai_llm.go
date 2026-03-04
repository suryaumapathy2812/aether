package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/config"
)

type OpenAILLMProvider struct {
	apiKey   string
	baseURL  string
	model    string
	http     *http.Client
	headers  map[string]string
	provider string
}

// NewOpenAILLMProvider creates a provider from centralized config.
func NewOpenAILLMProvider(cfg config.LLMConfig) *OpenAILLMProvider {
	apiKey := strings.TrimSpace(cfg.APIKey)
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/v1"
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = "google/gemini-3.1-flash-lite-preview"
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

	return &OpenAILLMProvider{
		apiKey:   apiKey,
		baseURL:  baseURL,
		model:    model,
		http:     &http.Client{Timeout: 120 * time.Second},
		headers:  headers,
		provider: provider,
	}
}

func (p *OpenAILLMProvider) Name() string {
	return p.provider
}

func (p *OpenAILLMProvider) StreamWithTools(ctx context.Context, opts GenerateOptions) (<-chan LLMStreamEvent, error) {
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
		"messages":    opts.Messages,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"stream":      true,
	}
	if len(opts.Tools) > 0 {
		body["tools"] = opts.Tools
		body["tool_choice"] = "auto"
	}

	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(b))
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
		return nil, fmt.Errorf("LLM request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes[:n])))
	}

	out := make(chan LLMStreamEvent, 64)
	go func() {
		defer close(out)
		defer resp.Body.Close()

		type partial struct {
			ID        string
			Name      string
			Arguments string
		}
		pending := map[int]*partial{}
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		finishReason := "stop"
		hasToolCalls := false

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
					Delta struct {
						Content   string `json:"content"`
						ToolCalls []struct {
							Index    int    `json:"index"`
							ID       string `json:"id"`
							Function struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					} `json:"delta"`
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				out <- LLMStreamEvent{Type: EventError, Err: fmt.Errorf("failed to parse LLM stream chunk: %w", err)}
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.Delta.Content != "" {
				out <- LLMStreamEvent{Type: EventToken, Content: choice.Delta.Content}
			}
			if len(choice.Delta.ToolCalls) > 0 {
				hasToolCalls = true
				for _, tc := range choice.Delta.ToolCalls {
					pItem, ok := pending[tc.Index]
					if !ok {
						pItem = &partial{}
						pending[tc.Index] = pItem
					}
					if tc.ID != "" {
						pItem.ID = tc.ID
					}
					if tc.Function.Name != "" {
						pItem.Name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						pItem.Arguments += tc.Function.Arguments
					}
				}
			}
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}
		if err := scanner.Err(); err != nil {
			out <- LLMStreamEvent{Type: EventError, Err: err}
			return
		}

		if hasToolCalls {
			calls := make([]LLMToolCall, 0, len(pending))
			for i := 0; i < len(pending); i++ {
				item, ok := pending[i]
				if !ok {
					continue
				}
				args := map[string]any{}
				if strings.TrimSpace(item.Arguments) != "" {
					if err := json.Unmarshal([]byte(item.Arguments), &args); err != nil {
						args = map[string]any{"_raw": item.Arguments, "_parse_error": err.Error()}
					}
				}
				if item.ID == "" {
					item.ID = "call_" + strconv.Itoa(i)
				}
				calls = append(calls, LLMToolCall{ID: item.ID, Name: item.Name, Arguments: args})
			}
			if len(calls) > 0 {
				out <- LLMStreamEvent{Type: EventToolCalls, ToolCalls: calls}
			}
		}

		if finishReason == "" {
			finishReason = "stop"
		}
		out <- LLMStreamEvent{Type: EventDone, FinishReason: finishReason}
	}()

	return out, nil
}
