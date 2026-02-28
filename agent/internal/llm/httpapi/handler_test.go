package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type fakeProvider struct{}

type echoTool struct{}

func (t *echoTool) Definition() tools.Definition {
	return tools.Definition{Name: "echo", Description: "echo text", Parameters: []tools.Param{{Name: "text", Type: "string", Required: true}}}
}

func (t *echoTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	v, _ := call.Args["text"].(string)
	return tools.Success("tool:"+v, nil)
}

type roundTripProvider struct {
	test          *testing.T
	calls         int
	sawAssistant  bool
	sawToolResult bool
}

func (p *fakeProvider) Name() string { return "fake" }

func (p *fakeProvider) StreamWithTools(ctx context.Context, opts providers.GenerateOptions) (<-chan providers.LLMStreamEvent, error) {
	_ = ctx
	_ = opts
	out := make(chan providers.LLMStreamEvent, 3)
	go func() {
		defer close(out)
		out <- providers.LLMStreamEvent{Type: providers.EventToken, Content: "hello"}
		out <- providers.LLMStreamEvent{Type: providers.EventToken, Content: " world"}
		out <- providers.LLMStreamEvent{Type: providers.EventDone, FinishReason: "stop"}
	}()
	return out, nil
}

func (p *roundTripProvider) Name() string { return "roundtrip" }

func (p *roundTripProvider) StreamWithTools(ctx context.Context, opts providers.GenerateOptions) (<-chan providers.LLMStreamEvent, error) {
	_ = ctx
	out := make(chan providers.LLMStreamEvent, 4)
	p.calls++
	call := p.calls
	go func() {
		defer close(out)
		if call == 1 {
			out <- providers.LLMStreamEvent{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{
				ID:        "call_1",
				Name:      "echo",
				Arguments: map[string]any{"text": "ping"},
			}}}
			out <- providers.LLMStreamEvent{Type: providers.EventDone, FinishReason: "tool_calls"}
			return
		}

		for _, msg := range opts.Messages {
			role, _ := msg["role"].(string)
			if role == "assistant" {
				if calls, ok := msg["tool_calls"].([]map[string]any); ok && len(calls) > 0 {
					fn, _ := calls[0]["function"].(map[string]any)
					if fn["name"] == "echo" {
						if argStr, ok := fn["arguments"].(string); ok && strings.Contains(argStr, "ping") {
							p.sawAssistant = true
						}
					}
				} else if rawCalls, ok := msg["tool_calls"].([]any); ok && len(rawCalls) > 0 {
					if c0, ok := rawCalls[0].(map[string]any); ok {
						fn, _ := c0["function"].(map[string]any)
						if fn["name"] == "echo" {
							if argStr, ok := fn["arguments"].(string); ok && strings.Contains(argStr, "ping") {
								p.sawAssistant = true
							}
						}
					}
				}
			}
			if role == "tool" {
				if msg["tool_call_id"] == "call_1" {
					if content, _ := msg["content"].(string); strings.Contains(content, "tool:ping") {
						p.sawToolResult = true
					}
				}
			}
		}

		out <- providers.LLMStreamEvent{Type: providers.EventToken, Content: "roundtrip ok"}
		out <- providers.LLMStreamEvent{Type: providers.EventDone, FinishReason: "stop"}
	}()
	return out, nil
}

func buildHandler() *Handler {
	r := tools.NewRegistry()
	b := llm.NewContextBuilder(r, nil, nil, nil)
	c := llm.NewCore(&fakeProvider{}, tools.NewOrchestrator(r, tools.ExecContext{}))
	return New(Options{Core: c, Builder: b})
}

func TestModelsEndpoint(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "\"object\":\"list\"") {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

func TestChatCompletionsSync(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "hi",
		}},
		"stream": false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hello world") {
		t.Fatalf("expected model text in response: %s", w.Body.String())
	}
}

func TestChatCompletionsStream(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "hi",
		}},
		"stream": true,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	resp := w.Body.String()
	if !strings.Contains(resp, "chat.completion.chunk") {
		t.Fatalf("expected chunk events, got: %s", resp)
	}
	if !strings.Contains(resp, "data: [DONE]") {
		t.Fatalf("expected done event, got: %s", resp)
	}
}

func TestChatCompletionsSyncWithMultimodalInput(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "what is in this image"},
				{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,aGVsbG8="}},
			},
		}},
		"stream": false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hello world") {
		t.Fatalf("expected model text in response: %s", w.Body.String())
	}
}

func TestChatCompletionsRejectInvalidAudioFormat(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "transcribe this"},
				{"type": "input_audio", "input_audio": map[string]any{"data": "aGVsbG8=", "format": "exe"}},
			},
		}},
		"stream": false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "unsupported audio format") {
		t.Fatalf("expected validation message in response: %s", w.Body.String())
	}
}

func TestChatCompletionsToolCallRoundTrip(t *testing.T) {
	r := tools.NewRegistry()
	if err := r.Register(&echoTool{}, ""); err != nil {
		t.Fatalf("register tool: %v", err)
	}
	p := &roundTripProvider{test: t}
	b := llm.NewContextBuilder(r, nil, nil, nil)
	c := llm.NewCore(p, tools.NewOrchestrator(r, tools.ExecContext{}))
	h := New(Options{Core: c, Builder: b})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "use a tool",
		}},
		"stream": false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "roundtrip ok") {
		t.Fatalf("expected final output after tool roundtrip, got: %s", w.Body.String())
	}
	if !p.sawAssistant {
		t.Fatalf("expected assistant tool_call message serialization in second provider call")
	}
	if !p.sawToolResult {
		t.Fatalf("expected tool role message with tool_call_id and output in second provider call")
	}
}
