package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	agentcfg "github.com/suryaumapathy2812/core-ai/agent/internal/config"
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

func (p *fakeProvider) Capabilities() providers.ProviderCapabilities {
	return providers.DefaultCapabilities
}

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

func (p *roundTripProvider) Capabilities() providers.ProviderCapabilities {
	return providers.DefaultCapabilities
}

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
	b := llm.NewContextBuilder(r, nil, nil, nil, llm.ContextBuilderConfig{})
	c := llm.NewCore(&fakeProvider{}, tools.NewOrchestrator(r, tools.ExecContext{}))
	return New(Options{Core: c, Builder: b, Model: "test-model", MediaLimits: agentcfg.MediaLimitsConfig{
		MaxImageBytes: 5 * 1024 * 1024, MaxAudioBytes: 12 * 1024 * 1024,
		MaxTotalMediaBytes: 20 * 1024 * 1024, MaxMediaParts: 4,
	}})
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
	b := llm.NewContextBuilder(r, nil, nil, nil, llm.ContextBuilderConfig{})
	c := llm.NewCore(p, tools.NewOrchestrator(r, tools.ExecContext{}))
	h := New(Options{Core: c, Builder: b, Model: "test-model", MediaLimits: agentcfg.MediaLimitsConfig{
		MaxImageBytes: 5 * 1024 * 1024, MaxAudioBytes: 12 * 1024 * 1024,
		MaxTotalMediaBytes: 20 * 1024 * 1024, MaxMediaParts: 4,
	}})
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

func TestResponsesSync(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"input": []map[string]any{{
			"type":    "message",
			"role":    "user",
			"content": "hi",
		}},
		"stream": false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid json response: %v", err)
	}
	if resp["object"] != "response" {
		t.Fatalf("expected object=response, got %v", resp["object"])
	}
	if !strings.HasPrefix(resp["id"].(string), "resp-") {
		t.Fatalf("expected id to start with resp-, got %v", resp["id"])
	}
	output, ok := resp["output"].([]any)
	if !ok || len(output) == 0 {
		t.Fatalf("expected non-empty output array, got %v", resp["output"])
	}
	item := output[0].(map[string]any)
	if item["type"] != "output_text" {
		t.Fatalf("expected output_text item, got %v", item["type"])
	}
	if !strings.Contains(item["content"].(string), "hello world") {
		t.Fatalf("expected model text in output, got: %s", item["content"])
	}
}

func TestResponsesStream(t *testing.T) {
	h := buildHandler()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"input": []map[string]any{{
			"type":    "message",
			"role":    "user",
			"content": "hi",
		}},
		"stream": true,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	resp := w.Body.String()
	if !strings.Contains(resp, "response.created") {
		t.Fatalf("expected response.created event, got: %s", resp)
	}
	if !strings.Contains(resp, "response.output_text.delta") {
		t.Fatalf("expected response.output_text.delta event, got: %s", resp)
	}
	if !strings.Contains(resp, "response.completed") {
		t.Fatalf("expected response.completed event, got: %s", resp)
	}
	if !strings.Contains(resp, "data: [DONE]") {
		t.Fatalf("expected done event, got: %s", resp)
	}
}

func TestResponsesSyncToolCallRoundTrip(t *testing.T) {
	r := tools.NewRegistry()
	if err := r.Register(&echoTool{}, ""); err != nil {
		t.Fatalf("register tool: %v", err)
	}
	p := &roundTripProvider{test: t}
	b := llm.NewContextBuilder(r, nil, nil, nil, llm.ContextBuilderConfig{})
	c := llm.NewCore(p, tools.NewOrchestrator(r, tools.ExecContext{}))
	h := New(Options{Core: c, Builder: b, Model: "test-model", MediaLimits: agentcfg.MediaLimitsConfig{
		MaxImageBytes: 5 * 1024 * 1024, MaxAudioBytes: 12 * 1024 * 1024,
		MaxTotalMediaBytes: 20 * 1024 * 1024, MaxMediaParts: 4,
	}})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{
		"model": "aether",
		"input": []map[string]any{{
			"type":    "message",
			"role":    "user",
			"content": "use a tool",
		}},
		"stream": false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid json response: %v", err)
	}
	if resp["object"] != "response" {
		t.Fatalf("expected object=response, got %v", resp["object"])
	}
	output, ok := resp["output"].([]any)
	if !ok {
		t.Fatalf("expected output array, got %v", resp["output"])
	}
	// Should have function_call, function_call_output, and output_text items
	hasToolCall := false
	hasToolResult := false
	hasText := false
	for _, item := range output {
		m := item.(map[string]any)
		switch m["type"] {
		case "function_call":
			hasToolCall = true
			if m["name"] != "echo" {
				t.Fatalf("expected tool name echo, got %v", m["name"])
			}
		case "function_call_output":
			hasToolResult = true
			if out, _ := m["output"].(string); !strings.Contains(out, "tool:ping") {
				t.Fatalf("expected tool output containing tool:ping, got %v", m["output"])
			}
		case "output_text":
			hasText = true
			if !strings.Contains(m["content"].(string), "roundtrip ok") {
				t.Fatalf("expected roundtrip ok in text output, got %v", m["content"])
			}
		}
	}
	if !hasToolCall {
		t.Fatalf("expected function_call item in output")
	}
	if !hasToolResult {
		t.Fatalf("expected function_call_output item in output")
	}
	if !hasText {
		t.Fatalf("expected output_text item in output")
	}
}

func TestNormalizeResponsesInput(t *testing.T) {
	input := []map[string]any{
		{"type": "message", "role": "user", "content": "hello"},
		{"type": "function_call", "id": "call_1", "name": "echo", "arguments": `{"text":"hi"}`},
		{"type": "function_call_output", "call_id": "call_1", "output": "echoed"},
		{"content": "plain text"},
	}
	out := normalizeResponsesInput(input)
	if len(out) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(out))
	}
	// message → user role
	if out[0]["role"] != "user" || out[0]["content"] != "hello" {
		t.Fatalf("message normalization failed: %v", out[0])
	}
	// function_call → assistant with tool_calls
	if out[1]["role"] != "assistant" {
		t.Fatalf("function_call should become assistant role: %v", out[1])
	}
	tcs, ok := out[1]["tool_calls"].([]map[string]any)
	if !ok || len(tcs) != 1 {
		t.Fatalf("expected tool_calls array with 1 item: %v", out[1])
	}
	fn, _ := tcs[0]["function"].(map[string]any)
	if fn["name"] != "echo" {
		t.Fatalf("expected tool name echo: %v", fn)
	}
	// function_call_output → tool role
	if out[2]["role"] != "tool" || out[2]["tool_call_id"] != "call_1" || out[2]["content"] != "echoed" {
		t.Fatalf("function_call_output normalization failed: %v", out[2])
	}
	// default → user role
	if out[3]["role"] != "user" || out[3]["content"] != "plain text" {
		t.Fatalf("default normalization failed: %v", out[3])
	}
}
