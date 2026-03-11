package providers

import (
	"context"
	"encoding/json"
)

type LLMToolCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

type LLMStreamEventType string

const (
	EventToken      LLMStreamEventType = "token"
	EventToolCalls  LLMStreamEventType = "tool_calls"
	EventToolCall   LLMStreamEventType = "function_call"
	EventToolResult LLMStreamEventType = "function_call_output"
	EventDone       LLMStreamEventType = "done"
	EventError      LLMStreamEventType = "error"
)

// Usage holds token usage statistics returned by the LLM provider.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type LLMStreamEvent struct {
	Type         LLMStreamEventType
	Content      string
	ToolCalls    []LLMToolCall
	FinishReason string
	Err          error
	CallID       string
	Usage        *Usage // nil when usage is not available
}

type GenerateOptions struct {
	Messages    []map[string]any
	Tools       []map[string]any
	Model       string
	MaxTokens   int
	Temperature float64
}

type ProviderCapabilities struct {
	SupportsChat           bool
	SupportsResponses      bool
	SupportsResponsesWS    bool
	SupportsRealtime       bool
	SupportsStructuredJSON bool
}

var DefaultCapabilities = ProviderCapabilities{
	SupportsChat:           true,
	SupportsResponses:      false,
	SupportsResponsesWS:    false,
	SupportsRealtime:       false,
	SupportsStructuredJSON: false,
}

type LLMProvider interface {
	Name() string
	Capabilities() ProviderCapabilities
	StreamWithTools(ctx context.Context, opts GenerateOptions) (<-chan LLMStreamEvent, error)
}

type ResponsesProvider interface {
	LLMProvider
	StreamResponses(ctx context.Context, opts ResponsesOptions) (<-chan LLMStreamEvent, error)
}

type ResponsesOptions struct {
	Model              string
	Input              []map[string]any
	Tools              []map[string]any
	Instructions       string
	MaxTokens          int
	Temperature        float64
	Store              bool
	PreviousResponseID string
}

type ResponsesStreamEvent struct {
	Type           string
	ResponseID     string
	OutputIndex    int
	Content        string
	CallID         string
	FunctionName   string
	Arguments      string
	ArgumentsDelta string
	FinishReason   string
	Err            error
}

func (e ResponsesStreamEvent) ToLLMStreamEvent() LLMStreamEvent {
	switch e.Type {
	case "response.output_text.delta":
		return LLMStreamEvent{Type: EventToken, Content: e.Content}
	case "response.function_call.done":
		return LLMStreamEvent{
			Type:         EventToolCalls,
			ToolCalls:    []LLMToolCall{{ID: e.CallID, Name: e.FunctionName, Arguments: parseJSON(e.Arguments)}},
			FinishReason: "tool_calls",
		}
	case "response.function_call_output":
		return LLMStreamEvent{Type: EventToolResult, CallID: e.CallID, Content: e.Content}
	case "response.completed", "done":
		return LLMStreamEvent{Type: EventDone, FinishReason: e.FinishReason}
	case "error":
		return LLMStreamEvent{Type: EventError, Err: e.Err}
	default:
		return LLMStreamEvent{}
	}
}

func parseJSON(s string) map[string]any {
	var m map[string]any
	_ = json.Unmarshal([]byte(s), &m)
	return m
}
