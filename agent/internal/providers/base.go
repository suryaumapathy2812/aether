package providers

import "context"

type LLMToolCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

type LLMStreamEventType string

const (
	EventToken     LLMStreamEventType = "token"
	EventToolCalls LLMStreamEventType = "tool_calls"
	EventDone      LLMStreamEventType = "done"
	EventError     LLMStreamEventType = "error"
)

type LLMStreamEvent struct {
	Type         LLMStreamEventType
	Content      string
	ToolCalls    []LLMToolCall
	FinishReason string
	Err          error
}

type GenerateOptions struct {
	Messages    []map[string]any
	Tools       []map[string]any
	Model       string
	MaxTokens   int
	Temperature float64
}

type LLMProvider interface {
	Name() string
	StreamWithTools(ctx context.Context, opts GenerateOptions) (<-chan LLMStreamEvent, error)
}
