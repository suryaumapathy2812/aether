package llm

import (
	"context"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type scriptedProvider struct {
	steps [][]providers.LLMStreamEvent
	idx   int
}

func (p *scriptedProvider) Name() string { return "scripted" }

func (p *scriptedProvider) Capabilities() providers.ProviderCapabilities {
	return providers.DefaultCapabilities
}

func (p *scriptedProvider) StreamWithTools(ctx context.Context, opts providers.GenerateOptions) (<-chan providers.LLMStreamEvent, error) {
	_ = ctx
	_ = opts
	out := make(chan providers.LLMStreamEvent, 8)
	i := p.idx
	p.idx++
	go func() {
		defer close(out)
		for _, ev := range p.steps[i] {
			out <- ev
		}
	}()
	return out, nil
}

type echoTool struct{}

func (t *echoTool) Definition() tools.Definition {
	return tools.Definition{Name: "echo", Parameters: []tools.Param{{Name: "text", Type: "string", Required: true}}}
}

func (t *echoTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	v, _ := call.Args["text"].(string)
	return tools.Success("tool:"+v, nil)
}

func TestGenerateWithToolsLoop(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c1", Name: "echo", Arguments: map[string]any{"text": "hi"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "done"}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&echoTool{}, ""); err != nil {
		t.Fatalf("register tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "hello"}}, r.OpenAISchemas()).Normalize()
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	seenTool := false
	seenText := false
	seenEnd := false
	for _, ev := range events {
		switch ev.EventType {
		case EventToolResult:
			seenTool = true
		case EventTextDelta:
			if ev.Payload["delta"] == "done" {
				seenText = true
			}
		case EventFinish:
			seenEnd = true
		}
	}
	if !seenTool {
		t.Fatalf("expected tool result event")
	}
	if !seenText {
		t.Fatalf("expected final text event")
	}
	if !seenEnd {
		t.Fatalf("expected stream end event")
	}
}
