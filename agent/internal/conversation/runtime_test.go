package conversation

import (
	"context"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type scriptedProvider struct {
	steps [][]providers.LLMStreamEvent
	idx   int
}

func (p *scriptedProvider) Name() string { return "scripted" }

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

func TestConversationAckAndAnswer(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToken, Content: "here you go"}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := llm.NewCore(p, o)
	rt := NewRuntime(RuntimeOptions{Core: core})

	env := llm.NewBasicEnvelope([]map[string]any{{"role": "user", "content": "hello"}}, r.OpenAISchemas()).Normalize()
	ackSeen := false
	answerSeen := false
	for ev := range rt.Run(context.Background(), env, RunOptions{AckFallback: "Working..."}) {
		switch ev.EventType {
		case EventAck:
			ackSeen = true
		case EventAnswer:
			if ev.Payload["text"] == "here you go" {
				answerSeen = true
			}
		}
	}
	if !ackSeen {
		t.Fatalf("expected ack event")
	}
	if !answerSeen {
		t.Fatalf("expected answer event")
	}
}

func TestConversationToolFlow(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c1", Name: "echo", Arguments: map[string]any{"text": "hi"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "done"}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&echoTool{}, ""); err != nil {
		t.Fatalf("register tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := llm.NewCore(p, o)
	rt := NewRuntime(RuntimeOptions{Core: core})

	env := llm.NewBasicEnvelope([]map[string]any{{"role": "user", "content": "hello"}}, r.OpenAISchemas()).Normalize()
	ackSeen := false
	toolSeen := false
	answerSeen := false
	for ev := range rt.Run(context.Background(), env, RunOptions{AckFallback: "Checking"}) {
		switch ev.EventType {
		case EventAck:
			ackSeen = true
		case EventToolResult:
			toolSeen = true
		case EventAnswer:
			if ev.Payload["text"] == "done" {
				answerSeen = true
			}
		}
	}
	if !ackSeen {
		t.Fatalf("expected ack event")
	}
	if !toolSeen {
		t.Fatalf("expected tool result event")
	}
	if !answerSeen {
		t.Fatalf("expected answer event")
	}
}
