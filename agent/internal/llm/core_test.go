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

type worldTimeTestTool struct{}

func (t *worldTimeTestTool) Definition() tools.Definition {
	return tools.Definition{Name: "world_time", Parameters: []tools.Param{{Name: "timezone", Type: "string", Required: false}}}
}

func (t *worldTimeTestTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	tz, _ := call.Args["timezone"].(string)
	if tz == "" {
		tz = "UTC"
	}
	return tools.Success("time:"+tz, map[string]any{"timezone": tz})
}

type upcomingEventsTestTool struct{}

func (t *upcomingEventsTestTool) Definition() tools.Definition {
	return tools.Definition{Name: "upcoming_events", Parameters: []tools.Param{{Name: "days", Type: "integer", Required: false}}}
}

func (t *upcomingEventsTestTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	_ = call
	return tools.Success(`[{"summary":"Standup"}]`, nil)
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
		case EventToolOutputAvailable:
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

func TestGenerateWithTools_InjectsWorldTimeBeforeUpcomingEventsForRelativeDate(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-upcoming", Name: "upcoming_events", Arguments: map[string]any{"days": 1}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "You have one event tomorrow."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&worldTimeTestTool{}, ""); err != nil {
		t.Fatalf("register world_time tool: %v", err)
	}
	if err := r.Register(&upcomingEventsTestTool{}, "google-calendar"); err != nil {
		t.Fatalf("register upcoming_events tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "What's my calendar looking like for tomorrow?"}}, r.OpenAISchemas()).Normalize()
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	toolInputs := []string{}
	for _, ev := range events {
		if ev.EventType != EventToolInputAvailable {
			continue
		}
		name, _ := ev.Payload["toolName"].(string)
		toolInputs = append(toolInputs, name)
	}
	if len(toolInputs) < 2 {
		t.Fatalf("expected at least 2 tool inputs, got %v", toolInputs)
	}
	if toolInputs[0] != "world_time" {
		t.Fatalf("expected world_time first, got %v", toolInputs)
	}
	if toolInputs[1] != "upcoming_events" {
		t.Fatalf("expected upcoming_events second, got %v", toolInputs)
	}
}

func TestGenerateWithTools_InjectsWorldTimeWhenAssistantAsksForCurrentDate(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToken, Content: "Could you please confirm the current date so I can check tomorrow accurately?"}, {Type: providers.EventDone, FinishReason: "stop"}},
		{{Type: providers.EventToken, Content: "Tomorrow looks clear."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&worldTimeTestTool{}, ""); err != nil {
		t.Fatalf("register world_time tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "What's on my calendar tomorrow?"}}, r.OpenAISchemas()).Normalize()
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	seenWorldTime := false
	seenFinal := false
	for _, ev := range events {
		if ev.EventType == EventToolInputAvailable {
			name, _ := ev.Payload["toolName"].(string)
			if name == "world_time" {
				seenWorldTime = true
			}
		}
		if ev.EventType == EventTextDelta {
			delta, _ := ev.Payload["delta"].(string)
			if delta == "Tomorrow looks clear." {
				seenFinal = true
			}
		}
	}
	if !seenWorldTime {
		t.Fatalf("expected injected world_time tool call")
	}
	if !seenFinal {
		t.Fatalf("expected final answer after retry")
	}
}
