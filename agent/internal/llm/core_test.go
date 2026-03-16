package llm

import (
	"context"
	"sync/atomic"
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

type failingTool struct{}

func (t *failingTool) Definition() tools.Definition {
	return tools.Definition{Name: "always_fail", Parameters: []tools.Param{{Name: "q", Type: "string", Required: false}}}
}

func (t *failingTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	_ = call
	return tools.Fail("temporary tool failure", nil)
}

type deniedTool struct{}

func (t *deniedTool) Definition() tools.Definition {
	return tools.Definition{Name: "always_denied", Parameters: []tools.Param{{Name: "q", Type: "string", Required: false}}}
}

func (t *deniedTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	_ = call
	return tools.Fail("permission denied", nil)
}

type flakyRetryableTool struct {
	calls int32
}

func (t *flakyRetryableTool) Definition() tools.Definition {
	return tools.Definition{Name: "flaky_retryable", Parameters: []tools.Param{{Name: "q", Type: "string", Required: false}}}
}

func (t *flakyRetryableTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	_ = call
	if atomic.AddInt32(&t.calls, 1) == 1 {
		return tools.Fail("timeout while calling upstream", nil)
	}
	return tools.Success("ok", nil)
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

func TestGenerateWithTools_ContinuesOnceAfterToolErrorWhenModelStopsEarly(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-fail", Name: "always_fail", Arguments: map[string]any{"q": "x"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "I could not complete that."}, {Type: providers.EventDone, FinishReason: "stop"}},
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-echo", Name: "echo", Arguments: map[string]any{"text": "retry"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "Recovered successfully."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&failingTool{}, ""); err != nil {
		t.Fatalf("register failing tool: %v", err)
	}
	if err := r.Register(&echoTool{}, ""); err != nil {
		t.Fatalf("register echo tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "find the file"}}, r.OpenAISchemas()).Normalize()
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	seenFail := false
	seenRetryCall := false
	seenRecovered := false
	for _, ev := range events {
		if ev.EventType == EventType("tool-output-error") {
			seenFail = true
		}
		if ev.EventType == EventToolInputAvailable {
			name, _ := ev.Payload["toolName"].(string)
			if name == "echo" {
				seenRetryCall = true
			}
		}
		if ev.EventType == EventTextDelta {
			delta, _ := ev.Payload["delta"].(string)
			if delta == "Recovered successfully." {
				seenRecovered = true
			}
		}
	}
	if !seenFail {
		t.Fatalf("expected tool output error")
	}
	if !seenRetryCall {
		t.Fatalf("expected a retry tool call after tool error")
	}
	if !seenRecovered {
		t.Fatalf("expected recovered final response")
	}
}

func TestGenerateWithTools_DoesNotForceRecoveryOnNonRecoverableToolError(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-denied", Name: "always_denied", Arguments: map[string]any{"q": "x"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "I do not have permission for that tool."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&deniedTool{}, ""); err != nil {
		t.Fatalf("register denied tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "run denied tool"}}, r.OpenAISchemas()).Normalize()
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	toolInputCount := 0
	seenFinal := false
	for _, ev := range events {
		if ev.EventType == EventToolInputAvailable {
			toolInputCount++
		}
		if ev.EventType == EventTextDelta {
			delta, _ := ev.Payload["delta"].(string)
			if delta == "I do not have permission for that tool." {
				seenFinal = true
			}
		}
	}
	if toolInputCount != 1 {
		t.Fatalf("expected exactly one tool attempt, got %d", toolInputCount)
	}
	if !seenFinal {
		t.Fatalf("expected final response without forced recovery")
	}
}

func TestGenerateWithTools_ContinueLoopOnDenyWhenPolicyEnabled(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-denied", Name: "always_denied", Arguments: map[string]any{"q": "x"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "I cannot access that."}, {Type: providers.EventDone, FinishReason: "stop"}},
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-echo", Name: "echo", Arguments: map[string]any{"text": "fallback"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "Used fallback path."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	if err := r.Register(&deniedTool{}, ""); err != nil {
		t.Fatalf("register denied tool: %v", err)
	}
	if err := r.Register(&echoTool{}, ""); err != nil {
		t.Fatalf("register echo tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "run denied tool"}}, r.OpenAISchemas()).Normalize()
	env.Policy["continue_loop_on_deny"] = true
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	seenFallbackTool := false
	seenFinal := false
	for _, ev := range events {
		if ev.EventType == EventToolInputAvailable {
			name, _ := ev.Payload["toolName"].(string)
			if name == "echo" {
				seenFallbackTool = true
			}
		}
		if ev.EventType == EventTextDelta {
			delta, _ := ev.Payload["delta"].(string)
			if delta == "Used fallback path." {
				seenFinal = true
			}
		}
	}
	if !seenFallbackTool {
		t.Fatalf("expected fallback tool call when continue_loop_on_deny is true")
	}
	if !seenFinal {
		t.Fatalf("expected final fallback response")
	}
}

func TestGenerateWithTools_RetriesRetryableToolErrorBeforeReportingFailure(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-flaky", Name: "flaky_retryable", Arguments: map[string]any{"q": "x"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "Recovered after internal retry."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	flaky := &flakyRetryableTool{}
	if err := r.Register(flaky, ""); err != nil {
		t.Fatalf("register flaky tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "try flaky tool"}}, r.OpenAISchemas()).Normalize()
	env.Policy["tool_retry_attempts"] = 2
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	if got := atomic.LoadInt32(&flaky.calls); got < 2 {
		t.Fatalf("expected at least one internal retry, got calls=%d", got)
	}
	seenToolError := false
	for _, ev := range events {
		if ev.EventType == EventType("tool-output-error") {
			seenToolError = true
		}
	}
	if seenToolError {
		t.Fatalf("expected internal retry to recover without emitting tool-output-error")
	}
}

func TestGenerateWithTools_DoesNotRetryToolWhenRetryPolicyIsZero(t *testing.T) {
	p := &scriptedProvider{steps: [][]providers.LLMStreamEvent{
		{{Type: providers.EventToolCalls, ToolCalls: []providers.LLMToolCall{{ID: "c-flaky", Name: "flaky_retryable", Arguments: map[string]any{"q": "x"}}}}, {Type: providers.EventDone, FinishReason: "tool_calls"}},
		{{Type: providers.EventToken, Content: "Could not complete."}, {Type: providers.EventDone, FinishReason: "stop"}},
	}}
	r := tools.NewRegistry()
	flaky := &flakyRetryableTool{}
	if err := r.Register(flaky, ""); err != nil {
		t.Fatalf("register flaky tool: %v", err)
	}
	o := tools.NewOrchestrator(r, tools.ExecContext{})
	core := NewCore(p, o)

	env := NewBasicEnvelope([]map[string]any{{"role": "user", "content": "try flaky tool"}}, r.OpenAISchemas()).Normalize()
	env.Policy["tool_retry_attempts"] = 0
	env.Policy["tool_recovery_attempts"] = 0
	events := []LLMEventEnvelope{}
	for ev := range core.GenerateWithTools(context.Background(), env) {
		events = append(events, ev)
	}

	if got := atomic.LoadInt32(&flaky.calls); got != 1 {
		t.Fatalf("expected no internal retries, got calls=%d", got)
	}
	seenToolError := false
	for _, ev := range events {
		if ev.EventType == EventType("tool-output-error") {
			seenToolError = true
			break
		}
	}
	if !seenToolError {
		t.Fatalf("expected tool-output-error when retries are disabled")
	}
}
