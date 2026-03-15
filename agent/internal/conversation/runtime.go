package conversation

import (
	"context"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
)

type Runtime struct {
	core *llm.Core
}

type RuntimeOptions struct {
	Core *llm.Core
}

func NewRuntime(opts RuntimeOptions) *Runtime {
	return &Runtime{core: opts.Core}
}

type RunOptions struct{}

// Run streams LLM events through as conversation events.
// The LLM core handles start/step/text/tool/finish lifecycle;
// the runtime maps them to conversation event types.
func (r *Runtime) Run(ctx context.Context, env llm.LLMRequestEnvelope, opts RunOptions) <-chan Event {
	out := make(chan Event, 256)
	go func() {
		defer close(out)
		if r == nil || r.core == nil {
			out <- NewEvent(env.RequestID, 1, EventError, map[string]any{"errorText": "conversation runtime unavailable"})
			out <- NewEvent(env.RequestID, 2, EventFinish, map[string]any{"finishReason": "error"})
			return
		}

		env = env.Normalize()
		seq := 0

		for ev := range r.core.GenerateWithTools(ctx, env) {
			seq++
			switch ev.EventType {
			case llm.EventStart:
				out <- NewEvent(env.RequestID, seq, EventStart, ev.Payload)
			case llm.EventStartStep:
				out <- NewEvent(env.RequestID, seq, EventStartStep, ev.Payload)
			case llm.EventTextDelta:
				delta, _ := ev.Payload["delta"].(string)
				if strings.TrimSpace(delta) == "" {
					continue
				}
				out <- NewEvent(env.RequestID, seq, EventTextDelta, map[string]any{"delta": delta})
			case llm.EventToolCall:
				out <- NewEvent(env.RequestID, seq, EventToolCall, ev.Payload)
			case llm.EventToolResult:
				out <- NewEvent(env.RequestID, seq, EventToolResult, ev.Payload)
			case llm.EventFinishStep:
				out <- NewEvent(env.RequestID, seq, EventFinishStep, ev.Payload)
			case llm.EventFinish:
				out <- NewEvent(env.RequestID, seq, EventFinish, ev.Payload)
			case llm.EventError:
				out <- NewEvent(env.RequestID, seq, EventError, ev.Payload)
			}
		}
	}()
	return out
}

func ProviderToolCalls(events []providers.LLMStreamEvent) bool {
	for _, ev := range events {
		if ev.Type == providers.EventToolCalls {
			return true
		}
	}
	return false
}
