package conversation

import (
	"context"
	"strings"
	"time"

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

type RunOptions struct {
	AckFallback string
}

func (r *Runtime) Run(ctx context.Context, env llm.LLMRequestEnvelope, opts RunOptions) <-chan Event {
	out := make(chan Event, 256)
	go func() {
		defer close(out)
		if r == nil || r.core == nil {
			out <- NewEvent(env.RequestID, 1, EventError, map[string]any{"message": "conversation runtime unavailable"})
			out <- NewEvent(env.RequestID, 2, EventDone, map[string]any{"finish_reason": "error"})
			return
		}

		env = env.Normalize()
		seq := 0
		ackEmitted := false
		answerParts := []string{}
		finishReason := "stop"
		toolArgs := map[string]map[string]any{}

		fallback := strings.TrimSpace(opts.AckFallback)
		if fallback == "" {
			fallback = "Working on that now."
		}

		emitAck := func(text string) {
			text = strings.TrimSpace(text)
			if text == "" || ackEmitted {
				return
			}
			ackEmitted = true
			seq++
			out <- NewEvent(env.RequestID, seq, EventAck, map[string]any{"text": text})
		}

		emitAnswer := func(text string) {
			text = strings.TrimSpace(text)
			if text == "" {
				return
			}
			seq++
			out <- NewEvent(env.RequestID, seq, EventAnswer, map[string]any{"text": text})
		}

		emitAck(fallback)

		for ev := range r.core.GenerateWithTools(ctx, env) {
			switch ev.EventType {
			case llm.EventTextChunk:
				chunk, _ := ev.Payload["text"].(string)
				if strings.TrimSpace(chunk) == "" {
					continue
				}
				answerParts = append(answerParts, chunk)
			case llm.EventToolCall:
				name, _ := ev.Payload["tool_name"].(string)
				callID, _ := ev.Payload["call_id"].(string)
				args, _ := ev.Payload["arguments"].(map[string]any)
				if strings.TrimSpace(callID) != "" {
					toolArgs[callID] = args
				}
				seq++
				out <- NewEvent(env.RequestID, seq, EventToolCall, map[string]any{"tool_name": name, "call_id": callID, "arguments": args})
			case llm.EventToolResult:
				name, _ := ev.Payload["tool_name"].(string)
				errFlag, _ := ev.Payload["error"].(bool)
				output, _ := ev.Payload["output"].(string)
				callID, _ := ev.Payload["call_id"].(string)
				seq++
				out <- NewEvent(env.RequestID, seq, EventToolResult, map[string]any{"tool_name": name, "call_id": callID, "output": output, "error": errFlag, "arguments": toolArgs[callID]})
			case llm.EventStatus:
				seq++
				out <- NewEvent(env.RequestID, seq, EventStatus, ev.Payload)
			case llm.EventError:
				msg, _ := ev.Payload["message"].(string)
				if msg == "" {
					msg = "unknown error"
				}
				seq++
				out <- NewEvent(env.RequestID, seq, EventError, map[string]any{"message": msg})
			case llm.EventStreamEnd:
				if fr, ok := ev.Payload["finish_reason"].(string); ok && strings.TrimSpace(fr) != "" {
					finishReason = fr
				}
			}
		}

		if len(answerParts) > 0 {
			emitAnswer(strings.Join(answerParts, ""))
		}

		seq++
		out <- NewEvent(env.RequestID, seq, EventDone, map[string]any{"finish_reason": finishReason, "ts": time.Now().UTC().Format(time.RFC3339)})
	}()
	return out
}

func BuildAckFallback(messages []map[string]any) string {
	text := strings.TrimSpace(llm.LatestUserMessageText(messages))
	if text == "" {
		return "Working on that now."
	}
	return "Working on that: " + text
}

func ProviderToolCalls(events []providers.LLMStreamEvent) bool {
	for _, ev := range events {
		if ev.Type == providers.EventToolCalls {
			return true
		}
	}
	return false
}
