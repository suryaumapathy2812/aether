package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

const DefaultMaxToolIterations = 50
const defaultStreamRetryAttempts = 3

type Core struct {
	provider          providers.LLMProvider
	orchestrator      *tools.Orchestrator
	maxToolIterations int
}

func NewCore(provider providers.LLMProvider, orchestrator *tools.Orchestrator) *Core {
	return &Core{provider: provider, orchestrator: orchestrator, maxToolIterations: DefaultMaxToolIterations}
}

func (c *Core) Generate(ctx context.Context, envelope LLMRequestEnvelope) <-chan LLMEventEnvelope {
	out := make(chan LLMEventEnvelope, 128)
	go func() {
		defer close(out)
		env := envelope.Normalize()
		seq := 0
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStart, seq, nil)
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStartStep, seq, nil)

		opts := providers.GenerateOptions{
			Messages:    env.Messages,
			Tools:       env.Tools,
			Model:       policyString(env.Policy, "model"),
			MaxTokens:   policyInt(env.Policy, "max_tokens", 8192),
			Temperature: policyFloat(env.Policy, "temperature", 0.2),
		}
		attempt := 0
		for {
			attempt++
			stream, err := c.provider.StreamWithTools(ctx, opts)
			if err != nil {
				if attempt < defaultStreamRetryAttempts && isRetryableProviderError(err) {
					if !sleepWithContext(ctx, retryDelay(attempt)) {
						return
					}
					continue
				}
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "provider_error", "message": err.Error(), "recoverable": false})
				return
			}

			hadOutput := false
			streamErr := error(nil)
			finishReason := "stop"
			var usage map[string]any
			for ev := range stream {
				switch ev.Type {
				case providers.EventToken:
					if strings.TrimSpace(ev.Content) == "" {
						continue
					}
					hadOutput = true
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventTextDelta, seq, map[string]any{"delta": ev.Content})
				case providers.EventToolCalls:
					hadOutput = true
					for _, tc := range ev.ToolCalls {
						seq++
						out <- NewEvent(env.RequestID, env.JobID, EventToolCall, seq, map[string]any{"toolName": tc.Name, "input": tc.Arguments, "toolCallId": tc.ID})
					}
				case providers.EventError:
					if ev.Err != nil {
						streamErr = ev.Err
					} else {
						streamErr = fmt.Errorf("stream error")
					}
				case providers.EventDone:
					finishReason = firstNonEmpty(ev.FinishReason, "stop")
					if ev.Usage != nil {
						usage = map[string]any{
							"prompt_tokens":     ev.Usage.PromptTokens,
							"completion_tokens": ev.Usage.CompletionTokens,
							"total_tokens":      ev.Usage.TotalTokens,
						}
					}
				}
			}
			if streamErr != nil && !hadOutput && attempt < defaultStreamRetryAttempts && isRetryableProviderError(streamErr) {
				if !sleepWithContext(ctx, retryDelay(attempt)) {
					return
				}
				continue
			}
			if streamErr != nil {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "stream_error", "message": streamErr.Error(), "recoverable": false})
				return
			}
			seq++
			stepPayload := map[string]any{"finishReason": finishReason}
			if usage != nil {
				stepPayload["usage"] = usage
			}
			out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, stepPayload)
			seq++
			finishPayload := map[string]any{"finishReason": finishReason}
			if usage != nil {
				finishPayload["usage"] = usage
			}
			out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, finishPayload)
			return
		}
	}()
	return out
}

func (c *Core) GenerateWithTools(ctx context.Context, envelope LLMRequestEnvelope) <-chan LLMEventEnvelope {
	out := make(chan LLMEventEnvelope, 256)
	go func() {
		defer close(out)
		env := envelope.Normalize()
		seq := 0
		if c.provider == nil {
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "provider_missing", "message": "LLM provider is not configured"})
			return
		}

		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStart, seq, nil)

		maxIter := c.maxToolIterations
		if maxIter <= 0 {
			maxIter = DefaultMaxToolIterations
		}
		recentSignatures := []string{}

		totalPromptTokens := 0
		totalCompletionTokens := 0
		totalTokens := 0

		for iteration := 0; iteration < maxIter; iteration++ {
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventStartStep, seq, nil)

			opts := providers.GenerateOptions{
				Messages:    env.Messages,
				Tools:       env.Tools,
				Model:       policyString(env.Policy, "model"),
				MaxTokens:   policyInt(env.Policy, "max_tokens", 8192),
				Temperature: policyFloat(env.Policy, "temperature", 0.2),
			}
			assistantText := strings.Builder{}
			pendingToolCalls := make([]providers.LLMToolCall, 0)
			finishReason := "stop"
			streamErr := error(nil)
			hadOutput := false
			hasUsage := false
			attempt := 0
			for {
				attempt++
				stream, err := c.provider.StreamWithTools(ctx, opts)
				if err != nil {
					if attempt < defaultStreamRetryAttempts && isRetryableProviderError(err) {
						if !sleepWithContext(ctx, retryDelay(attempt)) {
							return
						}
						continue
					}
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "provider_error", "message": err.Error(), "recoverable": false})
					return
				}

				for ev := range stream {
					switch ev.Type {
					case providers.EventToken:
						if ev.Content == "" {
							continue
						}
						hadOutput = true
						assistantText.WriteString(ev.Content)
						seq++
						out <- NewEvent(env.RequestID, env.JobID, EventTextDelta, seq, map[string]any{"delta": ev.Content})
					case providers.EventToolCalls:
						hadOutput = true
						pendingToolCalls = append(pendingToolCalls, ev.ToolCalls...)
					case providers.EventDone:
						finishReason = firstNonEmpty(ev.FinishReason, "stop")
						if ev.Usage != nil {
							hasUsage = true
							totalPromptTokens += ev.Usage.PromptTokens
							totalCompletionTokens += ev.Usage.CompletionTokens
							totalTokens += ev.Usage.TotalTokens
						}
					case providers.EventError:
						if ev.Err != nil {
							streamErr = ev.Err
						} else {
							streamErr = fmt.Errorf("stream error")
						}
					}
				}
				if streamErr != nil && !hadOutput && attempt < defaultStreamRetryAttempts && isRetryableProviderError(streamErr) {
					if !sleepWithContext(ctx, retryDelay(attempt)) {
						return
					}
					streamErr = nil
					continue
				}
				break
			}
			if streamErr != nil {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "stream_error", "message": streamErr.Error(), "recoverable": false})
				return
			}

			if len(pendingToolCalls) == 0 {
				// No tool calls — this step is done, session is complete.
				seq++
				stepPayload := map[string]any{"finishReason": finishReason}
				if hasUsage || totalTokens > 0 {
					stepPayload["usage"] = map[string]any{
						"prompt_tokens":     totalPromptTokens,
						"completion_tokens": totalCompletionTokens,
						"total_tokens":      totalTokens,
					}
				}
				out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, stepPayload)
				seq++
				finishPayload := map[string]any{"finishReason": finishReason}
				if hasUsage || totalTokens > 0 {
					finishPayload["usage"] = map[string]any{
						"prompt_tokens":     totalPromptTokens,
						"completion_tokens": totalCompletionTokens,
						"total_tokens":      totalTokens,
					}
				}
				out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, finishPayload)
				return
			}

			env.Messages = append(env.Messages, assistantMessage(assistantText.String(), pendingToolCalls))

			sig := toolCallsSignature(pendingToolCalls)
			recentSignatures = append(recentSignatures, sig)
			if len(recentSignatures) > 3 {
				recentSignatures = recentSignatures[len(recentSignatures)-3:]
			}
			if len(recentSignatures) == 3 && recentSignatures[0] == recentSignatures[1] && recentSignatures[1] == recentSignatures[2] {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "doom_loop", "message": "Detected repeated identical tool calls", "recoverable": false})
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": "doom_loop"})
				return
			}

			if c.orchestrator == nil {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "orchestrator_missing", "message": "Tool execution is not configured"})
				return
			}

			// Emit tool calls.
			for _, tc := range pendingToolCalls {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventToolCall, seq, map[string]any{"toolName": tc.Name, "input": tc.Arguments, "toolCallId": tc.ID})
			}

			// Execute all tool calls concurrently.
			type toolResult struct {
				tc     providers.LLMToolCall
				result tools.Result
			}
			results := make([]toolResult, len(pendingToolCalls))
			var wg sync.WaitGroup
			for i, tc := range pendingToolCalls {
				wg.Add(1)
				go func(idx int, call providers.LLMToolCall) {
					defer wg.Done()
					results[idx] = toolResult{tc: call, result: c.orchestrator.Execute(ctx, call.Name, call.Arguments, call.ID)}
				}(i, tc)
			}
			wg.Wait()

			// Emit results and append to messages.
			for _, tr := range results {
				toolText := tr.result.Output
				if len(toolText) > 12000 {
					toolText = toolText[:12000] + "\n...truncated"
				}
				toolPayload := map[string]any{"toolName": tr.tc.Name, "output": toolText, "toolCallId": tr.tc.ID, "error": tr.result.Error}
				if len(tr.result.Metadata) > 0 {
					toolPayload["metadata"] = tr.result.Metadata
				}
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventToolResult, seq, toolPayload)

				env.Messages = append(env.Messages, map[string]any{
					"role":         "tool",
					"tool_call_id": tr.tc.ID,
					"content":      tr.result.Output,
				})
			}

			// Emit finish-step for this iteration (tools were called, looping back).
			seq++
			stepPayload := map[string]any{"finishReason": "tool-calls"}
			if hasUsage || totalTokens > 0 {
				stepPayload["usage"] = map[string]any{
					"prompt_tokens":     totalPromptTokens,
					"completion_tokens": totalCompletionTokens,
					"total_tokens":      totalTokens,
				}
			}
			out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, stepPayload)
		}

		// Max iterations reached.
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventTextDelta, seq, map[string]any{"delta": "I've done many tool steps and will stop here."})
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": "max_iterations"})
	}()

	return out
}

func assistantMessage(content string, toolCalls []providers.LLMToolCall) map[string]any {
	out := map[string]any{"role": "assistant"}
	if strings.TrimSpace(content) != "" {
		out["content"] = content
	}
	tc := make([]map[string]any, 0, len(toolCalls))
	for _, call := range toolCalls {
		args, _ := json.Marshal(call.Arguments)
		tc = append(tc, map[string]any{
			"id":   call.ID,
			"type": "function",
			"function": map[string]any{
				"name":      call.Name,
				"arguments": string(args),
			},
		})
	}
	out["tool_calls"] = tc
	return out
}

func toolCallsSignature(calls []providers.LLMToolCall) string {
	parts := make([]string, 0, len(calls))
	for _, call := range calls {
		b, _ := json.Marshal(call.Arguments)
		parts = append(parts, call.Name+":"+string(b))
	}
	return strings.Join(parts, "|")
}

func policyString(policy map[string]any, key string) string {
	v, _ := policy[key].(string)
	return strings.TrimSpace(v)
}

func policyInt(policy map[string]any, key string, fallback int) int {
	v, ok := policy[key]
	if !ok {
		return fallback
	}
	switch n := v.(type) {
	case int:
		if n > 0 {
			return n
		}
	case int64:
		if n > 0 {
			return int(n)
		}
	case float64:
		if n > 0 {
			return int(n)
		}
	}
	return fallback
}

func policyFloat(policy map[string]any, key string, fallback float64) float64 {
	v, ok := policy[key]
	if !ok {
		return fallback
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	}
	return fallback
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func retryDelay(attempt int) time.Duration {
	if attempt <= 1 {
		return 300 * time.Millisecond
	}
	if attempt == 2 {
		return 800 * time.Millisecond
	}
	return 1600 * time.Millisecond
}

func sleepWithContext(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func isRetryableProviderError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "429") || strings.Contains(msg, "rate limit") || strings.Contains(msg, "timeout") || strings.Contains(msg, "temporar") || strings.Contains(msg, "connection reset") || strings.Contains(msg, "503") || strings.Contains(msg, "502") || strings.Contains(msg, "504") || strings.Contains(msg, "overloaded") {
		return true
	}
	return false
}

func NewBasicEnvelope(messages []map[string]any, tools []map[string]any) LLMRequestEnvelope {
	return LLMRequestEnvelope{
		Kind:       "reply_text",
		Modality:   "text",
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
		Policy: map[string]any{
			"max_tokens":  8192,
			"temperature": 0.2,
		},
	}
}

func ParseChatMessages(raw []map[string]any) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(raw))
	for i, msg := range raw {
		role, _ := msg["role"].(string)
		if strings.TrimSpace(role) == "" {
			return nil, fmt.Errorf("messages[%d].role is required", i)
		}
		if _, ok := msg["content"]; !ok && role != "assistant" {
			return nil, fmt.Errorf("messages[%d].content is required", i)
		}
		out = append(out, msg)
	}
	return out, nil
}
