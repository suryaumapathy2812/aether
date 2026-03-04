package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

const DefaultMaxToolIterations = 30
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
		opts := providers.GenerateOptions{
			Messages:    env.Messages,
			Tools:       env.Tools,
			Model:       policyString(env.Policy, "model"),
			MaxTokens:   policyInt(env.Policy, "max_tokens", 1200),
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
			for ev := range stream {
				switch ev.Type {
				case providers.EventToken:
					if strings.TrimSpace(ev.Content) == "" {
						continue
					}
					hadOutput = true
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventTextChunk, seq, map[string]any{"text": ev.Content, "role": "assistant"})
				case providers.EventToolCalls:
					hadOutput = true
					for _, tc := range ev.ToolCalls {
						seq++
						out <- NewEvent(env.RequestID, env.JobID, EventToolCall, seq, map[string]any{"tool_name": tc.Name, "arguments": tc.Arguments, "call_id": tc.ID})
					}
				case providers.EventError:
					if ev.Err != nil {
						streamErr = ev.Err
					} else {
						streamErr = fmt.Errorf("stream error")
					}
				case providers.EventDone:
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventStreamEnd, seq, map[string]any{"finish_reason": firstNonEmpty(ev.FinishReason, "stop")})
					return
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
			out <- NewEvent(env.RequestID, env.JobID, EventStreamEnd, seq, map[string]any{"finish_reason": "stop"})
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

		maxIter := c.maxToolIterations
		if maxIter <= 0 {
			maxIter = DefaultMaxToolIterations
		}
		recentSignatures := []string{}

		for iteration := 0; iteration < maxIter; iteration++ {
			opts := providers.GenerateOptions{
				Messages:    env.Messages,
				Tools:       env.Tools,
				Model:       policyString(env.Policy, "model"),
				MaxTokens:   policyInt(env.Policy, "max_tokens", 1200),
				Temperature: policyFloat(env.Policy, "temperature", 0.2),
			}
			assistantText := strings.Builder{}
			pendingToolCalls := make([]providers.LLMToolCall, 0)
			finishReason := "stop"
			streamErr := error(nil)
			hadOutput := false
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
						out <- NewEvent(env.RequestID, env.JobID, EventTextChunk, seq, map[string]any{"text": ev.Content, "role": "assistant"})
					case providers.EventToolCalls:
						hadOutput = true
						pendingToolCalls = append(pendingToolCalls, ev.ToolCalls...)
					case providers.EventDone:
						finishReason = firstNonEmpty(ev.FinishReason, "stop")
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
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventStreamEnd, seq, map[string]any{"finish_reason": finishReason})
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
				out <- NewEvent(env.RequestID, env.JobID, EventStreamEnd, seq, map[string]any{"finish_reason": "doom_loop"})
				return
			}

			if c.orchestrator == nil {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"code": "orchestrator_missing", "message": "Tool execution is not configured"})
				return
			}

			for _, tc := range pendingToolCalls {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventToolCall, seq, map[string]any{"tool_name": tc.Name, "arguments": tc.Arguments, "call_id": tc.ID})

				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventStatus, seq, map[string]any{"message": "Using " + tc.Name + "...", "tool_name": tc.Name})

				result := c.orchestrator.Execute(ctx, tc.Name, tc.Arguments, tc.ID)
				toolText := result.Output
				if len(toolText) > 2000 {
					toolText = toolText[:2000] + "\n...truncated"
				}
				awaitHuman, _ := result.Metadata["await_human"].(bool)
				toolPayload := map[string]any{"tool_name": tc.Name, "output": toolText, "call_id": tc.ID, "error": result.Error, "arguments": tc.Arguments}
				if len(result.Metadata) > 0 {
					toolPayload["metadata"] = result.Metadata
				}
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventToolResult, seq, toolPayload)

				env.Messages = append(env.Messages, map[string]any{
					"role":         "tool",
					"tool_call_id": tc.ID,
					"content":      result.Output,
				})
				if awaitHuman {
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventStreamEnd, seq, map[string]any{"finish_reason": "waiting_input"})
					return
				}
			}
		}

		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventTextChunk, seq, map[string]any{"text": "I've done many tool steps and will stop here.", "role": "assistant"})
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStreamEnd, seq, map[string]any{"finish_reason": "max_iterations"})
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
			"max_tokens":  1200,
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
