package llm

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

const DefaultMaxToolIterations = 500

func randomPartID() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

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
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": err.Error()})
				return
			}

			hadOutput := false
			streamErr := error(nil)
			finishReason := "stop"
			textPartID := ""
			for ev := range stream {
				switch ev.Type {
				case providers.EventToken:
					if ev.Content == "" {
						continue
					}
					hadOutput = true
					if textPartID == "" {
						textPartID = randomPartID()
						seq++
						out <- NewEvent(env.RequestID, env.JobID, EventTextStart, seq, map[string]any{"id": textPartID})
					}
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventTextDelta, seq, map[string]any{"id": textPartID, "delta": ev.Content})
				case providers.EventToolCalls:
					hadOutput = true
					for _, tc := range ev.ToolCalls {
						seq++
						out <- NewEvent(env.RequestID, env.JobID, EventToolInputAvailable, seq, map[string]any{"toolName": tc.Name, "input": tc.Arguments, "toolCallId": tc.ID})
					}
				case providers.EventError:
					if ev.Err != nil {
						streamErr = ev.Err
					} else {
						streamErr = fmt.Errorf("stream error")
					}
				case providers.EventDone:
					finishReason = firstNonEmpty(ev.FinishReason, "stop")
				}
			}
			if textPartID != "" {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventTextEnd, seq, map[string]any{"id": textPartID})
			}
			if streamErr != nil && !hadOutput && attempt < defaultStreamRetryAttempts && isRetryableProviderError(streamErr) {
				if !sleepWithContext(ctx, retryDelay(attempt)) {
					return
				}
				continue
			}
			if streamErr != nil {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": streamErr.Error()})
				return
			}
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, nil)
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": finishReason})
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
			out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": "LLM provider is not configured"})
			return
		}

		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStart, seq, nil)

		maxIter := c.maxToolIterations
		if maxIter <= 0 {
			maxIter = DefaultMaxToolIterations
		}
		recentSignatures := []string{}

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
			textPartID := ""
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
					out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": err.Error()})
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
						if textPartID == "" {
							textPartID = randomPartID()
							seq++
							out <- NewEvent(env.RequestID, env.JobID, EventTextStart, seq, map[string]any{"id": textPartID})
						}
						seq++
						out <- NewEvent(env.RequestID, env.JobID, EventTextDelta, seq, map[string]any{"id": textPartID, "delta": ev.Content})
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
				if textPartID != "" {
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventTextEnd, seq, map[string]any{"id": textPartID})
					textPartID = ""
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
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": streamErr.Error()})
				return
			}

			if len(pendingToolCalls) == 0 {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, nil)
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": finishReason})
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
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": "Detected repeated identical tool calls (doom loop)"})
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": "error"})
				return
			}

			if c.orchestrator == nil {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": "Tool execution is not configured"})
				return
			}

			// Emit tool calls.
			for _, tc := range pendingToolCalls {
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventToolInputAvailable, seq, map[string]any{"toolName": tc.Name, "input": tc.Arguments, "toolCallId": tc.ID})
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
				if tr.result.Error {
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventType("tool-output-error"), seq, map[string]any{"toolCallId": tr.tc.ID, "errorText": toolText})
				} else {
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventToolOutputAvailable, seq, map[string]any{"toolCallId": tr.tc.ID, "output": toolText})
				}

				env.Messages = append(env.Messages, map[string]any{
					"role":         "tool",
					"tool_call_id": tr.tc.ID,
					"content":      tr.result.Output,
				})
			}

			// Emit finish-step for this iteration (tools were called, looping back).
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, nil)
		}

		// Max iterations reached — inject a summarize-and-stop instruction
		// as the last assistant message (like OpenCode's max-steps pattern).
		env.Messages = append(env.Messages, map[string]any{
			"role":    "assistant",
			"content": "MAXIMUM STEPS REACHED. Tools are now disabled. I must provide a text-only summary of what was accomplished, what remains, and recommendations for next steps.",
		})

		// Run one final LLM call with no tools to generate the summary.
		summaryOpts := providers.GenerateOptions{
			Messages:    env.Messages,
			Tools:       []map[string]any{}, // no tools available
			Model:       policyString(env.Policy, "model"),
			MaxTokens:   policyInt(env.Policy, "max_tokens", 8192),
			Temperature: policyFloat(env.Policy, "temperature", 0.2),
		}
		summaryStream, summaryErr := c.provider.StreamWithTools(ctx, summaryOpts)
		if summaryErr != nil {
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": "Max steps reached and summary generation failed: " + summaryErr.Error()})
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": "length"})
			return
		}
		summaryTextID := randomPartID()
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStartStep, seq, nil)
		emittedTextStart := false
		for ev := range summaryStream {
			if ev.Type == providers.EventToken && ev.Content != "" {
				if !emittedTextStart {
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventTextStart, seq, map[string]any{"id": summaryTextID})
					emittedTextStart = true
				}
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventTextDelta, seq, map[string]any{"id": summaryTextID, "delta": ev.Content})
			}
		}
		if emittedTextStart {
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventTextEnd, seq, map[string]any{"id": summaryTextID})
		}
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, nil)
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": "length"})
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
