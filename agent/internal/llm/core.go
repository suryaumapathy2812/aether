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

type toolErrorClass string

const (
	toolErrorRetryable toolErrorClass = "retryable"
	toolErrorFixable   toolErrorClass = "fixable_input"
	toolErrorDenied    toolErrorClass = "permission"
	toolErrorAuth      toolErrorClass = "auth"
	toolErrorConfig    toolErrorClass = "config"
	toolErrorFatal     toolErrorClass = "fatal"
	toolErrorUnknown   toolErrorClass = "unknown"
)

const (
	loopStateRunning    = "running"
	loopStateRetrying   = "retrying"
	loopStateRecovering = "recovering"
	loopStateBlocked    = "blocked"
	loopStateStopped    = "stopped"
)

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
		emitLoopState(out, env, &seq, loopStateRunning, "")
		seq++
		out <- NewEvent(env.RequestID, env.JobID, EventStartStep, seq, nil)

		opts := providers.GenerateOptions{
			Messages:    env.Messages,
			Tools:       env.Tools,
			Model:       policyString(env.Policy, "model"),
			MaxTokens:   policyInt(env.Policy, "max_tokens", 8192),
			Temperature: policyFloat(env.Policy, "temperature", 0.7),
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
		pendingRecovery := false
		remainingRecovery := policyNonNegativeInt(env.Policy, "tool_recovery_attempts", 1)
		continueLoopOnDeny := policyBool(env.Policy, "continue_loop_on_deny", false)
		toolRetryAttempts := policyNonNegativeInt(env.Policy, "tool_retry_attempts", 1)
		toolRetryBaseDelayMs := policyNonNegativeInt(env.Policy, "tool_retry_base_delay_ms", 400)
		if remainingRecovery < 0 {
			remainingRecovery = 0
		}
		if toolRetryAttempts < 0 {
			toolRetryAttempts = 0
		}
		if toolRetryBaseDelayMs < 50 {
			toolRetryBaseDelayMs = 50
		}

		contextWindowLimit := policyInt(env.Policy, "context_window_limit", 100000)
		compactThreshold := policyFloat(env.Policy, "context_compact_threshold", 0.8)

		for iteration := 0; iteration < maxIter; iteration++ {
			env.Messages = c.compactIfNeeded(ctx, env, out, &seq, contextWindowLimit, compactThreshold)

			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventStartStep, seq, nil)

			opts := providers.GenerateOptions{
				Messages:    env.Messages,
				Tools:       env.Tools,
				Model:       policyString(env.Policy, "model"),
				MaxTokens:   policyInt(env.Policy, "max_tokens", 8192),
				Temperature: policyFloat(env.Policy, "temperature", 0.7),
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
				if attempt > 1 {
					emitLoopState(out, env, &seq, loopStateRetrying, "provider_stream")
				}
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

			pendingToolCalls = c.repairToolCalls(env, pendingToolCalls)

			if len(pendingToolCalls) == 0 {
				if pendingRecovery && remainingRecovery > 0 {
					emitLoopState(out, env, &seq, loopStateRecovering, "tool_error")
					remainingRecovery--
					pendingRecovery = false
					env.Messages = append(env.Messages, map[string]any{
						"role":    "system",
						"content": "A previous tool call failed. If the request is not complete, attempt one corrected tool call (adjust tool choice or arguments) before stopping.",
					})
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, nil)
					continue
				}
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventFinishStep, seq, nil)
				seq++
				out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": finishReason})
				emitLoopState(out, env, &seq, loopStateStopped, finishReason)
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
				emitLoopState(out, env, &seq, loopStateStopped, "error")
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
			results := c.executeToolsConcurrently(ctx, pendingToolCalls, toolRetryAttempts, time.Duration(toolRetryBaseDelayMs)*time.Millisecond)

			// Emit results and append to messages.
			iterationRecoverableToolError := false
			iterationBlocked := false
			for _, tr := range results {
				toolText := tr.result.Output
				if len(toolText) > 12000 {
					toolText = toolText[:12000] + "\n...truncated"
				}
				if tr.result.Error {
					classification := classifyToolError(toolText)
					if classification == toolErrorDenied || classification == toolErrorAuth || classification == toolErrorConfig || classification == toolErrorFatal {
						iterationBlocked = true
					}
					if shouldAttemptRecoveryForToolError(classification, continueLoopOnDeny) {
						iterationRecoverableToolError = true
					}
					seq++
					out <- NewEvent(env.RequestID, env.JobID, EventType("tool-output-error"), seq, map[string]any{"toolCallId": tr.tc.ID, "errorText": toolText, "class": string(classification)})
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
			pendingRecovery = iterationRecoverableToolError
			if iterationBlocked && !iterationRecoverableToolError {
				emitLoopState(out, env, &seq, loopStateBlocked, "tool_error")
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
			Temperature: policyFloat(env.Policy, "temperature", 0.7),
		}
		summaryStream, summaryErr := c.provider.StreamWithTools(ctx, summaryOpts)
		if summaryErr != nil {
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventError, seq, map[string]any{"errorText": "Max steps reached and summary generation failed: " + summaryErr.Error()})
			seq++
			out <- NewEvent(env.RequestID, env.JobID, EventFinish, seq, map[string]any{"finishReason": "length"})
			emitLoopState(out, env, &seq, loopStateStopped, "length")
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
		emitLoopState(out, env, &seq, loopStateStopped, "length")
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

func policyNonNegativeInt(policy map[string]any, key string, fallback int) int {
	v, ok := policy[key]
	if !ok {
		return fallback
	}
	switch n := v.(type) {
	case int:
		if n >= 0 {
			return n
		}
	case int64:
		if n >= 0 {
			return int(n)
		}
	case float64:
		if n >= 0 {
			return int(n)
		}
	}
	return fallback
}

func policyBool(policy map[string]any, key string, fallback bool) bool {
	v, ok := policy[key]
	if !ok {
		return fallback
	}
	b, ok := v.(bool)
	if !ok {
		return fallback
	}
	return b
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

func (c *Core) repairToolCalls(env LLMRequestEnvelope, calls []providers.LLMToolCall) []providers.LLMToolCall {
	if len(calls) == 0 || c.orchestrator == nil {
		return calls
	}
	toolNames := c.orchestrator.ToolNames()
	nameIndex := map[string]string{}
	for _, name := range toolNames {
		nameIndex[strings.ToLower(strings.TrimSpace(name))] = name
	}

	repaired := make([]providers.LLMToolCall, 0, len(calls))
	for _, tc := range calls {
		tc.Name = strings.TrimSpace(tc.Name)

		// 1. Exact match — no repair needed.
		if _, ok := nameIndex[strings.ToLower(tc.Name)]; ok {
			tc.Name = nameIndex[strings.ToLower(tc.Name)]
			repaired = append(repaired, tc)
			continue
		}

		// 2. Case-insensitive match (LLM says "World_Time", registry has "world_time").
		lower := strings.ToLower(tc.Name)
		if canonical, ok := nameIndex[lower]; ok {
			tc.Name = canonical
			repaired = append(repaired, tc)
			continue
		}

		// 3. Plugin-prefixed match (LLM says "google_calendar_upcoming_events", registry has "upcoming_events").
		matched := false
		for lowered, canonical := range nameIndex {
			if strings.HasSuffix(lower, "_"+lowered) || strings.HasSuffix(lower, "."+lowered) {
				tc.Name = canonical
				repaired = append(repaired, tc)
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		// 4. Underscore/hyphen normalization (LLM says "web-search", registry has "web_search").
		normalized := strings.ReplaceAll(lower, "-", "_")
		if canonical, ok := nameIndex[normalized]; ok {
			tc.Name = canonical
			repaired = append(repaired, tc)
			continue
		}

		// 5. No match found — pass through unchanged (orchestrator will return "Unknown tool" error).
		repaired = append(repaired, tc)
	}

	// Strip unknown args for each repaired call.
	for i, tc := range repaired {
		def := c.orchestrator.DefinitionForTool(tc.Name)
		if def == nil || len(def.Parameters) == 0 {
			continue
		}
		knownParams := map[string]bool{}
		for _, p := range def.Parameters {
			knownParams[p.Name] = true
		}
		cleaned := map[string]any{}
		for key, val := range tc.Arguments {
			if knownParams[key] {
				cleaned[key] = val
			}
		}
		// Fill missing required params with defaults.
		for _, p := range def.Parameters {
			if _, ok := cleaned[p.Name]; !ok && p.Default != nil {
				cleaned[p.Name] = p.Default
			}
		}
		repaired[i].Arguments = cleaned
	}

	return repaired
}

const loopStateCompacting = "compacting"

func estimateTokenCount(messages []map[string]any) int {
	total := 0
	for _, msg := range messages {
		content, _ := msg["content"].(string)
		total += len(content) / 4
		if calls, ok := msg["tool_calls"]; ok {
			b, _ := json.Marshal(calls)
			total += len(b) / 4
		}
	}
	return total
}

func (c *Core) compactIfNeeded(ctx context.Context, env LLMRequestEnvelope, out chan<- LLMEventEnvelope, seq *int, windowLimit int, threshold float64) []map[string]any {
	if c.provider == nil || windowLimit <= 0 || threshold <= 0 {
		return env.Messages
	}
	estimated := estimateTokenCount(env.Messages)
	limit := int(float64(windowLimit) * threshold)
	if estimated <= limit {
		return env.Messages
	}

	// Keep system messages and last 4 turns. Summarize everything in between.
	var systemMsgs []map[string]any
	var middle []map[string]any
	var tail []map[string]any

	for _, msg := range env.Messages {
		role, _ := msg["role"].(string)
		if role == "system" {
			systemMsgs = append(systemMsgs, msg)
		} else {
			middle = append(middle, msg)
		}
	}

	keepTail := 4
	if len(middle) <= keepTail+2 {
		return env.Messages
	}

	tail = middle[len(middle)-keepTail:]
	toSummarize := middle[:len(middle)-keepTail]

	emitLoopState(out, env, seq, loopStateCompacting, "context_overflow")

	var summaryBuf strings.Builder
	for _, msg := range toSummarize {
		role, _ := msg["role"].(string)
		content, _ := msg["content"].(string)
		if strings.TrimSpace(content) == "" {
			continue
		}
		summaryBuf.WriteString(role)
		summaryBuf.WriteString(": ")
		if len(content) > 200 {
			summaryBuf.WriteString(content[:200])
			summaryBuf.WriteString("...")
		} else {
			summaryBuf.WriteString(content)
		}
		summaryBuf.WriteString("\n")
	}

	summaryPrompt := []map[string]any{
		{"role": "system", "content": "Summarize the following conversation excerpt in 2-3 sentences. Preserve key facts, decisions, and tool outcomes. Be concise."},
		{"role": "user", "content": summaryBuf.String()},
	}

	summaryOpts := providers.GenerateOptions{
		Messages:    summaryPrompt,
		Tools:       nil,
		Model:       policyString(env.Policy, "model"),
		MaxTokens:   300,
		Temperature: 0.0,
	}

	summaryText := ""
	summaryCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	stream, err := c.provider.StreamWithTools(summaryCtx, summaryOpts)
	if err == nil {
		var sb strings.Builder
		for ev := range stream {
			if ev.Type == providers.EventToken {
				sb.WriteString(ev.Content)
			}
		}
		summaryText = strings.TrimSpace(sb.String())
	}

	if summaryText == "" {
		return env.Messages
	}

	compactedMsg := map[string]any{
		"role":    "system",
		"content": "[Compacted conversation summary]\n" + summaryText,
	}

	result := make([]map[string]any, 0, len(systemMsgs)+1+len(tail))
	result = append(result, systemMsgs...)
	result = append(result, compactedMsg)
	result = append(result, tail...)
	return result
}

func emitLoopState(out chan<- LLMEventEnvelope, env LLMRequestEnvelope, seq *int, state, reason string) {
	*seq = *seq + 1
	payload := map[string]any{"state": state}
	if strings.TrimSpace(reason) != "" {
		payload["reason"] = reason
	}
	out <- NewEvent(env.RequestID, env.JobID, EventType("loop-state"), *seq, payload)
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
	return hasAnyMarker(strings.ToLower(err.Error()), []string{
		"429", "rate limit", "timeout", "temporar", "connection reset", "503", "502", "504", "overloaded",
	})
}

func classifyToolError(text string) toolErrorClass {
	t := strings.ToLower(strings.TrimSpace(text))
	if t == "" {
		return toolErrorUnknown
	}
	if hasAnyMarker(t, []string{"timed out", "timeout", "temporar", "rate limit", "too many requests", "connection reset", "503", "502", "504", "overloaded", "econnreset", "network"}) {
		return toolErrorRetryable
	}
	if hasAnyMarker(t, []string{"invalid value", "invalid query", "invalid argument", "bad request", "malformed", "missing required", "must be"}) {
		return toolErrorFixable
	}
	if hasAnyMarker(t, []string{"permission denied", "access denied", "not allowed", "approval required", "rejected"}) {
		return toolErrorDenied
	}
	if hasAnyMarker(t, []string{"unauthorized", "forbidden", "invalid api key", "invalid credential", "token refresh failed", "oauth", "authentication"}) {
		return toolErrorAuth
	}
	if hasAnyMarker(t, []string{"tool execution is not configured", "disabled plugin", "not connected", "missing config", "not configured"}) {
		return toolErrorConfig
	}
	if hasAnyMarker(t, []string{"fatal", "panic", "segmentation fault"}) {
		return toolErrorFatal
	}
	return toolErrorUnknown
}

func shouldAttemptRecoveryForToolError(class toolErrorClass, continueLoopOnDeny bool) bool {
	switch class {
	case toolErrorRetryable, toolErrorFixable, toolErrorUnknown:
		return true
	case toolErrorDenied:
		return continueLoopOnDeny
	case toolErrorAuth, toolErrorConfig, toolErrorFatal:
		return false
	default:
		return false
	}
}

func hasAnyMarker(text string, markers []string) bool {
	for _, marker := range markers {
		if strings.Contains(text, marker) {
			return true
		}
	}
	return false
}

// toolOutcome holds the input call and execution result from executeToolsConcurrently.
type toolOutcome struct {
	tc     providers.LLMToolCall
	result tools.Result
}

// executeToolsConcurrently runs all tool calls in parallel and waits for completion.
func (c *Core) executeToolsConcurrently(ctx context.Context, calls []providers.LLMToolCall, retryAttempts int, baseDelay time.Duration) []toolOutcome {
	outcomes := make([]toolOutcome, len(calls))
	var wg sync.WaitGroup
	for i, tc := range calls {
		wg.Add(1)
		go func(idx int, call providers.LLMToolCall) {
			defer wg.Done()
			outcomes[idx] = toolOutcome{tc: call, result: c.executeToolWithRetry(ctx, call, retryAttempts, baseDelay)}
		}(i, tc)
	}
	wg.Wait()
	return outcomes
}

func (c *Core) executeToolWithRetry(ctx context.Context, call providers.LLMToolCall, maxRetries int, baseDelay time.Duration) tools.Result {
	if c.orchestrator == nil {
		return tools.Fail("Tool execution is not configured", nil)
	}
	attempt := 0
	for {
		attempt++
		result := c.orchestrator.Execute(ctx, call.Name, call.Arguments, call.ID)
		if !result.Error {
			if attempt > 1 {
				if result.Metadata == nil {
					result.Metadata = map[string]any{}
				}
				result.Metadata["retry_count"] = attempt - 1
			}
			return result
		}
		if attempt >= maxRetries+1 {
			if result.Metadata == nil {
				result.Metadata = map[string]any{}
			}
			result.Metadata["retry_count"] = attempt - 1
			return result
		}
		if classifyToolError(result.Output) != toolErrorRetryable {
			if result.Metadata == nil {
				result.Metadata = map[string]any{}
			}
			result.Metadata["retry_count"] = attempt - 1
			return result
		}
		delay := toolRetryDelay(baseDelay, attempt)
		if retryAfterMs, ok := result.Metadata["retry_after_ms"]; ok {
			if ms, isNum := retryAfterMs.(int64); isNum && ms > 0 {
				serverDelay := time.Duration(ms) * time.Millisecond
				if serverDelay > 30*time.Second {
					serverDelay = 30 * time.Second
				}
				if serverDelay > delay {
					delay = serverDelay
				}
			}
		}
		if !sleepWithContext(ctx, delay) {
			return tools.Fail("Tool retry aborted", map[string]any{"retry_count": attempt - 1})
		}
	}
}

func toolRetryDelay(base time.Duration, attempt int) time.Duration {
	if attempt <= 1 {
		return base
	}
	if attempt == 2 {
		return base * 2
	}
	return base * 4
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
			"temperature": 0.7,
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
