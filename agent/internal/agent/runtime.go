package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type Runtime struct {
	store      *db.Store
	core       *llm.Core
	builder    *llm.ContextBuilder
	subPrompt  string
	window     ContextWindowManager
	workers    int
	pollEvery  time.Duration
	leaseFor   time.Duration
	notifier   Notifier
	memory     *memory.Service
	wg         sync.WaitGroup
	cancelFunc context.CancelFunc
}

type RuntimeOptions struct {
	Store     *db.Store
	Core      *llm.Core
	Builder   *llm.ContextBuilder
	AssetsDir string
	Window    ContextWindowManager
	Workers   int
	PollEvery time.Duration
	LeaseFor  time.Duration
	Notifier  Notifier
	Memory    *memory.Service
}

func NewRuntime(opts RuntimeOptions) *Runtime {
	workers := opts.Workers
	if workers <= 0 {
		workers = 2
	}
	pollEvery := opts.PollEvery
	if pollEvery <= 0 {
		pollEvery = 750 * time.Millisecond
	}
	leaseFor := opts.LeaseFor
	if leaseFor <= 0 {
		leaseFor = 45 * time.Second
	}
	window := opts.Window
	if window.HardChars == 0 && window.SoftChars == 0 {
		window = DefaultContextWindow()
	}
	notifier := opts.Notifier
	if notifier == nil {
		notifier = NoopNotifier{}
	}
	return &Runtime{
		store:     opts.Store,
		core:      opts.Core,
		builder:   opts.Builder,
		subPrompt: loadSubAgentPrompt(opts.AssetsDir),
		window:    window,
		workers:   workers,
		pollEvery: pollEvery,
		leaseFor:  leaseFor,
		notifier:  notifier,
		memory:    opts.Memory,
	}
}

func (r *Runtime) Start(ctx context.Context) error {
	if r.store == nil || r.core == nil || r.builder == nil {
		return fmt.Errorf("agent runtime dependencies are not configured")
	}
	if r.cancelFunc != nil {
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	r.cancelFunc = cancel
	for i := 0; i < r.workers; i++ {
		r.wg.Add(1)
		go func(workerID int) {
			defer r.wg.Done()
			r.workerLoop(runCtx, workerID)
		}(i + 1)
	}
	return nil
}

func (r *Runtime) Stop(ctx context.Context) error {
	if r.cancelFunc != nil {
		r.cancelFunc()
	}
	done := make(chan struct{})
	go func() {
		r.wg.Wait()
		close(done)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		return nil
	}
}

func (r *Runtime) workerLoop(ctx context.Context, workerID int) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		task, err := r.store.ClaimNextAgentTask(ctx, time.Now().UTC(), r.leaseFor)
		if err == db.ErrNotFound {
			if !sleepWithContext(ctx, r.pollEvery) {
				return
			}
			continue
		}
		if err != nil {
			_ = sleepWithContext(ctx, r.pollEvery)
			continue
		}
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"worker_id": workerID, "status": string(task.Status), "message": "Task claimed by runtime worker"})
		r.notify(ctx, task, "claimed", map[string]any{"worker_id": workerID})
		if task.Status == db.AgentTaskVerifying {
			r.runVerificationTask(ctx, task)
			continue
		}
		r.runTask(ctx, task)
	}
}

func (r *Runtime) runTask(ctx context.Context, task db.AgentTaskRecord) {
	leaseDeadline := time.Now().UTC().Add(r.leaseFor)
	_ = r.store.RenewAgentTaskLease(ctx, task.ID, task.LockToken, leaseDeadline)
	if task.MaxSteps > 0 && task.StepCount >= task.MaxSteps {
		errMsg := fmt.Sprintf("max execution cycles reached (%d/%d)", task.StepCount, task.MaxSteps)
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, errMsg)
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "error", map[string]any{"message": errMsg, "code": "max_steps_reached"})
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": errMsg, "phase": "max_steps"})
		return
	}

	if cancel, err := r.store.IsAgentTaskCancelRequested(ctx, task.ID); err == nil && cancel {
		_ = r.store.CancelAgentTask(ctx, task.ID, task.LockToken)
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskCancelled), "message": "Task cancelled before execution"})
		task.Status = db.AgentTaskCancelled
		r.notify(ctx, task, "cancelled", map[string]any{"phase": "pre_execution"})
		return
	}

	messages, err := r.store.ListAgentTaskMessages(ctx, task.ID, 2000)
	if err != nil {
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "failed to load task history: "+err.Error())
		return
	}
	llmMessages := make([]map[string]any, 0, len(messages))
	for _, m := range messages {
		var payload map[string]any
		if err := json.Unmarshal([]byte(m.ContentJSON), &payload); err != nil {
			continue
		}
		llmMessages = append(llmMessages, payload)
	}
	if len(llmMessages) == 0 {
		llmMessages = append(llmMessages, map[string]any{"role": "user", "content": task.Goal})
	}
	bounded, compactNote := r.window.Apply(llmMessages)
	subPrompt := strings.TrimSpace(r.subPrompt)
	if subPrompt == "" {
		subPrompt = "You are the delegated background worker for this task. First plan the work by creating a concise todo/task checklist, then execute the checklist step-by-step. Keep updating progress mentally as you complete items. Do not call delegate_task. Do not say that you delegated the task to someone else. Return concrete execution results with clear outcomes."
	}
	bounded = append([]map[string]any{{
		"role":    "system",
		"content": subPrompt,
	}}, bounded...)
	if strings.TrimSpace(compactNote) != "" {
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": "compacted", "message": compactNote})
	}

	env := r.builder.Build(bounded, r.userModelPolicy(ctx, task.UserID), task.UserID, firstNonEmpty(task.SessionID, task.ID))
	env.Kind = "delegated_task"

	_ = r.store.IncrementAgentTaskStep(ctx, task.ID, task.LockToken)
	assistantParts := []string{}
	hadError := ""
	finishReason := "stop"
	pendingToolCalls := []map[string]any{}
	pendingToolCallIDs := map[string]struct{}{}
	assistantFlushedForToolBatch := false
	toolCallCount := 0
	taskCtx := tools.WithTaskRuntimeContext(ctx, tools.TaskRuntimeContext{TaskID: task.ID, LockToken: task.LockToken, UserID: task.UserID})
	for ev := range r.core.GenerateWithTools(taskCtx, env) {
		if time.Now().UTC().After(leaseDeadline.Add(-10 * time.Second)) {
			leaseDeadline = time.Now().UTC().Add(r.leaseFor)
			_ = r.store.RenewAgentTaskLease(ctx, task.ID, task.LockToken, leaseDeadline)
		}
		switch ev.EventType {
		case llm.EventTextChunk:
			chunk, _ := ev.Payload["text"].(string)
			if strings.TrimSpace(chunk) != "" {
				assistantParts = append(assistantParts, chunk)
			}
		case llm.EventToolCall:
			toolCallCount++
			_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "tool_call", ev.Payload)
			toolName, _ := ev.Payload["tool_name"].(string)
			callID, _ := ev.Payload["call_id"].(string)
			args := ev.Payload["arguments"]
			argsJSON, _ := json.Marshal(args)
			pendingToolCalls = append(pendingToolCalls, map[string]any{
				"id":   callID,
				"type": "function",
				"function": map[string]any{
					"name":      toolName,
					"arguments": string(argsJSON),
				},
			})
			if strings.TrimSpace(callID) != "" {
				pendingToolCallIDs[callID] = struct{}{}
			}
			assistantFlushedForToolBatch = false
		case llm.EventToolResult:
			_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "tool_result", ev.Payload)
			if !assistantFlushedForToolBatch && len(pendingToolCalls) > 0 {
				assistantMsg := map[string]any{
					"role":       "assistant",
					"tool_calls": pendingToolCalls,
				}
				assistantText := strings.TrimSpace(strings.Join(assistantParts, ""))
				if assistantText != "" {
					assistantMsg["content"] = assistantText
				}
				_ = r.store.AppendAgentTaskMessage(ctx, task.ID, "assistant", assistantMsg)
				assistantFlushedForToolBatch = true
				assistantParts = nil
			}
			toolName, _ := ev.Payload["tool_name"].(string)
			toolOutput, _ := ev.Payload["output"].(string)
			toolErr, _ := ev.Payload["error"].(bool)
			callID, _ := ev.Payload["call_id"].(string)
			args := map[string]any{}
			if v, ok := ev.Payload["arguments"].(map[string]any); ok {
				args = v
			}
			if r.memory != nil {
				r.memory.RecordAction(context.Background(), task.UserID, firstNonEmpty(task.SessionID, task.ID), toolName, args, toolOutput, toolErr)
			}
			if strings.TrimSpace(toolOutput) != "" {
				content := toolOutput
				if toolErr {
					content = "[tool_error] " + content
				}
				_ = r.store.AppendAgentTaskMessage(ctx, task.ID, "tool", map[string]any{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      content,
				})
			}
			if strings.TrimSpace(callID) != "" {
				delete(pendingToolCallIDs, callID)
			}
			if len(pendingToolCallIDs) == 0 {
				pendingToolCalls = nil
				assistantFlushedForToolBatch = false
			}
		case llm.EventStatus:
			_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", ev.Payload)
		case llm.EventError:
			hadError, _ = ev.Payload["message"].(string)
			if hadError == "" {
				hadError = "task execution failed"
			}
			_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "error", ev.Payload)
		case llm.EventStreamEnd:
			if fr, ok := ev.Payload["finish_reason"].(string); ok && strings.TrimSpace(fr) != "" {
				finishReason = fr
			}
		}
		if cancel, err := r.store.IsAgentTaskCancelRequested(ctx, task.ID); err == nil && cancel {
			_ = r.store.CancelAgentTask(ctx, task.ID, task.LockToken)
			_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskCancelled), "message": "Task cancelled while running"})
			task.Status = db.AgentTaskCancelled
			r.notify(ctx, task, "cancelled", map[string]any{"phase": "running"})
			return
		}
	}
	assistant := strings.TrimSpace(strings.Join(assistantParts, ""))
	if assistant != "" {
		_ = r.store.AppendAgentTaskMessage(ctx, task.ID, "assistant", map[string]any{"role": "assistant", "content": assistant})
	}
	if finishReason == "waiting_input" {
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskWaitingInput), "message": "Waiting for human input"})
		latest, err := r.store.GetAgentTask(ctx, task.ID)
		if err == nil {
			task = latest
		}
		task.Status = db.AgentTaskWaitingInput
		r.notify(ctx, task, "waiting_input", map[string]any{"question": task.ResultSummary})
		return
	}

	if finishReason == "stop" && likelyNeedsHumanInput(assistant) {
		if err := r.store.SetAgentTaskWaitingInput(ctx, task.ID, task.LockToken, assistant, map[string]any{"question": assistant, "source": "heuristic_fallback"}); err == nil {
			task.Status = db.AgentTaskWaitingInput
			r.notify(ctx, task, "waiting_input", map[string]any{"question": assistant, "source": "heuristic_fallback"})
			return
		}
	}
	if hadError != "" {
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, hadError)
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": hadError})
		return
	}
	if finishReason == "max_iterations" {
		feedback := "Execution reached the tool-call iteration limit for this run. Continue from current progress, process the next batch, and provide concrete results."
		_ = r.store.AppendAgentTaskMessage(ctx, task.ID, "user", map[string]any{
			"role":    "user",
			"content": "System feedback: " + feedback,
		})
		if err := r.store.SetAgentTaskNeedsMoreWork(ctx, task.ID, task.LockToken, feedback); err != nil {
			_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "failed to requeue after max_iterations: "+err.Error())
			task.Status = db.AgentTaskFailed
			r.notify(ctx, task, "failed", map[string]any{"error": err.Error(), "phase": "max_iterations"})
			return
		}
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskNeedsMoreWork), "message": "Execution reached max tool iterations; queued for continued work"})
		task.Status = db.AgentTaskNeedsMoreWork
		r.notify(ctx, task, "needs_more_work", map[string]any{"reason": "max_iterations"})
		return
	}
	if toolCallCount == 0 && likelyMetaDelegationResponse(assistant) {
		errMsg := "delegated task completed without execution: assistant acknowledged delegation instead of performing the work"
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "error", map[string]any{"message": errMsg, "assistant": assistant})
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, errMsg)
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": errMsg})
		return
	}
	result := map[string]any{"summary": assistant}
	if err := r.store.SetAgentTaskVerifyPending(ctx, task.ID, task.LockToken, assistant, result); err != nil {
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "failed to enqueue verifier: "+err.Error())
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": err.Error()})
		return
	}
	_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskVerifyPending), "message": "Task execution finished; queued for verification"})
	task.Status = db.AgentTaskVerifyPending
	if r.memory != nil && assistant != "" {
		taskStarted := task.CreatedAt
		if task.StartedAt != nil {
			taskStarted = *task.StartedAt
		}
		r.memory.RecordSessionSummary(context.Background(), task.UserID, firstNonEmpty(task.SessionID, task.ID), assistant, taskStarted, time.Now().UTC(), task.StepCount, []string{"delegated_task"})
	}
	r.notify(ctx, task, "verify_pending", map[string]any{"summary": assistant})
}

func (r *Runtime) runVerificationTask(ctx context.Context, task db.AgentTaskRecord) {
	leaseDeadline := time.Now().UTC().Add(r.leaseFor)
	_ = r.store.RenewAgentTaskLease(ctx, task.ID, task.LockToken, leaseDeadline)

	decision, err := r.verifyTaskWithLLM(ctx, task)
	if err != nil {
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "verification failed: "+err.Error())
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": err.Error(), "phase": "verification"})
		return
	}
	_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "verification_decision", map[string]any{
		"decision": decision.Decision,
		"reason":   decision.Reason,
		"comments": decision.Comments,
	})
	events, _ := r.store.ListAgentTaskEvents(ctx, task.ID, 500)

	switch decision.Decision {
	case "completed":
		summary := strings.TrimSpace(decision.RevisedSummary)
		if summary == "" {
			summary = strings.TrimSpace(task.ResultSummary)
		}
		result := map[string]any{"summary": summary, "verified": true, "verification_reason": decision.Reason}
		if err := r.store.CompleteAgentTask(ctx, task.ID, task.LockToken, summary, result); err != nil {
			_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "failed to finalize verified task: "+err.Error())
			task.Status = db.AgentTaskFailed
			r.notify(ctx, task, "failed", map[string]any{"error": err.Error(), "phase": "verification_finalize"})
			return
		}
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskCompleted), "message": "Task verified and completed"})
		task.Status = db.AgentTaskCompleted
		r.notify(ctx, task, "completed", map[string]any{"summary": summary, "verified": true})
		return
	case "needs_more_work":
		if countNeedsMoreWorkDecisions(events) >= 3 {
			reason := "verification requested more work too many times (3); failing to avoid infinite loop"
			_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, reason)
			task.Status = db.AgentTaskFailed
			r.notify(ctx, task, "failed", map[string]any{"error": reason, "phase": "verification_retry_exhausted"})
			return
		}
		feedback := strings.TrimSpace(decision.Comments)
		if feedback == "" {
			feedback = strings.TrimSpace(decision.Reason)
		}
		if feedback == "" {
			feedback = "Verification requires additional concrete work and a clearer final result. Continue execution and provide specific outcomes."
		}
		_ = r.store.AppendAgentTaskMessage(ctx, task.ID, "user", map[string]any{
			"role":    "user",
			"content": "Verifier feedback: " + feedback + "\nContinue working on the same task and return a concrete final result.",
		})
		if err := r.store.SetAgentTaskNeedsMoreWork(ctx, task.ID, task.LockToken, feedback); err != nil {
			_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "failed to queue task for more work: "+err.Error())
			task.Status = db.AgentTaskFailed
			r.notify(ctx, task, "failed", map[string]any{"error": err.Error(), "phase": "verification_requeue"})
			return
		}
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskNeedsMoreWork), "message": "Verifier requested more work", "comments": feedback})
		task.Status = db.AgentTaskNeedsMoreWork
		r.notify(ctx, task, "needs_more_work", map[string]any{"comments": feedback})
		return
	default:
		reason := strings.TrimSpace(decision.Reason)
		if reason == "" {
			reason = "verification marked task as failed"
		}
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, reason)
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": reason, "phase": "verification"})
	}
}

func (r *Runtime) notify(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	if r == nil || r.notifier == nil {
		return
	}
	r.notifier.OnTaskUpdate(ctx, task, event, payload)
}

type verificationDecision struct {
	Decision       string `json:"decision"`
	Reason         string `json:"reason"`
	Comments       string `json:"comments"`
	RevisedSummary string `json:"revised_summary"`
}

func (r *Runtime) verifyTaskWithLLM(ctx context.Context, task db.AgentTaskRecord) (verificationDecision, error) {
	decision := verificationDecision{}
	events, _ := r.store.ListAgentTaskEvents(ctx, task.ID, 400)
	evidence := summarizeTaskEvidence(events)
	payload, _ := json.MarshalIndent(map[string]any{
		"task_id":        task.ID,
		"title":          task.Title,
		"goal":           task.Goal,
		"step_count":     task.StepCount,
		"max_steps":      task.MaxSteps,
		"result_summary": task.ResultSummary,
		"evidence":       evidence,
	}, "", "  ")
	messages := []map[string]any{
		{
			"role":    "system",
			"content": "You are a strict delegated-task verifier. Decide whether execution actually completed the requested goal. Return only JSON with keys: decision, reason, comments, revised_summary. decision must be one of: completed, needs_more_work, failed.",
		},
		{
			"role":    "user",
			"content": "Verify this delegated task result and decide completion. Mark needs_more_work if output is vague, incomplete, or missing concrete deliverables.\n\nTask payload:\n" + string(payload),
		},
	}
	env := r.builder.Build(messages, r.userModelPolicy(ctx, task.UserID), task.UserID, firstNonEmpty(task.SessionID, task.ID)+":verify")
	env.Kind = "delegated_task_verifier"
	env.Tools = nil
	env.ToolChoice = "none"
	buf := strings.Builder{}
	hadErr := ""
	for ev := range r.core.Generate(ctx, env) {
		switch ev.EventType {
		case llm.EventTextChunk:
			chunk, _ := ev.Payload["text"].(string)
			buf.WriteString(chunk)
		case llm.EventError:
			hadErr, _ = ev.Payload["message"].(string)
		}
	}
	if strings.TrimSpace(hadErr) != "" {
		return decision, fmt.Errorf(hadErr)
	}
	raw := strings.TrimSpace(buf.String())
	if raw == "" {
		return decision, fmt.Errorf("empty verifier response")
	}
	if err := json.Unmarshal([]byte(raw), &decision); err != nil {
		start := strings.Index(raw, "{")
		end := strings.LastIndex(raw, "}")
		if start >= 0 && end > start {
			if err2 := json.Unmarshal([]byte(raw[start:end+1]), &decision); err2 == nil {
				err = nil
			}
		}
		if err != nil {
			return decision, fmt.Errorf("invalid verifier response: %w", err)
		}
	}
	decision.Decision = strings.TrimSpace(strings.ToLower(decision.Decision))
	if decision.Decision != "completed" && decision.Decision != "needs_more_work" && decision.Decision != "failed" {
		return decision, fmt.Errorf("invalid verifier decision: %q", decision.Decision)
	}
	return decision, nil
}

func summarizeTaskEvidence(events []db.AgentTaskEvent) map[string]any {
	toolCounts := map[string]int{}
	toolErrors := 0
	statusTrail := make([]string, 0, 12)
	notable := make([]string, 0, 12)
	for _, ev := range events {
		payload := map[string]any{}
		_ = json.Unmarshal([]byte(ev.PayloadJSON), &payload)
		switch ev.Kind {
		case "tool_call":
			tool, _ := payload["tool_name"].(string)
			if strings.TrimSpace(tool) != "" {
				toolCounts[tool]++
			}
		case "tool_result":
			if isErr, _ := payload["error"].(bool); isErr {
				toolErrors++
			}
			tool, _ := payload["tool_name"].(string)
			out, _ := payload["output"].(string)
			out = strings.TrimSpace(out)
			if len(notable) < 10 && strings.TrimSpace(tool) != "" && out != "" {
				if len(out) > 180 {
					out = out[:180] + "..."
				}
				notable = append(notable, fmt.Sprintf("%s: %s", tool, out))
			}
		case "status":
			status, _ := payload["status"].(string)
			msg, _ := payload["message"].(string)
			line := strings.TrimSpace(strings.Join([]string{status, msg}, " "))
			if line != "" && len(statusTrail) < 20 {
				statusTrail = append(statusTrail, line)
			}
		}
	}
	return map[string]any{
		"event_count":    len(events),
		"tool_counts":    toolCounts,
		"tool_errors":    toolErrors,
		"status_trail":   statusTrail,
		"notable_output": notable,
	}
}

func countNeedsMoreWorkDecisions(events []db.AgentTaskEvent) int {
	n := 0
	for _, ev := range events {
		if ev.Kind != "verification_decision" {
			continue
		}
		payload := map[string]any{}
		if err := json.Unmarshal([]byte(ev.PayloadJSON), &payload); err != nil {
			continue
		}
		decision, _ := payload["decision"].(string)
		if strings.EqualFold(strings.TrimSpace(decision), "needs_more_work") {
			n++
		}
	}
	return n
}

func loadSubAgentPrompt(assetsDir string) string {
	promptPath := ""
	if strings.TrimSpace(assetsDir) != "" {
		promptPath = filepath.Join(assetsDir, "sub-agent.md")
	} else {
		wd, err := os.Getwd()
		if err == nil {
			promptPath = filepath.Join(wd, "assets", "sub-agent.md")
		}
	}
	if strings.TrimSpace(promptPath) == "" {
		return ""
	}
	b, err := os.ReadFile(promptPath)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
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

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func likelyNeedsHumanInput(text string) bool {
	v := strings.ToLower(strings.TrimSpace(text))
	if v == "" {
		return false
	}
	if strings.HasSuffix(v, "?") {
		return true
	}
	phrases := []string{
		"please provide",
		"could you provide",
		"what should",
		"what is the title",
		"need the title",
		"waiting for your input",
		"let me know",
	}
	for _, p := range phrases {
		if strings.Contains(v, p) {
			return true
		}
	}
	return false
}

func (r *Runtime) userModelPolicy(ctx context.Context, userID string) map[string]any {
	policy := map[string]any{"max_tokens": 1200, "temperature": 0.2}
	if userID == "" {
		return policy
	}
	model, err := r.store.GetUserPreference(ctx, userID, "model")
	if err != nil || model == "" {
		return policy
	}
	policy["model"] = model
	return policy
}

func likelyMetaDelegationResponse(text string) bool {
	v := strings.ToLower(strings.TrimSpace(text))
	if v == "" {
		return false
	}
	phrases := []string{
		"i've delegated",
		"i have delegated",
		"delegated a task",
		"task delegated",
		"background agent",
		"i will notify you once",
		"i'll notify you once",
	}
	for _, p := range phrases {
		if strings.Contains(v, p) {
			return true
		}
	}
	return false
}
