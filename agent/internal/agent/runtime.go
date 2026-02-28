package agent

import (
	"context"
	"encoding/json"
	"fmt"
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
		r.runTask(ctx, task)
	}
}

func (r *Runtime) runTask(ctx context.Context, task db.AgentTaskRecord) {
	leaseDeadline := time.Now().UTC().Add(r.leaseFor)
	_ = r.store.RenewAgentTaskLease(ctx, task.ID, task.LockToken, leaseDeadline)

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
	if strings.TrimSpace(compactNote) != "" {
		_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": "compacted", "message": compactNote})
	}

	env := r.builder.Build(bounded, map[string]any{"max_tokens": 1200, "temperature": 0.2}, task.UserID, firstNonEmpty(task.SessionID, task.ID))
	env.Kind = "delegated_task"

	_ = r.store.IncrementAgentTaskStep(ctx, task.ID, task.LockToken)
	assistantParts := []string{}
	hadError := ""
	finishReason := "stop"
	pendingToolCalls := []map[string]any{}
	pendingToolCallIDs := map[string]struct{}{}
	assistantFlushedForToolBatch := false
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
	result := map[string]any{"summary": assistant}
	if err := r.store.CompleteAgentTask(ctx, task.ID, task.LockToken, assistant, result); err != nil {
		_ = r.store.FailAgentTask(ctx, task.ID, task.LockToken, "failed to finalize task: "+err.Error())
		task.Status = db.AgentTaskFailed
		r.notify(ctx, task, "failed", map[string]any{"error": err.Error()})
		return
	}
	_ = r.store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"status": string(db.AgentTaskCompleted), "message": "Task completed"})
	task.Status = db.AgentTaskCompleted
	if r.memory != nil && assistant != "" {
		taskStarted := task.CreatedAt
		if task.StartedAt != nil {
			taskStarted = *task.StartedAt
		}
		r.memory.RecordSessionSummary(context.Background(), task.UserID, firstNonEmpty(task.SessionID, task.ID), assistant, taskStarted, time.Now().UTC(), task.StepCount, []string{"delegated_task"})
	}
	r.notify(ctx, task, "completed", map[string]any{"summary": assistant})
}

func (r *Runtime) notify(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	if r == nil || r.notifier == nil {
		return
	}
	r.notifier.OnTaskUpdate(ctx, task, event, payload)
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
