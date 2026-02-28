package db

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestAgentTaskLifecycle(t *testing.T) {
	ctx := context.Background()
	store, err := Open(filepath.Join(t.TempDir(), "state.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	task, err := store.CreateAgentTask(ctx, AgentTaskCreate{UserID: "u1", SessionID: "s1", Title: "title", Goal: "goal", MaxSteps: 5})
	if err != nil {
		t.Fatalf("create task: %v", err)
	}
	if task.Status != AgentTaskQueued {
		t.Fatalf("unexpected initial status: %s", task.Status)
	}

	claimed, err := store.ClaimNextAgentTask(ctx, time.Now().UTC(), 30*time.Second)
	if err != nil {
		t.Fatalf("claim task: %v", err)
	}
	if claimed.ID != task.ID {
		t.Fatalf("claimed unexpected task: %s", claimed.ID)
	}
	if claimed.Status != AgentTaskRunning {
		t.Fatalf("expected running status, got: %s", claimed.Status)
	}

	if err := store.AppendAgentTaskMessage(ctx, task.ID, "assistant", map[string]any{"role": "assistant", "content": "working"}); err != nil {
		t.Fatalf("append task message: %v", err)
	}
	messages, err := store.ListAgentTaskMessages(ctx, task.ID, 100)
	if err != nil {
		t.Fatalf("list messages: %v", err)
	}
	if len(messages) < 2 {
		t.Fatalf("expected at least 2 messages, got %d", len(messages))
	}

	if err := store.AppendAgentTaskEvent(ctx, task.ID, "status", map[string]any{"message": "working"}); err != nil {
		t.Fatalf("append event: %v", err)
	}
	events, err := store.ListAgentTaskEvents(ctx, task.ID, 100)
	if err != nil {
		t.Fatalf("list events: %v", err)
	}
	if len(events) == 0 {
		t.Fatalf("expected events")
	}

	if err := store.CompleteAgentTask(ctx, task.ID, claimed.LockToken, "done", map[string]any{"ok": true}); err != nil {
		t.Fatalf("complete task: %v", err)
	}
	finalTask, err := store.GetAgentTask(ctx, task.ID)
	if err != nil {
		t.Fatalf("get final task: %v", err)
	}
	if finalTask.Status != AgentTaskCompleted {
		t.Fatalf("expected completed status, got: %s", finalTask.Status)
	}
}

func TestAgentTaskCancelRequest(t *testing.T) {
	ctx := context.Background()
	store, err := Open(filepath.Join(t.TempDir(), "state.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	task, err := store.CreateAgentTask(ctx, AgentTaskCreate{UserID: "u2", Title: "t", Goal: "g"})
	if err != nil {
		t.Fatalf("create task: %v", err)
	}
	if err := store.RequestCancelAgentTask(ctx, task.ID); err != nil {
		t.Fatalf("request cancel: %v", err)
	}
	cancel, err := store.IsAgentTaskCancelRequested(ctx, task.ID)
	if err != nil {
		t.Fatalf("check cancel: %v", err)
	}
	if !cancel {
		t.Fatalf("expected cancel request")
	}
}

func TestAgentTaskWaitingAndResume(t *testing.T) {
	ctx := context.Background()
	store, err := Open(filepath.Join(t.TempDir(), "state.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	task, err := store.CreateAgentTask(ctx, AgentTaskCreate{UserID: "u3", SessionID: "s3", Title: "approve", Goal: "need approval"})
	if err != nil {
		t.Fatalf("create task: %v", err)
	}
	claimed, err := store.ClaimNextAgentTask(ctx, time.Now().UTC(), 20*time.Second)
	if err != nil {
		t.Fatalf("claim task: %v", err)
	}
	if err := store.SetAgentTaskWaitingInput(ctx, task.ID, claimed.LockToken, "Approve this?", map[string]any{"question": "Approve this?"}); err != nil {
		t.Fatalf("set waiting input: %v", err)
	}
	waitingTask, err := store.GetAgentTask(ctx, task.ID)
	if err != nil {
		t.Fatalf("get waiting task: %v", err)
	}
	if waitingTask.Status != AgentTaskWaitingInput {
		t.Fatalf("expected waiting_input status, got %s", waitingTask.Status)
	}

	resumed, err := store.ResumeAgentTask(ctx, task.ID, "u3", "Approved. continue")
	if err != nil {
		t.Fatalf("resume task: %v", err)
	}
	if resumed.Status != AgentTaskQueued {
		t.Fatalf("expected queued status after resume, got %s", resumed.Status)
	}
	waitingItems, err := store.ListAgentTasksByUserWithStatus(ctx, "u3", string(AgentTaskWaitingInput), 20)
	if err != nil {
		t.Fatalf("list waiting: %v", err)
	}
	if len(waitingItems) != 0 {
		t.Fatalf("expected no waiting tasks after resume")
	}
}

func TestAgentTaskReject(t *testing.T) {
	ctx := context.Background()
	store, err := Open(filepath.Join(t.TempDir(), "state.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	task, err := store.CreateAgentTask(ctx, AgentTaskCreate{UserID: "u4", SessionID: "s4", Title: "reject", Goal: "wait and reject"})
	if err != nil {
		t.Fatalf("create task: %v", err)
	}
	claimed, err := store.ClaimNextAgentTask(ctx, time.Now().UTC(), 20*time.Second)
	if err != nil {
		t.Fatalf("claim task: %v", err)
	}
	if err := store.SetAgentTaskWaitingInput(ctx, task.ID, claimed.LockToken, "Approve?", map[string]any{"question": "Approve?"}); err != nil {
		t.Fatalf("set waiting input: %v", err)
	}
	rejected, err := store.RejectAgentTask(ctx, task.ID, "u4", "not approved")
	if err != nil {
		t.Fatalf("reject task: %v", err)
	}
	if rejected.Status != AgentTaskCancelled {
		t.Fatalf("expected cancelled status, got %s", rejected.Status)
	}
}
