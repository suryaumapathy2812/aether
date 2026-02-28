package ws

import (
	"context"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

// TaskNotifier implements the agent.Notifier interface by broadcasting
// task lifecycle events to connected WebSocket clients.
type TaskNotifier struct {
	hub *Hub
}

// NewTaskNotifier creates a notifier that pushes via the WS hub.
func NewTaskNotifier(hub *Hub) *TaskNotifier {
	return &TaskNotifier{hub: hub}
}

// OnTaskUpdate broadcasts a task_update message to the task's owning user.
func (n *TaskNotifier) OnTaskUpdate(_ context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	if n == nil || n.hub == nil {
		return
	}
	msg := Message{
		Type: "task_update",
		Payload: map[string]any{
			"event":          event,
			"task_id":        task.ID,
			"user_id":        task.UserID,
			"status":         string(task.Status),
			"title":          task.Title,
			"result_summary": task.ResultSummary,
			"step_count":     task.StepCount,
			"max_steps":      task.MaxSteps,
			"extra":          payload,
		},
	}
	n.hub.Broadcast(task.UserID, msg)
}

// BroadcastNotification pushes a notification message to a specific user.
func (n *TaskNotifier) BroadcastNotification(userID string, notification map[string]any) {
	if n == nil || n.hub == nil {
		return
	}
	n.hub.Broadcast(userID, Message{
		Type:    "notification",
		Payload: notification,
	})
}
