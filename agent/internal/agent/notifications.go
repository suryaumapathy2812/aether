package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

type Notifier interface {
	OnTaskUpdate(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any)
}

type NoopNotifier struct{}

func (n NoopNotifier) OnTaskUpdate(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	_ = ctx
	_ = task
	_ = event
	_ = payload
}

type LogNotifier struct{}

func (n LogNotifier) OnTaskUpdate(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	_ = ctx
	b, _ := json.Marshal(payload)
	log.Printf("agent notifier event=%s task_id=%s status=%s payload=%s", event, task.ID, task.Status, string(b))
}

type WebhookNotifier struct {
	url  string
	http *http.Client
}

func NewWebhookNotifier(url string) *WebhookNotifier {
	return &WebhookNotifier{url: strings.TrimSpace(url), http: &http.Client{Timeout: 10 * time.Second}}
}

func (n *WebhookNotifier) OnTaskUpdate(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	if n == nil || strings.TrimSpace(n.url) == "" {
		return
	}
	body := map[string]any{
		"event":   event,
		"task_id": task.ID,
		"user_id": task.UserID,
		"status":  task.Status,
		"title":   task.Title,
		"payload": payload,
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, n.url, bytes.NewReader(b))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := n.http.Do(req)
	if err != nil {
		return
	}
	_ = resp.Body.Close()
}

type MultiNotifier struct {
	notifiers []Notifier
}

func (n MultiNotifier) OnTaskUpdate(ctx context.Context, task db.AgentTaskRecord, event string, payload map[string]any) {
	for _, item := range n.notifiers {
		if item == nil {
			continue
		}
		item.OnTaskUpdate(ctx, task, event, payload)
	}
}

// NewNotifierFromEnv creates a MultiNotifier from environment config.
// Pass optional extra notifiers (e.g. a WebSocketNotifier) to include them.
func NewNotifierFromEnv(extra ...Notifier) Notifier {
	notifiers := []Notifier{LogNotifier{}}
	if webhook := strings.TrimSpace(os.Getenv("AGENT_TASK_WEBHOOK_URL")); webhook != "" {
		notifiers = append(notifiers, NewWebhookNotifier(webhook))
	}
	notifiers = append(notifiers, extra...)
	return MultiNotifier{notifiers: notifiers}
}
