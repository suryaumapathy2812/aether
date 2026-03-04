package builtin

import (
	"context"
	"fmt"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// SendNotificationTool allows the agent to send push notifications to the user.
// It queues a notification in the store for delivery by the notification
// infrastructure and broadcasts via WebSocket when available.
type SendNotificationTool struct{}

func (t *SendNotificationTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "send_notification",
		Description: "Send a push notification to the user. Use this to deliver important proactive findings, briefing summaries, or urgent alerts.",
		StatusText:  "Sending notification...",
		Parameters: []tools.Param{
			{Name: "title", Type: "string", Description: "Notification title (short, attention-grabbing)", Required: true},
			{Name: "body", Type: "string", Description: "Notification body (concise summary of what the user needs to know)", Required: true},
			{Name: "tag", Type: "string", Description: "Notification tag for grouping (e.g. briefing, email, calendar, urgent)", Required: false, Default: "proactive"},
		},
	}
}

func (t *SendNotificationTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("Notification store is unavailable", nil)
	}

	title, _ := call.Args["title"].(string)
	body, _ := call.Args["body"].(string)
	tag, _ := call.Args["tag"].(string)

	title = strings.TrimSpace(title)
	body = strings.TrimSpace(body)
	tag = strings.TrimSpace(tag)

	if title == "" {
		return tools.Fail("title is required", nil)
	}
	if body == "" {
		return tools.Fail("body is required", nil)
	}
	if tag == "" {
		tag = "proactive"
	}

	// Build the notification text as "title: body" for the store
	notificationText := title + ": " + body

	// Queue the notification in the store for delivery
	notifID, err := call.Ctx.Store.QueueMemoryNotification(
		ctx,
		"default",        // userID
		notificationText, // text
		"push",           // deliveryType
		"proactive",      // source
		nil,              // deliverAt (immediate)
		map[string]any{"tag": tag, "title": title},
	)
	if err != nil {
		return tools.Fail("Failed to queue notification: "+err.Error(), nil)
	}

	return tools.Success(
		fmt.Sprintf("Notification sent to user: %q (id=%d, tag=%s)", title, notifID, tag),
		map[string]any{"notification_id": notifID, "tag": tag},
	)
}

var _ tools.Tool = (*SendNotificationTool)(nil)
