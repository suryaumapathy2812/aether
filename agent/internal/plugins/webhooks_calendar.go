package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type calendarWebhookHandler struct{}

func newCalendarWebhookHandler() *calendarWebhookHandler {
	return &calendarWebhookHandler{}
}

func (h *calendarWebhookHandler) BuildTask(_ context.Context, req WebhookRequest) (*WebhookTask, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("invalid calendar payload: %w", err)
	}

	notification, ok := payload["notification"].(map[string]interface{})
	if !ok {
		return &WebhookTask{
			SessionID: "calendar-webhook",
			Title:     "Process Calendar notification",
			Goal:      "A Google Calendar push notification arrived. Call handle_calendar_event with the provided payload to check for calendar changes and decide if any action is needed.",
			Metadata: map[string]interface{}{
				"source":        "calendar_webhook",
				"plugin":        "google-calendar",
				"device_id":     req.DeviceID,
				"calendar_data": payload,
			},
		}, nil
	}

	eventData := parseCalendarNotification(notification)

	goal := fmt.Sprintf(
		"A new Calendar notification arrived. Summary: %s. First call handle_calendar_event with the provided payload to fetch details and decide if any action is needed.",
		eventData["summary"],
	)

	return &WebhookTask{
		SessionID: "calendar-notification-" + eventData["event_id"],
		Title:     "Calendar: " + truncateForTitle(eventData["summary"]),
		Goal:      goal,
		Metadata: map[string]interface{}{
			"source":        "calendar_webhook",
			"plugin":        "google-calendar",
			"device_id":     req.DeviceID,
			"event_id":      eventData["event_id"],
			"summary":       eventData["summary"],
			"calendar_data": payload,
		},
	}, nil
}

func parseCalendarNotification(notification map[string]interface{}) map[string]string {
	result := map[string]string{
		"event_id": "",
		"summary":  "",
		"kind":     "",
	}

	if attrs, ok := notification["attributes"].(map[string]interface{}); ok {
		if summary, ok := attrs["summary"].(string); ok {
			result["summary"] = summary
		}
		if eventId, ok := attrs["event_id"].(string); ok {
			result["event_id"] = eventId
		}
		if kind, ok := attrs["kind"].(string); ok {
			result["kind"] = kind
		}
	}

	return result
}

func truncateForCalendarTitle(text string) string {
	text = strings.TrimSpace(text)
	if len(text) <= 48 {
		return text
	}
	return text[:45] + "..."
}

func init() {
	DefaultWebhookRegistry().Register("google-calendar", newCalendarWebhookHandler())
}
