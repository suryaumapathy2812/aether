package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type googleWorkspaceWebhookHandler struct{}

func newGoogleWorkspaceWebhookHandler() *googleWorkspaceWebhookHandler {
	return &googleWorkspaceWebhookHandler{}
}

func (h *googleWorkspaceWebhookHandler) BuildTask(_ context.Context, req WebhookRequest) (*WebhookTask, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("invalid google-workspace payload: %w", err)
	}

	// Dispatch based on payload content.
	// Gmail payloads have notification.notification.emailAddress or notification.historyId
	// Calendar payloads have eventType or calendar-specific attributes
	if isGmailWebhook(payload) {
		return buildGmailTask(req, payload)
	}
	if isCalendarWebhook(payload) {
		return buildCalendarTask(req, payload)
	}

	// Generic fallback
	return &WebhookTask{
		SessionID: "google-workspace-webhook",
		Title:     "Process Google Workspace notification",
		Goal:      "A Google Workspace push notification arrived. Inspect the payload and determine the appropriate action.",
		Metadata: map[string]interface{}{
			"source":       "google-workspace_webhook",
			"plugin":       "google-workspace",
			"device_id":    req.DeviceID,
			"service_data": payload,
		},
	}, nil
}

func isGmailWebhook(payload map[string]interface{}) bool {
	if notification, ok := payload["notification"].(map[string]interface{}); ok {
		if data, ok := notification["notification"].(map[string]interface{}); ok {
			if _, ok := data["emailAddress"]; ok {
				return true
			}
			if _, ok := data["historyId"]; ok {
				return true
			}
		}
	}
	if _, ok := payload["historyId"]; ok {
		return true
	}
	return false
}

func isCalendarWebhook(payload map[string]interface{}) bool {
	if eventType, ok := payload["eventType"].(string); ok && eventType != "" {
		return true
	}
	if notification, ok := payload["notification"].(map[string]interface{}); ok {
		if attrs, ok := notification["attributes"].(map[string]interface{}); ok {
			if _, ok := attrs["summary"]; ok {
				return true
			}
			if kind, ok := attrs["kind"].(string); ok {
				if strings.Contains(strings.ToLower(kind), "calendar") || strings.Contains(strings.ToLower(kind), "event") {
					return true
				}
			}
		}
	}
	return false
}

func buildGmailTask(req WebhookRequest, payload map[string]interface{}) (*WebhookTask, error) {
	notification, ok := payload["notification"].(map[string]interface{})
	if !ok {
		return &WebhookTask{
			SessionID: "gmail-webhook",
			Title:     "Process Gmail notification",
			Goal:      "A Gmail push notification arrived. Call handle_gmail_event with the provided payload to check for new emails and decide if any action is needed.",
			Metadata: map[string]interface{}{
				"source":     "gmail_webhook",
				"plugin":     "google-workspace",
				"device_id":  req.DeviceID,
				"gmail_data": payload,
			},
		}, nil
	}

	emailData := parseGmailNotification(notification)

	goal := fmt.Sprintf(
		"A new Gmail notification arrived. Email from: %s, Subject: %s. First call handle_gmail_event with the provided payload to fetch details and decide if any action is needed.",
		emailData["from"],
		emailData["subject"],
	)

	return &WebhookTask{
		SessionID: "gmail-notification-" + emailData["message_id"],
		Title:     "Gmail: " + truncateForTitle(emailData["subject"]),
		Goal:      goal,
		Metadata: map[string]interface{}{
			"source":     "gmail_webhook",
			"plugin":     "google-workspace",
			"device_id":  req.DeviceID,
			"message_id": emailData["message_id"],
			"from":       emailData["from"],
			"subject":    emailData["subject"],
			"gmail_data": payload,
		},
	}, nil
}

func buildCalendarTask(req WebhookRequest, payload map[string]interface{}) (*WebhookTask, error) {
	notification, ok := payload["notification"].(map[string]interface{})
	if !ok {
		return &WebhookTask{
			SessionID: "calendar-webhook",
			Title:     "Process Calendar notification",
			Goal:      "A Google Calendar push notification arrived. Call handle_calendar_event with the provided payload to check for calendar changes and decide if any action is needed.",
			Metadata: map[string]interface{}{
				"source":        "calendar_webhook",
				"plugin":        "google-workspace",
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
			"plugin":        "google-workspace",
			"device_id":     req.DeviceID,
			"event_id":      eventData["event_id"],
			"summary":       eventData["summary"],
			"calendar_data": payload,
		},
	}, nil
}

func init() {
	DefaultWebhookRegistry().Register("google-workspace", newGoogleWorkspaceWebhookHandler())
}

func parseGmailNotification(notification map[string]interface{}) map[string]string {
	result := map[string]string{
		"message_id": "",
		"from":       "",
		"subject":    "",
		"snippet":    "",
	}

	if data, ok := notification["notification"].(map[string]interface{}); ok {
		if emailAddress, ok := data["emailAddress"].(string); ok {
			result["from"] = emailAddress
		}
		if historyId, ok := data["historyId"].(string); ok {
			result["history_id"] = historyId
		}
	}

	if attrs, ok := notification["attributes"].(map[string]interface{}); ok {
		if subject, ok := attrs["Subject"].(string); ok {
			result["subject"] = subject
		}
		if from, ok := attrs["From"].(string); ok {
			result["from"] = from
		}
	}

	return result
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
