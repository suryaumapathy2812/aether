package plugins

import (
	"context"
	"encoding/json"
	"fmt"
)

type gmailWebhookHandler struct{}

func newGmailWebhookHandler() *gmailWebhookHandler {
	return &gmailWebhookHandler{}
}

func (h *gmailWebhookHandler) BuildTask(_ context.Context, req WebhookRequest) (*WebhookTask, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("invalid gmail payload: %w", err)
	}

	notification, ok := payload["notification"].(map[string]interface{})
	if !ok {
		return &WebhookTask{
			SessionID: "gmail-webhook",
			Title:     "Process Gmail notification",
			Goal:      "A Gmail push notification arrived. Call handle_gmail_event with the provided payload to check for new emails and decide if any action is needed.",
			Metadata: map[string]interface{}{
				"source":     "gmail_webhook",
				"plugin":     "gmail",
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
			"plugin":     "gmail",
			"device_id":  req.DeviceID,
			"message_id": emailData["message_id"],
			"from":       emailData["from"],
			"subject":    emailData["subject"],
			"gmail_data": payload,
		},
	}, nil
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

func init() {
	DefaultWebhookRegistry().Register("gmail", newGmailWebhookHandler())
}
