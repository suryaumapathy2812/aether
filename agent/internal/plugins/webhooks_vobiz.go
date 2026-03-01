package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type vobizWebhookHandler struct{}

func (h *vobizWebhookHandler) BuildTask(_ context.Context, req WebhookRequest) (*WebhookTask, error) {
	var payload map[string]any
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("invalid vobiz payload: %w", err)
	}

	from := strings.TrimSpace(stringify(payload["from"]))
	to := strings.TrimSpace(stringify(payload["to"]))
	callID := strings.TrimSpace(firstNonEmpty(stringify(payload["call_uuid"]), stringify(payload["call_id"]), stringify(payload["uuid"])))
	if callID == "" {
		callID = "unknown"
	}

	goal := "A VoBiz telephony webhook event arrived. Review metadata.vobiz_data and determine if a call-related follow-up is needed. Use make_phone_call only when explicitly required."
	if from != "" || to != "" {
		goal = fmt.Sprintf("A VoBiz telephony webhook event arrived (from=%s to=%s call_id=%s). Review metadata.vobiz_data and decide if any follow-up action is needed.", from, to, callID)
	}

	return &WebhookTask{
		SessionID: "vobiz-call-" + callID,
		Title:     "VoBiz event " + callID,
		Goal:      goal,
		Metadata: map[string]any{
			"source":     "vobiz_webhook",
			"plugin":     "vobiz",
			"device_id":  req.DeviceID,
			"call_id":    callID,
			"from":       from,
			"to":         to,
			"vobiz_data": payload,
		},
	}, nil
}

func init() {
	DefaultWebhookRegistry().Register("vobiz", &vobizWebhookHandler{})
}
