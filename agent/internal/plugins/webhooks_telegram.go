package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
)

type telegramWebhookHandler struct {
	mu           sync.Mutex
	lastUpdateBy map[string]int64
}

func newTelegramWebhookHandler() *telegramWebhookHandler {
	return &telegramWebhookHandler{lastUpdateBy: map[string]int64{}}
}

func (h *telegramWebhookHandler) BuildTask(_ context.Context, req WebhookRequest) (*WebhookTask, error) {
	var payload map[string]any
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("invalid telegram payload: %w", err)
	}

	message := firstMap(payload, "message", "edited_message", "channel_post")
	if len(message) == 0 {
		return &WebhookTask{
			SessionID: "telegram-system",
			Title:     "Process Telegram update",
			Goal:      "A Telegram update arrived without a message body. Call handle_telegram_event with the provided payload to normalize it and decide if any action is required.",
			Metadata: map[string]any{
				"source":        "telegram_webhook",
				"device_id":     req.DeviceID,
				"plugin":        "telegram",
				"telegram_data": payload,
			},
		}, nil
	}

	chat := asMap(message["chat"])
	from := asMap(message["from"])
	chatID := stringify(chat["id"])
	if chatID == "" {
		chatID = "unknown"
	}

	if updateID, ok := asInt64(payload["update_id"]); ok {
		h.mu.Lock()
		last := h.lastUpdateBy[chatID]
		if updateID <= last {
			h.mu.Unlock()
			return nil, nil
		}
		h.lastUpdateBy[chatID] = updateID
		h.mu.Unlock()
	}

	text := strings.TrimSpace(stringify(message["text"]))
	if text == "" {
		text = strings.TrimSpace(stringify(message["caption"]))
	}
	if text == "" {
		text = "(non-text message)"
	}

	sender := strings.TrimSpace(stringify(from["first_name"]))
	if sender == "" {
		sender = strings.TrimSpace(stringify(from["username"]))
	}
	if sender == "" {
		sender = "unknown sender"
	}

	goal := fmt.Sprintf(
		"A new Telegram webhook event arrived from %s in chat_id=%s. First call handle_telegram_event with payload exactly from metadata.telegram_data. Then decide if a response is needed. If needed, send it using telegram_send_typing and telegram_send_message. Message text: %q",
		sender,
		chatID,
		text,
	)

	return &WebhookTask{
		SessionID: "telegram-chat-" + chatID,
		Title:     "Telegram message: " + truncateForTitle(text),
		Goal:      goal,
		Metadata: map[string]any{
			"source":        "telegram_webhook",
			"device_id":     req.DeviceID,
			"plugin":        "telegram",
			"chat_id":       chatID,
			"sender":        sender,
			"text":          text,
			"telegram_data": payload,
		},
	}, nil
}

func firstMap(payload map[string]any, keys ...string) map[string]any {
	for _, key := range keys {
		if m := asMap(payload[key]); len(m) > 0 {
			return m
		}
	}
	return nil
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func stringify(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case json.Number:
		return x.String()
	case float64:
		return strconv.FormatInt(int64(x), 10)
	case int64:
		return strconv.FormatInt(x, 10)
	case int:
		return strconv.Itoa(x)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func asInt64(v any) (int64, bool) {
	s := strings.TrimSpace(stringify(v))
	if s == "" || s == "<nil>" {
		return 0, false
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

func truncateForTitle(text string) string {
	text = strings.TrimSpace(text)
	if len(text) <= 48 {
		return text
	}
	return text[:45] + "..."
}

func init() {
	DefaultWebhookRegistry().Register("telegram", newTelegramWebhookHandler())
}
