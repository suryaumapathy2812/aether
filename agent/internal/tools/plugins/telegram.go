package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

type SendMessageTool struct{}
type SendPhotoTool struct{}
type GetChatInfoTool struct{}
type HandleTelegramEventTool struct{}
type SendTypingIndicatorTool struct{}

func (t *SendMessageTool) Definition() tools.Definition {
	return tools.Definition{Name: "telegram_send_message", Description: "Send Telegram message.", StatusText: "Sending Telegram message...", Parameters: []tools.Param{{Name: "chat_id", Type: "string", Required: true}, {Name: "text", Type: "string", Required: true}, {Name: "parse_mode", Type: "string", Required: false, Default: "Markdown", Enum: []string{"Markdown", "HTML", "MarkdownV2"}}, {Name: "reply_to_message_id", Type: "integer", Required: false}}}
}

func (t *SendMessageTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	chatID, _ := call.Args["chat_id"].(string)
	text, _ := call.Args["text"].(string)
	parseMode, _ := call.Args["parse_mode"].(string)
	if parseMode == "" {
		parseMode = "Markdown"
	}
	payload := map[string]any{"chat_id": chatID, "text": text, "parse_mode": parseMode}
	if id, err := asInt(call.Args["reply_to_message_id"]); err == nil && id > 0 {
		payload["reply_to_message_id"] = id
	}
	obj, err := telegramRequest(ctx, call, "sendMessage", payload)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Message sent.", map[string]any{"chat_id": chatID, "message_id": nestedValue(obj, "result", "message_id")})
}

func (t *SendPhotoTool) Definition() tools.Definition {
	return tools.Definition{Name: "telegram_send_photo", Description: "Send Telegram photo by URL.", StatusText: "Sending photo...", Parameters: []tools.Param{{Name: "chat_id", Type: "string", Required: true}, {Name: "photo_url", Type: "string", Required: true}, {Name: "caption", Type: "string", Required: false}}}
}

func (t *SendPhotoTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	chatID, _ := call.Args["chat_id"].(string)
	photoURL, _ := call.Args["photo_url"].(string)
	caption, _ := call.Args["caption"].(string)
	payload := map[string]any{"chat_id": chatID, "photo": photoURL}
	if strings.TrimSpace(caption) != "" {
		payload["caption"] = caption
		payload["parse_mode"] = "Markdown"
	}
	obj, err := telegramRequest(ctx, call, "sendPhoto", payload)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Photo sent.", map[string]any{"chat_id": chatID, "message_id": nestedValue(obj, "result", "message_id")})
}

func (t *GetChatInfoTool) Definition() tools.Definition {
	return tools.Definition{Name: "telegram_get_chat", Description: "Get Telegram chat information.", StatusText: "Loading chat info...", Parameters: []tools.Param{{Name: "chat_id", Type: "string", Required: true}}}
}

func (t *GetChatInfoTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	chatID, _ := call.Args["chat_id"].(string)
	obj, err := telegramRequest(ctx, call, "getChat", map[string]any{"chat_id": chatID})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["result"])
	return tools.Success(string(b), map[string]any{"chat_id": chatID})
}

func (t *HandleTelegramEventTool) Definition() tools.Definition {
	return tools.Definition{Name: "handle_telegram_event", Description: "Parse incoming Telegram webhook payload.", StatusText: "Processing Telegram event...", Parameters: []tools.Param{{Name: "payload", Type: "object", Required: true}}}
}

func (t *HandleTelegramEventTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	payload, _ := call.Args["payload"].(map[string]any)
	message := payload["message"]
	if message == nil {
		message = payload["edited_message"]
	}
	if message == nil {
		message = payload["channel_post"]
	}
	msg, _ := message.(map[string]any)
	if len(msg) == 0 {
		return tools.Success("Telegram update received with no message payload.", map[string]any{"update_id": payload["update_id"]})
	}
	chat, _ := msg["chat"].(map[string]any)
	from, _ := msg["from"].(map[string]any)
	text, _ := msg["text"].(string)
	if text == "" {
		text, _ = msg["caption"].(string)
	}
	if text == "" {
		text = "(non-text message)"
	}
	chatID := fmt.Sprintf("%v", chat["id"])
	sender := fmt.Sprintf("%v", from["first_name"])
	if strings.TrimSpace(sender) == "" {
		sender = fmt.Sprintf("%v", from["username"])
	}
	out := fmt.Sprintf("Telegram message from %s (chat_id=%s): %s", sender, chatID, text)
	return tools.Success(out, map[string]any{"chat_id": chatID, "text": text, "sender": sender, "message_id": msg["message_id"]})
}

func (t *SendTypingIndicatorTool) Definition() tools.Definition {
	return tools.Definition{Name: "telegram_send_typing", Description: "Show typing indicator in Telegram chat.", StatusText: "Sending typing indicator...", Parameters: []tools.Param{{Name: "chat_id", Type: "string", Required: true}}}
}

func (t *SendTypingIndicatorTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	chatID, _ := call.Args["chat_id"].(string)
	_, err := telegramRequest(ctx, call, "sendChatAction", map[string]any{"chat_id": chatID, "action": "typing"})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Typing indicator sent.", map[string]any{"chat_id": chatID})
}

func telegramRequest(ctx context.Context, call tools.Call, method string, payload map[string]any) (map[string]any, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return nil, err
	}
	botToken, err := requireString(cfg, "bot_token")
	if err != nil {
		return nil, fmt.Errorf("Telegram is not configured: %w", err)
	}
	allowed := strings.TrimSpace(cfg["allowed_chat_ids"])
	if allowed != "" {
		chatID := fmt.Sprintf("%v", payload["chat_id"])
		if chatID != "" {
			ok := false
			for _, c := range strings.Split(allowed, ",") {
				if strings.TrimSpace(c) == chatID {
					ok = true
					break
				}
			}
			if !ok {
				return nil, fmt.Errorf("chat_id %s is not allowed by allowed_chat_ids", chatID)
			}
		}
	}
	b, _ := json.Marshal(payload)
	reqURL := "https://api.telegram.org/bot" + url.PathEscape(botToken) + "/" + method
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, strings.NewReader(string(b)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 15 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Telegram API request failed with status %d", resp.StatusCode)
	}
	if ok, _ := obj["ok"].(bool); !ok {
		return nil, fmt.Errorf("Telegram API returned non-ok response")
	}
	return obj, nil
}

func nestedValue(obj map[string]any, path ...string) any {
	var cur any = obj
	for _, p := range path {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur = m[p]
	}
	return cur
}

var (
	_ tools.Tool = (*SendMessageTool)(nil)
	_ tools.Tool = (*SendPhotoTool)(nil)
	_ tools.Tool = (*GetChatInfoTool)(nil)
	_ tools.Tool = (*HandleTelegramEventTool)(nil)
	_ tools.Tool = (*SendTypingIndicatorTool)(nil)
)
