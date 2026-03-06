package channels

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// HTTPHandler handles incoming webhook requests from different channels
type HTTPHandler struct {
	manager       *Manager
	webhookSecret string
}

// NewHTTPHandler creates a new HTTP handler for channel webhooks
func NewHTTPHandler(manager *Manager, webhookSecret string) *HTTPHandler {
	return &HTTPHandler{
		manager:       manager,
		webhookSecret: webhookSecret,
	}
}

// RegisterRoutes registers the webhook routes to the given mux
func (h *HTTPHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/channels/telegram/webhook", h.handleTelegramWebhook)
}

// handleTelegramWebhook handles incoming Telegram updates
func (h *HTTPHandler) handleTelegramWebhook(w http.ResponseWriter, r *http.Request) {
	// Verify the request is a POST
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Verify webhook secret if configured
	if h.webhookSecret != "" {
		secret := r.Header.Get("X-Telegram-Bot-Api-Secret-Token")
		if secret != h.webhookSecret {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
	}

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Get the Telegram channel
	ch, ok := h.manager.Get(ChannelTypeTelegram)
	if !ok {
		http.Error(w, "telegram channel not configured", http.StatusServiceUnavailable)
		return
	}

	// Parse the update
	update, err := parseTelegramUpdate(body)
	if err != nil {
		log.Printf("failed to parse telegram update: %v\n", err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	// Extract the inbound message
	msg, ok := extractTelegramInboundMessage(update)
	if !ok {
		// Not a message we can process (e.g., callback query, inline query)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Handle the message via the channel
	if err := ch.HandleMessage(r.Context(), msg); err != nil {
		log.Printf("failed to handle telegram message: %v\n", err)
		// Still return 200 to Telegram to avoid retries
	}

	w.WriteHeader(http.StatusOK)
}

// parseTelegramUpdate parses a Telegram update
func parseTelegramUpdate(body []byte) (map[string]any, error) {
	var update map[string]any
	if err := json.Unmarshal(body, &update); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}
	return update, nil
}

// extractTelegramInboundMessage extracts an InboundMessage from a Telegram update
func extractTelegramInboundMessage(update map[string]any) (InboundMessage, bool) {
	// Get message from update
	msg, ok := update["message"].(map[string]any)
	if !ok {
		// Try edited_message
		msg, ok = update["edited_message"].(map[string]any)
		if !ok {
			return InboundMessage{}, false
		}
	}

	// Extract text
	text := ""
	if textIface := msg["text"]; textIface != nil {
		text = strings.TrimSpace(textIface.(string))
	}
	// Also check caption
	if text == "" {
		if captionIface := msg["caption"]; captionIface != nil {
			text = strings.TrimSpace(captionIface.(string))
		}
	}

	// If no text and no media, skip
	if text == "" {
		hasMedia := msg["photo"] != nil || msg["voice"] != nil || msg["video"] != nil || msg["audio"] != nil
		if !hasMedia {
			return InboundMessage{}, false
		}
	}

	// Extract chat ID
	chat, ok := msg["chat"].(map[string]any)
	if !ok {
		return InboundMessage{}, false
	}
	chatIDIface, ok := chat["id"]
	if !ok {
		return InboundMessage{}, false
	}
	var chatID string
	switch v := chatIDIface.(type) {
	case float64:
		chatID = fmt.Sprintf("%.0f", v)
	case string:
		chatID = v
	default:
		return InboundMessage{}, false
	}

	// Extract message ID
	messageID := ""
	if msgID, ok := msg["message_id"].(float64); ok {
		messageID = fmt.Sprintf("%.0f", msgID)
	}

	// Extract date
	timestamp := time.Now()
	if date, ok := msg["date"].(float64); ok {
		timestamp = time.Unix(int64(date), 0)
	}

	// Extract user info
	raw := make(map[string]any)
	if from, ok := msg["from"].(map[string]any); ok {
		if userID, ok := from["id"].(float64); ok {
			raw["user_id"] = fmt.Sprintf("%.0f", userID)
		}
		if username, ok := from["username"].(string); ok {
			raw["username"] = username
		}
		if firstName, ok := from["first_name"].(string); ok {
			raw["first_name"] = firstName
		}
	}

	// Handle /start command with payload
	if strings.HasPrefix(text, "/start ") {
		payload := strings.TrimSpace(strings.TrimPrefix(text, "/start "))
		raw["start_payload"] = payload
		text = "" // Clear text so handler knows this is a /start command
	}

	// Handle media
	mediaURL := ""
	if photo, ok := msg["photo"].([]any); ok && len(photo) > 0 {
		if p, ok := photo[len(photo)-1].(map[string]any); ok {
			if fileID, ok := p["file_id"].(string); ok {
				mediaURL = fileID
			}
		}
	}
	if voice, ok := msg["voice"].(map[string]any); ok {
		if fileID, ok := voice["file_id"].(string); ok {
			mediaURL = fileID
		}
	}

	return InboundMessage{
		Channel:   string(ChannelTypeTelegram),
		ChannelID: chatID,
		MessageID: messageID,
		Text:      text,
		MediaURL:  mediaURL,
		Timestamp: timestamp,
		Raw:       raw,
	}, true
}

// JSONError writes a JSON error response
func JSONError(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{
		"error": message,
	})
}
