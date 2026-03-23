package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/channels"
	"github.com/suryaumapathy2812/core-ai/agent/internal/channels/telegram"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

// Handler handles channel API requests
type Handler struct {
	store          *db.Store
	messageHandler channels.MessageHandler
	webhookBaseURL string // e.g. "https://xyz.trycloudflare.com"
	agentID        string // agent identity for multi-agent webhook routing
}

// NewHandler creates a new channel HTTP handler.
// webhookBaseURL is the public base URL for webhooks (e.g. CHANNELS_WEBHOOK_URL).
// If empty, the webhook URL is derived from the incoming request (won't work behind tunnels).
// agentID is the orchestrator-assigned agent identifier; when set, it is appended to
// the webhook URL so the orchestrator can route inbound webhooks to the correct agent.
func NewHandler(store *db.Store, messageHandler channels.MessageHandler, webhookBaseURL, agentID string) *Handler {
	return &Handler{
		store:          store,
		messageHandler: messageHandler,
		webhookBaseURL: strings.TrimRight(webhookBaseURL, "/"),
		agentID:        strings.TrimSpace(agentID),
	}
}

// RegisterRoutes registers channel routes
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	// Channel management
	for _, path := range []string{"/agent/v1/channels", "/api/channels"} {
		mux.HandleFunc(path, h.handleListChannels)
	}
	for _, path := range []string{"/agent/v1/channels/ios/connect", "/api/channels/ios/connect"} {
		mux.HandleFunc(path, h.handleIOSConnect)
	}
	for _, path := range []string{"/agent/v1/channels/telegram/connect", "/api/channels/telegram/connect"} {
		mux.HandleFunc(path, h.handleTelegramConnect)
	}
	for _, path := range []string{"/agent/v1/channels/telegram/disconnect", "/api/channels/telegram/disconnect"} {
		mux.HandleFunc(path, h.handleTelegramDisconnect)
	}

	// Webhook endpoint for incoming messages
	for _, path := range []string{"/agent/v1/channels/telegram/webhook", "/api/channels/telegram/webhook"} {
		mux.HandleFunc(path, h.handleTelegramWebhook)
	}

	// Channel actions
	for _, path := range []string{"/agent/v1/channels/", "/api/channels/"} {
		mux.HandleFunc(path, h.handleChannelAction)
	}
}

// handleIOSConnect handles POST /agent/v1/channels/ios/connect
func (h *Handler) handleIOSConnect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		UserID      string `json:"user_id"`
		ChannelID   string `json:"channel_id"`
		DisplayName string `json:"display_name"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		channels.JSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	userID := strings.TrimSpace(req.UserID)
	if userID == "" {
		userID = "default"
	}
	channelID := strings.TrimSpace(req.ChannelID)
	if channelID == "" {
		channels.JSONError(w, http.StatusBadRequest, "channel_id is required")
		return
	}
	displayName := strings.TrimSpace(req.DisplayName)
	if displayName == "" {
		displayName = "iOS Device"
	}

	record := db.ChannelRecord{
		UserID:      userID,
		ChannelType: string(channels.ChannelTypeIOS),
		ChannelID:   channelID,
		DisplayName: displayName,
		Enabled:     true,
	}

	saved, err := h.store.UpsertChannel(r.Context(), record)
	if err != nil {
		channels.JSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to save channel: %v", err))
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"success": true,
		"channel": saved,
	})
}

// handleListChannels handles GET /agent/v1/channels
func (h *Handler) handleListChannels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.URL.Query().Get("user_id")
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}

	chs, err := h.store.ListChannels(r.Context(), userID)
	if err != nil {
		channels.JSONError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"channels": chs,
	})
}

// handleTelegramConnect handles POST /agent/v1/channels/telegram/connect
func (h *Handler) handleTelegramConnect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		UserID   string `json:"user_id"`
		BotToken string `json:"bot_token"`
		ChatID   string `json:"chat_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		channels.JSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if strings.TrimSpace(req.BotToken) == "" {
		channels.JSONError(w, http.StatusBadRequest, "bot_token is required")
		return
	}

	userID := strings.TrimSpace(req.UserID)
	if userID == "" {
		userID = "default"
	}

	// Validate the bot token by getting bot info
	ch := telegram.NewTelegramChannel(req.BotToken, h.messageHandler)
	botInfo, err := ch.GetBotInfo(r.Context(), map[string]string{"bot_token": req.BotToken})
	if err != nil {
		channels.JSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid bot token: %v", err))
		return
	}

	// Determine the channel ID to use
	channelID := req.ChatID
	if channelID == "" {
		channels.JSONError(w, http.StatusBadRequest, "chat_id is required. Start a chat with your bot and provide the chat_id")
		return
	}

	// Store the channel configuration
	record := db.ChannelRecord{
		UserID:      userID,
		ChannelType: string(channels.ChannelTypeTelegram),
		ChannelID:   channelID,
		BotToken:    req.BotToken,
		DisplayName: fmt.Sprintf("Telegram (%s)", botInfo.Username),
		Enabled:     true,
	}

	saved, err := h.store.UpsertChannel(r.Context(), record)
	if err != nil {
		channels.JSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to save channel: %v", err))
		return
	}

	// Set the webhook
	webhookURL := h.getWebhookURL(r, channels.ChannelTypeTelegram, userID)
	if webhookURL != "" {
		_ = ch.SetWebhook(r.Context(), webhookURL)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"success":  true,
		"channel":  saved,
		"bot_info": botInfo,
	})
}

// handleTelegramDisconnect handles POST /agent/v1/channels/telegram/disconnect
func (h *Handler) handleTelegramDisconnect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ChannelID string `json:"channel_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		channels.JSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if strings.TrimSpace(req.ChannelID) == "" {
		channels.JSONError(w, http.StatusBadRequest, "channel_id is required")
		return
	}

	// Get the channel first to delete the webhook
	channel, err := h.store.GetChannel(r.Context(), req.ChannelID)
	if err != nil {
		channels.JSONError(w, http.StatusNotFound, "channel not found")
		return
	}

	// Delete webhook
	if channel.ChannelType == string(channels.ChannelTypeTelegram) && channel.BotToken != "" {
		ch := telegram.NewTelegramChannel(channel.BotToken, h.messageHandler)
		_ = ch.SetWebhook(r.Context(), "") // Clear webhook
	}

	// Delete the channel
	if err := h.store.DeleteChannel(r.Context(), req.ChannelID); err != nil {
		channels.JSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to delete channel: %v", err))
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"success": true,
	})
}

// handleChannelAction handles channel-specific actions
func (h *Handler) handleChannelAction(w http.ResponseWriter, r *http.Request) {
	// Extract channel ID from path
	path := strings.TrimPrefix(strings.TrimPrefix(r.URL.Path, "/agent/v1/channels/"), "/api/channels/")
	parts := strings.Split(path, "/")
	if len(parts) < 1 {
		channels.JSONError(w, http.StatusBadRequest, "invalid path")
		return
	}

	channelID := parts[0]
	action := ""
	if len(parts) > 1 {
		action = parts[1]
	}

	switch r.Method {
	case http.MethodGet:
		// Get channel info
		channel, err := h.store.GetChannel(r.Context(), channelID)
		if err != nil {
			channels.JSONError(w, http.StatusNotFound, "channel not found")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(channel)

	case http.MethodPost:
		switch action {
		case "enable":
			if err := h.store.SetChannelEnabled(r.Context(), channelID, true); err != nil {
				channels.JSONError(w, http.StatusInternalServerError, err.Error())
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"success": true})

		case "disable":
			if err := h.store.SetChannelEnabled(r.Context(), channelID, false); err != nil {
				channels.JSONError(w, http.StatusInternalServerError, err.Error())
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"success": true})

		case "send":
			var req struct {
				Text string `json:"text"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				channels.JSONError(w, http.StatusBadRequest, "invalid request body")
				return
			}

			channel, err := h.store.GetChannel(r.Context(), channelID)
			if err != nil {
				channels.JSONError(w, http.StatusNotFound, "channel not found")
				return
			}

			if !channel.Enabled {
				channels.JSONError(w, http.StatusBadRequest, "channel is disabled")
				return
			}

			// Send message via the appropriate channel
			if channel.ChannelType == string(channels.ChannelTypeTelegram) {
				ch := telegram.NewTelegramChannel(channel.BotToken, h.messageHandler)
				err = ch.SendMessage(r.Context(), channels.OutboundMessage{
					ChannelID: channel.ChannelID,
					Text:      req.Text,
				})
				if err != nil {
					channels.JSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to send message: %v", err))
					return
				}
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"success": true})

		default:
			http.Error(w, "unknown action", http.StatusBadRequest)
		}

	case http.MethodDelete:
		if err := h.store.DeleteChannel(r.Context(), channelID); err != nil {
			channels.JSONError(w, http.StatusInternalServerError, err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleTelegramWebhook handles incoming Telegram webhook updates.
// It parses the update, looks up the channel record by chat_id,
// and dispatches the message to the messageHandler with full metadata.
func (h *Handler) handleTelegramWebhook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parse the Telegram update using the typed parser
	update, err := telegram.ParseUpdate(body)
	if err != nil {
		log.Printf("telegram webhook: failed to parse update: %v", err)
		w.WriteHeader(http.StatusOK) // Always 200 to Telegram
		return
	}

	// Extract inbound message
	msg, ok := telegram.ExtractInboundMessage(update)
	if !ok {
		// Not a processable message (callback query, inline query, etc.)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Look up the channel record by chat_id to find the user mapping
	chatID := msg.ChannelID
	matched, err := h.store.GetChannelByTypeAndChatID(r.Context(), string(channels.ChannelTypeTelegram), chatID)
	if err != nil {
		log.Printf("telegram webhook: no channel record for chat_id=%s: %v", chatID, err)
		w.WriteHeader(http.StatusOK)
		return
	}
	if !matched.Enabled {
		log.Printf("telegram webhook: channel disabled for chat_id=%s", chatID)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Build metadata for the message handler
	metadata := map[string]any{
		"channel_type":  string(channels.ChannelTypeTelegram),
		"chat_id":       chatID,
		"bot_token":     matched.BotToken,
		"channel_id_db": matched.ID,
	}
	// Merge raw metadata from the parsed message
	for k, v := range msg.Raw {
		metadata[k] = v
	}

	text := strings.TrimSpace(msg.Text)
	if text == "" && msg.MediaURL != "" {
		text = "[media]"
	}

	// Dispatch to the message handler in a background goroutine.
	// Use a detached context because the HTTP request context will be
	// cancelled once we return 200 to Telegram.
	go func() {
		bgCtx := context.Background()
		if err := h.messageHandler(bgCtx, matched.UserID, text, metadata); err != nil {
			log.Printf("telegram webhook: handler error: user=%s err=%v", matched.UserID, err)
		}
	}()

	// Always return 200 to Telegram immediately to avoid retries
	w.WriteHeader(http.StatusOK)
}

// getWebhookURL returns the public webhook URL for a channel type.
// Uses the configured webhookBaseURL if set, otherwise falls back to the request host.
// Preferred URL pattern:
// /go/v1/{user_id}/channels/{channel_type}/webhook/{agent_id}
func (h *Handler) getWebhookURL(r *http.Request, channelType channels.ChannelType, userID string) string {
	var base string
	if h.webhookBaseURL != "" {
		base = h.webhookBaseURL
	} else {
		// Fallback: derive from request (won't work behind tunnels/proxies)
		scheme := "https"
		if r.TLS == nil {
			scheme = "http"
		}
		base = fmt.Sprintf("%s://%s", scheme, r.Host)
	}
	base = strings.TrimRight(base, "/")
	uid := strings.TrimSpace(userID)
	if uid == "" {
		// Backward-compatible fallback when user_id is unavailable.
		return fmt.Sprintf("%s/agent/v1/channels/%s/webhook", base, channelType)
	}
	agentID := strings.TrimSpace(h.agentID)
	if agentID == "" {
		agentID = "unassigned"
	}
	return fmt.Sprintf("%s/go/v1/%s/channels/%s/webhook/%s", base, url.PathEscape(uid), channelType, url.PathEscape(agentID))
}
