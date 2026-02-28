package ws

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/suryaumapathy/core-ai/agent/internal/db"
)

// PushHandler serves REST endpoints for push subscription management.
type PushHandler struct {
	store  *db.Store
	sender *PushSender
}

// NewPushHandler creates a push handler.
func NewPushHandler(store *db.Store, sender *PushSender) *PushHandler {
	return &PushHandler{store: store, sender: sender}
}

// RegisterRoutes registers push-related HTTP endpoints.
func (h *PushHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/push/vapid-key", h.handleVAPIDKey)
	mux.HandleFunc("/api/push/subscribe", h.handleSubscription)
}

func (h *PushHandler) handleVAPIDKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	key := ""
	if h.sender != nil {
		key = h.sender.VAPIDPublicKey()
	}
	writeJSON(w, http.StatusOK, map[string]any{"public_key": key})
}

func (h *PushHandler) handleSubscription(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		h.subscribe(w, r)
	case http.MethodDelete:
		h.unsubscribe(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (h *PushHandler) subscribe(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID       string           `json:"user_id"`
		Subscription PushSubscription `json:"subscription"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if strings.TrimSpace(req.UserID) == "" {
		req.UserID = "default"
	}
	if strings.TrimSpace(req.Subscription.Endpoint) == "" {
		writeError(w, http.StatusBadRequest, "subscription endpoint is required")
		return
	}
	err := h.store.SavePushSubscription(r.Context(), req.UserID, db.PushSubscriptionRecord{
		Endpoint:  req.Subscription.Endpoint,
		KeyP256dh: req.Subscription.Keys.P256dh,
		KeyAuth:   req.Subscription.Keys.Auth,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "subscribed"})
}

func (h *PushHandler) unsubscribe(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID   string `json:"user_id"`
		Endpoint string `json:"endpoint"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if strings.TrimSpace(req.UserID) == "" {
		req.UserID = "default"
	}
	if strings.TrimSpace(req.Endpoint) == "" {
		writeError(w, http.StatusBadRequest, "endpoint is required")
		return
	}
	err := h.store.DeletePushSubscription(r.Context(), req.UserID, req.Endpoint)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "unsubscribed"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]any{"error": detail})
}
