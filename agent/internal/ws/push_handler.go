package ws

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
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
	mux.HandleFunc("/api/push/test", h.handleTestPush)
}

func (h *PushHandler) handleVAPIDKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	key := ""
	if h.sender != nil {
		key = h.sender.VAPIDPublicKey()
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"public_key": key})
}

func (h *PushHandler) handleSubscription(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		h.subscribe(w, r)
	case http.MethodDelete:
		h.unsubscribe(w, r)
	default:
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (h *PushHandler) subscribe(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID       string           `json:"user_id"`
		Subscription PushSubscription `json:"subscription"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if strings.TrimSpace(req.UserID) == "" {
		req.UserID = "default"
	}
	if strings.TrimSpace(req.Subscription.Endpoint) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "subscription endpoint is required")
		return
	}
	err := h.store.SavePushSubscription(r.Context(), req.UserID, db.PushSubscriptionRecord{
		Endpoint:  req.Subscription.Endpoint,
		KeyP256dh: req.Subscription.Keys.P256dh,
		KeyAuth:   req.Subscription.Keys.Auth,
	})
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "subscribed"})
}

func (h *PushHandler) unsubscribe(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID   string `json:"user_id"`
		Endpoint string `json:"endpoint"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if strings.TrimSpace(req.UserID) == "" {
		req.UserID = "default"
	}
	if strings.TrimSpace(req.Endpoint) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "endpoint is required")
		return
	}
	err := h.store.DeletePushSubscription(r.Context(), req.UserID, req.Endpoint)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "unsubscribed"})
}

func (h *PushHandler) handleTestPush(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.sender == nil {
		httputil.WriteError(w, http.StatusServiceUnavailable, "push sender not configured (VAPID keys missing)")
		return
	}
	if h.store == nil {
		httputil.WriteError(w, http.StatusServiceUnavailable, "store unavailable")
		return
	}

	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
	if userID == "" {
		userID = "default"
	}

	subs, err := h.store.GetPushSubscriptions(r.Context(), userID)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, "failed to get subscriptions: "+err.Error())
		return
	}
	if len(subs) == 0 {
		httputil.WriteError(w, http.StatusNotFound, "no push subscriptions found for user")
		return
	}

	pushSubs := make([]PushSubscription, 0, len(subs))
	for _, sub := range subs {
		pushSubs = append(pushSubs, PushSubscription{
			Endpoint: sub.Endpoint,
			Keys: struct {
				P256dh string `json:"p256dh"`
				Auth   string `json:"auth"`
			}{P256dh: sub.KeyP256dh, Auth: sub.KeyAuth},
		})
	}

	results := h.sender.SendToAll(pushSubs, PushPayload{
		Title: "Hello from Aether",
		Body:  "Push notifications are working!",
		Tag:   "test",
	})

	succeeded := 0
	failed := 0
	for _, r := range results {
		if r.Success {
			succeeded++
		} else {
			failed++
		}
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"status":        "sent",
		"subscriptions": len(pushSubs),
		"succeeded":     succeeded,
		"failed":        failed,
		"results":       results,
	})
}


