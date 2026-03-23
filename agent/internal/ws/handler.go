package ws

import (
	"log"
	"net/http"
	"strings"

	"github.com/gorilla/websocket"
	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
)

// Handler serves the WebSocket notification endpoint.
type Handler struct {
	hub       *Hub
	validator *agentauth.Validator
	upgrader  websocket.Upgrader
}

// NewHandler creates a handler backed by the given hub.
func NewHandler(hub *Hub, validator *agentauth.Validator) *Handler {
	return &Handler{
		hub:       hub,
		validator: validator,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			// Allow all origins in dev. In production, lock this down.
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}
}

// RegisterRoutes registers the WebSocket endpoint on the mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	for _, path := range []string{"/agent/v1/ws/notifications", "/ws/notifications"} {
		mux.HandleFunc(path, h.handleWS)
	}
}

func (h *Handler) handleWS(w http.ResponseWriter, r *http.Request) {
	token := strings.TrimSpace(r.URL.Query().Get("token"))
	if token == "" {
		http.Error(w, "missing token", http.StatusUnauthorized)
		return
	}

	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
	if h.validator != nil && (agentauth.RequiresDirectToken(r) || agentauth.IsDirectToken(token)) {
		claims, err := h.validator.ValidateRequest(r)
		if err != nil {
			http.Error(w, "invalid token", http.StatusUnauthorized)
			return
		}
		userID = claims.UserID
	}
	if userID == "" {
		userID = "default"
	}

	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws handler: upgrade failed: %v", err)
		return
	}

	h.hub.newClient(userID, conn)
}
