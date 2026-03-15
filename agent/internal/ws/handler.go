package ws

import (
	"log"
	"net/http"
	"strings"

	"github.com/gorilla/websocket"
)

// Handler serves the WebSocket notification endpoint.
type Handler struct {
	hub      *Hub
	upgrader websocket.Upgrader
}

// NewHandler creates a handler backed by the given hub.
func NewHandler(hub *Hub) *Handler {
	return &Handler{
		hub: hub,
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
	mux.HandleFunc("/ws/notifications", h.handleWS)
}

func (h *Handler) handleWS(w http.ResponseWriter, r *http.Request) {
	// The orchestrator proxy validates auth before forwarding to the agent.
	// Here we just require a non-empty token as a guard against accidental
	// direct connections bypassing the proxy.
	token := strings.TrimSpace(r.URL.Query().Get("token"))
	if token == "" {
		http.Error(w, "missing token", http.StatusUnauthorized)
		return
	}

	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
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
