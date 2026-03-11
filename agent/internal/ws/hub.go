// Package ws provides a WebSocket notification hub for real-time push
// of agent task events and memory notifications to connected dashboard clients.
package ws

import (
	"encoding/json"
	"log"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// Message is the JSON envelope sent over the WebSocket.
type Message struct {
	Type    string `json:"type"`              // e.g. "notification", "task_update", "pong"
	Payload any    `json:"payload,omitempty"` // varies by type
}

// Client wraps a single WebSocket connection for a user.
type Client struct {
	UserID string
	conn   *websocket.Conn
	send   chan []byte
	hub    *Hub
}

// Hub maintains per-user sets of connected WebSocket clients.
type Hub struct {
	mu      sync.RWMutex
	clients map[string]map[*Client]struct{} // userID → set of clients
}

// NewHub creates a ready-to-use hub.
func NewHub() *Hub {
	return &Hub{
		clients: make(map[string]map[*Client]struct{}),
	}
}

// Register adds a client to the hub.
func (h *Hub) Register(c *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.clients[c.UserID] == nil {
		h.clients[c.UserID] = make(map[*Client]struct{})
	}
	h.clients[c.UserID][c] = struct{}{}
	log.Printf("ws hub: registered client for user=%s (total=%d)", c.UserID, len(h.clients[c.UserID]))
}

// Unregister removes a client from the hub and closes its send channel.
func (h *Hub) Unregister(c *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if set, ok := h.clients[c.UserID]; ok {
		if _, exists := set[c]; exists {
			delete(set, c)
			close(c.send)
			if len(set) == 0 {
				delete(h.clients, c.UserID)
			}
		}
	}
	log.Printf("ws hub: unregistered client for user=%s", c.UserID)
}

// Broadcast sends a message to all connected clients for a given user.
// The read lock is held for the entire iteration to prevent concurrent
// modification of the client set (e.g. by Unregister) which would cause
// a data race on the map and potential send-on-closed-channel panic.
func (h *Hub) Broadcast(userID string, msg Message) {
	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("ws hub: marshal error: %v", err)
		return
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	set := h.clients[userID]
	for c := range set {
		select {
		case c.send <- data:
		default:
			log.Printf("ws hub: dropping message for slow client user=%s", userID)
		}
	}
}

// BroadcastAll sends a message to all connected clients regardless of user.
func (h *Hub) BroadcastAll(msg Message) {
	data, err := json.Marshal(msg)
	if err != nil {
		return
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	for _, set := range h.clients {
		for c := range set {
			select {
			case c.send <- data:
			default:
			}
		}
	}
}

// ConnectedUsers returns the number of distinct users with at least one connection.
func (h *Hub) ConnectedUsers() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.clients)
}

// newClient creates a client and starts its read/write goroutines.
func (h *Hub) newClient(userID string, conn *websocket.Conn) *Client {
	c := &Client{
		UserID: userID,
		conn:   conn,
		send:   make(chan []byte, 64),
		hub:    h,
	}
	h.Register(c)
	go c.writePump()
	go c.readPump()
	return c
}

const (
	writeWait  = 10 * time.Second
	pongWait   = 60 * time.Second
	pingPeriod = (pongWait * 9) / 10
	maxMsgSize = 4096
)

// readPump reads messages from the client (ping, notification_feedback).
func (c *Client) readPump() {
	defer func() {
		c.hub.Unregister(c)
		_ = c.conn.Close()
	}()
	c.conn.SetReadLimit(maxMsgSize)
	_ = c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		_ = c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})
	for {
		_, raw, err := c.conn.ReadMessage()
		if err != nil {
			break
		}
		var msg struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(raw, &msg) != nil {
			continue
		}
		switch msg.Type {
		case "ping":
			select {
			case c.send <- mustJSON(Message{Type: "pong"}):
			default:
			}
		// notification_feedback could be handled here in the future.
		default:
			// Unknown client message — ignore.
		}
	}
}

// writePump sends queued messages and periodic pings to the client.
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		_ = c.conn.Close()
	}()
	for {
		select {
		case data, ok := <-c.send:
			_ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				_ = c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
				return
			}
		case <-ticker.C:
			_ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}
