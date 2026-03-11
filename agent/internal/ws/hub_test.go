package ws

import (
	"encoding/json"
	"sync"
	"testing"
	"time"
)

// makeTestClient creates a Client suitable for hub-level testing.
// The conn field is nil because we never start readPump/writePump.
func makeTestClient(hub *Hub, userID string, bufSize int) *Client {
	return &Client{
		UserID: userID,
		send:   make(chan []byte, bufSize),
		hub:    hub,
		conn:   nil,
	}
}

// drainOne reads a single message from the client's send channel with a
// short timeout. Returns the raw bytes and true, or nil and false on timeout.
func drainOne(c *Client, timeout time.Duration) ([]byte, bool) {
	select {
	case data := <-c.send:
		return data, true
	case <-time.After(timeout):
		return nil, false
	}
}

// TestHubRegisterUnregister verifies basic register/unregister lifecycle.
func TestHubRegisterUnregister(t *testing.T) {
	h := NewHub()

	// Register 3 clients for "alice".
	clients := make([]*Client, 3)
	for i := range clients {
		clients[i] = makeTestClient(h, "alice", 64)
		h.Register(clients[i])
	}

	if got := h.ConnectedUsers(); got != 1 {
		t.Fatalf("ConnectedUsers after 3 registrations: got %d, want 1", got)
	}

	// Unregister the first client.
	h.Unregister(clients[0])

	// Verify the send channel is closed.
	select {
	case _, ok := <-clients[0].send:
		if ok {
			t.Fatal("expected send channel to be closed after unregister")
		}
	default:
		t.Fatal("send channel should be closed (readable with ok=false), not blocking")
	}

	// Still 1 user because alice has 2 remaining clients.
	if got := h.ConnectedUsers(); got != 1 {
		t.Fatalf("ConnectedUsers after 1 unregister: got %d, want 1", got)
	}

	// Unregister remaining clients.
	h.Unregister(clients[1])
	h.Unregister(clients[2])

	if got := h.ConnectedUsers(); got != 0 {
		t.Fatalf("ConnectedUsers after all unregistered: got %d, want 0", got)
	}
}

// TestHubBroadcast verifies targeted broadcast reaches only the intended user.
func TestHubBroadcast(t *testing.T) {
	h := NewHub()

	alice1 := makeTestClient(h, "alice", 64)
	alice2 := makeTestClient(h, "alice", 64)
	bob := makeTestClient(h, "bob", 64)
	h.Register(alice1)
	h.Register(alice2)
	h.Register(bob)

	msg := Message{Type: "test", Payload: "hello"}
	h.Broadcast("alice", msg)

	// Both alice clients should receive the message.
	for i, c := range []*Client{alice1, alice2} {
		data, ok := drainOne(c, 100*time.Millisecond)
		if !ok {
			t.Fatalf("alice client %d: expected message, got timeout", i)
		}
		var got Message
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("alice client %d: unmarshal error: %v", i, err)
		}
		if got.Type != "test" {
			t.Fatalf("alice client %d: got type %q, want %q", i, got.Type, "test")
		}
	}

	// Bob should NOT have received anything.
	if _, ok := drainOne(bob, 50*time.Millisecond); ok {
		t.Fatal("bob should not have received a message from alice broadcast")
	}
}

// TestHubBroadcastAll verifies broadcast to all connected users.
func TestHubBroadcastAll(t *testing.T) {
	h := NewHub()

	alice := makeTestClient(h, "alice", 64)
	bob := makeTestClient(h, "bob", 64)
	carol := makeTestClient(h, "carol", 64)
	h.Register(alice)
	h.Register(bob)
	h.Register(carol)

	msg := Message{Type: "global", Payload: "announcement"}
	h.BroadcastAll(msg)

	for _, tc := range []struct {
		name   string
		client *Client
	}{
		{"alice", alice},
		{"bob", bob},
		{"carol", carol},
	} {
		data, ok := drainOne(tc.client, 100*time.Millisecond)
		if !ok {
			t.Fatalf("%s: expected message, got timeout", tc.name)
		}
		var got Message
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("%s: unmarshal error: %v", tc.name, err)
		}
		if got.Type != "global" {
			t.Fatalf("%s: got type %q, want %q", tc.name, got.Type, "global")
		}
	}
}

// TestHubConcurrentBroadcastAndUnregister is the primary race-condition test.
// It must pass with -race enabled.
func TestHubConcurrentBroadcastAndUnregister(t *testing.T) {
	h := NewHub()

	// Register 10 long-lived clients for "alice".
	baseClients := make([]*Client, 10)
	for i := range baseClients {
		baseClients[i] = makeTestClient(h, "alice", 256)
		h.Register(baseClients[i])
	}

	var wg sync.WaitGroup

	// 5 goroutines broadcasting 100 times each.
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				h.Broadcast("alice", Message{Type: "stress", Payload: id*1000 + i})
			}
		}(g)
	}

	// 5 goroutines rapidly registering and unregistering new clients.
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				c := makeTestClient(h, "alice", 256)
				h.Register(c)
				// Drain any messages that arrived to prevent channel fill-up.
			drainLoop:
				for {
					select {
					case <-c.send:
					default:
						break drainLoop
					}
				}
				h.Unregister(c)
			}
		}()
	}

	wg.Wait()

	// Clean up: unregister base clients.
	for _, c := range baseClients {
		h.Unregister(c)
	}

	if got := h.ConnectedUsers(); got != 0 {
		t.Fatalf("ConnectedUsers after cleanup: got %d, want 0", got)
	}
}

// TestHubBroadcastToSlowClient verifies that a slow client (full send buffer)
// does not block the broadcast; the message is dropped instead.
func TestHubBroadcastToSlowClient(t *testing.T) {
	h := NewHub()

	// Buffer size of 1 — easy to fill.
	slow := makeTestClient(h, "alice", 1)
	h.Register(slow)

	// Fill the buffer.
	h.Broadcast("alice", Message{Type: "fill"})

	// This second broadcast should drop the message (buffer full) without blocking.
	done := make(chan struct{})
	go func() {
		h.Broadcast("alice", Message{Type: "dropped"})
		close(done)
	}()

	select {
	case <-done:
		// Good — broadcast returned without blocking.
	case <-time.After(2 * time.Second):
		t.Fatal("Broadcast blocked on slow client — should have dropped the message")
	}

	// The channel should contain exactly the first message.
	data, ok := drainOne(slow, 50*time.Millisecond)
	if !ok {
		t.Fatal("expected the first message in the buffer")
	}
	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if got.Type != "fill" {
		t.Fatalf("got type %q, want %q", got.Type, "fill")
	}

	// No second message should be present.
	if _, ok := drainOne(slow, 50*time.Millisecond); ok {
		t.Fatal("expected dropped message, but got a second message")
	}

	h.Unregister(slow)
}

// TestHubConcurrentRegisterAndBroadcast exercises all hub operations
// concurrently. Must pass with -race.
func TestHubConcurrentRegisterAndBroadcast(t *testing.T) {
	h := NewHub()

	var wg sync.WaitGroup
	users := []string{"alice", "bob", "carol", "dave"}

	// Concurrent Register.
	registered := make(chan *Client, 200)
	for _, u := range users {
		wg.Add(1)
		go func(userID string) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				c := makeTestClient(h, userID, 128)
				h.Register(c)
				registered <- c
			}
		}(u)
	}

	// Concurrent Broadcast per user.
	for _, u := range users {
		wg.Add(1)
		go func(userID string) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				h.Broadcast(userID, Message{Type: "targeted"})
			}
		}(u)
	}

	// Concurrent BroadcastAll.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			h.BroadcastAll(Message{Type: "global"})
		}
	}()

	// Concurrent ConnectedUsers.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_ = h.ConnectedUsers()
		}
	}()

	wg.Wait()
	close(registered)

	// Concurrent Unregister of everything we registered.
	var wg2 sync.WaitGroup
	for c := range registered {
		wg2.Add(1)
		go func(cl *Client) {
			defer wg2.Done()
			h.Unregister(cl)
		}(c)
	}
	wg2.Wait()

	if got := h.ConnectedUsers(); got != 0 {
		t.Fatalf("ConnectedUsers after full cleanup: got %d, want 0", got)
	}
}
