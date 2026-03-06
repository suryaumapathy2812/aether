package channels

import (
	"context"
	"fmt"
	"sync"
)

// Manager manages all registered channel implementations
type Manager struct {
	mu          sync.RWMutex
	channels    map[ChannelType]Channel
	httpHandler *HTTPHandler
}

// NewManager creates a new channel manager
func NewManager() *Manager {
	return &Manager{
		channels: make(map[ChannelType]Channel),
	}
}

// Register adds a channel implementation to the manager
func (m *Manager) Register(ch Channel) error {
	if ch == nil {
		return fmt.Errorf("channel cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	channelType := ch.Type()
	if _, exists := m.channels[channelType]; exists {
		return fmt.Errorf("channel type %s already registered", channelType)
	}

	m.channels[channelType] = ch
	return nil
}

// Get returns a channel by type
func (m *Manager) Get(channelType ChannelType) (Channel, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ch, ok := m.channels[channelType]
	return ch, ok
}

// List returns all registered channel types
func (m *Manager) List() []ChannelType {
	m.mu.RLock()
	defer m.mu.RUnlock()

	types := make([]ChannelType, 0, len(m.channels))
	for t := range m.channels {
		types = append(types, t)
	}
	return types
}

// HTTPHandler returns the HTTP handler for incoming webhooks
func (m *Manager) HTTPHandler() *HTTPHandler {
	return m.httpHandler
}

// SetHTTPHandler sets the HTTP handler for incoming webhooks
func (m *Manager) SetHTTPHandler(handler *HTTPHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.httpHandler = handler
}

// ChannelMessageHandlerFunc is the function type for handling processed messages
type ChannelMessageHandlerFunc func(ctx context.Context, userID, text string, metadata map[string]any) error

// SetupWithHandler configures all channels with a message handler
func (m *Manager) SetupWithHandler(handler ChannelMessageHandlerFunc) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for channelType, ch := range m.channels {
		// Create a wrapper that converts the function to the interface
		wrapped := &channelHandlerWrapper{
			ch:      ch,
			handler: handler,
		}

		// Re-register with the wrapped handler
		// Note: This is a bit of a hack; ideally we'd have a better way
		// For now, each channel implementation will need to be created with the handler
		_ = wrapped // unused for now
		_ = channelType
	}

	return nil
}

// channelHandlerWrapper wraps a Channel with a handler function
type channelHandlerWrapper struct {
	ch      Channel
	handler ChannelMessageHandlerFunc
}
