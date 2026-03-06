package channels

import (
	"context"
	"time"
)

// ChannelType represents the type of communication channel
type ChannelType string

const (
	ChannelTypeTelegram ChannelType = "telegram"
	ChannelTypeWhatsApp ChannelType = "whatsapp"
	ChannelTypeSlack    ChannelType = "slack"
	ChannelTypeEmail    ChannelType = "email"
)

// InboundMessage represents an incoming message from a channel
type InboundMessage struct {
	Channel   string
	ChannelID string // Platform-specific identifier (e.g., Telegram chat_id)
	UserID    string // Aether user ID (mapped)
	MessageID string // Platform message ID
	Text      string
	MediaURL  string
	Timestamp time.Time
	Raw       map[string]any // Platform-specific payload
}

// OutboundMessage represents an outgoing message to a channel
type OutboundMessage struct {
	ChannelID string
	Text      string
	ParseMode string // e.g., "Markdown", "HTML"
	ReplyTo   string // Message ID to reply to
}

// Channel is the interface all channel implementations must satisfy
type Channel interface {
	// Type returns the channel type
	Type() ChannelType

	// DisplayName returns a human-readable name
	DisplayName() string

	// HandleMessage processes an incoming message from the channel
	HandleMessage(ctx context.Context, msg InboundMessage) error

	// SendMessage sends a message to the user via the channel
	SendMessage(ctx context.Context, msg OutboundMessage) error

	// ValidateConfig validates the channel configuration
	ValidateConfig(ctx context.Context, config map[string]string) error

	// SetWebhook configures the webhook for this channel
	// Returns the webhook URL that should be set
	SetWebhook(ctx context.Context, webhookURL string) error

	// GetBotInfo returns information about the bot/account
	GetBotInfo(ctx context.Context, config map[string]string) (BotInfo, error)
}

// BotInfo represents information about a bot account
type BotInfo struct {
	ID        int64  `json:"id"`
	FirstName string `json:"first_name"`
	Username  string `json:"username"`
	Name      string `json:"name"`
}

// Config holds channel configuration
type Config struct {
	BotToken   string
	WebhookURL string
	ChannelID  string // For sending messages (e.g., chat_id for Telegram)
}

// MessageHandler is the function type for handling processed messages
type MessageHandler func(ctx context.Context, userID, text string, metadata map[string]any) error
