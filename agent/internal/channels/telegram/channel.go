package telegram

import (
	"context"
	"fmt"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/channels"
)

// TelegramChannel implements the Channel interface for Telegram
type TelegramChannel struct {
	botToken       string
	messageHandler channels.MessageHandler
}

// NewTelegramChannel creates a new Telegram channel handler
func NewTelegramChannel(botToken string, handler channels.MessageHandler) *TelegramChannel {
	return &TelegramChannel{
		botToken:       botToken,
		messageHandler: handler,
	}
}

// Type returns the channel type
func (t *TelegramChannel) Type() channels.ChannelType {
	return channels.ChannelTypeTelegram
}

// DisplayName returns a human-readable name
func (t *TelegramChannel) DisplayName() string {
	return "Telegram"
}

// HandleMessage processes an incoming message from Telegram
func (t *TelegramChannel) HandleMessage(ctx context.Context, msg channels.InboundMessage) error {
	if t.messageHandler == nil {
		return fmt.Errorf("no message handler configured")
	}

	// Extract user ID from start payload if available
	userID := msg.UserID
	if payload, ok := msg.Raw["start_payload"].(string); ok && payload != "" {
		// Payload contains the session token or user identifier
		// This is used to link the Telegram chat to the Aether user
		userID = payload
	}

	// If still no user ID, we can't process the message
	if userID == "" {
		return fmt.Errorf("user not linked: please start the bot with /start <session_token>")
	}

	return t.messageHandler(ctx, userID, msg.Text, msg.Raw)
}

// SendMessage sends a message to the user via Telegram
func (t *TelegramChannel) SendMessage(ctx context.Context, msg channels.OutboundMessage) error {
	if msg.ChannelID == "" {
		return fmt.Errorf("channel_id is required")
	}
	if msg.Text == "" {
		return fmt.Errorf("text is required")
	}

	client := NewClient(t.botToken)

	var parseMode string
	switch msg.ParseMode {
	case "HTML", "Markdown":
		parseMode = msg.ParseMode
	default:
		parseMode = "Markdown"
	}

	opts := []SendOption{
		WithParseMode(parseMode),
	}

	if msg.ReplyTo != "" {
		opts = append(opts, ReplyToMessage(0)) // TODO: parse message ID
	}

	return client.SendMessage(ctx, msg.ChannelID, msg.Text, opts...)
}

// ValidateConfig validates the channel configuration
func (t *TelegramChannel) ValidateConfig(ctx context.Context, config map[string]string) error {
	token := config["bot_token"]
	if strings.TrimSpace(token) == "" {
		return fmt.Errorf("bot_token is required")
	}

	// Validate the token by calling getMe
	client := NewClient(token)
	me, err := client.GetMe(ctx)
	if err != nil {
		return fmt.Errorf("failed to validate bot token: %w", err)
	}
	if !me.Ok || me.Result == nil {
		return fmt.Errorf("invalid bot token")
	}

	return nil
}

// SetWebhook configures the webhook for this channel
func (t *TelegramChannel) SetWebhook(ctx context.Context, webhookURL string) error {
	if strings.TrimSpace(webhookURL) == "" {
		return fmt.Errorf("webhook URL is required")
	}

	client := NewClient(t.botToken)
	return client.SetWebhook(ctx, webhookURL)
}

// GetBotInfo returns information about the bot
func (t *TelegramChannel) GetBotInfo(ctx context.Context, config map[string]string) (channels.BotInfo, error) {
	token := config["bot_token"]
	if strings.TrimSpace(token) == "" {
		return channels.BotInfo{}, fmt.Errorf("bot_token is required")
	}

	client := NewClient(token)
	me, err := client.GetMe(ctx)
	if err != nil {
		return channels.BotInfo{}, err
	}
	if !me.Ok || me.Result == nil {
		return channels.BotInfo{}, fmt.Errorf("failed to get bot info")
	}

	name := me.Result.FirstName
	if me.Result.Username != "" {
		name = fmt.Sprintf("@%s (%s)", me.Result.Username, me.Result.FirstName)
	}

	return channels.BotInfo{
		ID:        me.Result.ID,
		FirstName: me.Result.FirstName,
		Username:  me.Result.Username,
		Name:      name,
	}, nil
}

// VerifyWebhook verifies that the webhook request comes from Telegram
// by checking the secret token if one was set
func VerifyWebhook(secretToken, expectedToken string) bool {
	if expectedToken == "" {
		return true // No token configured, skip verification
	}
	return secretToken == expectedToken
}
