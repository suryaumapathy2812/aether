package telegram

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/channels"
)

// Client handles communication with the Telegram Bot API
type Client struct {
	botToken   string
	httpClient *http.Client
	apiBase    string
}

// NewClient creates a new Telegram client
func NewClient(botToken string) *Client {
	return &Client{
		botToken:   botToken,
		httpClient: &http.Client{Timeout: 30 * time.Second},
		apiBase:    "https://api.telegram.org",
	}
}

// SetAPIBase sets a custom API base URL (useful for testing or proxies)
func (c *Client) SetAPIBase(base string) {
	c.apiBase = strings.TrimRight(base, "/")
}

// SendMessage sends a message to a chat
// https://core.telegram.org/bots/api#sendmessage
func (c *Client) SendMessage(ctx context.Context, chatID, text string, opts ...SendOption) error {
	if chatID == "" {
		return fmt.Errorf("chat_id is required")
	}
	if text == "" {
		return fmt.Errorf("text is required")
	}

	req := SendMessageRequest{
		ChatID:                chatID,
		Text:                  text,
		ParseMode:             "Markdown",
		DisableWebPagePreview: false,
	}

	for _, opt := range opts {
		opt(&req)
	}

	return c.post(ctx, "sendMessage", req, nil)
}

// SendMessageWithKeyboard sends a message with inline keyboard
// https://core.telegram.org/bots/api#sendmessage
func (c *Client) SendMessageWithKeyboard(ctx context.Context, chatID, text string, keyboard [][]InlineKeyboardButton) error {
	if chatID == "" {
		return fmt.Errorf("chat_id is required")
	}
	if text == "" {
		return fmt.Errorf("text is required")
	}

	req := SendMessageRequest{
		ChatID:    chatID,
		Text:      text,
		ParseMode: "Markdown",
		ReplyMarkup: &InlineKeyboardMarkup{
			InlineKeyboard: keyboard,
		},
	}

	return c.post(ctx, "sendMessage", req, nil)
}

// GetMe returns information about the bot
// https://core.telegram.org/bots/api#getme
func (c *Client) GetMe(ctx context.Context) (GetMeResponse, error) {
	var resp GetMeResponse
	err := c.get(ctx, "getMe", nil, &resp)
	return resp, err
}

// SetWebhook configures the webhook for the bot
// https://core.telegram.org/bots/api#setwebhook
func (c *Client) SetWebhook(ctx context.Context, webhookURL string, opts ...WebhookOption) error {
	req := SetWebhookRequest{
		URL:                webhookURL,
		DropPendingUpdates: false,
		MaxConnections:     40,
	}

	for _, opt := range opts {
		opt(&req)
	}

	var resp APIResponse
	err := c.post(ctx, "setWebhook", req, &resp)
	if err != nil {
		return err
	}
	if !resp.Ok {
		return fmt.Errorf("telegram API error: %s", resp.Description)
	}
	return nil
}

// DeleteWebhook removes the webhook
// https://core.telegram.org/bots/api#deletewebhook
func (c *Client) DeleteWebhook(ctx context.Context) error {
	var resp APIResponse
	err := c.get(ctx, "deleteWebhook", nil, &resp)
	if err != nil {
		return err
	}
	if !resp.Ok {
		return fmt.Errorf("telegram API error: %s", resp.Description)
	}
	return nil
}

// GetWebhookInfo returns current webhook status
// https://core.telegram.org/bots/api#getwebhookinfo
func (c *Client) GetWebhookInfo(ctx context.Context) (GetWebhookInfoResponse, error) {
	var resp GetWebhookInfoResponse
	err := c.get(ctx, "getWebhookInfo", nil, &resp)
	return resp, err
}

// ParseUpdate parses a Telegram update from the request body
func ParseUpdate(body []byte) (Update, error) {
	var update Update
	err := json.Unmarshal(body, &update)
	return update, err
}

// ExtractInboundMessage extracts an InboundMessage from a Telegram update
func ExtractInboundMessage(update Update) (channels.InboundMessage, bool) {
	msg := update.Message
	if msg == nil {
		// Handle edited messages
		if update.EditedMessage != nil {
			msg = update.EditedMessage
		} else {
			return channels.InboundMessage{}, false
		}
	}

	if msg.Text == "" && msg.MediaGroupID == "" && msg.Photo == nil && msg.Voice == nil {
		return channels.InboundMessage{}, false
	}

	inbound := channels.InboundMessage{
		Channel:   string(channels.ChannelTypeTelegram),
		ChannelID: fmt.Sprintf("%d", msg.Chat.ID),
		MessageID: fmt.Sprintf("%d", msg.MessageID),
		Text:      msg.Text,
		Timestamp: time.Unix(int64(msg.Date), 0),
		Raw:       make(map[string]any),
	}

	// Extract user info if available
	if msg.From != nil {
		inbound.Raw["user_id"] = msg.From.ID
		inbound.Raw["username"] = msg.From.Username
		inbound.Raw["first_name"] = msg.From.FirstName
	}

	// Handle media
	if msg.Photo != nil && len(*msg.Photo) > 0 {
		photo := (*msg.Photo)[len(*msg.Photo)-1] // Get largest photo
		inbound.MediaURL = photo.FileID
	}
	if msg.Voice != nil {
		inbound.MediaURL = msg.Voice.FileID
	}

	// Handle /start command with payload (for user linking)
	if strings.HasPrefix(msg.Text, "/start ") {
		payload := strings.TrimSpace(strings.TrimPrefix(msg.Text, "/start "))
		inbound.Raw["start_payload"] = payload
	}

	return inbound, true
}

// ─────────────────────────────────────────────────────────────────────
// API Types
// ─────────────────────────────────────────────────────────────────────

// APIResponse represents a generic Telegram API response
type APIResponse struct {
	Ok          bool            `json:"ok"`
	ErrorCode   int             `json:"error_code,omitempty"`
	Description string          `json:"description,omitempty"`
	Result      json.RawMessage `json:"result,omitempty"`
}

// Update represents an incoming update
// https://core.telegram.org/bots/api#update
type Update struct {
	UpdateID           int64               `json:"update_id"`
	Message            *Message            `json:"message,omitempty"`
	EditedMessage      *Message            `json:"edited_message,omitempty"`
	ChannelPost        *Message            `json:"channel_post,omitempty"`
	EditedChannelPost  *Message            `json:"edited_channel_post,omitempty"`
	InlineQuery        *InlineQuery        `json:"inline_query,omitempty"`
	ChosenInlineResult *ChosenInlineResult `json:"chosen_inline_result,omitempty"`
	CallbackQuery      *CallbackQuery      `json:"callback_query,omitempty"`
}

// Message represents a message
// https://core.telegram.org/bots/api#message
type Message struct {
	MessageID             int64                 `json:"message_id"`
	From                  *User                 `json:"from,omitempty"`
	Date                  int                   `json:"date"`
	Chat                  *Chat                 `json:"chat"`
	ForwardFrom           *User                 `json:"forward_from,omitempty"`
	ForwardFromChat       *Chat                 `json:"forward_from_chat,omitempty"`
	ForwardDate           int                   `json:"forward_date,omitempty"`
	ReplyToMessage        *Message              `json:"reply_to_message,omitempty"`
	EditDate              int                   `json:"edit_date,omitempty"`
	Text                  string                `json:"text,omitempty"`
	MediaGroupID          string                `json:"media_group_id,omitempty"`
	Photo                 *[]PhotoSize          `json:"photo,omitempty"`
	Voice                 *Voice                `json:"voice,omitempty"`
	Caption               string                `json:"caption,omitempty"`
	Contact               *Contact              `json:"contact,omitempty"`
	Location              *Location             `json:"location,omitempty"`
	Venue                 *Venue                `json:"venue,omitempty"`
	NewChatMembers        []User                `json:"new_chat_members,omitempty"`
	LeftChatMember        *User                 `json:"left_chat_member,omitempty"`
	NewChatTitle          string                `json:"new_chat_title,omitempty"`
	NewChatPhoto          *[]PhotoSize          `json:"new_chat_photo,omitempty"`
	DeleteChatPhoto       bool                  `json:"delete_chat_photo,omitempty"`
	GroupChatCreated      bool                  `json:"group_chat_created,omitempty"`
	SupergroupChatCreated bool                  `json:"supergroup_chat_created,omitempty"`
	ChannelChatCreated    bool                  `json:"channel_chat_created,omitempty"`
	MigrateToChatID       int64                 `json:"migrate_to_chat_id,omitempty"`
	MigrateFromChatID     int64                 `json:"migrate_from_chat_id,omitempty"`
	PinnedMessage         *Message              `json:"pinned_message,omitempty"`
	Invoice               *Invoice              `json:"invoice,omitempty"`
	SuccessfulPayment     *SuccessfulPayment    `json:"successful_payment,omitempty"`
	PassportData          *PassportData         `json:"passport_data,omitempty"`
	ReplyMarkup           *InlineKeyboardMarkup `json:"reply_markup,omitempty"`
}

// User represents a user
// https://core.telegram.org/bots/api#user
type User struct {
	ID           int64  `json:"id"`
	IsBot        bool   `json:"is_bot"`
	FirstName    string `json:"first_name"`
	LastName     string `json:"last_name,omitempty"`
	Username     string `json:"username,omitempty"`
	LanguageCode string `json:"language_code,omitempty"`
}

// Chat represents a chat
// https://core.telegram.org/bots/api#chat
type Chat struct {
	ID        int64  `json:"id"`
	Type      string `json:"type"`
	Title     string `json:"title,omitempty"`
	Username  string `json:"username,omitempty"`
	FirstName string `json:"first_name,omitempty"`
	LastName  string `json:"last_name,omitempty"`
}

// PhotoSize represents a photo
// https://core.telegram.org/bots/api#photosize
type PhotoSize struct {
	FileID   string `json:"file_id"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	FileSize int    `json:"file_size,omitempty"`
}

// Voice represents a voice note
// https://core.telegram.org/bots/api#voice
type Voice struct {
	FileID   string `json:"file_id"`
	Duration int    `json:"duration"`
	MimeType string `json:"mime_type,omitempty"`
	FileSize int    `json:"file_size,omitempty"`
}

// Contact represents a contact
// https://core.telegram.org/bots/api#contact
type Contact struct {
	PhoneNumber string `json:"phone_number"`
	FirstName   string `json:"first_name"`
	LastName    string `json:"last_name,omitempty"`
	UserID      int64  `json:"user_id,omitempty"`
}

// Location represents a location
// https://core.telegram.org/bots/api#location
type Location struct {
	Longitude float64 `json:"longitude"`
	Latitude  float64 `json:"latitude"`
}

// Venue represents a venue
// https://core.telegram.org/bots/api#venue
type Venue struct {
	Location     Location `json:"location"`
	Title        string   `json:"title"`
	Address      string   `json:"address"`
	FoursquareID string   `json:"foursquare_id,omitempty"`
}

// InlineKeyboardButton represents a button for an inline keyboard
// https://core.telegram.org/bots/api#inlinekeyboardbutton
type InlineKeyboardButton struct {
	Text         string    `json:"text"`
	URL          string    `json:"url,omitempty"`
	CallbackData string    `json:"callback_data,omitempty"`
	LoginURL     *LoginURL `json:"login_url,omitempty"`
}

// LoginURL represents a parameter for the login URL
// https://core.telegram.org/bots/api#loginurl
type LoginURL struct {
	URL                string `json:"url"`
	ForwardText        string `json:"forward_text,omitempty"`
	BotUsername        string `json:"bot_username,omitempty"`
	RequestWriteAccess bool   `json:"request_write_access,omitempty"`
}

// InlineKeyboardMarkup represents an inline keyboard
// https://core.telegram.org/bots/api#inlinekeyboardmarkup
type InlineKeyboardMarkup struct {
	InlineKeyboard [][]InlineKeyboardButton `json:"inline_keyboard"`
}

// Invoice represents an invoice
// https://core.telegram.org/bots/api#invoice
type Invoice struct {
	Title          string `json:"title"`
	Description    string `json:"description"`
	StartParameter string `json:"start_parameter"`
	Currency       string `json:"currency"`
	TotalAmount    int    `json:"total_amount"`
}

// SuccessfulPayment represents a successful payment
// https://core.telegram.org/bots/api#successfulpayment
type SuccessfulPayment struct {
	Currency                string     `json:"currency"`
	TotalAmount             int        `json:"total_amount"`
	InvoicePayload          string     `json:"invoice_payload"`
	ShippingOptionID        string     `json:"shipping_option_id,omitempty"`
	OrderInfo               *OrderInfo `json:"order_info,omitempty"`
	TelegramPaymentChargeID string     `json:"telegram_payment_charge_id"`
	ProviderPaymentChargeID string     `json:"provider_payment_charge_id"`
}

// OrderInfo represents order info
// https://core.telegram.org/bots/api#orderinfo
type OrderInfo struct {
	Name            string           `json:"name,omitempty"`
	PhoneNumber     string           `json:"phone_number,omitempty"`
	Email           string           `json:"email,omitempty"`
	ShippingAddress *ShippingAddress `json:"shipping_address,omitempty"`
}

// ShippingAddress represents a shipping address
// https://core.telegram.org/bots/api#shippingaddress
type ShippingAddress struct {
	CountryCode string `json:"country_code"`
	State       string `json:"state,omitempty"`
	City        string `json:"city"`
	StreetLine1 string `json:"street_line1"`
	StreetLine2 string `json:"street_line2,omitempty"`
	PostCode    string `json:"post_code"`
}

// PassportData represents a passport
// https://core.telegram.org/bots/api#passportdata
type PassportData struct {
	Data        []EncryptedPassportElement `json:"data"`
	Credentials EncryptedCredentials       `json:"credentials"`
}

// EncryptedPassportElement represents an encrypted passport element
// https://core.telegram.org/bots/api#encryptedpassportelement
type EncryptedPassportElement struct {
	Type        string         `json:"type"`
	Data        string         `json:"data,omitempty"`
	PhoneNumber string         `json:"phone_number,omitempty"`
	Email       string         `json:"email,omitempty"`
	Files       []PassportFile `json:"files,omitempty"`
	FrontSide   *PassportFile  `json:"front_side,omitempty"`
	ReverseSide *PassportFile  `json:"reverse_side,omitempty"`
	Selfie      *PassportFile  `json:"selfie,omitempty"`
}

// PassportFile represents a passport file
// https://core.telegram.org/bots/api#passportfile
type PassportFile struct {
	FileID   string `json:"file_id"`
	FileSize int    `json:"file_size"`
	FileDate int    `json:"file_date"`
}

// EncryptedCredentials represents encrypted credentials
// https://core.telegram.org/bots/api#encryptedcredentials
type EncryptedCredentials struct {
	Data   string `json:"data"`
	Hash   string `json:"hash"`
	Secret string `json:"secret"`
}

// InlineQuery represents an inline query
// https://core.telegram.org/bots/api#inlinequery
type InlineQuery struct {
	ID       string    `json:"id"`
	From     *User     `json:"from"`
	Query    string    `json:"query"`
	Offset   string    `json:"offset"`
	Location *Location `json:"location,omitempty"`
}

// ChosenInlineResult represents a result of an inline query
// https://core.telegram.org/bots/api#choseninlineresult
type ChosenInlineResult struct {
	ResultID        string    `json:"result_id"`
	From            *User     `json:"from"`
	Query           string    `json:"query"`
	Location        *Location `json:"location,omitempty"`
	InlineMessageID string    `json:"inline_message_id,omitempty"`
}

// CallbackQuery represents an incoming callback query
// https://core.telegram.org/bots/api#callbackquery
type CallbackQuery struct {
	ID              string   `json:"id"`
	From            *User    `json:"from"`
	Message         *Message `json:"message,omitempty"`
	InlineMessageID string   `json:"inline_message_id,omitempty"`
	ChatInstance    string   `json:"chat_instance"`
	Data            string   `json:"data,omitempty"`
	GameShortName   string   `json:"game_short_name,omitempty"`
}

// ─────────────────────────────────────────────────────────────────────
// Request/Response Types
// ─────────────────────────────────────────────────────────────────────

// SendMessageRequest represents the request parameters for sendMessage
type SendMessageRequest struct {
	ChatID                any                   `json:"chat_id"`
	Text                  string                `json:"text"`
	ParseMode             string                `json:"parse_mode,omitempty"`
	DisableWebPagePreview bool                  `json:"disable_web_page_preview,omitempty"`
	DisableNotification   bool                  `json:"disable_notification,omitempty"`
	ReplyToMessageID      int64                 `json:"reply_to_message_id,omitempty"`
	ReplyMarkup           *InlineKeyboardMarkup `json:"reply_markup,omitempty"`
}

// SendOption is a functional option for SendMessage
type SendOption func(*SendMessageRequest)

// WithParseMode sets the parse mode
func WithParseMode(mode string) SendOption {
	return func(req *SendMessageRequest) {
		req.ParseMode = mode
	}
}

// WithHTMLParseMode sets HTML parse mode
func WithHTMLParseMode() SendOption {
	return func(req *SendMessageRequest) {
		req.ParseMode = "HTML"
	}
}

// DisableWebPreview disables link previews
func DisableWebPreview() SendOption {
	return func(req *SendMessageRequest) {
		req.DisableWebPagePreview = true
	}
}

// ReplyToMessage replies to a specific message
func ReplyToMessage(messageID int64) SendOption {
	return func(req *SendMessageRequest) {
		req.ReplyToMessageID = messageID
	}
}

// SetWebhookRequest represents the request parameters for setWebhook
type SetWebhookRequest struct {
	URL                string   `json:"url"`
	Certificate        any      `json:"certificate,omitempty"`
	MaxConnections     int      `json:"max_connections,omitempty"`
	AllowedUpdates     []string `json:"allowed_updates,omitempty"`
	DropPendingUpdates bool     `json:"drop_pending_updates"`
	SecretToken        string   `json:"secret_token,omitempty"`
}

// WebhookOption is a functional option for SetWebhook
type WebhookOption func(*SetWebhookRequest)

// WithAllowedUpdates sets the allowed updates
func WithAllowedUpdates(updates []string) WebhookOption {
	return func(req *SetWebhookRequest) {
		req.AllowedUpdates = updates
	}
}

// WithSecretToken sets a secret token for webhook verification
func WithSecretToken(token string) WebhookOption {
	return func(req *SetWebhookRequest) {
		req.SecretToken = token
	}
}

// GetMeResponse represents the response from getMe
type GetMeResponse struct {
	Ok     bool  `json:"ok"`
	Result *User `json:"result"`
}

// GetWebhookInfoResponse represents the response from getWebhookInfo
type GetWebhookInfoResponse struct {
	Ok     bool         `json:"ok"`
	Result *WebhookInfo `json:"result"`
}

// WebhookInfo represents the current webhook status
// https://core.telegram.org/bots/api#webhookinfo
type WebhookInfo struct {
	URL                          string   `json:"url"`
	HasCustomCertificate         bool     `json:"has_custom_certificate"`
	PendingUpdateCount           int      `json:"pending_update_count"`
	IPAddress                    string   `json:"ip_address,omitempty"`
	LastErrorDate                int      `json:"last_error_date,omitempty"`
	LastErrorMessage             string   `json:"last_error_message,omitempty"`
	LastSynchronizationErrorDate int      `json:"last_synchronization_error_date,omitempty"`
	MaxConnections               int      `json:"max_connections"`
	AllowedUpdates               []string `json:"allowed_updates,omitempty"`
}

// ─────────────────────────────────────────────────────────────────────
// HTTP Methods
// ─────────────────────────────────────────────────────────────────────

func (c *Client) post(ctx context.Context, method string, req, resp any) error {
	apiURL := fmt.Sprintf("%s/bot%s/%s", c.apiBase, c.botToken, method)

	var body io.Reader
	if req != nil {
		data, err := json.Marshal(req)
		if err != nil {
			return fmt.Errorf("failed to marshal request: %w", err)
		}
		body = bytes.NewReader(data)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, body)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	respData, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer respData.Body.Close()

	respBody, err := io.ReadAll(respData.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if respData.StatusCode != http.StatusOK {
		return fmt.Errorf("telegram API returned status %d: %s", respData.StatusCode, string(respBody))
	}

	if resp != nil {
		if err := json.Unmarshal(respBody, resp); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}

	return nil
}

func (c *Client) get(ctx context.Context, method string, params map[string]string, resp any) error {
	apiURL := fmt.Sprintf("%s/bot%s/%s", c.apiBase, c.botToken, method)

	if len(params) > 0 {
		q := url.Values{}
		for k, v := range params {
			q.Set(k, v)
		}
		apiURL += "?" + q.Encode()
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	respData, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer respData.Body.Close()

	respBody, err := io.ReadAll(respData.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if respData.StatusCode != http.StatusOK {
		return fmt.Errorf("telegram API returned status %d: %s", respData.StatusCode, string(respBody))
	}

	if resp != nil {
		if err := json.Unmarshal(respBody, resp); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}

	return nil
}
