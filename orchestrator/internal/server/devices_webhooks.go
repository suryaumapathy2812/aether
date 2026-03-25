package server

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/agent"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/auth"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/proxy"
)

type deviceRecord struct {
	ID         string            `json:"id"`
	UserID     string            `json:"user_id,omitempty"`
	Name       string            `json:"name"`
	DeviceType string            `json:"device_type"`
	PluginName string            `json:"plugin_name,omitempty"`
	Config     map[string]string `json:"config,omitempty"`
	PairedAt   *time.Time        `json:"paired_at,omitempty"`
	LastSeen   *time.Time        `json:"last_seen,omitempty"`
}

func (s *Server) handleDevices(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	rows, err := s.db.Query(r.Context(), `
		SELECT id, name, device_type, COALESCE(plugin_name,''), COALESCE(config_json,'{}'), paired_at, last_seen
		FROM devices
		WHERE user_id = $1
		ORDER BY paired_at DESC
	`, id.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	out := make([]deviceRecord, 0)
	for rows.Next() {
		var rec deviceRecord
		var configJSON string
		if err := rows.Scan(&rec.ID, &rec.Name, &rec.DeviceType, &rec.PluginName, &configJSON, &rec.PairedAt, &rec.LastSeen); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		rec.Config = parseConfigJSON(configJSON)
		scrubDeviceConfig(rec.Config)
		out = append(out, rec)
	}
	writeJSON(w, http.StatusOK, map[string]any{"devices": out})
}

func (s *Server) handleTelegramDevice(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req struct {
		BotToken       string `json:"bot_token"`
		AllowedChatIDs string `json:"allowed_chat_ids"`
		Name           string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	botToken := strings.TrimSpace(req.BotToken)
	if botToken == "" {
		writeError(w, http.StatusBadRequest, "bot_token is required")
		return
	}
	deviceID, err := randomID("dev")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create device id")
		return
	}
	deviceToken, err := randomID("tok")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create device token")
		return
	}
	secretToken, err := randomID("sec")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create webhook secret")
		return
	}

	baseURL := resolvePublicBaseURL(r)
	if baseURL == "" {
		writeError(w, http.StatusBadRequest, "missing public base url (set AETHER_PUBLIC_BASE_URL)")
		return
	}
	webhookURL := strings.TrimRight(baseURL, "/") + orchestratorAPIPrefix + "/hooks/telegram/" + deviceID
	if err := setTelegramWebhook(r.Context(), botToken, webhookURL, secretToken); err != nil {
		writeError(w, http.StatusBadGateway, "failed to configure Telegram webhook: "+err.Error())
		return
	}

	cfg := map[string]string{
		"bot_token":        botToken,
		"secret_token":     secretToken,
		"allowed_chat_ids": strings.TrimSpace(req.AllowedChatIDs),
		"webhook_url":      webhookURL,
		"base_url":         strings.TrimRight(baseURL, "/"),
	}
	configJSON, _ := json.Marshal(cfg)
	name := strings.TrimSpace(req.Name)
	if name == "" {
		name = "Telegram Bot"
	}

	_, err = s.db.Exec(r.Context(), `
		INSERT INTO devices (id, user_id, token, name, device_type, plugin_name, config_json, paired_at, last_seen)
		VALUES ($1, $2, $3, $4, 'telegram', 'telegram', $5, now(), now())
	`, deviceID, id.UserID, deviceToken, name, string(configJSON))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":      "registered",
		"device_id":   deviceID,
		"webhook_url": webhookURL,
	})
}

func (s *Server) handleDeviceByID(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	deviceID := strings.TrimSpace(trimAnyPrefix(r.URL.Path, orchestratorAPIPrefix+"/devices/", "/api/devices/"))
	if deviceID == "" {
		writeError(w, http.StatusBadRequest, "device id is required")
		return
	}
	if r.Method != http.MethodDelete {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var rec deviceRecord
	var configJSON string
	err := s.db.QueryRow(r.Context(), `
		SELECT id, user_id, name, device_type, COALESCE(plugin_name,''), COALESCE(config_json,'{}')
		FROM devices
		WHERE id = $1 AND user_id = $2
	`, deviceID, id.UserID).Scan(&rec.ID, &rec.UserID, &rec.Name, &rec.DeviceType, &rec.PluginName, &configJSON)
	if err != nil {
		if errorsIsNoRows(err) {
			writeError(w, http.StatusNotFound, "device not found")
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	rec.Config = parseConfigJSON(configJSON)

	if rec.DeviceType == "telegram" || rec.PluginName == "telegram" {
		botToken := strings.TrimSpace(rec.Config["bot_token"])
		if botToken != "" {
			_ = deleteTelegramWebhook(r.Context(), botToken)
		}
	}

	_, err = s.db.Exec(r.Context(), `DELETE FROM devices WHERE id = $1 AND user_id = $2`, deviceID, id.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "deleted"})
}

func (s *Server) handlePluginWebhookIngress(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	path := strings.Trim(trimAnyPrefix(r.URL.Path, orchestratorAPIPrefix+"/hooks/", "/api/hooks/"), "/")
	parts := strings.Split(path, "/")
	if len(parts) != 2 {
		writeError(w, http.StatusNotFound, "not found")
		return
	}
	pluginName := strings.TrimSpace(parts[0])
	deviceID := strings.TrimSpace(parts[1])
	if pluginName == "" || deviceID == "" {
		writeError(w, http.StatusBadRequest, "plugin and device id are required")
		return
	}

	var userID string
	var configJSON string
	err := s.db.QueryRow(r.Context(), `
		SELECT user_id, COALESCE(config_json,'{}')
		FROM devices
		WHERE id = $1 AND (device_type = $2 OR plugin_name = $2)
	`, deviceID, pluginName).Scan(&userID, &configJSON)
	if err != nil {
		log.Printf("hooks: unknown device plugin=%s device_id=%s err=%v", pluginName, deviceID, err)
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	cfg := parseConfigJSON(configJSON)
	if !validateWebhookSignature(pluginName, r, cfg) {
		log.Printf("hooks: invalid signature plugin=%s device_id=%s", pluginName, deviceID)
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 2*1024*1024))
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	target, err := s.resolveAgent(r.Context(), userID)
	if err != nil {
		log.Printf("hooks: resolveAgent failed plugin=%s user=%s err=%v", pluginName, userID, err)
		writeJSON(w, http.StatusOK, map[string]any{"status": "queued", "downstream": false})
		return
	}

	if err := s.forwardHookToAgent(r.Context(), target, pluginName, userID, deviceID, body, r.Header); err != nil {
		log.Printf("hooks: forward failed plugin=%s user=%s device=%s err=%v", pluginName, userID, deviceID, err)
		writeJSON(w, http.StatusOK, map[string]any{"status": "queued", "downstream": false})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "queued", "downstream": true})
}

func (s *Server) forwardHookToAgent(ctx context.Context, target agent.Target, pluginName, userID, deviceID string, body []byte, headers http.Header) error {
	url := fmt.Sprintf("http://%s:%d/internal/hooks/%s", target.Host, target.Port, pluginName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	proxy.CopyRequestHeaders(req.Header, headers)
	req.Header.Set("X-Aether-User-ID", userID)
	req.Header.Set("X-Aether-Plugin", pluginName)
	req.Header.Set("X-Aether-Device-ID", deviceID)
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("agent webhook status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return nil
}

func validateWebhookSignature(pluginName string, r *http.Request, cfg map[string]string) bool {
	switch strings.ToLower(strings.TrimSpace(pluginName)) {
	case "telegram":
		received := strings.TrimSpace(r.Header.Get("X-Telegram-Bot-Api-Secret-Token"))
		expected := strings.TrimSpace(cfg["secret_token"])
		if expected == "" {
			return false
		}
		return received == expected
	default:
		return true
	}
}

func setTelegramWebhook(ctx context.Context, botToken, webhookURL, secretToken string) error {
	payload := map[string]any{
		"url":          webhookURL,
		"secret_token": secretToken,
	}
	b, _ := json.Marshal(payload)
	apiURL := "https://api.telegram.org/bot" + strings.TrimSpace(botToken) + "/setWebhook"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("telegram setWebhook failed status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func deleteTelegramWebhook(ctx context.Context, botToken string) error {
	apiURL := "https://api.telegram.org/bot" + strings.TrimSpace(botToken) + "/deleteWebhook?drop_pending_updates=true"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, nil)
	if err != nil {
		return err
	}
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("telegram deleteWebhook failed status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func parseConfigJSON(raw string) map[string]string {
	out := map[string]string{}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return out
	}
	_ = json.Unmarshal([]byte(raw), &out)
	return out
}

func scrubDeviceConfig(cfg map[string]string) {
	for _, key := range []string{"bot_token", "secret_token", "token", "api_key", "auth_token"} {
		if strings.TrimSpace(cfg[key]) != "" {
			cfg[key] = "__configured__"
		}
	}
}

func resolvePublicBaseURL(r *http.Request) string {
	if env := strings.TrimSpace(os.Getenv("AETHER_PUBLIC_BASE_URL")); env != "" {
		return strings.TrimRight(env, "/")
	}
	forwardedHost := strings.TrimSpace(r.Header.Get("X-Forwarded-Host"))
	if forwardedHost != "" {
		proto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto"))
		if proto == "" {
			proto = "https"
		}
		return proto + "://" + forwardedHost
	}
	if host := strings.TrimSpace(r.Host); host != "" {
		return "https://" + host
	}
	return ""
}

func randomID(prefix string) (string, error) {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	if strings.TrimSpace(prefix) == "" {
		return hex.EncodeToString(b), nil
	}
	return prefix + "_" + hex.EncodeToString(b), nil
}

func errorsIsNoRows(err error) bool {
	return err == pgx.ErrNoRows
}

// Pub/Sub webhook handler for Google push notifications (Gmail, Calendar, etc.)
// Route: /go/v1/hooks/pubsub/{plugin_name}

type pubsubMessage struct {
	Data        string            `json:"data"`
	Attributes  map[string]string `json:"attributes"`
	MessageID   string            `json:"messageId"`
	PublishTime string            `json:"publishTime"`
}

type pubsubPushMessage struct {
	Message      pubsubMessage `json:"message"`
	Subscription string        `json:"subscription"`
}

func (s *Server) handlePubsubWebhookIngress(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	path := strings.Trim(trimAnyPrefix(r.URL.Path, orchestratorAPIPrefix+"/hooks/pubsub/", "/api/hooks/pubsub/"), "/")
	pluginName := strings.TrimSpace(path)
	if pluginName == "" {
		writeError(w, http.StatusBadRequest, "plugin name is required")
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 2*1024*1024))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var pushMsg pubsubPushMessage
	if err := json.Unmarshal(body, &pushMsg); err != nil {
		log.Printf("pubsub: invalid message format: %v", err)
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	if pushMsg.Message.Data == "" {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	notificationData, err := decodePubsubData(pushMsg.Message.Data)
	if err != nil {
		log.Printf("pubsub: failed to decode message data: %v", err)
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	// Fan-out: resolve ALL users with this email address in email_mappings
	emailAddress := notificationData["emailAddress"]
	userIDs := lookupUsersByEmail(r.Context(), s.db, emailAddress, pluginName)
	if len(userIDs) == 0 {
		// Fallback: check notification payload for user_id, then LIMIT 1 lookup
		if uid := notificationData["user_id"]; uid != "" {
			userIDs = []string{uid}
		} else {
			if uid := lookupUserByPlugin(r.Context(), s.db, pluginName); uid != "" {
				userIDs = []string{uid}
			}
		}
	}
	if len(userIDs) == 0 {
		log.Printf("pubsub: no user found for plugin %s email=%s", pluginName, emailAddress)
		writeJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	notificationBody, _ := json.Marshal(map[string]any{
		"plugin":       pluginName,
		"source":       "pubsub",
		"notification": notificationData,
		"attributes":   pushMsg.Message.Attributes,
		"message_id":   pushMsg.Message.MessageID,
	})

	delivered := 0
	for _, userID := range userIDs {
		deviceID := lookupDeviceByPlugin(r.Context(), s.db, pluginName, userID)
		target, err := s.resolveAgent(r.Context(), userID)
		if err != nil {
			log.Printf("pubsub: resolveAgent failed plugin=%s user=%s err=%v", pluginName, userID, err)
			continue
		}
		if err := s.forwardHookToAgent(r.Context(), target, pluginName, userID, deviceID, notificationBody, r.Header); err != nil {
			log.Printf("pubsub: forward failed plugin=%s user=%s err=%v", pluginName, userID, err)
			continue
		}
		delivered++
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "queued", "downstream": delivered > 0, "users": len(userIDs), "delivered": delivered})
}

func decodePubsubData(encoded string) (map[string]string, error) {
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("base64 decode failed: %w", err)
	}

	var data map[string]string
	if err := json.Unmarshal(decoded, &data); err != nil {
		return nil, fmt.Errorf("json decode failed: %w", err)
	}

	return data, nil
}

func lookupUserByPlugin(ctx context.Context, db *pgxpool.Pool, pluginName string) string {
	var userID string
	err := db.QueryRow(ctx, `
		SELECT user_id FROM plugins WHERE name = $1 AND enabled = true LIMIT 1
	`, pluginName).Scan(&userID)
	if err != nil {
		return ""
	}
	return userID
}

func lookupUsersByEmail(ctx context.Context, db *pgxpool.Pool, email, pluginName string) []string {
	email = strings.TrimSpace(email)
	if email == "" {
		return nil
	}
	rows, err := db.Query(ctx, `
		SELECT user_id FROM email_mappings WHERE email = $1 AND plugin_name = $2
	`, email, pluginName)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var userIDs []string
	for rows.Next() {
		var uid string
		if err := rows.Scan(&uid); err == nil {
			userIDs = append(userIDs, uid)
		}
	}
	return userIDs
}

func (s *Server) handleEmailMappings(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var req struct {
			Email      string `json:"email"`
			UserID     string `json:"user_id"`
			PluginName string `json:"plugin_name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json body")
			return
		}
		email := strings.TrimSpace(req.Email)
		userID := strings.TrimSpace(req.UserID)
		pluginName := strings.TrimSpace(req.PluginName)
		if email == "" || userID == "" {
			writeError(w, http.StatusBadRequest, "email and user_id are required")
			return
		}
		if pluginName == "" {
			pluginName = "google-workspace"
		}
		_, err := s.db.Exec(r.Context(), `
			INSERT INTO email_mappings (email, user_id, plugin_name, created_at)
			VALUES ($1, $2, $3, now())
			ON CONFLICT (email, user_id) DO UPDATE SET plugin_name = $3, created_at = now()
		`, email, userID, pluginName)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
		return
	}

	if r.Method == http.MethodDelete {
		var req struct {
			Email      string `json:"email"`
			UserID     string `json:"user_id"`
			PluginName string `json:"plugin_name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json body")
			return
		}
		email := strings.TrimSpace(req.Email)
		userID := strings.TrimSpace(req.UserID)
		pluginName := strings.TrimSpace(req.PluginName)
		if pluginName == "" {
			pluginName = "google-workspace"
		}
		if email == "" || userID == "" {
			writeError(w, http.StatusBadRequest, "email and user_id are required")
			return
		}
		_, _ = s.db.Exec(r.Context(), `
			DELETE FROM email_mappings WHERE email = $1 AND user_id = $2 AND plugin_name = $3
		`, email, userID, pluginName)
		writeJSON(w, http.StatusOK, map[string]any{"status": "deleted"})
		return
	}

	writeError(w, http.StatusMethodNotAllowed, "method not allowed")
}

func lookupDeviceByPlugin(ctx context.Context, db *pgxpool.Pool, pluginName, userID string) string {
	var deviceID string
	err := db.QueryRow(ctx, `
		SELECT id FROM devices WHERE user_id = $1 AND plugin_name = $2 LIMIT 1
	`, userID, pluginName).Scan(&deviceID)
	if err != nil {
		return ""
	}
	return deviceID
}
