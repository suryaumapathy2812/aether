package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/auth"
)

func (s *Server) handlePairRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req struct {
		Code       string `json:"code"`
		DeviceType string `json:"device_type"`
		DeviceName string `json:"device_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	code := normalizePairCode(req.Code)
	if code == "" {
		writeError(w, http.StatusBadRequest, "code is required")
		return
	}
	deviceType := strings.ToLower(strings.TrimSpace(req.DeviceType))
	if deviceType == "" {
		deviceType = "ios"
	}
	deviceName := strings.TrimSpace(req.DeviceName)
	if deviceName == "" {
		deviceName = "Unknown Device"
	}

	_, err := s.db.Exec(r.Context(), `
		INSERT INTO pair_requests (code, device_type, device_name, created_at, expires_at, claimed_by)
		VALUES ($1, $2, $3, now(), now() + interval '10 minutes', NULL)
		ON CONFLICT (code) DO UPDATE SET
			device_type = EXCLUDED.device_type,
			device_name = EXCLUDED.device_name,
			created_at = now(),
			expires_at = now() + interval '10 minutes',
			claimed_by = NULL,
			issued_channel_id = NULL,
			issued_token = NULL,
			issued_at = NULL
	`, code, deviceType, deviceName)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save pairing request")
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "code": code})
}

func (s *Server) handlePairClaim(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req struct {
		Code string `json:"code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	code := normalizePairCode(req.Code)
	if code == "" {
		writeError(w, http.StatusBadRequest, "code is required")
		return
	}

	var claimedBy string
	var expiresAt time.Time
	err := s.db.QueryRow(r.Context(), `
		SELECT COALESCE(claimed_by, ''), expires_at
		FROM pair_requests
		WHERE code = $1
	`, code).Scan(&claimedBy, &expiresAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeError(w, http.StatusNotFound, "pairing code not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to read pairing request")
		return
	}
	if time.Now().After(expiresAt) {
		_, _ = s.db.Exec(r.Context(), `DELETE FROM pair_requests WHERE code = $1`, code)
		writeError(w, http.StatusGone, "pairing code expired")
		return
	}
	if strings.TrimSpace(claimedBy) != "" {
		writeError(w, http.StatusConflict, "pairing code already claimed")
		return
	}

	if _, err := s.db.Exec(r.Context(), `
		UPDATE pair_requests
		SET claimed_by = $1,
			issued_channel_id = NULL,
			issued_token = NULL,
			issued_at = NULL
		WHERE code = $2
	`, id.UserID, code); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to claim pairing code")
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "claimed"})
}

func (s *Server) handlePairStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	code := normalizePairCode(strings.TrimPrefix(r.URL.Path, "/api/pair/status/"))
	if code == "" {
		writeError(w, http.StatusBadRequest, "code is required")
		return
	}

	tx, err := s.db.BeginTx(r.Context(), pgx.TxOptions{})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to begin pairing transaction")
		return
	}
	defer func() { _ = tx.Rollback(r.Context()) }()

	var claimedBy string
	var deviceType string
	var deviceName string
	var expiresAt time.Time
	var issuedChannelID string
	var issuedToken string
	var issuedAt *time.Time
	err = tx.QueryRow(r.Context(), `
		SELECT
			COALESCE(claimed_by, ''),
			COALESCE(device_type, 'ios'),
			COALESCE(device_name, 'Unknown Device'),
			expires_at,
			COALESCE(issued_channel_id, ''),
			COALESCE(issued_token, ''),
			issued_at
		FROM pair_requests
		WHERE code = $1
		FOR UPDATE
	`, code).Scan(&claimedBy, &deviceType, &deviceName, &expiresAt, &issuedChannelID, &issuedToken, &issuedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeJSON(w, http.StatusOK, map[string]any{"status": "not_found"})
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to read pairing request")
		return
	}

	if time.Now().After(expiresAt) {
		_, _ = tx.Exec(r.Context(), `DELETE FROM pair_requests WHERE code = $1`, code)
		_ = tx.Commit(r.Context())
		writeJSON(w, http.StatusOK, map[string]any{"status": "expired"})
		return
	}

	if strings.TrimSpace(claimedBy) == "" {
		_ = tx.Commit(r.Context())
		writeJSON(w, http.StatusOK, map[string]any{"status": "pending"})
		return
	}

	if strings.TrimSpace(issuedToken) != "" && strings.TrimSpace(issuedChannelID) != "" {
		if issuedAt != nil && time.Since(*issuedAt) <= 10*time.Minute {
			_ = tx.Commit(r.Context())
			writeJSON(w, http.StatusOK, map[string]any{
				"status":       "paired",
				"device_token": issuedToken,
			})
			return
		}
	}

	channelID, err := randomID("ios")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to generate channel id")
		return
	}
	if err := s.registerIOSChannel(r.Context(), claimedBy, channelID, deviceName); err != nil {
		writeError(w, http.StatusBadGateway, "failed to register channel")
		return
	}

	token, err := s.auth.MintChannelToken(claimedBy, channelID, deviceType, 365*24*time.Hour)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to issue channel token")
		return
	}

	if _, err := tx.Exec(r.Context(), `
		UPDATE pair_requests
		SET issued_channel_id = $1,
			issued_token = $2,
			issued_at = now(),
			expires_at = now() + interval '10 minutes'
		WHERE code = $3
	`, channelID, token, code); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to persist pairing issuance")
		return
	}
	if err := tx.Commit(r.Context()); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to finalize pairing")
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "paired",
		"device_token": token,
	})
}

func (s *Server) registerIOSChannel(ctx context.Context, userID, channelID, deviceName string) error {
	target, err := s.resolveAgent(ctx, userID)
	if err != nil {
		return err
	}
	body, err := json.Marshal(map[string]any{
		"user_id":      userID,
		"channel_id":   channelID,
		"display_name": strings.TrimSpace(deviceName),
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://%s:%d/api/channels/ios/connect", target.Host, target.Port), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("ios channel registration failed: %s", resp.Status)
	}
	return nil
}

func normalizePairCode(v string) string {
	return strings.ToUpper(strings.TrimSpace(v))
}
