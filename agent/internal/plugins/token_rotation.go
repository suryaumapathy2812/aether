package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

func RegisterDefaultTokenRotators(registry *CronRegistry) {
	if registry == nil {
		return
	}
	googlePlugins := []string{"gmail", "google-calendar", "google-contacts", "google-drive"}
	for _, pluginName := range googlePlugins {
		registry.RegisterTokenRotator(pluginName, oauthTokenRotator("https://oauth2.googleapis.com/token", false))
	}
	registry.RegisterTokenRotator("spotify", oauthTokenRotator("https://accounts.spotify.com/api/token", true))
}

func oauthTokenRotator(tokenURL string, useBasicAuth bool) TokenRotator {
	return func(ctx context.Context, state PluginState, payload map[string]any) error {
		_ = payload
		cfg, err := state.Config(ctx)
		if err != nil {
			return err
		}
		if cfg == nil {
			cfg = map[string]string{}
		}

		refreshTokenRaw := firstNonEmpty(cfg["refresh_token"], cfg["oauth_refresh_token"])
		refreshToken, err := maybeDecrypt(state, refreshTokenRaw)
		if err != nil {
			_ = persistRefreshFailure(ctx, state, cfg, "failed", "failed to decrypt refresh token: "+err.Error())
			return err
		}
		if strings.TrimSpace(refreshToken) == "" {
			err := fmt.Errorf("missing refresh token")
			_ = persistRefreshFailure(ctx, state, cfg, "failed", err.Error())
			return err
		}

		clientID := firstNonEmpty(cfg["client_id"], cfg["google_client_id"], cfg["spotify_client_id"])
		clientSecret := firstNonEmpty(cfg["client_secret"], cfg["google_client_secret"], cfg["spotify_client_secret"])
		if strings.TrimSpace(clientID) == "" || strings.TrimSpace(clientSecret) == "" {
			err := fmt.Errorf("missing client_id or client_secret")
			_ = persistRefreshFailure(ctx, state, cfg, "failed", err.Error())
			return err
		}

		form := url.Values{}
		form.Set("grant_type", "refresh_token")
		form.Set("refresh_token", refreshToken)
		if !useBasicAuth {
			form.Set("client_id", clientID)
			form.Set("client_secret", clientSecret)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
		if err != nil {
			_ = persistRefreshFailure(ctx, state, cfg, "failed", err.Error())
			return err
		}
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		if useBasicAuth {
			req.SetBasicAuth(clientID, clientSecret)
		}

		httpClient := &http.Client{Timeout: 20 * time.Second}
		resp, err := httpClient.Do(req)
		if err != nil {
			_ = persistRefreshFailure(ctx, state, cfg, "failed", err.Error())
			return err
		}
		defer resp.Body.Close()

		var payloadResp struct {
			AccessToken  string `json:"access_token"`
			RefreshToken string `json:"refresh_token"`
			ExpiresIn    int64  `json:"expires_in"`
			TokenType    string `json:"token_type"`
			Scope        string `json:"scope"`
			Error        string `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&payloadResp); err != nil {
			_ = persistRefreshFailure(ctx, state, cfg, "failed", "failed to decode token response")
			return err
		}
		if resp.StatusCode != http.StatusOK {
			errMsg := payloadResp.Error
			if strings.TrimSpace(errMsg) == "" {
				errMsg = fmt.Sprintf("token endpoint status %d", resp.StatusCode)
			}
			_ = persistRefreshFailure(ctx, state, cfg, "failed", errMsg)
			return fmt.Errorf("token refresh failed: %s", errMsg)
		}
		if strings.TrimSpace(payloadResp.AccessToken) == "" {
			err := fmt.Errorf("token response missing access_token")
			_ = persistRefreshFailure(ctx, state, cfg, "failed", err.Error())
			return err
		}

		now := time.Now().UTC()
		expiresIn := payloadResp.ExpiresIn
		if expiresIn <= 0 {
			expiresIn = 3600
		}
		expiresAt := now.Add(time.Duration(expiresIn) * time.Second)

		encAccess, encErr := state.EncryptString(payloadResp.AccessToken)
		if encErr == nil {
			cfg["access_token"] = encAccess
		} else {
			cfg["access_token"] = payloadResp.AccessToken
		}
		if payloadResp.RefreshToken != "" {
			encRefresh, err := state.EncryptString(payloadResp.RefreshToken)
			if err == nil {
				cfg["refresh_token"] = encRefresh
			} else {
				cfg["refresh_token"] = payloadResp.RefreshToken
			}
		}
		if payloadResp.TokenType != "" {
			cfg["token_type"] = payloadResp.TokenType
		}
		if payloadResp.Scope != "" {
			cfg["scope"] = payloadResp.Scope
		}
		cfg["expires_at"] = strconv.FormatInt(expiresAt.Unix(), 10)
		cfg["last_refresh_at"] = now.Format(time.RFC3339)
		cfg["last_refresh_status"] = "ok"
		cfg["last_refresh_error"] = ""
		cfg["refresh_fail_count"] = "0"
		cfg["next_refresh_at"] = expiresAt.Add(-5 * time.Minute).Format(time.RFC3339)

		return state.SetConfig(ctx, cfg)
	}
}

func persistRefreshFailure(ctx context.Context, state PluginState, cfg map[string]string, status string, errMsg string) error {
	now := time.Now().UTC()
	fails := 0
	if v := strings.TrimSpace(cfg["refresh_fail_count"]); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			fails = n
		}
	}
	fails++
	cfg["last_refresh_at"] = now.Format(time.RFC3339)
	cfg["last_refresh_status"] = status
	cfg["last_refresh_error"] = errMsg
	cfg["refresh_fail_count"] = strconv.Itoa(fails)
	return state.SetConfig(ctx, cfg)
}

func maybeDecrypt(state PluginState, value string) (string, error) {
	v := strings.TrimSpace(value)
	if !strings.HasPrefix(v, "enc:v1:") {
		return v, nil
	}
	return state.DecryptString(v)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
