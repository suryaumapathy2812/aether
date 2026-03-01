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

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

func refreshOAuthAccessToken(ctx context.Context, call tools.Call, tokenURL string, useBasicAuth bool) tools.Result {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	rawRefresh := cfg["refresh_token"]
	if strings.TrimSpace(rawRefresh) == "" {
		rawRefresh = cfg["oauth_refresh_token"]
	}
	refreshToken, err := maybeDecryptFromState(ctx, call, rawRefresh)
	if err != nil {
		return tools.Fail("Failed to decrypt refresh token: "+err.Error(), nil)
	}
	if strings.TrimSpace(refreshToken) == "" {
		return tools.Fail("Missing refresh token in plugin config", nil)
	}

	clientID := firstNonEmpty(cfg["client_id"], cfg["google_client_id"], cfg["spotify_client_id"])
	clientSecret := firstNonEmpty(cfg["client_secret"], cfg["google_client_secret"], cfg["spotify_client_secret"])
	if strings.HasPrefix(strings.TrimSpace(clientSecret), "enc:v1:") {
		if decrypted, decErr := maybeDecryptFromState(ctx, call, clientSecret); decErr == nil {
			clientSecret = decrypted
		}
	}
	if strings.TrimSpace(clientID) == "" || strings.TrimSpace(clientSecret) == "" {
		return tools.Fail("Missing client_id/client_secret in plugin config", nil)
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
		return tools.Fail("Failed to build refresh request: "+err.Error(), nil)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if useBasicAuth {
		req.SetBasicAuth(clientID, clientSecret)
	}

	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return tools.Fail("Token refresh failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()

	var data struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int64  `json:"expires_in"`
		TokenType    string `json:"token_type"`
		Scope        string `json:"scope"`
		Error        string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return tools.Fail("Failed to decode refresh response", nil)
	}
	if resp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(data.Error)
		if msg == "" {
			msg = fmt.Sprintf("token endpoint status %d", resp.StatusCode)
		}
		return tools.Fail("Token refresh failed: "+msg, nil)
	}
	if strings.TrimSpace(data.AccessToken) == "" {
		return tools.Fail("Refresh response missing access_token", nil)
	}

	accessToken := data.AccessToken
	if call.Ctx.PluginState != nil {
		if enc, err := call.Ctx.PluginState.EncryptString(data.AccessToken); err == nil {
			accessToken = enc
		}
	}
	cfg["access_token"] = accessToken
	if strings.TrimSpace(data.RefreshToken) != "" {
		refresh := data.RefreshToken
		if call.Ctx.PluginState != nil {
			if enc, err := call.Ctx.PluginState.EncryptString(data.RefreshToken); err == nil {
				refresh = enc
			}
		}
		cfg["refresh_token"] = refresh
	}
	if strings.TrimSpace(data.TokenType) != "" {
		cfg["token_type"] = data.TokenType
	}
	if strings.TrimSpace(data.Scope) != "" {
		cfg["scope"] = data.Scope
	}
	expiresIn := data.ExpiresIn
	if expiresIn <= 0 {
		expiresIn = 3600
	}
	expiresAt := time.Now().UTC().Add(time.Duration(expiresIn) * time.Second)
	cfg["expires_at"] = strconv.FormatInt(expiresAt.Unix(), 10)
	cfg["last_refresh_at"] = time.Now().UTC().Format(time.RFC3339)
	cfg["last_refresh_status"] = "ok"
	cfg["last_refresh_error"] = ""
	if call.Ctx.PluginState != nil {
		if err := call.Ctx.PluginState.SetConfig(ctx, cfg); err != nil {
			return tools.Fail("Failed to persist refreshed token: "+err.Error(), nil)
		}
	}
	return tools.Success("Token refreshed successfully.", map[string]any{"expires_at": cfg["expires_at"]})
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
