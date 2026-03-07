package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const channelTokenPrefix = "aether_chan.v1"

type ChannelTokenClaims struct {
	UserID      string `json:"sub"`
	ChannelID   string `json:"ch"`
	ChannelType string `json:"typ"`
	IssuedAt    int64  `json:"iat"`
	ExpiresAt   int64  `json:"exp"`
}

func (a *Authenticator) MintChannelToken(userID, channelID, channelType string, ttl time.Duration) (string, error) {
	if strings.TrimSpace(a.secret) == "" {
		return "", fmt.Errorf("channel token secret is not configured")
	}
	uid := strings.TrimSpace(userID)
	if uid == "" {
		return "", fmt.Errorf("user id is required")
	}
	chid := strings.TrimSpace(channelID)
	if chid == "" {
		return "", fmt.Errorf("channel id is required")
	}
	typ := strings.TrimSpace(channelType)
	if typ == "" {
		typ = "ios"
	}
	if ttl <= 0 {
		ttl = 365 * 24 * time.Hour
	}
	now := time.Now().UTC()
	claims := ChannelTokenClaims{
		UserID:      uid,
		ChannelID:   chid,
		ChannelType: typ,
		IssuedAt:    now.Unix(),
		ExpiresAt:   now.Add(ttl).Unix(),
	}
	b, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	payload := base64.RawURLEncoding.EncodeToString(b)
	sig := signChannelPayload(a.secret, payload)
	return channelTokenPrefix + "." + payload + "." + sig, nil
}

func (a *Authenticator) VerifyChannelToken(token string) (ChannelTokenClaims, bool) {
	var claims ChannelTokenClaims
	if strings.TrimSpace(a.secret) == "" {
		return claims, false
	}
	t := strings.TrimSpace(token)
	prefix := channelTokenPrefix + "."
	if !strings.HasPrefix(t, prefix) {
		return claims, false
	}
	parts := strings.Split(t[len(prefix):], ".")
	if len(parts) != 2 {
		return claims, false
	}
	payload := parts[0]
	providedSig := parts[1]
	expected := signChannelPayload(a.secret, payload)
	if !hmac.Equal([]byte(providedSig), []byte(expected)) {
		return claims, false
	}
	decoded, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil {
		return claims, false
	}
	if err := json.Unmarshal(decoded, &claims); err != nil {
		return claims, false
	}
	if strings.TrimSpace(claims.UserID) == "" || strings.TrimSpace(claims.ChannelID) == "" {
		return claims, false
	}
	return claims, true
}

func signChannelPayload(secret, payload string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(payload))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}
