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

const directTokenPrefix = "aether_direct.v1"

type DirectTokenClaims struct {
	UserID    string `json:"sub"`
	Prefix    string `json:"pfx"`
	AgentID   string `json:"aid"`
	Audience  string `json:"aud"`
	IssuedAt  int64  `json:"iat"`
	ExpiresAt int64  `json:"exp"`
}

func (a *Authenticator) MintDirectToken(userID, prefix, agentID string, ttl time.Duration) (string, DirectTokenClaims, error) {
	var claims DirectTokenClaims
	if strings.TrimSpace(a.directSecret) == "" {
		return "", claims, fmt.Errorf("direct token secret is not configured")
	}
	uid := strings.TrimSpace(userID)
	if uid == "" {
		return "", claims, fmt.Errorf("user id is required")
	}
	pfx := strings.TrimSpace(prefix)
	if pfx == "" {
		return "", claims, fmt.Errorf("prefix is required")
	}
	if ttl <= 0 {
		ttl = time.Hour
	}
	now := time.Now().UTC()
	claims = DirectTokenClaims{
		UserID:    uid,
		Prefix:    pfx,
		AgentID:   strings.TrimSpace(agentID),
		Audience:  "agent",
		IssuedAt:  now.Unix(),
		ExpiresAt: now.Add(ttl).Unix(),
	}
	b, err := json.Marshal(claims)
	if err != nil {
		return "", DirectTokenClaims{}, err
	}
	payload := base64.RawURLEncoding.EncodeToString(b)
	sig := signDirectPayload(a.directSecret, payload)
	return directTokenPrefix + "." + payload + "." + sig, claims, nil
}

func (a *Authenticator) VerifyDirectToken(token string) (DirectTokenClaims, bool) {
	var claims DirectTokenClaims
	if strings.TrimSpace(a.directSecret) == "" {
		return claims, false
	}
	t := strings.TrimSpace(token)
	prefix := directTokenPrefix + "."
	if !strings.HasPrefix(t, prefix) {
		return claims, false
	}
	parts := strings.Split(t[len(prefix):], ".")
	if len(parts) != 2 {
		return claims, false
	}
	payload := parts[0]
	providedSig := parts[1]
	expected := signDirectPayload(a.directSecret, payload)
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
	if strings.TrimSpace(claims.UserID) == "" || strings.TrimSpace(claims.Prefix) == "" {
		return DirectTokenClaims{}, false
	}
	if claims.Audience != "agent" {
		return DirectTokenClaims{}, false
	}
	if claims.ExpiresAt > 0 && time.Now().UTC().Unix() > claims.ExpiresAt {
		return DirectTokenClaims{}, false
	}
	return claims, true
}

func signDirectPayload(secret, payload string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(payload))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}
