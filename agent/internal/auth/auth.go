package auth

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"
)

const (
	DirectTokenPrefix = "aether_direct.v1"
	DirectHostSuffix  = ".aether.suryaumapathy.in"
)

type contextKey string

const userIDContextKey contextKey = "agent.auth.user_id"

var (
	ErrMissingToken  = errors.New("missing token")
	ErrInvalidToken  = errors.New("invalid direct token")
	ErrExpiredToken  = errors.New("expired direct token")
	ErrHostMismatch  = errors.New("direct token host mismatch")
	ErrAgentMismatch = errors.New("direct token agent mismatch")
)

type Claims struct {
	UserID    string `json:"sub"`
	Prefix    string `json:"pfx"`
	AgentID   string `json:"aid"`
	Audience  string `json:"aud"`
	IssuedAt  int64  `json:"iat"`
	ExpiresAt int64  `json:"exp"`
}

type Validator struct {
	secret  string
	agentID string
}

func NewValidator(secret string, agentID string) *Validator {
	return &Validator{secret: strings.TrimSpace(secret), agentID: strings.TrimSpace(agentID)}
}

func (v *Validator) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		claims, err := v.ValidateRequest(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r.WithContext(WithUserID(r.Context(), claims.UserID)))
	})
}

func (v *Validator) ValidateRequest(r *http.Request) (Claims, error) {
	token := ExtractToken(r)
	if token == "" {
		return Claims{}, ErrMissingToken
	}
	if strings.TrimSpace(v.secret) == "" {
		return Claims{}, ErrInvalidToken
	}
	if !IsDirectToken(token) {
		return Claims{}, ErrInvalidToken
	}
	parts := strings.Split(strings.TrimPrefix(strings.TrimSpace(token), DirectTokenPrefix+"."), ".")
	if len(parts) != 2 {
		return Claims{}, ErrInvalidToken
	}
	payload := parts[0]
	providedSig := parts[1]
	expectedSig := signPayload(v.secret, payload)
	if !hmac.Equal([]byte(providedSig), []byte(expectedSig)) {
		return Claims{}, ErrInvalidToken
	}
	decoded, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil {
		return Claims{}, ErrInvalidToken
	}
	var claims Claims
	if err := json.Unmarshal(decoded, &claims); err != nil {
		return Claims{}, ErrInvalidToken
	}
	if strings.TrimSpace(claims.UserID) == "" || strings.TrimSpace(claims.Prefix) == "" || claims.Audience != "agent" {
		return Claims{}, ErrInvalidToken
	}
	if claims.ExpiresAt > 0 && time.Now().UTC().Unix() > claims.ExpiresAt {
		return Claims{}, ErrExpiredToken
	}
	if v.agentID != "" && strings.TrimSpace(claims.AgentID) != "" && strings.TrimSpace(claims.AgentID) != v.agentID {
		return Claims{}, ErrAgentMismatch
	}
	if !hostMatchesPrefix(r.Host, claims.Prefix) {
		return Claims{}, ErrHostMismatch
	}
	return claims, nil
}

func IsDirectToken(token string) bool {
	return strings.HasPrefix(strings.TrimSpace(token), DirectTokenPrefix+".")
}

func ExtractToken(r *http.Request) string {
	h := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(h), "bearer ") {
		return strings.TrimSpace(h[7:])
	}
	if q := strings.TrimSpace(r.URL.Query().Get("token")); q != "" {
		return q
	}
	for _, name := range []string{"aether_direct_token", "__Secure-aether_direct_token"} {
		if c, err := r.Cookie(name); err == nil {
			if value := strings.TrimSpace(c.Value); value != "" {
				return value
			}
		}
	}
	return ""
}

func WithUserID(ctx context.Context, userID string) context.Context {
	return context.WithValue(ctx, userIDContextKey, strings.TrimSpace(userID))
}

func UserIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	value, _ := ctx.Value(userIDContextKey).(string)
	return strings.TrimSpace(value)
}

func RequestUserID(r *http.Request, fallbackValues ...string) string {
	if userID := UserIDFromContext(r.Context()); userID != "" {
		return userID
	}
	for _, value := range fallbackValues {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return "default"
}

func ResolveDirectUserID(r *http.Request, validator *Validator, fallbackValues ...string) (string, error) {
	if userID := UserIDFromContext(r.Context()); userID != "" {
		return userID, nil
	}
	if validator == nil {
		return RequestUserID(r, fallbackValues...), nil
	}
	if RequiresDirectToken(r) || IsDirectToken(ExtractToken(r)) {
		claims, err := validator.ValidateRequest(r)
		if err != nil {
			return "", err
		}
		return claims.UserID, nil
	}
	return RequestUserID(r, fallbackValues...), nil
}

func AuthorizeDirectRequest(r *http.Request, validator *Validator) error {
	if validator == nil {
		return nil
	}
	if RequiresDirectToken(r) || IsDirectToken(ExtractToken(r)) {
		_, err := validator.ValidateRequest(r)
		return err
	}
	return nil
}

func signPayload(secret, payload string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(payload))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func hostMatchesPrefix(host, prefix string) bool {
	trimmedHost := strings.TrimSpace(host)
	trimmedPrefix := strings.TrimSpace(prefix)
	if trimmedHost == "" || trimmedPrefix == "" {
		return false
	}
	hostOnly := trimmedHost
	if strings.Contains(hostOnly, ":") {
		if parts := strings.Split(hostOnly, ":"); len(parts) > 0 {
			hostOnly = parts[0]
		}
	}
	hostOnly = strings.Trim(strings.ToLower(hostOnly), ".")
	trimmedPrefix = strings.ToLower(trimmedPrefix)
	if trimmedPrefix == "local" {
		return hostOnly == "localhost" || hostOnly == "127.0.0.1"
	}
	return strings.HasPrefix(hostOnly, trimmedPrefix+".")
}

func RequiresDirectToken(r *http.Request) bool {
	if r == nil {
		return false
	}
	host := strings.ToLower(strings.TrimSpace(r.Host))
	if host == "" {
		return false
	}
	if strings.Contains(host, ":") {
		parts := strings.Split(host, ":")
		host = parts[0]
	}
	host = strings.Trim(host, ".")
	return strings.HasSuffix(host, DirectHostSuffix)
}
