package auth

import (
	"encoding/base64"
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"
)

func TestValidateRequestAcceptsDirectToken(t *testing.T) {
	t.Parallel()
	v := NewValidator("secret", "agent-1")
	token := mustDirectToken(t, Claims{
		UserID:    "user-1",
		Prefix:    "abc12345",
		AgentID:   "agent-1",
		Audience:  "agent",
		IssuedAt:  time.Now().UTC().Unix(),
		ExpiresAt: time.Now().UTC().Add(time.Hour).Unix(),
	}, "secret")
	req := httptest.NewRequest("GET", "http://abc12345.example.com/test?token="+token, nil)
	claims, err := v.ValidateRequest(req)
	if err != nil {
		t.Fatalf("validate request: %v", err)
	}
	if claims.UserID != "user-1" {
		t.Fatalf("expected user-1, got %q", claims.UserID)
	}
}

func TestValidateRequestRejectsWrongHost(t *testing.T) {
	t.Parallel()
	v := NewValidator("secret", "agent-1")
	token := mustDirectToken(t, Claims{
		UserID:    "user-1",
		Prefix:    "abc12345",
		AgentID:   "agent-1",
		Audience:  "agent",
		IssuedAt:  time.Now().UTC().Unix(),
		ExpiresAt: time.Now().UTC().Add(time.Hour).Unix(),
	}, "secret")
	req := httptest.NewRequest("GET", "http://wrong.example.com/test?token="+token, nil)
	if _, err := v.ValidateRequest(req); err != ErrHostMismatch {
		t.Fatalf("expected host mismatch, got %v", err)
	}
}

func mustDirectToken(t *testing.T, claims Claims, secret string) string {
	t.Helper()
	b, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	payload := base64.RawURLEncoding.EncodeToString(b)
	return DirectTokenPrefix + "." + payload + "." + signPayload(secret, payload)
}
