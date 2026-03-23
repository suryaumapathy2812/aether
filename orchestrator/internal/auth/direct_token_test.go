package auth

import (
	"testing"
	"time"
)

func TestDirectTokenRoundTrip(t *testing.T) {
	t.Parallel()
	a := &Authenticator{directSecret: "test-secret"}
	token, claims, err := a.MintDirectToken("user-1", "abc12345", "agent-1", time.Hour)
	if err != nil {
		t.Fatalf("mint token: %v", err)
	}
	if token == "" {
		t.Fatal("expected token")
	}
	got, ok := a.VerifyDirectToken(token)
	if !ok {
		t.Fatal("expected token verification success")
	}
	if got.UserID != claims.UserID || got.Prefix != claims.Prefix || got.AgentID != claims.AgentID {
		t.Fatalf("unexpected claims: %#v", got)
	}
}

func TestDirectTokenRejectsExpiry(t *testing.T) {
	t.Parallel()
	a := &Authenticator{directSecret: "test-secret"}
	token, _, err := a.MintDirectToken("user-1", "abc12345", "agent-1", time.Second)
	if err != nil {
		t.Fatalf("mint token: %v", err)
	}
	time.Sleep(2100 * time.Millisecond)
	if _, ok := a.VerifyDirectToken(token); ok {
		t.Fatal("expected expired token to fail")
	}
}
