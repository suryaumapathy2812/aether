package server

import (
	"net/url"
	"testing"

	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/agent"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/auth"
)

func TestBuildAgentWSURL_InjectsIdentityAndPreservesQuery(t *testing.T) {
	t.Parallel()

	target := agent.Target{Host: "127.0.0.1", Port: 9000}
	incoming := url.Values{
		"foo":     []string{"bar"},
		"user_id": []string{"wrong"},
		"token":   []string{"old"},
	}
	id := auth.Identity{UserID: "user-123", Token: "tok-abc"}

	raw := buildAgentWSURL(target, "/ws/conversation", incoming, id)
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	if got := u.Scheme; got != "ws" {
		t.Fatalf("expected ws scheme, got %q", got)
	}
	if got := u.Host; got != "127.0.0.1:9000" {
		t.Fatalf("expected host 127.0.0.1:9000, got %q", got)
	}
	if got := u.Path; got != "/ws/conversation" {
		t.Fatalf("expected /ws/conversation path, got %q", got)
	}
	q := u.Query()
	if got := q.Get("foo"); got != "bar" {
		t.Fatalf("expected foo=bar, got %q", got)
	}
	if got := q.Get("user_id"); got != "user-123" {
		t.Fatalf("expected user_id override, got %q", got)
	}
	if got := q.Get("token"); got != "tok-abc" {
		t.Fatalf("expected token override, got %q", got)
	}
}

func TestBuildAgentWSURL_LeavesTokenUnsetWhenMissing(t *testing.T) {
	t.Parallel()

	raw := buildAgentWSURL(
		agent.Target{Host: "agent.local", Port: 8080},
		"/ws/notifications",
		nil,
		auth.Identity{UserID: "user-1"},
	)
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	q := u.Query()
	if got := q.Get("user_id"); got != "user-1" {
		t.Fatalf("expected user_id, got %q", got)
	}
	if _, ok := q["token"]; ok {
		t.Fatal("expected token query to be omitted")
	}
}
