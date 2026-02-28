package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestRewriteQueryUserID(t *testing.T) {
	q := url.Values{"limit": []string{"20"}, "user_id": []string{"wrong"}}
	out := rewriteQueryUserID("/api/memory/conversations", q, "user-1")
	if got := out.Get("user_id"); got != "user-1" {
		t.Fatalf("expected forced user_id user-1, got %q", got)
	}
	if got := out.Get("limit"); got != "20" {
		t.Fatalf("expected limit to be preserved, got %q", got)
	}
}

func TestRewriteBodyUserID(t *testing.T) {
	body := []byte(`{"user_id":"wrong","kind":"image"}`)
	out := rewriteBodyUserID("/v1/media/upload/init", "application/json", body, "user-2")
	if !strings.Contains(string(out), `"user_id":"user-2"`) {
		t.Fatalf("expected user_id rewrite, got %s", string(out))
	}
}

func TestExtractTokensPriority(t *testing.T) {
	req := httptest.NewRequest("GET", "http://x.local/path?token=q-token", nil)
	req.Header.Set("Authorization", "Bearer h-token")
	req.AddCookie(&http.Cookie{Name: "better-auth.session_token", Value: "c-token.sig"})

	tokens := extractTokens(req)
	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d", len(tokens))
	}
	if tokens[0] != "h-token" || tokens[1] != "c-token" || tokens[2] != "q-token" {
		t.Fatalf("unexpected token order/content: %+v", tokens)
	}
}

func TestV1ProxyRewritesUserIDBody(t *testing.T) {
	t.Parallel()

	var gotBody map[string]any
	agent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/media/upload/init" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agent.Close()

	s := &server{
		cfg:        config{LocalAgentURL: agent.URL},
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	body := []byte(`{"user_id":"wrong-user","kind":"image","size":10}`)
	req := httptest.NewRequest(http.MethodPost, "http://orch.local/v1/media/upload/init", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleV1Proxy(w, req, identity{UserID: "user-correct"})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if gotBody["user_id"] != "user-correct" {
		t.Fatalf("expected rewritten user_id user-correct, got %#v", gotBody["user_id"])
	}
}

func TestMemoryProxyRewritesQueryUserID(t *testing.T) {
	t.Parallel()

	var gotUserID string
	agent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/memory/conversations" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		gotUserID = r.URL.Query().Get("user_id")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"conversations":[]}`))
	}))
	defer agent.Close()

	s := &server{
		cfg:        config{LocalAgentURL: agent.URL},
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	req := httptest.NewRequest(http.MethodGet, "http://orch.local/api/memory/conversations?user_id=wrong&limit=10", nil)
	w := httptest.NewRecorder()

	s.handleMemoryProxy(w, req, identity{UserID: "user-correct"})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if gotUserID != "user-correct" {
		t.Fatalf("expected rewritten query user_id user-correct, got %q", gotUserID)
	}
}
