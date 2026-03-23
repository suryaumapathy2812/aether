package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/auth"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/proxy"
)

func TestRewriteQueryUserID(t *testing.T) {
	q := url.Values{"limit": []string{"20"}, "user_id": []string{"wrong"}}
	out := proxy.RewriteQueryUserID("/agent/v1/memory/conversations", q, "user-1")
	if got := out.Get("user_id"); got != "user-1" {
		t.Fatalf("expected forced user_id user-1, got %q", got)
	}
	if got := out.Get("limit"); got != "20" {
		t.Fatalf("expected limit to be preserved, got %q", got)
	}
}

func TestRewriteBodyUserID(t *testing.T) {
	body := []byte(`{"user_id":"wrong","kind":"image"}`)
	out := proxy.RewriteBodyUserID("/agent/v1/media/upload/init", "application/json", body, "user-2")
	if !strings.Contains(string(out), `"user_id":"user-2"`) {
		t.Fatalf("expected user_id rewrite, got %s", string(out))
	}
}

func TestExtractTokensPriority(t *testing.T) {
	req := httptest.NewRequest("GET", "http://x.local/path?token=q-token", nil)
	req.Header.Set("Authorization", "Bearer h-token")
	req.AddCookie(&http.Cookie{Name: "better-auth.session_token", Value: "c-token.sig"})

	tokens := auth.ExtractTokens(req)
	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d", len(tokens))
	}
	if tokens[0] != "h-token" || tokens[1] != "c-token" || tokens[2] != "q-token" {
		t.Fatalf("unexpected token order/content: %+v", tokens)
	}
}

func TestHTTPStreamRewritesBodyUserID(t *testing.T) {
	t.Parallel()

	var gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b := new(bytes.Buffer)
		_, _ = b.ReadFrom(r.Body)
		gotBody = b.String()
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	u, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	host := u.Hostname()
	port := 80
	if p := u.Port(); p != "" {
		var convErr error
		port, convErr = strconv.Atoi(p)
		if convErr != nil {
			t.Fatal(convErr)
		}
	}

	req := httptest.NewRequest(http.MethodPost, "http://orch.local/agent/v1/media/upload/init", strings.NewReader(`{"user_id":"wrong"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	client := &http.Client{}

	ok := proxy.HTTPStream(client, w, req, host, port, "/agent/v1/media/upload/init", "user-correct", true)
	if !ok {
		t.Fatal("expected proxy stream success")
	}
	if !strings.Contains(gotBody, `"user_id":"user-correct"`) {
		t.Fatalf("expected rewritten user_id, got %s", gotBody)
	}
}
