package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCheckHealthURLAcceptsOK(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	mgr := &Manager{httpClient: srv.Client()}
	if err := mgr.checkHealthURL(context.Background(), srv.URL); err != nil {
		t.Fatalf("expected health check to pass, got %v", err)
	}
}

func TestCheckHealthURLRejectsNonOK(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	mgr := &Manager{httpClient: srv.Client()}
	if err := mgr.checkHealthURL(context.Background(), srv.URL); err == nil {
		t.Fatal("expected health check to fail")
	}
}

func TestDirectHealthURLUsesPrefixAndDomain(t *testing.T) {
	t.Parallel()

	got := directHealthURL("abc12345", "agents.example.com")
	want := "https://abc12345.agents.example.com/health"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}
