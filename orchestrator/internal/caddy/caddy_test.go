package caddy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

func TestAddRouteCreatesWhenIDMissing(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var requests []string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = r.Body.Close()
		mu.Lock()
		requests = append(requests, r.Method+" "+r.URL.Path+" "+string(body))
		mu.Unlock()

		switch {
		case r.Method == http.MethodPatch && r.URL.Path == "/id/aether-direct-demo":
			http.Error(w, `unknown object ID 'aether-direct-demo'`, http.StatusNotFound)
		case r.Method == http.MethodPut && r.URL.Path == dynamicRoutesConfigPath:
			w.WriteHeader(http.StatusOK)
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	rm := NewRouteManager(srv.URL, "agents.example.com")
	if err := rm.AddRoute("demo", "127.0.0.1", 8000); err != nil {
		t.Fatalf("add route: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}
	if requests[0][:6] != "PATCH " {
		t.Fatalf("expected PATCH first, got %q", requests[0])
	}
	if requests[1][:4] != "PUT " {
		t.Fatalf("expected PUT second, got %q", requests[1])
	}
}

func TestAddRouteUpdatesExistingByID(t *testing.T) {
	t.Parallel()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method != http.MethodPatch || r.URL.Path != "/id/aether-direct-demo" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	rm := NewRouteManager(srv.URL, "agents.example.com")
	if err := rm.AddRoute("demo", "127.0.0.1", 8000); err != nil {
		t.Fatalf("add route: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 request, got %d", calls)
	}
}
