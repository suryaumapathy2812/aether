package integrations

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
)

type WebhookRequest struct {
	Plugin   string
	UserID   string
	DeviceID string
	Body     []byte
	Header   http.Header
}

type WebhookTask struct {
	SessionID string
	Title     string
	Goal      string
	Metadata  map[string]any
}

type WebhookHandler interface {
	BuildTask(ctx context.Context, req WebhookRequest) (*WebhookTask, error)
}

type WebhookRegistry struct {
	mu       sync.RWMutex
	handlers map[string]WebhookHandler
}

func NewWebhookRegistry() *WebhookRegistry {
	return &WebhookRegistry{handlers: map[string]WebhookHandler{}}
}

func (r *WebhookRegistry) Register(plugin string, handler WebhookHandler) {
	if r == nil || handler == nil {
		return
	}
	key := normalizeName(plugin)
	if key == "" {
		return
	}
	r.mu.Lock()
	r.handlers[key] = handler
	r.mu.Unlock()
}

func (r *WebhookRegistry) Handler(plugin string) (WebhookHandler, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	h, ok := r.handlers[normalizeName(plugin)]
	return h, ok
}

func (r *WebhookRegistry) BuildTask(ctx context.Context, req WebhookRequest) (*WebhookTask, error) {
	h, ok := r.Handler(req.Plugin)
	if !ok {
		return nil, fmt.Errorf("no webhook handler registered for plugin %q", req.Plugin)
	}
	if strings.TrimSpace(req.UserID) == "" {
		return nil, fmt.Errorf("missing user id")
	}
	return h.BuildTask(ctx, req)
}

var defaultWebhookRegistry = NewWebhookRegistry()

func DefaultWebhookRegistry() *WebhookRegistry {
	return defaultWebhookRegistry
}
