package integrations

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

const (
	CronModulePlugins     = "plugins"
	CronJobTypeRotate     = "rotate_token"
	CronJobTypeRenewWatch = "renew_watch"
)

type TokenRotator func(ctx context.Context, state PluginState, payload map[string]any) error
type WatchRenewer func(ctx context.Context, state PluginState, payload map[string]any) error

type CronRegistry struct {
	mu            sync.RWMutex
	tokenRotators map[string]TokenRotator
	watchRenewers map[string]WatchRenewer
}

func NewCronRegistry() *CronRegistry {
	return &CronRegistry{
		tokenRotators: map[string]TokenRotator{},
		watchRenewers: map[string]WatchRenewer{},
	}
}

func (r *CronRegistry) RegisterTokenRotator(pluginName string, fn TokenRotator) {
	r.mu.Lock()
	r.tokenRotators[pluginName] = fn
	r.mu.Unlock()
}

func (r *CronRegistry) RegisterWatchRenewer(pluginName string, fn WatchRenewer) {
	r.mu.Lock()
	r.watchRenewers[pluginName] = fn
	r.mu.Unlock()
}

func (r *CronRegistry) tokenRotator(pluginName string) (TokenRotator, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	fn, ok := r.tokenRotators[pluginName]
	return fn, ok
}

func (r *CronRegistry) watchRenewer(pluginName string) (WatchRenewer, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	fn, ok := r.watchRenewers[pluginName]
	return fn, ok
}

func (r *CronRegistry) TokenRotatorCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tokenRotators)
}

func (r *CronRegistry) WatchRenewerCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.watchRenewers)
}

// RefreshTokenIfExpired checks whether an integration's access token has expired
// (or will expire within 2 minutes) and, if so, runs the registered token rotator.
// This ensures credentials are fresh before injection into tool execution.
// Returns nil if no rotation was needed, the token has no expiry, or no rotator
// is registered for this integration.
func (r *CronRegistry) RefreshTokenIfExpired(ctx context.Context, state PluginState, pluginName string) error {
	if r == nil {
		return nil
	}
	fn, ok := r.tokenRotator(pluginName)
	if !ok {
		return nil // no rotator registered; nothing to do (e.g. api_key integrations)
	}
	cfg, err := state.Config(ctx)
	if err != nil {
		return err
	}
	expiresAtRaw := cfg["expires_at"]
	if expiresAtRaw == "" {
		return nil // no expiry tracked — skip
	}
	var expiresAt int64
	if _, err := fmt.Sscanf(expiresAtRaw, "%d", &expiresAt); err != nil {
		return nil // unparseable — skip
	}
	// Refresh if expired or will expire within 2 minutes.
	if time.Now().UTC().Unix() < expiresAt-120 {
		return nil // still fresh
	}
	log.Printf("integrations: access token for %q expired or expiring soon, refreshing", pluginName)
	return fn(ctx, state, nil)
}

func RegisterCronHandlers(scheduler *cron.Scheduler, store *db.Store, registry *CronRegistry) {
	if scheduler == nil || store == nil || registry == nil {
		return
	}

	scheduler.RegisterHandler(CronModulePlugins, CronJobTypeRotate, func(ctx context.Context, job cron.Job) error {
		pluginName, err := pluginNameFromPayload(job.Payload)
		if err != nil {
			return err
		}
		fn, ok := registry.tokenRotator(pluginName)
		if !ok {
			return fmt.Errorf("no token rotator registered for plugin %q", pluginName)
		}
		state := NewPluginState(store, pluginName)
		enabled, err := state.Enabled(ctx)
		if err != nil {
			return err
		}
		if !enabled {
			log.Printf("cron/plugins: skip rotate_token for disabled plugin=%s", pluginName)
			return nil
		}
		return fn(ctx, state, job.Payload)
	})

	scheduler.RegisterHandler(CronModulePlugins, CronJobTypeRenewWatch, func(ctx context.Context, job cron.Job) error {
		pluginName, err := pluginNameFromPayload(job.Payload)
		if err != nil {
			return err
		}
		fn, ok := registry.watchRenewer(pluginName)
		if !ok {
			return fmt.Errorf("no watch renewer registered for plugin %q", pluginName)
		}
		state := NewPluginState(store, pluginName)
		enabled, err := state.Enabled(ctx)
		if err != nil {
			return err
		}
		if !enabled {
			log.Printf("cron/plugins: skip renew_watch for disabled plugin=%s", pluginName)
			return nil
		}
		return fn(ctx, state, job.Payload)
	})
}

func pluginNameFromPayload(payload map[string]any) (string, error) {
	if payload == nil {
		return "", fmt.Errorf("missing payload")
	}
	if v, ok := payload["plugin"]; ok {
		if s, ok := v.(string); ok && s != "" {
			return s, nil
		}
	}
	if v, ok := payload["plugin_name"]; ok {
		if s, ok := v.(string); ok && s != "" {
			return s, nil
		}
	}
	return "", fmt.Errorf("payload requires plugin or plugin_name")
}
