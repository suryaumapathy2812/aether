package plugins

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/suryaumapathy/core-ai/agent/internal/cron"
	"github.com/suryaumapathy/core-ai/agent/internal/db"
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
