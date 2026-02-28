package plugins

import (
	"context"

	"github.com/suryaumapathy/core-ai/agent/internal/db"
)

// PluginState gives a plugin module scoped access to only its own row.
type PluginState interface {
	Ensure(ctx context.Context, base db.PluginRecord) error
	Enabled(ctx context.Context) (bool, error)
	SetEnabled(ctx context.Context, enabled bool) error
	Config(ctx context.Context) (map[string]string, error)
	SetConfig(ctx context.Context, cfg map[string]string) error
	EncryptString(plaintext string) (string, error)
	DecryptString(ciphertext string) (string, error)
}

type pluginState struct {
	scope *db.PluginScope
}

func NewPluginState(store *db.Store, pluginName string) PluginState {
	return &pluginState{scope: store.ScopePlugin(pluginName)}
}

func (p *pluginState) Ensure(ctx context.Context, base db.PluginRecord) error {
	return p.scope.Ensure(ctx, base)
}

func (p *pluginState) Enabled(ctx context.Context) (bool, error) {
	return p.scope.IsEnabled(ctx)
}

func (p *pluginState) SetEnabled(ctx context.Context, enabled bool) error {
	return p.scope.SetEnabled(ctx, enabled)
}

func (p *pluginState) Config(ctx context.Context) (map[string]string, error) {
	return p.scope.GetConfig(ctx)
}

func (p *pluginState) SetConfig(ctx context.Context, cfg map[string]string) error {
	return p.scope.SetConfig(ctx, cfg)
}

func (p *pluginState) EncryptString(plaintext string) (string, error) {
	return p.scope.EncryptString(plaintext)
}

func (p *pluginState) DecryptString(ciphertext string) (string, error) {
	return p.scope.DecryptString(ciphertext)
}
