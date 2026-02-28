package plugins

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

func pluginConfig(ctx context.Context, call tools.Call) (map[string]string, error) {
	if call.Ctx.PluginState == nil {
		return nil, fmt.Errorf("plugin state is unavailable")
	}
	cfg, err := call.Ctx.PluginState.Config(ctx)
	if err != nil {
		return nil, err
	}
	if cfg == nil {
		cfg = map[string]string{}
	}
	return cfg, nil
}

func requireString(cfg map[string]string, key string) (string, error) {
	v := strings.TrimSpace(cfg[key])
	if v == "" {
		return "", fmt.Errorf("missing required config key: %s", key)
	}
	return v, nil
}

func maybeDecryptFromState(ctx context.Context, call tools.Call, value string) (string, error) {
	v := strings.TrimSpace(value)
	if !strings.HasPrefix(v, "enc:v1:") {
		return v, nil
	}
	if call.Ctx.PluginState == nil {
		return "", fmt.Errorf("plugin state is unavailable")
	}
	return call.Ctx.PluginState.DecryptString(v)
}

func requireToken(ctx context.Context, call tools.Call, cfg map[string]string) (string, error) {
	raw, err := requireString(cfg, "access_token")
	if err != nil {
		return "", err
	}
	return maybeDecryptFromState(ctx, call, raw)
}

func base64URL(s string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(s))
}
