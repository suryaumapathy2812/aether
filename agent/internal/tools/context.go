package tools

import (
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
)

type ExecContext struct {
	WorkingDir   string
	Store        *db.Store
	Skills       *skills.Manager
	Plugins      *plugins.Manager
	PluginName   string
	PluginState  plugins.PluginState
	RuntimeHints map[string]any
}
