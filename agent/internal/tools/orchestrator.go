package tools

import (
	"context"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
)

type Runtime struct {
	WorkingDir string
	Store      any
	Skills     any
	Plugins    any
}

type Orchestrator struct {
	registry *Registry
	runtime  ExecContext
}

func NewOrchestrator(registry *Registry, runtime ExecContext) *Orchestrator {
	return &Orchestrator{registry: registry, runtime: runtime}
}

func (o *Orchestrator) ToolNames() []string {
	if o == nil || o.registry == nil {
		return nil
	}
	return o.registry.ToolNames()
}

func (o *Orchestrator) DefinitionForTool(name string) *Definition {
	if o == nil || o.registry == nil {
		return nil
	}
	t, ok := o.registry.Get(name)
	if !ok {
		return nil
	}
	def := t.Definition()
	return &def
}

func (o *Orchestrator) Execute(ctx context.Context, toolName string, args map[string]any, callID string) Result {
	execCtx := o.runtime
	if pluginName := o.registry.PluginForTool(toolName); pluginName != "" && execCtx.Store != nil {
		rec, err := execCtx.Store.GetPlugin(ctx, pluginName)
		if err != nil {
			return Fail("Plugin is not configured: "+pluginName, map[string]any{"call_id": callID, "plugin": pluginName})
		}
		if !rec.Enabled {
			return Fail("Plugin is disabled: "+pluginName+". Enable it before using this tool.", map[string]any{"call_id": callID, "plugin": pluginName})
		}
		if execCtx.Plugins != nil {
			requiredKeys, err := execCtx.Plugins.RequiredConfigKeys(pluginName)
			if err == nil && len(requiredKeys) > 0 {
				missing := make([]string, 0)
				for _, key := range requiredKeys {
					if strings.TrimSpace(rec.Config[key]) == "" {
						missing = append(missing, key)
					}
				}
				if len(missing) > 0 {
					return Fail("Plugin config missing required keys: "+strings.Join(missing, ", "), map[string]any{"call_id": callID, "plugin": pluginName, "missing_keys": missing})
				}
			}
		}
		execCtx.PluginName = pluginName
		execCtx.PluginState = plugins.NewPluginState(execCtx.Store, pluginName)
	}

	tool, ok := o.registry.Get(toolName)
	if !ok {
		return Fail("Unknown tool: "+toolName, map[string]any{"call_id": callID})
	}

	result := safeExecute(ctx, tool, Call{ID: callID, Args: args, Ctx: execCtx})
	if result.Metadata == nil {
		result.Metadata = map[string]any{}
	}
	if callID != "" {
		result.Metadata["call_id"] = callID
	}
	return result
}
