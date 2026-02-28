package tools

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

type Registry struct {
	mu         sync.RWMutex
	tools      map[string]Tool
	toolPlugin map[string]string
}

func NewRegistry() *Registry {
	return &Registry{
		tools:      map[string]Tool{},
		toolPlugin: map[string]string{},
	}
}

func (r *Registry) Register(tool Tool, pluginName string) error {
	def := tool.Definition()
	if def.Name == "" {
		return fmt.Errorf("tool name is required")
	}
	r.mu.Lock()
	r.tools[def.Name] = tool
	if pluginName != "" {
		r.toolPlugin[def.Name] = pluginName
	}
	r.mu.Unlock()
	return nil
}

func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

func (r *Registry) PluginForTool(name string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.toolPlugin[name]
}

func (r *Registry) ToolNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.tools))
	for name := range r.tools {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func (r *Registry) Definitions() []Definition {
	names := r.ToolNames()
	out := make([]Definition, 0, len(names))
	for _, name := range names {
		if t, ok := r.Get(name); ok {
			out = append(out, t.Definition())
		}
	}
	return out
}

func (r *Registry) Dispatch(ctx context.Context, name string, args map[string]any, execCtx ExecContext) Result {
	t, ok := r.Get(name)
	if !ok {
		return Fail("Unknown tool: "+name, nil)
	}
	return safeExecute(ctx, t, Call{ID: "", Args: args, Ctx: execCtx})
}

func safeExecute(ctx context.Context, tool Tool, call Call) (res Result) {
	def := tool.Definition()
	cleaned, err := ValidateArgs(def.Parameters, call.Args)
	if err != nil {
		return Fail("Invalid arguments: "+err.Error(), nil)
	}
	call.Args = cleaned
	defer func() {
		if recovered := recover(); recovered != nil {
			res = Fail("Tool error: panic during execution", map[string]any{"tool": def.Name})
		}
	}()
	return tool.Execute(ctx, call)
}

func (r *Registry) OpenAISchemas() []map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]map[string]any, 0, len(r.tools))
	for _, tool := range r.tools {
		def := tool.Definition()
		props := map[string]any{}
		required := []string{}
		for _, p := range def.Parameters {
			entry := map[string]any{"type": p.Type, "description": p.Description}
			if len(p.Enum) > 0 {
				entry["enum"] = p.Enum
			}
			if p.Items != nil {
				entry["items"] = p.Items
			}
			props[p.Name] = entry
			if p.Required {
				required = append(required, p.Name)
			}
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        def.Name,
				"description": def.Description,
				"parameters": map[string]any{
					"type":       "object",
					"properties": props,
					"required":   required,
				},
			},
		})
	}
	return out
}
