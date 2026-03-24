package tools

import "context"

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
