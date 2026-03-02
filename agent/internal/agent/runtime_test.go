package agent

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type staticProvider struct{}

func (p *staticProvider) Name() string { return "static" }

func (p *staticProvider) StreamWithTools(ctx context.Context, opts providers.GenerateOptions) (<-chan providers.LLMStreamEvent, error) {
	_ = ctx
	_ = opts
	out := make(chan providers.LLMStreamEvent, 2)
	go func() {
		defer close(out)
		out <- providers.LLMStreamEvent{Type: providers.EventToken, Content: "completed delegated task"}
		out <- providers.LLMStreamEvent{Type: providers.EventDone, FinishReason: "stop"}
	}()
	return out, nil
}

func TestRuntimeCompletesQueuedTask(t *testing.T) {
	ctx := context.Background()
	store, err := db.Open(filepath.Join(t.TempDir(), "state.db"), "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	registry := tools.NewRegistry()
	orchestrator := tools.NewOrchestrator(registry, tools.ExecContext{Store: store})
	core := llm.NewCore(&staticProvider{}, orchestrator)
	builder := llm.NewContextBuilder(registry, nil, nil, store, llm.ContextBuilderConfig{})
	runtime := NewRuntime(RuntimeOptions{Store: store, Core: core, Builder: builder, Workers: 1, PollEvery: 20 * time.Millisecond})

	task, err := store.CreateAgentTask(ctx, db.AgentTaskCreate{UserID: "u1", Title: "Plan", Goal: "Do the thing", MaxSteps: 5})
	if err != nil {
		t.Fatalf("create task: %v", err)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	if err := runtime.Start(runCtx); err != nil {
		t.Fatalf("start runtime: %v", err)
	}
	defer runtime.Stop(context.Background())

	deadline := time.Now().Add(4 * time.Second)
	for time.Now().Before(deadline) {
		updated, err := store.GetAgentTask(ctx, task.ID)
		if err != nil {
			t.Fatalf("get task: %v", err)
		}
		if updated.Status == db.AgentTaskCompleted {
			if updated.ResultSummary == "" {
				t.Fatalf("expected result summary")
			}
			return
		}
		time.Sleep(40 * time.Millisecond)
	}
	t.Fatalf("task did not complete in time")
}
