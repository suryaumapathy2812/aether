package plugins

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/cron"
	"github.com/suryaumapathy/core-ai/agent/internal/db"
)

func TestPluginCronRotateHandler(t *testing.T) {
	ctx := context.Background()
	store := openPluginCronStore(t)
	defer store.Close()

	if err := store.UpsertPlugin(ctx, db.PluginRecord{Name: "weather", DisplayName: "Weather", Enabled: true}); err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}
	if err := store.SetPluginEnabled(ctx, "weather", true); err != nil {
		t.Fatalf("enable plugin: %v", err)
	}

	s := cron.NewScheduler(store, cron.SchedulerOptions{BatchSize: 10})
	reg := NewCronRegistry()
	called := false
	reg.RegisterTokenRotator("weather", func(ctx context.Context, state PluginState, payload map[string]any) error {
		called = true
		enc, err := state.EncryptString("abc")
		if err != nil {
			return err
		}
		cfg, err := state.Config(ctx)
		if err != nil {
			return err
		}
		cfg["token"] = enc
		return state.SetConfig(ctx, cfg)
	})
	RegisterCronHandlers(s, store, reg)

	module := store.ScopeCronModule(CronModulePlugins)
	if _, err := module.ScheduleOnce(ctx, CronJobTypeRotate, map[string]any{"plugin": "weather"}, time.Now().UTC().Add(-time.Second), 3); err != nil {
		t.Fatalf("schedule: %v", err)
	}

	if err := s.RunOnce(ctx); err != nil {
		t.Fatalf("run once: %v", err)
	}
	if !called {
		t.Fatalf("expected rotator to run")
	}

	jobs, err := module.List(ctx)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 1 || jobs[0].Status != db.CronStatusDone {
		t.Fatalf("unexpected job status: %#v", jobs)
	}
}

func openPluginCronStore(t *testing.T) *db.Store {
	t.Helper()
	t.Setenv("AGENT_STATE_KEY", "12345678901234567890123456789012")
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return store
}
