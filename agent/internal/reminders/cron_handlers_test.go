package reminders

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

func TestReminderCronHandler(t *testing.T) {
	ctx := context.Background()
	store := openReminderStore(t)
	defer store.Close()

	s := cron.NewScheduler(store, cron.SchedulerOptions{BatchSize: 10})
	reg := NewRegistry()
	called := false
	reg.Register(func(ctx context.Context, payload map[string]any) error {
		called = true
		if payload["message"] != "ping" {
			t.Fatalf("unexpected payload: %#v", payload)
		}
		return nil
	})
	RegisterCronHandlers(s, reg)

	module := store.ScopeCronModule(CronModuleReminders)
	if _, err := module.ScheduleOnce(ctx, CronJobTypeDeliver, map[string]any{"message": "ping"}, time.Now().UTC().Add(-time.Second), 2); err != nil {
		t.Fatalf("schedule: %v", err)
	}

	if err := s.RunOnce(ctx); err != nil {
		t.Fatalf("run once: %v", err)
	}
	if !called {
		t.Fatalf("expected delivery handler to be called")
	}
}

func openReminderStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return store
}
