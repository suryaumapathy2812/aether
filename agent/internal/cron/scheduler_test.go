package cron

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

func TestSchedulerRunsOneShotJob(t *testing.T) {
	ctx := context.Background()
	store := openStore(t)
	defer store.Close()

	called := false
	s := NewScheduler(store, SchedulerOptions{BatchSize: 5})
	s.RegisterHandler("reminders", "deliver", func(ctx context.Context, job Job) error {
		called = true
		if job.Payload["message"] != "hello" {
			t.Fatalf("unexpected payload: %#v", job.Payload)
		}
		return nil
	})

	_, err := store.ScheduleCronJob(ctx, db.CronJobCreate{
		Module:  "reminders",
		JobType: "deliver",
		Payload: map[string]any{"message": "hello"},
		RunAt:   time.Now().UTC().Add(-time.Second),
	})
	if err != nil {
		t.Fatalf("schedule: %v", err)
	}

	if err := s.RunOnce(ctx); err != nil {
		t.Fatalf("run once: %v", err)
	}
	if !called {
		t.Fatalf("expected handler call")
	}

	jobs, err := store.ListCronJobsByModule(ctx, "reminders")
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 1 || jobs[0].Status != db.CronStatusDone {
		t.Fatalf("unexpected job state: %#v", jobs)
	}
}

func TestSchedulerRetriesAndFailsAfterMaxAttempts(t *testing.T) {
	ctx := context.Background()
	store := openStore(t)
	defer store.Close()

	s := NewScheduler(store, SchedulerOptions{BatchSize: 10})
	s.RegisterHandler("plugins", "rotate", func(ctx context.Context, job Job) error {
		return errors.New("refresh failed")
	})

	_, err := store.ScheduleCronJob(ctx, db.CronJobCreate{
		Module:      "plugins",
		JobType:     "rotate",
		Payload:     map[string]any{"plugin": "gmail"},
		RunAt:       time.Now().UTC().Add(-time.Second),
		MaxAttempts: 1,
	})
	if err != nil {
		t.Fatalf("schedule: %v", err)
	}

	if err := s.RunOnce(ctx); err != nil {
		t.Fatalf("run once: %v", err)
	}

	jobs, err := store.ListCronJobsByModule(ctx, "plugins")
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 1 {
		t.Fatalf("expected one job")
	}
	if jobs[0].Status != db.CronStatusFailed || jobs[0].Enabled {
		t.Fatalf("expected failed disabled job: %#v", jobs[0])
	}
}

func openStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path, "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	return store
}
