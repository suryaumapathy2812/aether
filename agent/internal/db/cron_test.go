package db

import (
	"context"
	"testing"
	"time"
)

func TestCronScheduleAcquireAndCompleteOneShot(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	runAt := time.Now().UTC().Add(-time.Second)
	job, err := store.ScheduleCronJob(ctx, CronJobCreate{
		Module:  "reminders",
		JobType: "deliver_reminder",
		Payload: map[string]any{"message": "wake up"},
		RunAt:   runAt,
	})
	if err != nil {
		t.Fatalf("schedule: %v", err)
	}

	jobs, err := store.AcquireDueCronJobs(ctx, time.Now().UTC(), 10, 30*time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if len(jobs) != 1 || jobs[0].ID != job.ID {
		t.Fatalf("unexpected acquired jobs: %#v", jobs)
	}
	if jobs[0].AttemptCount != 1 {
		t.Fatalf("expected first attempt count=1, got %d", jobs[0].AttemptCount)
	}

	if err := store.MarkCronJobSuccess(ctx, job.ID, time.Now().UTC(), nil); err != nil {
		t.Fatalf("mark success: %v", err)
	}

	updated, err := store.GetCronJob(ctx, job.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if updated.Status != CronStatusDone || updated.Enabled {
		t.Fatalf("unexpected final state: %#v", updated)
	}
}

func TestCronRecurringRescheduleAndScopeIsolation(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	appScope := store.ScopeCronModule("plugins")
	otherScope := store.ScopeCronModule("reminders")
	runAt := time.Now().UTC().Add(-time.Second)
	job, err := appScope.ScheduleRecurring(ctx, "rotate_token", map[string]any{"plugin": "gmail"}, runAt, 60, 5)
	if err != nil {
		t.Fatalf("schedule recurring: %v", err)
	}

	acquired, err := store.AcquireDueCronJobs(ctx, time.Now().UTC(), 10, 30*time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if len(acquired) != 1 {
		t.Fatalf("expected one acquired job, got %d", len(acquired))
	}

	next := time.Now().UTC().Add(60 * time.Second)
	if err := store.MarkCronJobSuccess(ctx, job.ID, time.Now().UTC(), &next); err != nil {
		t.Fatalf("mark success recurring: %v", err)
	}

	updated, err := store.GetCronJob(ctx, job.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if updated.Status != CronStatusScheduled || !updated.Enabled {
		t.Fatalf("unexpected recurring state: %#v", updated)
	}
	if !updated.NextRunAt.Equal(next) {
		t.Fatalf("unexpected next_run_at: got=%s want=%s", updated.NextRunAt, next)
	}

	if err := otherScope.Cancel(ctx, job.ID); err != ErrNotFound {
		t.Fatalf("expected scoped not found, got %v", err)
	}
}
