package cron

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

type HandlerFunc func(ctx context.Context, job Job) error

type Job struct {
	ID           string
	Module       string
	JobType      string
	Payload      map[string]any
	RunAt        time.Time
	IntervalS    *int64
	AttemptCount int
	MaxAttempts  int
	NextRunAt    time.Time
}

type SchedulerOptions struct {
	PollInterval time.Duration
	LeaseFor     time.Duration
	BatchSize    int
	JobTimeout   time.Duration
}

type Scheduler struct {
	store    *db.Store
	options  SchedulerOptions
	handlers map[string]HandlerFunc

	mu     sync.RWMutex
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewScheduler(store *db.Store, opts SchedulerOptions) *Scheduler {
	if opts.PollInterval <= 0 {
		opts.PollInterval = time.Second
	}
	if opts.LeaseFor <= 0 {
		opts.LeaseFor = 30 * time.Second
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = 20
	}
	if opts.JobTimeout <= 0 {
		opts.JobTimeout = 60 * time.Second
	}
	return &Scheduler{
		store:    store,
		options:  opts,
		handlers: map[string]HandlerFunc{},
	}
}

func (s *Scheduler) RegisterHandler(module, jobType string, handler HandlerFunc) {
	key := handlerKey(module, jobType)
	s.mu.Lock()
	s.handlers[key] = handler
	s.mu.Unlock()
}

func (s *Scheduler) Start(ctx context.Context) error {
	if s.store == nil {
		return fmt.Errorf("scheduler store is required")
	}
	s.mu.Lock()
	if s.cancel != nil {
		s.mu.Unlock()
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	s.mu.Unlock()

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(s.options.PollInterval)
		defer ticker.Stop()
		_ = s.RunOnce(runCtx)
		for {
			select {
			case <-runCtx.Done():
				return
			case <-ticker.C:
				_ = s.RunOnce(runCtx)
			}
		}
	}()

	return nil
}

func (s *Scheduler) Stop(ctx context.Context) error {
	s.mu.Lock()
	cancel := s.cancel
	s.cancel = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}

	ch := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(ch)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ch:
		return nil
	}
}

func (s *Scheduler) RunOnce(ctx context.Context) error {
	if s.store == nil {
		return fmt.Errorf("scheduler store is required")
	}
	now := time.Now().UTC()
	jobs, err := s.store.AcquireDueCronJobs(ctx, now, s.options.BatchSize, s.options.LeaseFor)
	if err != nil {
		return err
	}
	for _, rec := range jobs {
		s.executeJob(ctx, rec)
	}
	return nil
}

func (s *Scheduler) executeJob(ctx context.Context, rec db.CronJobRecord) {
	now := time.Now().UTC()
	job := Job{
		ID:           rec.ID,
		Module:       rec.Module,
		JobType:      rec.JobType,
		RunAt:        rec.RunAt,
		IntervalS:    rec.IntervalS,
		AttemptCount: rec.AttemptCount,
		MaxAttempts:  rec.MaxAttempts,
		NextRunAt:    rec.NextRunAt,
	}
	if err := json.Unmarshal([]byte(rec.PayloadJSON), &job.Payload); err != nil {
		s.failJob(ctx, rec, now, fmt.Errorf("invalid payload: %w", err))
		return
	}

	handler, ok := s.getHandler(rec.Module, rec.JobType)
	if !ok {
		s.failJob(ctx, rec, now, fmt.Errorf("no handler registered for %s/%s", rec.Module, rec.JobType))
		return
	}

	jobCtx, cancel := context.WithTimeout(ctx, s.options.JobTimeout)
	err := handler(jobCtx, job)
	cancel()
	if err != nil {
		s.failJob(ctx, rec, now, err)
		return
	}

	if rec.IntervalS != nil && *rec.IntervalS > 0 {
		next := rec.NextRunAt
		interval := time.Duration(*rec.IntervalS) * time.Second
		for !next.After(now) {
			next = next.Add(interval)
		}
		_ = s.store.MarkCronJobSuccess(ctx, rec.ID, now, &next)
		return
	}
	_ = s.store.MarkCronJobSuccess(ctx, rec.ID, now, nil)
}

func (s *Scheduler) failJob(ctx context.Context, rec db.CronJobRecord, now time.Time, runErr error) {
	attempt := rec.AttemptCount
	terminal := attempt >= rec.MaxAttempts
	if terminal {
		_ = s.store.MarkCronJobFailure(ctx, rec.ID, now, runErr.Error(), nil, true)
		return
	}
	delay := retryBackoff(attempt)
	next := now.Add(delay)
	_ = s.store.MarkCronJobFailure(ctx, rec.ID, now, runErr.Error(), &next, false)
}

func retryBackoff(attempt int) time.Duration {
	if attempt <= 1 {
		return 30 * time.Second
	}
	backoff := 30 * time.Second
	for i := 1; i < attempt; i++ {
		backoff *= 2
		if backoff >= 15*time.Minute {
			return 15 * time.Minute
		}
	}
	if backoff > 15*time.Minute {
		return 15 * time.Minute
	}
	return backoff
}

func (s *Scheduler) getHandler(module, jobType string) (HandlerFunc, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	h, ok := s.handlers[handlerKey(module, jobType)]
	return h, ok
}

func handlerKey(module, jobType string) string {
	return module + "::" + jobType
}
