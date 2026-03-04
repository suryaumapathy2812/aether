package proactive

import (
	"context"
	"fmt"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
)

const (
	// CronModuleProactive is the cron module name for proactive planning jobs.
	CronModuleProactive = "proactive"
	// CronJobTypePlan is the job type for the recurring planning cycle.
	CronJobTypePlan = "plan"
)

// RegisterCronHandlers registers the proactive planning cron handler with the scheduler.
func RegisterCronHandlers(scheduler *cron.Scheduler, engine *Engine) {
	if scheduler == nil || engine == nil {
		return
	}
	scheduler.RegisterHandler(CronModuleProactive, CronJobTypePlan, func(ctx context.Context, job cron.Job) error {
		if engine == nil {
			return fmt.Errorf("proactive engine not initialized")
		}
		return engine.RunPlanningCycle(ctx, job)
	})
}
