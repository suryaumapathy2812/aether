package builtin

import (
	"context"
	"fmt"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/reminders"
	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

type ScheduleReminderTool struct{}

func (t *ScheduleReminderTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "schedule_reminder",
		Description: "Schedule a one-shot reminder at a specific future time.",
		StatusText:  "Scheduling reminder...",
		Parameters: []tools.Param{
			{Name: "message", Type: "string", Description: "Reminder message", Required: true},
			{Name: "iso_datetime", Type: "string", Description: "Future datetime in RFC3339/ISO-8601", Required: true},
		},
	}
}

func (t *ScheduleReminderTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("Scheduler store is unavailable", nil)
	}
	message, _ := call.Args["message"].(string)
	raw, _ := call.Args["iso_datetime"].(string)
	runAt, err := parseISOTime(raw)
	if err != nil {
		return tools.Fail("Invalid datetime format: "+raw+". Use ISO-8601, e.g. 2026-02-21T15:30:00Z", nil)
	}
	if !runAt.After(time.Now().UTC()) {
		return tools.Fail("The requested time is in the past. Please provide a future time.", nil)
	}
	scope := call.Ctx.Store.ScopeCronModule(reminders.CronModuleReminders)
	job, err := scope.ScheduleOnce(ctx, reminders.CronJobTypeDeliver, map[string]any{"message": message}, runAt, 5)
	if err != nil {
		return tools.Fail("Failed to schedule reminder: "+err.Error(), nil)
	}
	return tools.Success(
		fmt.Sprintf("Reminder scheduled for %s.", runAt.Format("January 02, 2006 at 15:04 MST")),
		map[string]any{"job_id": job.ID, "run_at": runAt.Format(time.RFC3339), "module": reminders.CronModuleReminders},
	)
}

func parseISOTime(v string) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339, v); err == nil {
		return t.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("invalid datetime")
}

var _ tools.Tool = (*ScheduleReminderTool)(nil)
