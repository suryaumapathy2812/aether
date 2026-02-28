package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type SearchSkillTool struct{}
type ReadSkillTool struct{}
type CreateSkillTool struct{}
type InstallSkillTool struct{}
type RemoveSkillTool struct{}

type ListPluginsTool struct{}
type ReadPluginManifestTool struct{}
type InstallPluginTool struct{}
type EnablePluginTool struct{}
type DisablePluginTool struct{}
type SetPluginConfigTool struct{}

type ListJobsTool struct{}
type ScheduleJobTool struct{}
type CancelJobTool struct{}
type PauseJobTool struct{}
type ResumeJobTool struct{}

type DelegateTaskTool struct{}
type GetTaskStatusTool struct{}
type ListTasksTool struct{}
type CancelTaskTool struct{}
type GetTaskResultTool struct{}
type RequestHumanApprovalTool struct{}
type ResumeTaskTool struct{}
type ListPendingApprovalsTool struct{}
type ApproveTaskTool struct{}
type RejectTaskTool struct{}

func (t *SearchSkillTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_skill", Description: "Search installed skills by keyword.", StatusText: "Searching skills...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Search query", Required: true}}}
}

func (t *SearchSkillTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Skills == nil {
		return tools.Fail("Skills manager is unavailable", nil)
	}
	_, _ = call.Ctx.Skills.Discover(ctx)
	query, _ := call.Args["query"].(string)
	items := call.Ctx.Skills.Search(query)
	if len(items) == 0 {
		return tools.Success("No matching skills found.", map[string]any{"count": 0})
	}
	lines := make([]string, 0, len(items))
	for _, s := range items {
		lines = append(lines, fmt.Sprintf("- %s [%s] %s", s.Name, s.Source, s.Description))
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"count": len(items)})
}

func (t *ReadSkillTool) Definition() tools.Definition {
	return tools.Definition{Name: "read_skill", Description: "Read full content of a skill.", StatusText: "Reading skill...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Skill name", Required: true}}}
}

func (t *ReadSkillTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Skills == nil {
		return tools.Fail("Skills manager is unavailable", nil)
	}
	_, _ = call.Ctx.Skills.Discover(ctx)
	name, _ := call.Args["name"].(string)
	body, err := call.Ctx.Skills.Read(name)
	if err != nil {
		return tools.Fail("Failed to read skill: "+err.Error(), nil)
	}
	return tools.Success(body, map[string]any{"name": name})
}

func (t *CreateSkillTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "create_skill",
		Description: "Create a new user skill.",
		StatusText:  "Creating skill...",
		Parameters: []tools.Param{
			{Name: "name", Type: "string", Description: "Skill name", Required: true},
			{Name: "description", Type: "string", Description: "Skill description", Required: true},
			{Name: "content", Type: "string", Description: "Markdown skill body", Required: true},
		},
	}
}

func (t *CreateSkillTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Skills == nil {
		return tools.Fail("Skills manager is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	desc, _ := call.Args["description"].(string)
	content, _ := call.Args["content"].(string)
	created, err := call.Ctx.Skills.Create(name, desc, content)
	if err != nil {
		return tools.Fail("Failed to create skill: "+err.Error(), nil)
	}
	return tools.Success("Skill created.", map[string]any{"name": created.Name, "path": created.Location})
}

func (t *InstallSkillTool) Definition() tools.Definition {
	return tools.Definition{Name: "install_skill", Description: "Install an external skill from source owner/repo[@skill].", StatusText: "Installing skill...", Parameters: []tools.Param{{Name: "source", Type: "string", Description: "External source", Required: true}}}
}

func (t *InstallSkillTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Skills == nil {
		return tools.Fail("Skills manager is unavailable", nil)
	}
	source, _ := call.Args["source"].(string)
	res, err := call.Ctx.Skills.InstallFromSource(ctx, source)
	if err != nil {
		return tools.Fail("Failed to install skill: "+err.Error(), nil)
	}
	return tools.Success("Skill installed.", map[string]any{"name": res.Installed.Name, "source": source, "remote_url": res.RemoteURL})
}

func (t *RemoveSkillTool) Definition() tools.Definition {
	return tools.Definition{Name: "remove_skill", Description: "Remove an installed skill (non-builtin).", StatusText: "Removing skill...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Skill name", Required: true}}}
}

func (t *RemoveSkillTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Skills == nil {
		return tools.Fail("Skills manager is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	if err := call.Ctx.Skills.Remove(name); err != nil {
		return tools.Fail("Failed to remove skill: "+err.Error(), nil)
	}
	return tools.Success("Skill removed.", map[string]any{"name": name})
}

func (t *ListPluginsTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_plugins", Description: "List discovered plugins and status.", StatusText: "Listing plugins..."}
}

func (t *ListPluginsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Plugins == nil {
		return tools.Fail("Plugins manager is unavailable", nil)
	}
	_, _ = call.Ctx.Plugins.Discover(ctx)
	items := call.Ctx.Plugins.List()
	if len(items) == 0 {
		return tools.Success("No plugins found.", map[string]any{"count": 0})
	}
	lines := make([]string, 0, len(items))
	for _, p := range items {
		enabled := false
		if call.Ctx.Store != nil {
			rec, err := call.Ctx.Store.GetPlugin(ctx, p.Name)
			if err == nil {
				enabled = rec.Enabled
			}
		}
		lines = append(lines, fmt.Sprintf("- %s [%s] enabled=%t %s", p.Name, p.Source, enabled, p.Description))
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"count": len(items)})
}

func (t *ReadPluginManifestTool) Definition() tools.Definition {
	return tools.Definition{Name: "read_plugin_manifest", Description: "Read plugin manifest as JSON.", StatusText: "Reading plugin manifest...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Plugin name", Required: true}}}
}

func (t *ReadPluginManifestTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Plugins == nil {
		return tools.Fail("Plugins manager is unavailable", nil)
	}
	_, _ = call.Ctx.Plugins.Discover(ctx)
	name, _ := call.Args["name"].(string)
	m, err := call.Ctx.Plugins.ReadManifest(name)
	if err != nil {
		return tools.Fail("Failed to read plugin manifest: "+err.Error(), nil)
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return tools.Fail("Failed to encode manifest: "+err.Error(), nil)
	}
	return tools.Success(string(b), map[string]any{"name": name})
}

func (t *InstallPluginTool) Definition() tools.Definition {
	return tools.Definition{Name: "install_plugin", Description: "Install an external plugin from source owner/repo[@plugin-path].", StatusText: "Installing plugin...", Parameters: []tools.Param{{Name: "source", Type: "string", Description: "External source", Required: true}}}
}

func (t *InstallPluginTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Plugins == nil {
		return tools.Fail("Plugins manager is unavailable", nil)
	}
	source, _ := call.Args["source"].(string)
	res, err := call.Ctx.Plugins.InstallFromSource(ctx, source)
	if err != nil {
		return tools.Fail("Failed to install plugin: "+err.Error(), nil)
	}
	_, _ = call.Ctx.Plugins.Discover(ctx)
	return tools.Success("Plugin installed.", map[string]any{"name": res.Installed.Name, "source": source, "remote_url": res.RemoteURL})
}

func (t *EnablePluginTool) Definition() tools.Definition {
	return tools.Definition{Name: "enable_plugin", Description: "Enable a plugin in persistent state.", StatusText: "Enabling plugin...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Plugin name", Required: true}}}
}

func (t *EnablePluginTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	if err := call.Ctx.Store.SetPluginEnabled(ctx, name, true); err != nil {
		return tools.Fail("Failed to enable plugin: "+err.Error(), nil)
	}
	return tools.Success("Plugin enabled.", map[string]any{"name": name})
}

func (t *DisablePluginTool) Definition() tools.Definition {
	return tools.Definition{Name: "disable_plugin", Description: "Disable a plugin in persistent state.", StatusText: "Disabling plugin...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Plugin name", Required: true}}}
}

func (t *DisablePluginTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	if err := call.Ctx.Store.SetPluginEnabled(ctx, name, false); err != nil {
		return tools.Fail("Failed to disable plugin: "+err.Error(), nil)
	}
	return tools.Success("Plugin disabled.", map[string]any{"name": name})
}

func (t *SetPluginConfigTool) Definition() tools.Definition {
	return tools.Definition{Name: "set_plugin_config", Description: "Set plugin config map (string values).", StatusText: "Updating plugin config...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Plugin name", Required: true}, {Name: "config", Type: "object", Description: "Configuration object", Required: true}}}
}

func (t *SetPluginConfigTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	rawCfg, _ := call.Args["config"].(map[string]any)
	cfg := map[string]string{}
	for k, v := range rawCfg {
		cfg[k] = fmt.Sprintf("%v", v)
	}
	if err := call.Ctx.Store.SetPluginConfig(ctx, name, cfg); err != nil {
		return tools.Fail("Failed to set plugin config: "+err.Error(), nil)
	}
	return tools.Success("Plugin config updated.", map[string]any{"name": name, "keys": sortedKeys(cfg)})
}

func (t *ListJobsTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_jobs", Description: "List scheduled cron jobs, optionally filtered by module.", StatusText: "Listing jobs...", Parameters: []tools.Param{{Name: "module", Type: "string", Description: "Optional module filter", Required: false, Default: ""}}}
}

func (t *ListJobsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	module, _ := call.Args["module"].(string)
	var jobs []db.CronJobRecord
	var err error
	if strings.TrimSpace(module) == "" {
		jobs, err = call.Ctx.Store.ListCronJobs(ctx)
	} else {
		jobs, err = call.Ctx.Store.ListCronJobsByModule(ctx, module)
	}
	if err != nil {
		return tools.Fail("Failed to list jobs: "+err.Error(), nil)
	}
	if len(jobs) == 0 {
		return tools.Success("No cron jobs found.", map[string]any{"count": 0})
	}
	lines := make([]string, 0, len(jobs))
	for _, j := range jobs {
		interval := "one-shot"
		if j.IntervalS != nil {
			interval = fmt.Sprintf("every %ds", *j.IntervalS)
		}
		lines = append(lines, fmt.Sprintf("- %s [%s/%s] status=%s next=%s %s", j.ID, j.Module, j.JobType, j.Status, j.NextRunAt.Format(time.RFC3339), interval))
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"count": len(jobs)})
}

func (t *ScheduleJobTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "schedule_job",
		Description: "Schedule a one-shot or recurring cron job.",
		StatusText:  "Scheduling job...",
		Parameters: []tools.Param{
			{Name: "module", Type: "string", Description: "Job module", Required: true},
			{Name: "job_type", Type: "string", Description: "Job type", Required: true},
			{Name: "iso_datetime", Type: "string", Description: "Run at datetime RFC3339", Required: true},
			{Name: "payload", Type: "object", Description: "Job payload", Required: false, Default: map[string]any{}},
			{Name: "interval_s", Type: "integer", Description: "Optional recurring interval seconds", Required: false, Default: 0},
			{Name: "max_attempts", Type: "integer", Description: "Max attempts", Required: false, Default: 5},
		},
	}
}

func (t *ScheduleJobTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	module, _ := call.Args["module"].(string)
	jobType, _ := call.Args["job_type"].(string)
	iso, _ := call.Args["iso_datetime"].(string)
	payload, _ := call.Args["payload"].(map[string]any)
	intervalS, _ := toolsAsInt(call.Args["interval_s"])
	maxAttempts, _ := toolsAsInt(call.Args["max_attempts"])
	runAt, err := time.Parse(time.RFC3339, iso)
	if err != nil {
		return tools.Fail("Invalid datetime format; use RFC3339", nil)
	}
	create := db.CronJobCreate{Module: module, JobType: jobType, Payload: payload, RunAt: runAt.UTC(), MaxAttempts: maxAttempts}
	if intervalS > 0 {
		v := int64(intervalS)
		create.IntervalS = &v
	}
	job, err := call.Ctx.Store.ScheduleCronJob(ctx, create)
	if err != nil {
		return tools.Fail("Failed to schedule job: "+err.Error(), nil)
	}
	return tools.Success("Job scheduled.", map[string]any{"job_id": job.ID, "module": job.Module, "job_type": job.JobType, "next_run_at": job.NextRunAt.Format(time.RFC3339)})
}

func (t *CancelJobTool) Definition() tools.Definition {
	return tools.Definition{Name: "cancel_job", Description: "Cancel a cron job.", StatusText: "Cancelling job...", Parameters: []tools.Param{{Name: "job_id", Type: "string", Description: "Job ID", Required: true}}}
}

func (t *CancelJobTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	id, _ := call.Args["job_id"].(string)
	if err := call.Ctx.Store.CancelCronJob(ctx, id); err != nil {
		return tools.Fail("Failed to cancel job: "+err.Error(), nil)
	}
	return tools.Success("Job cancelled.", map[string]any{"job_id": id})
}

func (t *PauseJobTool) Definition() tools.Definition {
	return tools.Definition{Name: "pause_job", Description: "Pause a cron job.", StatusText: "Pausing job...", Parameters: []tools.Param{{Name: "job_id", Type: "string", Description: "Job ID", Required: true}}}
}

func (t *PauseJobTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	id, _ := call.Args["job_id"].(string)
	if err := call.Ctx.Store.PauseCronJob(ctx, id); err != nil {
		return tools.Fail("Failed to pause job: "+err.Error(), nil)
	}
	return tools.Success("Job paused.", map[string]any{"job_id": id})
}

func (t *ResumeJobTool) Definition() tools.Definition {
	return tools.Definition{Name: "resume_job", Description: "Resume a paused cron job.", StatusText: "Resuming job...", Parameters: []tools.Param{{Name: "job_id", Type: "string", Description: "Job ID", Required: true}}}
}

func (t *ResumeJobTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	id, _ := call.Args["job_id"].(string)
	if err := call.Ctx.Store.ResumeCronJob(ctx, id); err != nil {
		return tools.Fail("Failed to resume job: "+err.Error(), nil)
	}
	return tools.Success("Job resumed.", map[string]any{"job_id": id})
}

func (t *DelegateTaskTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "delegate_task",
		Description: "Delegate a multi-step task to background agent runtime.",
		StatusText:  "Delegating task...",
		Parameters: []tools.Param{
			{Name: "title", Type: "string", Description: "Short task title", Required: true},
			{Name: "goal", Type: "string", Description: "Detailed task objective", Required: true},
			{Name: "user_id", Type: "string", Description: "Task owner user id", Required: false, Default: "default"},
			{Name: "session_id", Type: "string", Description: "Optional session id", Required: false, Default: ""},
			{Name: "priority", Type: "integer", Description: "Priority score (higher first)", Required: false, Default: 0},
			{Name: "max_steps", Type: "integer", Description: "Maximum autonomous loop steps", Required: false, Default: 10},
			{Name: "constraints", Type: "object", Description: "Optional constraints metadata", Required: false, Default: map[string]any{}},
		},
	}
}

func (t *DelegateTaskTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	// Prevent sub-agents from recursively delegating — stops infinite task cascade.
	if rtCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(rtCtx.TaskID) != "" {
		return tools.Fail("delegate_task cannot be called from inside a delegated task. Complete the work directly instead of re-delegating.", nil)
	}
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	title, _ := call.Args["title"].(string)
	goal, _ := call.Args["goal"].(string)

	// Prefer the authenticated user ID from context over the LLM-provided argument.
	// The LLM often hallucinates user_id values (e.g. "user" instead of the real ID).
	userID := ""
	if rtCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(rtCtx.UserID) != "" {
		userID = rtCtx.UserID
	}
	if strings.TrimSpace(userID) == "" {
		userID, _ = call.Args["user_id"].(string)
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	sessionID, _ := call.Args["session_id"].(string)
	priority, _ := toolsAsInt(call.Args["priority"])
	maxSteps, _ := toolsAsInt(call.Args["max_steps"])
	if maxSteps > 0 && maxSteps < 3 {
		maxSteps = 3
	}
	constraints, _ := call.Args["constraints"].(map[string]any)
	if constraints == nil {
		constraints = map[string]any{}
	}
	goal = strings.TrimSpace(goal)
	if goal != "" {
		goal += "\n\nDelegation execution rules:\n- If you need missing user input before continuing, call request_human_approval and wait instead of finishing with a plain-text question.\n- If user provides natural-language time (e.g. 'tomorrow 8pm IST' or 'in 10 minutes'), convert it to RFC3339/ISO first.\n- schedule_reminder requires a future RFC3339 datetime; use world_time when needed to validate timezone and future time before calling schedule_reminder."
	}
	task, err := call.Ctx.Store.CreateAgentTask(ctx, db.AgentTaskCreate{
		UserID:    userID,
		SessionID: sessionID,
		Title:     title,
		Goal:      goal,
		Priority:  priority,
		MaxSteps:  maxSteps,
		Metadata: map[string]any{
			"constraints": constraints,
			"source":      "delegate_task",
		},
	})
	if err != nil {
		return tools.Fail("Failed to delegate task: "+err.Error(), nil)
	}
	return tools.Success("Task delegated to background agent runtime.", map[string]any{"task_id": task.ID, "status": task.Status, "user_id": userID, "title": task.Title})
}

func (t *GetTaskStatusTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_task_status", Description: "Get delegated task status and progress.", StatusText: "Checking task status...", Parameters: []tools.Param{{Name: "task_id", Type: "string", Description: "Task id", Required: true}}}
}

func (t *GetTaskStatusTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskID, _ := call.Args["task_id"].(string)
	task, err := call.Ctx.Store.GetAgentTask(ctx, taskID)
	if err != nil {
		return tools.Fail("Failed to fetch task: "+err.Error(), nil)
	}
	events, _ := call.Ctx.Store.ListAgentTaskEvents(ctx, taskID, 200)
	last := map[string]any{}
	if len(events) > 0 {
		_ = json.Unmarshal([]byte(events[len(events)-1].PayloadJSON), &last)
	}
	out := map[string]any{
		"task_id":          task.ID,
		"status":           task.Status,
		"step_count":       task.StepCount,
		"max_steps":        task.MaxSteps,
		"cancel_requested": task.CancelRequested,
		"last_error":       task.LastError,
		"result_summary":   task.ResultSummary,
		"last_event":       last,
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return tools.Success(string(b), out)
}

func (t *ListTasksTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_tasks", Description: "List delegated tasks for a user.", StatusText: "Listing tasks...", Parameters: []tools.Param{{Name: "user_id", Type: "string", Description: "User id", Required: false, Default: "default"}, {Name: "limit", Type: "integer", Description: "Max results", Required: false, Default: 20}}}
}

func (t *ListTasksTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	userID, _ := call.Args["user_id"].(string)
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	limit, _ := toolsAsInt(call.Args["limit"])
	if limit <= 0 {
		limit = 20
	}
	items, err := call.Ctx.Store.ListAgentTasksByUser(ctx, userID, limit)
	if err != nil {
		return tools.Fail("Failed to list tasks: "+err.Error(), nil)
	}
	if len(items) == 0 {
		return tools.Success("No delegated tasks found.", map[string]any{"count": 0, "user_id": userID})
	}
	lines := make([]string, 0, len(items))
	for _, it := range items {
		lines = append(lines, fmt.Sprintf("- %s status=%s steps=%d/%d title=%s", it.ID, it.Status, it.StepCount, it.MaxSteps, it.Title))
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"count": len(items), "user_id": userID})
}

func (t *CancelTaskTool) Definition() tools.Definition {
	return tools.Definition{Name: "cancel_task", Description: "Request cancellation for a delegated task.", StatusText: "Cancelling task...", Parameters: []tools.Param{{Name: "task_id", Type: "string", Description: "Task id", Required: true}}}
}

func (t *CancelTaskTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskID, _ := call.Args["task_id"].(string)
	if err := call.Ctx.Store.RequestCancelAgentTask(ctx, taskID); err != nil {
		return tools.Fail("Failed to cancel task: "+err.Error(), nil)
	}
	return tools.Success("Task cancellation requested.", map[string]any{"task_id": taskID, "status": "cancel_requested"})
}

func (t *GetTaskResultTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_task_result", Description: "Get final result for a delegated task.", StatusText: "Fetching task result...", Parameters: []tools.Param{{Name: "task_id", Type: "string", Description: "Task id", Required: true}}}
}

func (t *GetTaskResultTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskID, _ := call.Args["task_id"].(string)
	task, err := call.Ctx.Store.GetAgentTask(ctx, taskID)
	if err != nil {
		return tools.Fail("Failed to fetch task: "+err.Error(), nil)
	}
	if task.Status != db.AgentTaskCompleted {
		return tools.Success("Task is not complete yet.", map[string]any{"task_id": task.ID, "status": task.Status})
	}
	out := map[string]any{"task_id": task.ID, "status": task.Status, "result_summary": task.ResultSummary, "result_json": task.ResultJSON}
	b, _ := json.MarshalIndent(out, "", "  ")
	return tools.Success(string(b), out)
}

func (t *RequestHumanApprovalTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "request_human_approval",
		Description: "Pause delegated task and request human confirmation before continuing.",
		StatusText:  "Requesting human approval...",
		Parameters: []tools.Param{
			{Name: "question", Type: "string", Description: "Question for the human", Required: true},
			{Name: "reason", Type: "string", Description: "Why approval is needed", Required: false, Default: ""},
			{Name: "options", Type: "array", Description: "Optional answer options", Required: false, Default: []any{}, Items: map[string]any{"type": "string"}},
			{Name: "suggested_default", Type: "string", Description: "Optional recommended option", Required: false, Default: ""},
		},
	}
}

func (t *RequestHumanApprovalTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx)
	if !ok || strings.TrimSpace(taskCtx.TaskID) == "" || strings.TrimSpace(taskCtx.LockToken) == "" {
		return tools.Fail("request_human_approval can only be used inside delegated task runtime", nil)
	}
	question, _ := call.Args["question"].(string)
	reason, _ := call.Args["reason"].(string)
	options := []string{}
	if raw, ok := call.Args["options"].([]any); ok {
		for _, v := range raw {
			options = append(options, fmt.Sprintf("%v", v))
		}
	}
	suggested, _ := call.Args["suggested_default"].(string)
	payload := map[string]any{
		"task_id":              taskCtx.TaskID,
		"question":             question,
		"reason":               reason,
		"options":              options,
		"suggested_default":    suggested,
		"requested_at":         time.Now().UTC().Format(time.RFC3339),
		"requested_by_runtime": true,
	}
	if err := call.Ctx.Store.SetAgentTaskWaitingInput(ctx, taskCtx.TaskID, taskCtx.LockToken, question, payload); err != nil {
		return tools.Fail("Failed to set task waiting for approval: "+err.Error(), nil)
	}
	return tools.Success("Paused and waiting for human approval: "+question, map[string]any{
		"await_human": true,
		"task_id":     taskCtx.TaskID,
		"question":    question,
		"options":     options,
	})
}

func (t *ResumeTaskTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "resume_task",
		Description: "Resume a delegated task waiting for human input.",
		StatusText:  "Resuming task...",
		Parameters: []tools.Param{
			{Name: "task_id", Type: "string", Description: "Task id", Required: true},
			{Name: "user_id", Type: "string", Description: "User id owner", Required: false, Default: "default"},
			{Name: "message", Type: "string", Description: "Human approval/instruction message", Required: false, Default: "Human approved. Continue."},
		},
	}
}

func (t *ResumeTaskTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskID, _ := call.Args["task_id"].(string)
	userID, _ := call.Args["user_id"].(string)
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	message, _ := call.Args["message"].(string)
	task, err := call.Ctx.Store.ResumeAgentTask(ctx, taskID, userID, message)
	if err != nil {
		return tools.Fail("Failed to resume task: "+err.Error(), nil)
	}
	return tools.Success("Task resumed and queued for continued execution.", map[string]any{"task_id": task.ID, "status": task.Status})
}

func (t *ListPendingApprovalsTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "list_pending_approvals",
		Description: "List delegated tasks waiting for human approval.",
		StatusText:  "Checking pending approvals...",
		Parameters: []tools.Param{
			{Name: "user_id", Type: "string", Description: "User id", Required: false, Default: "default"},
			{Name: "limit", Type: "integer", Description: "Max results", Required: false, Default: 20},
		},
	}
}

func (t *ListPendingApprovalsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	userID, _ := call.Args["user_id"].(string)
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	limit, _ := toolsAsInt(call.Args["limit"])
	if limit <= 0 {
		limit = 20
	}
	tasks, err := call.Ctx.Store.ListAgentTasksByUserWithStatus(ctx, userID, string(db.AgentTaskWaitingInput), limit)
	if err != nil {
		return tools.Fail("Failed to list pending approvals: "+err.Error(), nil)
	}
	if len(tasks) == 0 {
		return tools.Success("No pending approvals.", map[string]any{"count": 0, "user_id": userID})
	}
	lines := make([]string, 0, len(tasks))
	for _, task := range tasks {
		question := strings.TrimSpace(task.ResultSummary)
		if question == "" {
			question = "Awaiting approval"
		}
		lines = append(lines, fmt.Sprintf("- %s title=%s question=%s", task.ID, task.Title, question))
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"count": len(tasks), "user_id": userID})
}

func (t *ApproveTaskTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "approve_task",
		Description: "Approve a waiting delegated task and provide standardized approval payload.",
		StatusText:  "Approving task...",
		Parameters: []tools.Param{
			{Name: "task_id", Type: "string", Description: "Task id", Required: true},
			{Name: "user_id", Type: "string", Description: "User id", Required: false, Default: "default"},
			{Name: "decision", Type: "string", Description: "Decision label", Required: false, Default: "approved"},
			{Name: "reason", Type: "string", Description: "Reasoning for approval", Required: false, Default: "Approved"},
			{Name: "instructions", Type: "string", Description: "Additional instructions for the task", Required: false, Default: "Proceed with the plan."},
		},
	}
}

func (t *ApproveTaskTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskID, _ := call.Args["task_id"].(string)
	userID, _ := call.Args["user_id"].(string)
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	decision, _ := call.Args["decision"].(string)
	if strings.TrimSpace(decision) == "" {
		decision = "approved"
	}
	reason, _ := call.Args["reason"].(string)
	instructions, _ := call.Args["instructions"].(string)
	message := strings.TrimSpace(instructions)
	if message == "" {
		message = strings.TrimSpace(reason)
	}
	if message == "" {
		message = "Approved. Continue with best effort."
	}
	task, err := call.Ctx.Store.ResumeAgentTask(ctx, taskID, userID, message)
	if err != nil {
		return tools.Fail("Failed to approve task: "+err.Error(), nil)
	}
	_ = call.Ctx.Store.AppendAgentTaskEvent(ctx, taskID, "decision", map[string]any{"decision": decision, "reason": reason, "instructions": instructions, "approved": true})
	return tools.Success("Task approved and resumed.", map[string]any{"task_id": task.ID, "status": task.Status, "decision": decision})
}

func (t *RejectTaskTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "reject_task",
		Description: "Reject a waiting delegated task with standardized rejection payload.",
		StatusText:  "Rejecting task...",
		Parameters: []tools.Param{
			{Name: "task_id", Type: "string", Description: "Task id", Required: true},
			{Name: "user_id", Type: "string", Description: "User id", Required: false, Default: "default"},
			{Name: "reason", Type: "string", Description: "Reason for rejection", Required: true},
			{Name: "next_action", Type: "string", Description: "Suggested next action", Required: false, Default: "Stop and wait for new instructions."},
		},
	}
}

func (t *RejectTaskTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	taskID, _ := call.Args["task_id"].(string)
	userID, _ := call.Args["user_id"].(string)
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	reason, _ := call.Args["reason"].(string)
	nextAction, _ := call.Args["next_action"].(string)
	rejectMsg := fmt.Sprintf("Rejected by human. Reason: %s. Next action: %s", strings.TrimSpace(reason), strings.TrimSpace(nextAction))
	task, err := call.Ctx.Store.RejectAgentTask(ctx, taskID, userID, rejectMsg)
	if err != nil {
		return tools.Fail("Failed to reject task: "+err.Error(), nil)
	}
	return tools.Success("Task rejected and cancelled.", map[string]any{"task_id": task.ID, "status": task.Status})
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

var (
	_ tools.Tool = (*SearchSkillTool)(nil)
	_ tools.Tool = (*ReadSkillTool)(nil)
	_ tools.Tool = (*CreateSkillTool)(nil)
	_ tools.Tool = (*InstallSkillTool)(nil)
	_ tools.Tool = (*RemoveSkillTool)(nil)
	_ tools.Tool = (*ListPluginsTool)(nil)
	_ tools.Tool = (*ReadPluginManifestTool)(nil)
	_ tools.Tool = (*InstallPluginTool)(nil)
	_ tools.Tool = (*EnablePluginTool)(nil)
	_ tools.Tool = (*DisablePluginTool)(nil)
	_ tools.Tool = (*SetPluginConfigTool)(nil)
	_ tools.Tool = (*ListJobsTool)(nil)
	_ tools.Tool = (*ScheduleJobTool)(nil)
	_ tools.Tool = (*CancelJobTool)(nil)
	_ tools.Tool = (*PauseJobTool)(nil)
	_ tools.Tool = (*ResumeJobTool)(nil)
	_ tools.Tool = (*DelegateTaskTool)(nil)
	_ tools.Tool = (*GetTaskStatusTool)(nil)
	_ tools.Tool = (*ListTasksTool)(nil)
	_ tools.Tool = (*CancelTaskTool)(nil)
	_ tools.Tool = (*GetTaskResultTool)(nil)
	_ tools.Tool = (*RequestHumanApprovalTool)(nil)
	_ tools.Tool = (*ResumeTaskTool)(nil)
	_ tools.Tool = (*ListPendingApprovalsTool)(nil)
	_ tools.Tool = (*ApproveTaskTool)(nil)
	_ tools.Tool = (*RejectTaskTool)(nil)
)
