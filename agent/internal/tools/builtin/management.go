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

type ListIntegrationsTool struct{}
type ReadIntegrationManifestTool struct{}
type InstallIntegrationTool struct{}
type EnableIntegrationTool struct{}
type DisableIntegrationTool struct{}
type SetIntegrationConfigTool struct{}

type ListJobsTool struct{}
type ScheduleJobTool struct{}
type CancelJobTool struct{}
type PauseJobTool struct{}
type ResumeJobTool struct{}

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

func (t *ListIntegrationsTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_integrations", Description: "List discovered integrations and status.", StatusText: "Listing integrations..."}
}

func (t *ListIntegrationsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Integrations == nil {
		return tools.Fail("Integrations manager is unavailable", nil)
	}
	_, _ = call.Ctx.Integrations.Discover(ctx)
	items := call.Ctx.Integrations.List()
	if len(items) == 0 {
		return tools.Success("No integrations found.", map[string]any{"count": 0})
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

func (t *ReadIntegrationManifestTool) Definition() tools.Definition {
	return tools.Definition{Name: "read_integration_manifest", Description: "Read integration manifest as JSON.", StatusText: "Reading integration manifest...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Integration name", Required: true}}}
}

func (t *ReadIntegrationManifestTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Integrations == nil {
		return tools.Fail("Integrations manager is unavailable", nil)
	}
	_, _ = call.Ctx.Integrations.Discover(ctx)
	name, _ := call.Args["name"].(string)
	m, err := call.Ctx.Integrations.ReadManifest(name)
	if err != nil {
		return tools.Fail("Failed to read integration manifest: "+err.Error(), nil)
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return tools.Fail("Failed to encode manifest: "+err.Error(), nil)
	}
	return tools.Success(string(b), map[string]any{"name": name})
}

func (t *InstallIntegrationTool) Definition() tools.Definition {
	return tools.Definition{Name: "install_integration", Description: "Install an external integration from source owner/repo[@integration-path].", StatusText: "Installing integration...", Parameters: []tools.Param{{Name: "source", Type: "string", Description: "External source", Required: true}}}
}

func (t *InstallIntegrationTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Integrations == nil {
		return tools.Fail("Integrations manager is unavailable", nil)
	}
	source, _ := call.Args["source"].(string)
	res, err := call.Ctx.Integrations.InstallFromSource(ctx, source)
	if err != nil {
		return tools.Fail("Failed to install integration: "+err.Error(), nil)
	}
	_, _ = call.Ctx.Integrations.Discover(ctx)
	return tools.Success("Integration installed.", map[string]any{"name": res.Installed.Name, "source": source, "remote_url": res.RemoteURL})
}

func (t *EnableIntegrationTool) Definition() tools.Definition {
	return tools.Definition{Name: "enable_integration", Description: "Enable an integration in persistent state.", StatusText: "Enabling integration...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Integration name", Required: true}}}
}

func (t *EnableIntegrationTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	if err := call.Ctx.Store.SetPluginEnabled(ctx, name, true); err != nil {
		return tools.Fail("Failed to enable integration: "+err.Error(), nil)
	}
	return tools.Success("Integration enabled.", map[string]any{"name": name})
}

func (t *DisableIntegrationTool) Definition() tools.Definition {
	return tools.Definition{Name: "disable_integration", Description: "Disable an integration in persistent state.", StatusText: "Disabling integration...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Integration name", Required: true}}}
}

func (t *DisableIntegrationTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	if err := call.Ctx.Store.SetPluginEnabled(ctx, name, false); err != nil {
		return tools.Fail("Failed to disable integration: "+err.Error(), nil)
	}
	return tools.Success("Integration disabled.", map[string]any{"name": name})
}

func (t *SetIntegrationConfigTool) Definition() tools.Definition {
	return tools.Definition{Name: "set_integration_config", Description: "Set integration config map (string values).", StatusText: "Updating integration config...", Parameters: []tools.Param{{Name: "name", Type: "string", Description: "Integration name", Required: true}, {Name: "config", Type: "object", Description: "Configuration object", Required: true}}}
}

func (t *SetIntegrationConfigTool) Execute(ctx context.Context, call tools.Call) tools.Result {
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
		return tools.Fail("Failed to set integration config: "+err.Error(), nil)
	}
	return tools.Success("Integration config updated.", map[string]any{"name": name, "keys": sortedKeys(cfg)})
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
	_ tools.Tool = (*ListIntegrationsTool)(nil)
	_ tools.Tool = (*ReadIntegrationManifestTool)(nil)
	_ tools.Tool = (*InstallIntegrationTool)(nil)
	_ tools.Tool = (*EnableIntegrationTool)(nil)
	_ tools.Tool = (*DisableIntegrationTool)(nil)
	_ tools.Tool = (*SetIntegrationConfigTool)(nil)
	_ tools.Tool = (*ListJobsTool)(nil)
	_ tools.Tool = (*ScheduleJobTool)(nil)
	_ tools.Tool = (*CancelJobTool)(nil)
	_ tools.Tool = (*PauseJobTool)(nil)
	_ tools.Tool = (*ResumeJobTool)(nil)
)
