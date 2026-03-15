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
)
