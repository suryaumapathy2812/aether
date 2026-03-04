package proactive

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/ws"
)

// Engine is the proactive planning engine. It periodically asks the LLM
// to decide what background work the agent should perform, then creates
// agent tasks for each work item.
type Engine struct {
	store            *db.Store
	core             *llm.Core
	pushSender       *ws.PushSender
	wsHub            *ws.Hub
	assetsDir        string
	planIntervalSecs int
}

// Options configures a new Engine.
type Options struct {
	Store               *db.Store
	Core                *llm.Core
	PushSender          *ws.PushSender
	WSHub               *ws.Hub
	AssetsDir           string
	PlanIntervalSeconds int
}

// workItem represents a single proactive task planned by the LLM.
type workItem struct {
	Title    string   `json:"title"`
	Goal     string   `json:"goal"`
	Priority int      `json:"priority"`
	Tags     []string `json:"tags"`
	Notify   bool     `json:"notify"`
}

// New creates a new proactive Engine with the given options.
func New(opts Options) *Engine {
	if opts.PlanIntervalSeconds <= 0 {
		opts.PlanIntervalSeconds = 3600 // 1 hour
	}
	return &Engine{
		store:            opts.Store,
		core:             opts.Core,
		pushSender:       opts.PushSender,
		wsHub:            opts.WSHub,
		assetsDir:        opts.AssetsDir,
		planIntervalSecs: opts.PlanIntervalSeconds,
	}
}

// loadPrompt reads the PROACTIVE.md prompt file from the assets directory.
func (e *Engine) loadPrompt() string {
	if e.assetsDir == "" {
		return ""
	}
	promptPath := filepath.Join(e.assetsDir, "PROACTIVE.md")
	b, err := os.ReadFile(promptPath)
	if err != nil {
		return ""
	}
	return string(b)
}

// buildPlanningContext assembles the current state for the LLM planner.
func (e *Engine) buildPlanningContext(ctx context.Context) string {
	var sb strings.Builder

	// 1. Current time
	now := time.Now()
	sb.WriteString("### Current Time\n")
	sb.WriteString(now.Format("Monday, January 2, 2006 3:04 PM MST"))
	sb.WriteString("\n\n")

	// 2. Enabled plugins
	sb.WriteString("### Enabled Plugins\n")
	if e.store != nil {
		if plugins, err := e.store.ListPlugins(ctx); err == nil {
			found := false
			for _, p := range plugins {
				if p.Enabled {
					sb.WriteString("- ")
					sb.WriteString(p.Name)
					if p.DisplayName != "" {
						sb.WriteString(" (")
						sb.WriteString(p.DisplayName)
						sb.WriteString(")")
					}
					sb.WriteString("\n")
					found = true
				}
			}
			if !found {
				sb.WriteString("(none)\n")
			}
		} else {
			sb.WriteString("(unavailable)\n")
		}
	} else {
		sb.WriteString("(unavailable)\n")
	}
	sb.WriteString("\n")

	// 3. Recent proactive work (last 6 hours) — derived from agent tasks
	sb.WriteString("### Recent Proactive Work (last 6 hours)\n")
	if e.store != nil {
		cutoff := now.Add(-6 * time.Hour).UTC()
		tasks, err := e.store.ListAgentTasksByUser(ctx, "default", 30)
		if err == nil {
			found := false
			for _, t := range tasks {
				// Only include proactive tasks created in the last 6 hours
				if t.CreatedAt.Before(cutoff) {
					continue
				}
				if !strings.HasPrefix(t.SessionID, "proactive-") {
					continue
				}
				sb.WriteString("- [")
				sb.WriteString(string(t.Status))
				sb.WriteString("] ")
				sb.WriteString(t.Title)
				sb.WriteString(" (")
				sb.WriteString(t.CreatedAt.Format(time.RFC3339))
				sb.WriteString(")\n")
				found = true
			}
			if !found {
				sb.WriteString("(none)\n")
			}
		} else {
			sb.WriteString("(unavailable)\n")
		}
	} else {
		sb.WriteString("(unavailable)\n")
	}
	sb.WriteString("\n")

	// 4. Known entities (gracefully skip if table doesn't exist)
	sb.WriteString("### Known Entities\n")
	if e.store != nil {
		entities, err := e.store.ListEntities(ctx, "default", "", 20)
		if err == nil && len(entities) > 0 {
			for _, ent := range entities {
				sb.WriteString("- ")
				sb.WriteString(ent.EntityType)
				sb.WriteString(": ")
				sb.WriteString(ent.Name)
				if ent.Summary != "" {
					sb.WriteString(" — ")
					sb.WriteString(ent.Summary)
				}
				sb.WriteString("\n")
			}
		} else {
			sb.WriteString("(none)\n")
		}
	} else {
		sb.WriteString("(none)\n")
	}
	sb.WriteString("\n")

	// 5. User facts
	sb.WriteString("### User Facts\n")
	if e.store != nil {
		facts, err := e.store.GetMemoryFacts(ctx, "default")
		if err == nil && len(facts) > 0 {
			for _, f := range facts {
				sb.WriteString("- ")
				sb.WriteString(f)
				sb.WriteString("\n")
			}
		} else {
			sb.WriteString("(none)\n")
		}
	} else {
		sb.WriteString("(none)\n")
	}
	sb.WriteString("\n")

	// 6. User decisions
	sb.WriteString("### User Decisions & Preferences\n")
	if e.store != nil {
		decisions, err := e.store.ListDecisions(ctx, "default", "", true)
		if err == nil && len(decisions) > 0 {
			for _, d := range decisions {
				sb.WriteString("- [")
				sb.WriteString(d.Category)
				sb.WriteString("] ")
				sb.WriteString(d.Decision)
				sb.WriteString("\n")
			}
		} else {
			sb.WriteString("(none)\n")
		}
	} else {
		sb.WriteString("(none)\n")
	}

	return sb.String()
}

// RunPlanningCycle is the main planning function called by the cron scheduler.
// It asks the LLM to plan proactive work items and creates agent tasks for each.
func (e *Engine) RunPlanningCycle(ctx context.Context, job cron.Job) error {
	// Load the proactive prompt
	prompt := e.loadPrompt()
	if prompt == "" {
		return fmt.Errorf("proactive: PROACTIVE.md prompt not found in assets dir")
	}

	// Build planning context
	planningCtx := e.buildPlanningContext(ctx)

	// Assemble full prompt
	fullPrompt := prompt + "\n\n---\n\n## Current Context\n\n" + planningCtx

	// Call LLM
	env := llm.LLMRequestEnvelope{
		Kind:     "proactive_plan",
		Modality: "system",
		Messages: []map[string]any{{"role": "user", "content": fullPrompt}},
		Tools:    []map[string]any{},
		Policy:   map[string]any{"max_tokens": 800, "temperature": 0.2},
	}

	parts := []string{}
	runCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	for ev := range e.core.GenerateWithTools(runCtx, env) {
		if ev.EventType == llm.EventTextChunk {
			if t, ok := ev.Payload["text"].(string); ok {
				parts = append(parts, t)
			}
		}
	}
	content := strings.TrimSpace(strings.Join(parts, ""))

	// Parse response as work items
	items := parseWorkItems(content)
	if len(items) == 0 {
		log.Printf("proactive: planned 0 work items")
		return nil
	}

	// Cap at 5 items
	if len(items) > 5 {
		items = items[:5]
	}

	sessionID := "proactive-" + time.Now().Format("2006-01-02")

	for _, item := range items {
		// Clamp priority 1-100
		priority := item.Priority
		if priority < 1 {
			priority = 1
		}
		if priority > 100 {
			priority = 100
		}

		// Build goal with notification instruction if needed
		goal := item.Goal
		if item.Notify {
			goal += "\n\nIMPORTANT: When you have results, use the send_notification tool to deliver a summary to the user."
		}

		// Create agent task
		taskRec, err := e.store.CreateAgentTask(ctx, db.AgentTaskCreate{
			UserID:    "default",
			SessionID: sessionID,
			Title:     "Proactive: " + item.Title,
			Goal:      goal,
			Priority:  priority,
			MaxSteps:  10,
			Metadata:  map[string]any{"source": "proactive", "tags": item.Tags, "notify": item.Notify},
		})
		if err != nil {
			log.Printf("proactive: failed to create task %q: %v", item.Title, err)
			continue
		}

		// Record proactive event
		_, _ = e.store.RecordProactiveEvent(ctx, "default", taskRec.ID, "proactive", item.Title, "queued", map[string]any{
			"tags":   item.Tags,
			"notify": item.Notify,
		})
	}

	log.Printf("proactive: planned %d work items", len(items))
	return nil
}

// EnsureCronJobs schedules the recurring planning job if it doesn't already exist.
func (e *Engine) EnsureCronJobs(ctx context.Context) error {
	if e == nil || e.store == nil {
		return nil
	}
	scope := e.store.ScopeCronModule(CronModuleProactive)
	jobs, err := scope.List(ctx)
	if err != nil {
		return err
	}
	for _, j := range jobs {
		if j.JobType == CronJobTypePlan && (j.Status == db.CronStatusScheduled || j.Status == db.CronStatusRunning) {
			return nil // already scheduled
		}
	}
	interval := int64(e.planIntervalSecs)
	_, err = scope.ScheduleRecurring(ctx, CronJobTypePlan, map[string]any{}, time.Now().Add(30*time.Second), interval, 5)
	return err
}

// parseWorkItems extracts a JSON array of work items from the LLM response.
// Falls back to regex extraction if the response contains extra text.
func parseWorkItems(content string) []workItem {
	v := strings.TrimSpace(content)
	if v == "" {
		return nil
	}
	// Try direct parse first
	if !strings.HasPrefix(v, "[") {
		re := regexp.MustCompile(`\[[\s\S]*\]`)
		m := re.FindString(v)
		if m == "" {
			return nil
		}
		v = m
	}
	var items []workItem
	if err := json.Unmarshal([]byte(v), &items); err != nil {
		return nil
	}
	return items
}
