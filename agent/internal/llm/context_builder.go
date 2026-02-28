package llm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type ContextBuilder struct {
	registry     *tools.Registry
	skills       *skills.Manager
	plugins      *plugins.Manager
	store        *db.Store
	systemPrompt string
}

func NewContextBuilder(registry *tools.Registry, skillsManager *skills.Manager, pluginsManager *plugins.Manager, store *db.Store) *ContextBuilder {
	basePrompt := strings.TrimSpace(os.Getenv("AGENT_SYSTEM_PROMPT"))
	if basePrompt == "" {
		basePrompt = strings.TrimSpace(loadPromptFromFile())
	}
	if basePrompt == "" {
		basePrompt = "You are Aether, a helpful assistant. Use tools when needed."
	}
	basePrompt += "\n\nExecution policy:\n- For reminders or alarms, use schedule_reminder exactly once and then respond with confirmation.\n- For long multi-step work, or when user explicitly asks to test/try delegation, use delegate_task and return the task_id.\n- Avoid repeatedly calling the same tool with similar arguments; if a tool fails, explain the failure and stop."
	return &ContextBuilder{
		registry:     registry,
		skills:       skillsManager,
		plugins:      pluginsManager,
		store:        store,
		systemPrompt: basePrompt,
	}
}

func (b *ContextBuilder) Build(messages []map[string]any, policy map[string]any, userID, sessionID string) LLMRequestEnvelope {
	toolsSchema := []map[string]any{}
	if b != nil && b.registry != nil {
		enabled := map[string]bool{}
		for _, name := range b.enabledPluginNames() {
			enabled[name] = true
		}
		for _, schema := range b.registry.OpenAISchemas() {
			fn, ok := schema["function"].(map[string]any)
			if !ok {
				continue
			}
			toolName, _ := fn["name"].(string)
			if strings.TrimSpace(toolName) == "" {
				continue
			}
			pluginName := b.registry.PluginForTool(toolName)
			if pluginName == "" || enabled[pluginName] {
				toolsSchema = append(toolsSchema, schema)
			}
		}
	}
	promptParts := []string{}
	if b != nil {
		if strings.TrimSpace(b.systemPrompt) != "" {
			promptParts = append(promptParts, b.systemPrompt)
		}
		if section := b.skillsPromptSection(); strings.TrimSpace(section) != "" {
			promptParts = append(promptParts, section)
		}
		if section := b.pluginsPromptSection(); strings.TrimSpace(section) != "" {
			promptParts = append(promptParts, section)
		}
		if section := b.decisionsPromptSection(userID); strings.TrimSpace(section) != "" {
			promptParts = append(promptParts, section)
		}
		if section := b.memoryPromptSection(userID, messages); strings.TrimSpace(section) != "" {
			promptParts = append(promptParts, section)
		}
	}
	finalMessages := make([]map[string]any, 0, len(messages)+1)
	if len(promptParts) > 0 {
		finalMessages = append(finalMessages, map[string]any{
			"role":    "system",
			"content": strings.Join(promptParts, "\n\n"),
		})
	}
	finalMessages = append(finalMessages, messages...)

	env := LLMRequestEnvelope{
		Kind:       "reply_text",
		Modality:   "text",
		UserID:     strings.TrimSpace(userID),
		SessionID:  strings.TrimSpace(sessionID),
		Messages:   finalMessages,
		Tools:      toolsSchema,
		ToolChoice: "auto",
		Policy:     policy,
	}
	if env.Policy == nil {
		env.Policy = map[string]any{}
	}
	if _, ok := env.Policy["max_tokens"]; !ok {
		env.Policy["max_tokens"] = 1200
	}
	if _, ok := env.Policy["temperature"]; !ok {
		env.Policy["temperature"] = 0.2
	}
	return env.Normalize()
}

func (b *ContextBuilder) decisionsPromptSection(userID string) string {
	if b == nil || b.store == nil {
		return ""
	}
	decisions, err := b.store.ListDecisions(context.Background(), normalizedUserID(userID), "", true)
	if err != nil || len(decisions) == 0 {
		return ""
	}
	lines := []string{"Your learned rules for this user (follow unless explicitly overridden):"}
	for _, d := range decisions {
		if strings.TrimSpace(d.Decision) == "" {
			continue
		}
		lines = append(lines, "- "+strings.TrimSpace(d.Decision))
	}
	if len(lines) <= 1 {
		return ""
	}
	return strings.Join(lines, "\n")
}

func (b *ContextBuilder) memoryPromptSection(userID string, messages []map[string]any) string {
	if b == nil || b.store == nil {
		return ""
	}
	query := latestUserMessage(messages)
	if strings.TrimSpace(query) == "" {
		return ""
	}
	results, err := b.store.SearchMemory(context.Background(), normalizedUserID(userID), query, 6)
	if err != nil || len(results) == 0 {
		return ""
	}
	lines := []string{"Relevant context from past interactions:"}
	for _, r := range results {
		switch r.Type {
		case "fact":
			lines = append(lines, "- [Known fact] "+r.Fact)
		case "memory":
			lines = append(lines, "- [Memory ("+r.Category+")] "+r.Memory)
		case "decision":
			lines = append(lines, "- [Decision ("+r.Category+")] "+r.Decision)
		case "action":
			lines = append(lines, "- [Past action] Used "+r.ToolName+": "+truncateText(strings.TrimSpace(r.Output), 140))
		case "session":
			lines = append(lines, "- [Previous session] "+truncateText(strings.TrimSpace(r.Summary), 180))
		case "conversation":
			lines = append(lines, "- [Previous conversation] User: "+truncateText(strings.TrimSpace(r.UserMessage), 120)+" | Assistant: "+truncateText(strings.TrimSpace(r.AssistantMessage), 120))
		}
	}
	if len(lines) <= 1 {
		return ""
	}
	return strings.Join(lines, "\n")
}

func latestUserMessage(messages []map[string]any) string {
	return LatestUserMessageText(messages)
}

func truncateText(v string, max int) string {
	if max <= 0 || len(v) <= max {
		return v
	}
	return strings.TrimSpace(v[:max]) + "..."
}

func normalizedUserID(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return "default"
	}
	return v
}

func loadPromptFromFile() string {
	promptPath := strings.TrimSpace(os.Getenv("AGENT_PROMPT_FILE"))
	if promptPath == "" {
		assetsDir := strings.TrimSpace(os.Getenv("AGENT_ASSETS_DIR"))
		if assetsDir != "" {
			promptPath = filepath.Join(assetsDir, "PROMPT.md")
		} else {
			wd, err := os.Getwd()
			if err == nil {
				promptPath = filepath.Join(wd, "assets", "PROMPT.md")
			}
		}
	}
	if strings.TrimSpace(promptPath) == "" {
		return ""
	}
	b, err := os.ReadFile(promptPath)
	if err != nil {
		return ""
	}
	return string(b)
}

func (b *ContextBuilder) skillsPromptSection() string {
	if b == nil || b.skills == nil {
		return ""
	}
	all := b.skills.List()
	if len(all) == 0 {
		return ""
	}
	lines := []string{"Available skills (load with skill tools as needed):"}
	for _, s := range all {
		desc := strings.TrimSpace(s.Description)
		if desc == "" {
			desc = "(no description)"
		}
		lines = append(lines, fmt.Sprintf("- %s: %s", s.Name, desc))
	}
	return strings.Join(lines, "\n")
}

func (b *ContextBuilder) pluginsPromptSection() string {
	if b == nil || b.plugins == nil {
		return ""
	}
	enabled := b.enabledPluginNames()
	if len(enabled) == 0 {
		return ""
	}
	lines := []string{"Active plugins:"}
	for _, name := range enabled {
		meta, ok := b.plugins.Get(name)
		if !ok {
			continue
		}
		line := fmt.Sprintf("- %s (%s): %s", meta.Name, meta.DisplayName, meta.Description)
		lines = append(lines, strings.TrimSpace(line))
		if skillText, err := b.plugins.ReadSkill(name); err == nil {
			snippet := compressSnippet(skillText, 900)
			if snippet != "" {
				lines = append(lines, fmt.Sprintf("  Plugin guidance: %s", snippet))
			}
		}
	}
	if len(lines) <= 1 {
		return ""
	}
	return strings.Join(lines, "\n")
}

func (b *ContextBuilder) enabledPluginNames() []string {
	if b == nil || b.plugins == nil {
		return []string{}
	}
	if b.store == nil {
		metas := b.plugins.List()
		names := make([]string, 0, len(metas))
		for _, m := range metas {
			names = append(names, m.Name)
		}
		sort.Strings(names)
		return names
	}
	rows, err := b.store.ListPlugins(context.Background())
	if err != nil {
		return []string{}
	}
	names := make([]string, 0, len(rows))
	for _, r := range rows {
		if r.Enabled {
			names = append(names, r.Name)
		}
	}
	sort.Strings(names)
	return names
}

func compressSnippet(input string, max int) string {
	v := strings.TrimSpace(input)
	if v == "" {
		return ""
	}
	v = strings.ReplaceAll(v, "\r\n", "\n")
	v = strings.ReplaceAll(v, "\n", " ")
	v = strings.Join(strings.Fields(v), " ")
	if len(v) <= max {
		return v
	}
	return strings.TrimSpace(v[:max]) + "..."
}
