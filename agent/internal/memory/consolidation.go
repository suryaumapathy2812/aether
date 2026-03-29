package memory

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
)

const (
	CronJobTypeSessionConsolidation  = "session_consolidation"
	CronJobTypeWeeklyReconsolidation = "weekly_reconsolidation"

	DefaultSessionConsolidationIntervalSecs  = 300
	DefaultWeeklyReconsolidationIntervalSecs = 604800
	defaultSessionIdleThreshold              = 3 * time.Hour
	defaultSessionBatchSize                  = 10
	defaultSummaryMinMessageDelta            = 3
)

type ConsolidationOptions struct {
	Store                          *db.Store
	Core                           *llm.Core
	Embedder                       embedder
	Dedup                          *DedupEngine
	SessionIdleThreshold           time.Duration
	SessionConsolidationIntervalS  int
	WeeklyReconsolidationIntervalS int
	SessionBatchSize               int
}

type ConsolidationEngine struct {
	store                          *db.Store
	core                           *llm.Core
	embedder                       embedder
	dedup                          *DedupEngine
	sessionIdleThreshold           time.Duration
	sessionConsolidationIntervalS  int
	weeklyReconsolidationIntervalS int
	sessionBatchSize               int
}

func NewConsolidationEngine(opts ConsolidationOptions) *ConsolidationEngine {
	if opts.SessionIdleThreshold <= 0 {
		opts.SessionIdleThreshold = defaultSessionIdleThreshold
	}
	if opts.SessionConsolidationIntervalS <= 0 {
		opts.SessionConsolidationIntervalS = DefaultSessionConsolidationIntervalSecs
	}
	if opts.WeeklyReconsolidationIntervalS <= 0 {
		opts.WeeklyReconsolidationIntervalS = DefaultWeeklyReconsolidationIntervalSecs
	}
	if opts.SessionBatchSize <= 0 {
		opts.SessionBatchSize = defaultSessionBatchSize
	}
	return &ConsolidationEngine{
		store:                          opts.Store,
		core:                           opts.Core,
		embedder:                       opts.Embedder,
		dedup:                          opts.Dedup,
		sessionIdleThreshold:           opts.SessionIdleThreshold,
		sessionConsolidationIntervalS:  opts.SessionConsolidationIntervalS,
		weeklyReconsolidationIntervalS: opts.WeeklyReconsolidationIntervalS,
		sessionBatchSize:               opts.SessionBatchSize,
	}
}

func RegisterConsolidationCronHandlers(scheduler *cron.Scheduler, engine *ConsolidationEngine) {
	if scheduler == nil || engine == nil {
		return
	}
	scheduler.RegisterHandler(CronModuleMemory, CronJobTypeSessionConsolidation, func(ctx context.Context, job cron.Job) error {
		return engine.RunSessionConsolidation(ctx)
	})
	scheduler.RegisterHandler(CronModuleMemory, CronJobTypeWeeklyReconsolidation, func(ctx context.Context, job cron.Job) error {
		return engine.RunWeeklyReconsolidation(ctx)
	})
}

func (e *ConsolidationEngine) EnsureCronJobs(ctx context.Context) error {
	if e == nil || e.store == nil {
		return nil
	}
	scope := e.store.ScopeCronModule(CronModuleMemory)
	jobs, err := scope.List(ctx)
	if err != nil {
		return err
	}
	hasSessionJob := false
	hasWeeklyJob := false
	for _, job := range jobs {
		switch job.JobType {
		case CronJobTypeSessionConsolidation:
			hasSessionJob = job.Status == db.CronStatusScheduled || job.Status == db.CronStatusRunning
		case CronJobTypeWeeklyReconsolidation:
			hasWeeklyJob = job.Status == db.CronStatusScheduled || job.Status == db.CronStatusRunning
		}
	}
	if !hasSessionJob {
		interval := int64(e.sessionConsolidationIntervalS)
		if _, err := scope.ScheduleRecurring(ctx, CronJobTypeSessionConsolidation, map[string]any{}, time.Now().Add(30*time.Second), interval, 5); err != nil {
			return err
		}
	}
	if !hasWeeklyJob {
		interval := int64(e.weeklyReconsolidationIntervalS)
		if _, err := scope.ScheduleRecurring(ctx, CronJobTypeWeeklyReconsolidation, map[string]any{}, time.Now().Add(2*time.Minute), interval, 5); err != nil {
			return err
		}
	}
	return nil
}

func (e *ConsolidationEngine) RunSessionConsolidation(ctx context.Context) error {
	if e == nil || e.store == nil || e.core == nil {
		return fmt.Errorf("consolidation engine not initialized")
	}
	idleSince := time.Now().UTC().Add(-e.sessionIdleThreshold)
	sessions, err := e.store.ListIdleChatSessions(ctx, idleSince, e.sessionBatchSize)
	if err != nil {
		return err
	}
	for _, session := range sessions {
		if err := e.consolidateSession(ctx, session); err != nil {
			log.Printf("memory consolidation: session summary failed session=%s user=%s: %v", session.ID, session.UserID, err)
		}
	}
	return nil
}

func (e *ConsolidationEngine) RunWeeklyReconsolidation(ctx context.Context) error {
	if e == nil || e.store == nil {
		return fmt.Errorf("consolidation engine not initialized")
	}
	users, err := e.store.ListMemoryUsers(ctx)
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	for _, userID := range users {
		if err := e.reconsolidateUser(ctx, userID, now); err != nil {
			log.Printf("memory reconsolidation: user=%s err=%v", userID, err)
		}
	}
	return nil
}

func (e *ConsolidationEngine) consolidateSession(ctx context.Context, session db.ChatSession) error {
	messages, err := e.store.ListChatMessages(ctx, session.UserID, session.ID, 4000)
	if err != nil {
		return err
	}
	if len(messages) < defaultSummaryMinMessageDelta {
		return nil
	}

	var lastSummary *db.ChatSessionSummary
	lastSummary, err = e.store.GetLatestSessionSummary(ctx, session.ID)
	if err != nil && err != db.ErrNotFound {
		return err
	}
	if err == db.ErrNotFound {
		lastSummary = nil
	}

	lastCount := 0
	previousSummary := ""
	if lastSummary != nil {
		lastCount = lastSummary.MessageCount
		previousSummary = strings.TrimSpace(lastSummary.SummaryText)
	}
	if lastCount < 0 || lastCount > len(messages) {
		lastCount = 0
	}
	if len(messages)-lastCount < defaultSummaryMinMessageDelta {
		return nil
	}

	titleSuggestion, summaryText, err := e.generateSessionSummary(ctx, session, previousSummary, messages, lastCount)
	if err != nil {
		return err
	}
	if strings.TrimSpace(summaryText) == "" || !summaryMeaningfullyChanged(previousSummary, summaryText) {
		return nil
	}

	summaryID, err := e.store.AddChatSessionSummary(ctx, session.ID, session.UserID, summaryText, titleSuggestion, len(messages))
	if err != nil {
		return err
	}
	if strings.TrimSpace(titleSuggestion) != "" {
		if err := e.store.SetChatSessionTitleAuto(ctx, session.ID, titleSuggestion); err != nil {
			log.Printf("memory consolidation: auto-title failed session=%s: %v", session.ID, err)
		}
	}

	window := messages[lastCount:]
	if len(window) == 0 {
		window = messages
	}
	if err := e.persistSummaryMemory(ctx, session, summaryID, summaryText, window); err != nil {
		log.Printf("memory consolidation: persist summary memory failed session=%s: %v", session.ID, err)
	}
	if err := e.consolidateSessionMemories(ctx, session.UserID, session.ID); err != nil {
		return err
	}
	return nil
}

func (e *ConsolidationEngine) reconsolidateUser(ctx context.Context, userID string, now time.Time) error {
	if _, err := e.store.ArchiveMemoryItemsByScope(ctx, userID, string(db.ScopeVolatile), now.Add(-7*24*time.Hour)); err != nil {
		return err
	}
	if _, err := e.store.ArchiveMemoryItemsByScope(ctx, userID, string(db.ScopeEpisodic), now.Add(-30*24*time.Hour)); err != nil {
		return err
	}
	if _, err := e.store.ArchiveMemoryItemsByScope(ctx, userID, string(db.ScopeContextual), now.Add(-90*24*time.Hour)); err != nil {
		return err
	}

	items, err := e.store.ListMemoryItemsForReconsolidation(ctx, userID, 5000)
	if err != nil {
		return err
	}
	for _, item := range items {
		if !strings.EqualFold(item.Status, "active") {
			continue
		}
		if err := e.promoteMemoryItem(ctx, item); err != nil {
			log.Printf("memory reconsolidation: promote failed user=%s item=%d: %v", userID, item.ID, err)
		}
	}

	if err := e.refreshEntitySummaries(ctx, userID, now.Add(-7*24*time.Hour)); err != nil {
		return err
	}
	if e.dedup != nil {
		if _, err := e.dedup.dedupEntities(ctx, userID); err != nil {
			log.Printf("memory reconsolidation: entity grouping failed user=%s: %v", userID, err)
		}
	}
	if err := e.archiveStaleEntities(ctx, userID, now.Add(-180*24*time.Hour)); err != nil {
		return err
	}
	return nil
}

func (e *ConsolidationEngine) consolidateSessionMemories(ctx context.Context, userID, sessionID string) error {
	items, err := e.store.ListMemoryItemsBySession(ctx, userID, sessionID, 2000)
	if err != nil {
		return err
	}
	for _, item := range items {
		if err := e.store.RecordMemoryItemSession(ctx, item.ID, sessionID, userID); err != nil {
			log.Printf("memory consolidation: record memory-session link failed item=%d: %v", item.ID, err)
		}
	}
	for _, item := range items {
		if err := e.promoteMemoryItem(ctx, item); err != nil {
			log.Printf("memory consolidation: promote failed item=%d: %v", item.ID, err)
		}
	}
	return nil
}

func (e *ConsolidationEngine) promoteMemoryItem(ctx context.Context, item db.MemoryItemRecord) error {
	sessionCount, err := e.store.CountDistinctSessions(ctx, item.ID)
	if err != nil {
		return err
	}
	current := strings.TrimSpace(item.Scope)
	next := current

	switch current {
	case string(db.ScopeVolatile):
		if sessionCount >= 2 {
			next = string(db.ScopeContextual)
		}
	case string(db.ScopeContextual):
		baseScope := db.InferMemoryScope(item.Kind, item.Category, item.Content)
		if sessionCount >= 3 && (item.Kind == "fact" || item.Kind == "decision") && baseScope == db.ScopeGlobal {
			next = string(db.ScopeGlobal)
		}
	case string(db.ScopeEpisodic):
		if item.RecallCount >= 3 || sessionCount >= 3 {
			next = string(db.ScopeContextual)
		}
	}

	if next == "" || next == current {
		return nil
	}
	return e.store.UpdateMemoryItemScope(ctx, item.ID, next)
}

func (e *ConsolidationEngine) persistSummaryMemory(ctx context.Context, session db.ChatSession, summaryID int64, summaryText string, window []db.ChatMessageRecord) error {
	startedAt := time.Now().UTC()
	endedAt := startedAt
	if len(window) > 0 {
		startedAt = window[0].CreatedAt
		endedAt = window[len(window)-1].CreatedAt
	}
	turns := 0
	for _, msg := range window {
		if msg.Role == "user" {
			turns++
		}
	}
	if err := e.store.AddMemorySessionSummary(ctx, session.UserID, session.ID, summaryText, startedAt, endedAt, turns, nil); err != nil {
		return err
	}

	var embedding []float32
	if e.embedder != nil {
		embedding, _ = e.embedder.EmbedSingle(ctx, summaryText)
	}
	itemID, err := e.store.AddMemory(ctx, db.AddMemoryInput{
		UserID:     session.UserID,
		Kind:       "summary",
		Category:   "session",
		Content:    summaryText,
		Confidence: 0.92,
		Importance: 0.82,
		SourceType: "chat_session_summary",
		SourceID:   fmt.Sprintf("%d", summaryID),
		SessionID:  session.ID,
		Metadata: map[string]any{
			"chat_session_id": session.ID,
			"summary_id":      summaryID,
			"message_count":   len(window),
		},
		Embedding:  embedding,
		ObservedAt: endedAt,
	})
	if err != nil {
		return err
	}
	return e.store.RecordMemoryItemSession(ctx, itemID, session.ID, session.UserID)
}

func (e *ConsolidationEngine) generateSessionSummary(ctx context.Context, session db.ChatSession, previousSummary string, messages []db.ChatMessageRecord, fromIndex int) (string, string, error) {
	window := messages
	label := "Full conversation"
	if fromIndex > 0 && fromIndex < len(messages) {
		window = messages[fromIndex:]
		label = "New messages since the last summary"
	}
	prompt := strings.TrimSpace(`
You are summarizing a chat session for long-term continuity.

Return plain text in exactly this format:
Title: <5-8 word descriptive title>
Goal: <1-2 sentences>
Context: <1-3 sentences or None>
Decisions:
- <bullet or None>
Blockers:
- <bullet or None>
Actions Taken:
- <bullet or None>
Current State: <1-3 sentences or None>
Next Steps:
- <bullet or None>

Rules:
- Keep the summary concise but specific.
- Focus on user intent, major decisions, blockers, work completed, and what should happen next.
- Use "None" when a section has nothing meaningful.
- Do not include markdown code fences.
- Do not mention that this is a summary.
`)

	body := []string{prompt}
	if strings.TrimSpace(previousSummary) != "" {
		body = append(body, "Previous summary:\n"+previousSummary)
	}
	body = append(body, label+":\n"+formatChatMessages(window))

	content, err := e.generateText(ctx, []map[string]any{
		{"role": "system", "content": "You write structured text summaries for session continuity."},
		{"role": "user", "content": strings.Join(body, "\n\n")},
	}, 900)
	if err != nil {
		return "", "", err
	}
	title, summaryText := parseStructuredSessionSummary(content)
	if strings.TrimSpace(summaryText) == "" {
		return "", "", fmt.Errorf("empty session summary")
	}
	if strings.TrimSpace(title) == "" {
		title = llm.TruncateTitle(session.Title, 60)
	}
	return title, summaryText, nil
}

func (e *ConsolidationEngine) refreshEntitySummaries(ctx context.Context, userID string, olderThan time.Time) error {
	if e.core == nil {
		return nil
	}
	entities, err := e.store.ListEntitiesNeedingSummaryRefresh(ctx, userID, olderThan, 50)
	if err != nil {
		return err
	}
	for _, entity := range entities {
		observations, err := e.store.ListEntityObservations(ctx, entity.ID, 25)
		if err != nil || len(observations) == 0 {
			continue
		}
		summary, err := e.generateEntitySummary(ctx, entity, observations)
		if err != nil {
			log.Printf("memory reconsolidation: entity summary failed entity=%s: %v", entity.ID, err)
			continue
		}
		if strings.TrimSpace(summary) == "" {
			continue
		}
		if err := e.store.UpdateEntitySummary(ctx, entity.ID, summary); err != nil {
			log.Printf("memory reconsolidation: entity summary update failed entity=%s: %v", entity.ID, err)
		}
	}
	return nil
}

func (e *ConsolidationEngine) archiveStaleEntities(ctx context.Context, userID string, olderThan time.Time) error {
	entities, err := e.store.ListActiveEntities(ctx, userID, 5000)
	if err != nil {
		return err
	}
	for _, entity := range entities {
		if entity.InteractionCount > 2 {
			continue
		}
		if entity.LastSeenAt.After(olderThan) {
			continue
		}
		if err := e.store.ArchiveEntity(ctx, entity.ID); err != nil {
			log.Printf("memory reconsolidation: archive entity failed entity=%s: %v", entity.ID, err)
		}
	}
	return nil
}

func (e *ConsolidationEngine) generateEntitySummary(ctx context.Context, entity db.EntityRecord, observations []db.EntityObservationRecord) (string, error) {
	lines := make([]string, 0, len(observations))
	for _, observation := range observations {
		text := strings.TrimSpace(observation.Observation)
		if text == "" {
			continue
		}
		lines = append(lines, "- "+text)
	}
	if len(lines) == 0 {
		return "", nil
	}
	prompt := fmt.Sprintf(`Synthesize a concise 1-2 sentence summary of this entity from its observations.
Entity: %s (%s)
Observations:
%s

Return only the summary text.`, entity.Name, entity.EntityType, strings.Join(lines, "\n"))
	return e.generateText(ctx, []map[string]any{
		{"role": "system", "content": "You write concise entity summaries."},
		{"role": "user", "content": prompt},
	}, 180)
}

func (e *ConsolidationEngine) generateText(ctx context.Context, messages []map[string]any, maxTokens int) (string, error) {
	parts := make([]string, 0, 64)
	var streamErr error
	env := llm.LLMRequestEnvelope{
		Kind:     "memory_consolidation",
		Modality: "text",
		Messages: messages,
		Tools:    []map[string]any{},
		Policy: map[string]any{
			"max_tokens":  maxTokens,
			"temperature": 0.1,
		},
	}
	runCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	for event := range e.core.Generate(runCtx, env) {
		switch event.EventType {
		case llm.EventTextDelta:
			delta, _ := event.Payload["delta"].(string)
			if delta != "" {
				parts = append(parts, delta)
			}
		case llm.EventError:
			errText, _ := event.Payload["errorText"].(string)
			if strings.TrimSpace(errText) == "" {
				errText = "llm generation failed"
			}
			streamErr = fmt.Errorf("%s", errText)
		}
	}
	if streamErr != nil {
		return "", streamErr
	}
	if err := runCtx.Err(); err != nil && err != context.Canceled {
		return "", err
	}
	return strings.TrimSpace(strings.Join(parts, "")), nil
}

func formatChatMessages(messages []db.ChatMessageRecord) string {
	lines := make([]string, 0, len(messages))
	for _, message := range messages {
		content := chatMessageText(message)
		if content == "" {
			content = "[non-text content]"
		}
		lines = append(lines, fmt.Sprintf("%s: %s", strings.ToUpper(strings.TrimSpace(message.Role)), content))
	}
	return strings.Join(lines, "\n")
}

func chatMessageText(message db.ChatMessageRecord) string {
	if message.Content == nil {
		return ""
	}
	if content, ok := message.Content["content"]; ok {
		if text := llm.ExtractTextFromMessageContent(content); text != "" {
			return text
		}
		switch v := content.(type) {
		case string:
			return strings.TrimSpace(v)
		}
	}
	if text := llm.ExtractTextFromMessageContent(message.Content); text != "" {
		return text
	}
	return ""
}

func parseStructuredSessionSummary(content string) (string, string) {
	content = strings.TrimSpace(content)
	if content == "" {
		return "", ""
	}
	content = strings.Trim(strings.TrimSpace(content), "`")
	lines := strings.Split(content, "\n")
	title := ""
	if len(lines) > 0 {
		first := strings.TrimSpace(lines[0])
		if strings.HasPrefix(strings.ToLower(first), "title:") {
			title = strings.TrimSpace(first[len("title:"):])
			lines = lines[1:]
		}
	}
	summaryText := strings.TrimSpace(strings.Join(lines, "\n"))
	return normalizeSummaryTitle(title), normalizeSummarySections(summaryText)
}

func normalizeSummarySections(summary string) string {
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return ""
	}
	re := regexp.MustCompile(`(?i)^([a-z ]+):\s*$`)
	lines := strings.Split(summary, "\n")
	out := make([]string, 0, len(lines))
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if re.MatchString(line) {
			parts := re.FindStringSubmatch(line)
			label := strings.TrimSpace(parts[1])
			out = append(out, titleCase(label)+":")
			continue
		}
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

func titleCase(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return ""
	}
	runes := []rune(s)
	capNext := true
	for i, r := range runes {
		if capNext && unicode.IsLetter(r) {
			runes[i] = unicode.ToUpper(r)
			capNext = false
		}
		if unicode.IsSpace(r) {
			capNext = true
		}
	}
	return string(runes)
}

func normalizeSummaryTitle(title string) string {
	title = strings.TrimSpace(title)
	title = strings.Trim(title, `"'`)
	if title == "" {
		return ""
	}
	return llm.TruncateTitle(title, 60)
}

func summaryMeaningfullyChanged(previous, current string) bool {
	previous = normalizedSummaryForCompare(previous)
	current = normalizedSummaryForCompare(current)
	if current == "" {
		return false
	}
	if previous == "" {
		return true
	}
	if previous == current {
		return false
	}
	return true
}

func normalizedSummaryForCompare(summary string) string {
	summary = strings.ToLower(strings.TrimSpace(summary))
	if summary == "" {
		return ""
	}
	lines := strings.Split(summary, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}
