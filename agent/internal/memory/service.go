package memory

import (
	"context"
	"encoding/json"
	"log"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
)

type Service struct {
	store  *db.Store
	core   *llm.Core
	queue  chan extractionJob
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

type extractionJob struct {
	UserID           string
	ConversationID   int64
	UserMessage      string
	AssistantMessage string
}

func NewService(store *db.Store, core *llm.Core) *Service {
	return &Service{
		store: store,
		core:  core,
		queue: make(chan extractionJob, 256),
	}
}

const extractionWorkers = 3

func (s *Service) Start(ctx context.Context) {
	if s == nil || s.store == nil || s.core == nil || s.cancel != nil {
		return
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	for i := 0; i < extractionWorkers; i++ {
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.worker(runCtx)
		}()
	}
	log.Printf("memory extraction: started %d workers", extractionWorkers)
}

func (s *Service) Stop(ctx context.Context) error {
	if s == nil || s.cancel == nil {
		return nil
	}
	s.cancel()
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		return nil
	}
}

func (s *Service) RecordConversation(ctx context.Context, userID, sessionID, userMessage string, userContent any, assistantMessage string) {
	if s == nil || s.store == nil {
		return
	}
	if strings.TrimSpace(userMessage) == "" || strings.TrimSpace(assistantMessage) == "" {
		return
	}
	convID, err := s.store.AddMemoryConversation(ctx, userID, sessionID, userMessage, userContent, assistantMessage)
	if err != nil {
		log.Printf("memory conversation write failed: %v", err)
		return
	}
	job := extractionJob{UserID: defaultUser(userID), ConversationID: convID, UserMessage: userMessage, AssistantMessage: assistantMessage}
	select {
	case s.queue <- job:
	default:
		log.Printf("memory extraction queue full; dropping job conversation_id=%d", convID)
	}
}

func (s *Service) RecordAction(ctx context.Context, userID, sessionID, toolName string, arguments map[string]any, output string, isError bool) {
	if s == nil || s.store == nil || strings.TrimSpace(toolName) == "" {
		return
	}
	if err := s.store.AddMemoryAction(ctx, userID, sessionID, toolName, arguments, output, isError); err != nil {
		log.Printf("memory action write failed: %v", err)
	}
}

func (s *Service) RecordSessionSummary(ctx context.Context, userID, sessionID, summary string, startedAt, endedAt time.Time, turns int, toolsUsed []string) {
	if s == nil || s.store == nil || strings.TrimSpace(summary) == "" {
		return
	}
	if err := s.store.AddMemorySessionSummary(ctx, userID, sessionID, summary, startedAt, endedAt, turns, toolsUsed); err != nil {
		log.Printf("memory session write failed: %v", err)
	}
}

func (s *Service) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case job := <-s.queue:
			s.extractAndStore(ctx, job)
		}
	}
}

func (s *Service) extractAndStore(ctx context.Context, job extractionJob) {
	prompt := `You are Aether's memory extractor. After each conversation turn, extract three types of information:

## Facts
Objective, stable user information that is useful later.

## Memories
Contextual or episodic information from this interaction.

## Decisions
Behavior rules the assistant should follow for this user.

Rules:
- Return concise third-person statements.
- Skip small talk and low-value details.
- If nothing useful, return empty arrays.

Return ONLY JSON:
{"facts": ["..."], "memories": ["..."], "decisions": ["..."]}

Conversation:
User: ` + job.UserMessage + `
Assistant: ` + job.AssistantMessage

	env := llm.LLMRequestEnvelope{
		Kind:     "memory_fact_extract",
		Modality: "system",
		Messages: []map[string]any{{"role": "user", "content": prompt}},
		Tools:    []map[string]any{},
		Policy:   map[string]any{"max_tokens": 400, "temperature": 0.0},
	}

	parts := []string{}
	runCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	for ev := range s.core.GenerateWithTools(runCtx, env) {
		if ev.EventType == llm.EventTextChunk {
			if t, ok := ev.Payload["text"].(string); ok {
				parts = append(parts, t)
			}
		}
	}
	content := strings.TrimSpace(strings.Join(parts, ""))
	parsed := parseExtraction(content)
	for _, fact := range parsed.Facts {
		_ = s.store.StoreMemoryFact(context.Background(), job.UserID, fact, job.ConversationID)
	}
	for _, memory := range parsed.Memories {
		_ = s.store.StoreMemory(context.Background(), job.UserID, memory, "episodic", job.ConversationID, nil)
	}
	for _, decision := range parsed.Decisions {
		_ = s.store.StoreMemoryDecision(context.Background(), job.UserID, decision, "preference", "extracted", job.ConversationID)
	}
}

type extractionResult struct {
	Facts     []string `json:"facts"`
	Memories  []string `json:"memories"`
	Decisions []string `json:"decisions"`
}

func parseExtraction(content string) extractionResult {
	out := extractionResult{Facts: []string{}, Memories: []string{}, Decisions: []string{}}
	v := strings.TrimSpace(content)
	if v == "" {
		return out
	}
	if !strings.HasPrefix(v, "{") {
		re := regexp.MustCompile(`\{[\s\S]*\}`)
		m := re.FindString(v)
		if m == "" {
			return out
		}
		v = m
	}
	if err := json.Unmarshal([]byte(v), &out); err != nil {
		return extractionResult{Facts: []string{}, Memories: []string{}, Decisions: []string{}}
	}
	out.Facts = cleanStrings(out.Facts)
	out.Memories = cleanStrings(out.Memories)
	out.Decisions = cleanStrings(out.Decisions)
	return out
}

func cleanStrings(items []string) []string {
	out := make([]string, 0, len(items))
	for _, item := range items {
		v := strings.TrimSpace(item)
		if v == "" {
			continue
		}
		out = append(out, v)
	}
	return out
}

func defaultUser(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return "default"
	}
	return v
}
