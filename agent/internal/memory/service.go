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
	store    *db.Store
	core     *llm.Core
	queue    chan extractionJob
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	entityMu sync.Mutex // serializes entity resolution across workers to prevent duplicates
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
	prompt := `You are Aether's memory extractor. Analyze the conversation turn and extract ONLY high-value information.

## Facts
Objective, stable user information that is useful in future sessions.
Examples: "The user works at Deshpande Educational Trust", "The user prefers Go over Python".
NOT facts: "The user said done", "The assistant ran a command", "The user asked a question".

## Memories
Significant contextual information worth remembering across sessions.
Examples: "User is building an AI agent called Aether", "User had a meeting with Bhuvan about onboarding".
NOT memories: "The user said hello", "The assistant cannot execute curl", "The user asked 'done?'".
Skip: greetings, acknowledgments, conversational filler, assistant capability statements.

## Decisions
Explicit behavior rules or preferences the user stated.
Examples: "User prefers no test files", "User wants only free tools, no freemium".
NOT decisions: "User said ok", "User approved the change".

## Entities
Extract ONLY significant, identifiable entities that are worth tracking long-term.

### Entity Rules (STRICT):
1. ONLY extract: real people, named projects, named organizations, significant topics, physical places.
2. Use FULL CANONICAL NAMES. If you know "Bhuvan" is "Bhuvan T", use "Bhuvan T". Never use partial names if the full name is known or inferrable from context.
3. For people mentioned with an organization (e.g. "discussion with Bhuvan Faceprep"), separate them: the person is "Bhuvan" and the org is "Faceprep". Do NOT combine them as "Bhuvan Faceprep".

### Entity Type Guide:
- person: Named individuals (colleagues, contacts, public figures). NOT: "the user", "someone".
- project: Named software projects, products, initiatives. NOT: generic tasks like "meeting" or "onboarding".
- organization: Named companies, teams, institutions. NOT: generic terms like "the team".
- topic: Specific technical domains or recurring subjects. NOT: generic words like "code", "meeting", "discussion", "next week".
- place: Named physical locations. NOT: "here", "there", "office".
- tool: Software tools the user actively uses as tools. NOT: document references (Google Docs, Gist URLs), websites visited, or file formats.

### DO NOT Extract:
- URLs, file paths, or document references (Google Docs links, GitHub gists, etc.)
- Generic/temporal words: "meeting", "next week", "today", "discussion", "onboarding"
- Assistant capabilities or limitations
- Conversational fragments or filler

Types: person, project, organization, topic, place, tool

Rules:
- Return concise third-person statements.
- Be SELECTIVE. When in doubt, do NOT extract. Quality over quantity.
- If nothing significant, return empty arrays. Empty is better than junk.

Return ONLY JSON:
{"facts": ["..."], "memories": ["..."], "decisions": ["..."], "entities": [{"name": "...", "type": "...", "observations": ["..."]}]}

Conversation:
User: ` + job.UserMessage + `
Assistant: ` + job.AssistantMessage

	env := llm.LLMRequestEnvelope{
		Kind:     "memory_fact_extract",
		Modality: "system",
		Messages: []map[string]any{{"role": "user", "content": prompt}},
		Tools:    []map[string]any{},
		Policy:   map[string]any{"max_tokens": 600, "temperature": 0.0},
	}

	parts := []string{}
	runCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	for ev := range s.core.GenerateWithTools(runCtx, env) {
		if ev.EventType == llm.EventTextDelta {
			if t, ok := ev.Payload["delta"].(string); ok {
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
	// Store extracted entities with dedup: serialize entity resolution across
	// workers via mutex to prevent race conditions where two workers both
	// search, both find nothing, and both create the same entity.
	s.entityMu.Lock()
	for _, entity := range parsed.Entities {
		entity.Name = strings.TrimSpace(entity.Name)
		entity.Type = strings.TrimSpace(entity.Type)
		if entity.Name == "" || entity.Type == "" {
			continue
		}
		entityID := s.resolveOrCreateEntity(ctx, job.UserID, entity.Type, entity.Name)
		if entityID == "" {
			continue
		}
		for _, obs := range entity.Observations {
			obs = strings.TrimSpace(obs)
			if obs == "" {
				continue
			}
			_ = s.store.AddEntityObservation(context.Background(), entityID, job.UserID, obs, "trait", "extracted")
		}
	}
	s.entityMu.Unlock()
}

// resolveOrCreateEntity searches for an existing entity by exact name, alias,
// or fuzzy token match before creating a new one. This prevents duplicates like
// "Bhuvan", "Bhuvan T", and "Bhuvan Faceprep" from becoming separate entities.
func (s *Service) resolveOrCreateEntity(ctx context.Context, userID, entityType, name string) string {
	// 1. Try exact name match via UpsertEntity (case-insensitive).
	//    This handles the simple case where the name already exists exactly.
	//    But first, check if a DIFFERENT entity type already has this name —
	//    we search broadly before blindly upserting.

	// 2. Try alias lookup — checks name column AND aliases JSON.
	existing, err := s.store.FindEntityByAlias(ctx, userID, name)
	if err == nil && existing != nil {
		// Found by alias. Update last_seen and return.
		_ = s.store.TouchEntity(ctx, existing.ID)
		return existing.ID
	}

	// 3. Fuzzy search — tokenize the incoming name and compare against existing entities.
	//    This catches cases like "Bhuvan T" matching "Bhuvan" or
	//    "DET" matching "Deshpande Educational Trust" (if DET is an alias).
	candidates, err := s.store.SearchEntities(ctx, userID, name, 5)
	if err == nil && len(candidates) > 0 {
		match := findBestEntityMatch(name, entityType, candidates)
		if match != nil {
			// Merge: add the incoming name as an alias if it differs from the canonical name.
			if !strings.EqualFold(match.Name, name) {
				_ = s.store.AddEntityAlias(ctx, match.ID, name)
			}
			_ = s.store.TouchEntity(ctx, match.ID)
			return match.ID
		}
	}

	// 4. No match found — create new entity.
	entityID, err := s.store.UpsertEntity(context.Background(), userID, entityType, name, nil, nil)
	if err != nil || entityID == "" {
		return ""
	}
	return entityID
}

// findBestEntityMatch picks the best matching entity from candidates.
// It requires: same entity type AND high name similarity (>0.6 overlap score
// OR one name is a substring of the other).
func findBestEntityMatch(name, entityType string, candidates []db.EntityRecord) *db.EntityRecord {
	nameLower := strings.ToLower(strings.TrimSpace(name))
	nameTokens := tokenize(nameLower)

	var bestMatch *db.EntityRecord
	bestScore := 0.0

	for i, c := range candidates {
		// Must be the same entity type.
		if !strings.EqualFold(c.EntityType, entityType) {
			continue
		}

		candidateLower := strings.ToLower(c.Name)

		// Exact match (case-insensitive) — best possible.
		if candidateLower == nameLower {
			return &candidates[i]
		}

		// Substring check: "Bhuvan" is contained in "Bhuvan T" or vice versa.
		if strings.Contains(candidateLower, nameLower) || strings.Contains(nameLower, candidateLower) {
			// Substring match is very strong — return immediately if types match.
			return &candidates[i]
		}

		// Also check aliases for substring match.
		for _, alias := range c.Aliases {
			aliasLower := strings.ToLower(alias)
			if aliasLower == nameLower || strings.Contains(aliasLower, nameLower) || strings.Contains(nameLower, aliasLower) {
				return &candidates[i]
			}
		}

		// Token overlap score.
		candidateTokens := tokenize(candidateLower)
		score := tokenOverlap(nameTokens, candidateTokens)
		if score > bestScore {
			bestScore = score
			bestMatch = &candidates[i]
		}
	}

	// Only accept fuzzy matches above a high threshold to avoid false positives.
	if bestScore >= 0.6 && bestMatch != nil {
		return bestMatch
	}
	return nil
}

// tokenize splits a string into lowercase alphanumeric tokens.
func tokenize(v string) map[string]struct{} {
	re := regexp.MustCompile(`[a-z0-9]+`)
	parts := re.FindAllString(strings.ToLower(v), -1)
	out := map[string]struct{}{}
	for _, p := range parts {
		out[p] = struct{}{}
	}
	return out
}

// tokenOverlap computes the overlap score between two token sets.
func tokenOverlap(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	common := 0
	for k := range a {
		if _, ok := b[k]; ok {
			common++
		}
	}
	if common == 0 {
		return 0
	}
	// Use the smaller set as denominator for a stricter match.
	smaller := len(a)
	if len(b) < smaller {
		smaller = len(b)
	}
	return float64(common) / float64(smaller)
}

type extractedEntity struct {
	Name         string   `json:"name"`
	Type         string   `json:"type"`
	Observations []string `json:"observations"`
}

type extractionResult struct {
	Facts     []string          `json:"facts"`
	Memories  []string          `json:"memories"`
	Decisions []string          `json:"decisions"`
	Entities  []extractedEntity `json:"entities"`
}

func parseExtraction(content string) extractionResult {
	out := extractionResult{Facts: []string{}, Memories: []string{}, Decisions: []string{}, Entities: []extractedEntity{}}
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
		return extractionResult{Facts: []string{}, Memories: []string{}, Decisions: []string{}, Entities: []extractedEntity{}}
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
