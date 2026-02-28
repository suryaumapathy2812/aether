package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"
)

type MemoryConversation struct {
	ID               int64
	UserID           string
	SessionID        string
	UserMessage      string
	UserContent      any `json:"user_content,omitempty"`
	AssistantMessage string
	Timestamp        time.Time
}

type MemorySession struct {
	SessionID string
	Summary   string
	StartedAt time.Time
	EndedAt   time.Time
	Turns     int
	ToolsUsed []string
}

type MemoryRecord struct {
	ID         int64
	Memory     string
	Category   string
	Confidence float64
	CreatedAt  time.Time
	ExpiresAt  *time.Time
}

type DecisionRecord struct {
	ID         int64
	Decision   string
	Category   string
	Source     string
	Active     bool
	Confidence float64
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

type MemoryNotification struct {
	ID               int64
	Text             string
	DeliveryType     string
	Status           string
	Source           string
	DeliverAt        *time.Time
	DeliveredAt      *time.Time
	DeliveryAttempts int
	LastAttemptAt    *time.Time
	NextRetryAt      *time.Time
	LastError        string
	ExpiresAt        *time.Time
	CreatedAt        time.Time
	Metadata         map[string]any
}

type MemorySearchResult struct {
	Type       string
	Similarity float64
	Timestamp  time.Time

	Fact             string
	Memory           string
	Decision         string
	Category         string
	Source           string
	ToolName         string
	Arguments        string
	Output           string
	Error            bool
	Summary          string
	UserMessage      string
	AssistantMessage string
	Confidence       float64
}

func (s *Store) AddMemoryConversation(ctx context.Context, userID, sessionID, userMessage string, userContent any, assistantMessage string) (int64, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	userContentJSON := ""
	if userContent != nil {
		if b, err := json.Marshal(userContent); err == nil {
			userContentJSON = string(b)
		}
	}
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO conversations(user_id, session_id, user_message, user_content_json, assistant_message, timestamp)
		VALUES(?, ?, ?, ?, ?, ?)
	`, userID, sessionID, userMessage, userContentJSON, assistantMessage, formatTS(time.Now().UTC()))
	if err != nil {
		return 0, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	return id, nil
}

func (s *Store) AddMemoryAction(ctx context.Context, userID, sessionID, toolName string, arguments map[string]any, output string, isError bool) error {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	if arguments == nil {
		arguments = map[string]any{}
	}
	b, err := json.Marshal(arguments)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO actions(user_id, session_id, tool_name, arguments, output, error, timestamp)
		VALUES(?, ?, ?, ?, ?, ?, ?)
	`, userID, sessionID, toolName, string(b), output, boolToInt(isError), formatTS(time.Now().UTC()))
	return err
}

func (s *Store) AddMemorySessionSummary(ctx context.Context, userID, sessionID, summary string, startedAt, endedAt time.Time, turns int, toolsUsed []string) error {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	b, _ := json.Marshal(toolsUsed)
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO sessions(user_id, session_id, summary, started_at, ended_at, turns, tools_used)
		VALUES(?, ?, ?, ?, ?, ?, ?)
	`, userID, sessionID, summary, formatTS(startedAt.UTC()), formatTS(endedAt.UTC()), turns, string(b))
	return err
}

func (s *Store) StoreMemoryFact(ctx context.Context, userID, fact string, sourceConversationID int64) error {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	fact = strings.TrimSpace(fact)
	if fact == "" {
		return nil
	}
	key := normalizeMemoryKey(fact)
	now := formatTS(time.Now().UTC())
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO facts(user_id, fact, fact_key, source_conversation_id, created_at, updated_at)
		VALUES(?, ?, ?, ?, ?, ?)
		ON CONFLICT(user_id, fact_key) DO UPDATE SET
			fact = excluded.fact,
			source_conversation_id = excluded.source_conversation_id,
			updated_at = excluded.updated_at
	`, userID, fact, key, sourceConversationID, now, now)
	return err
}

func (s *Store) StoreMemory(ctx context.Context, userID, memory, category string, sourceConversationID int64, expiresAt *time.Time) error {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	memory = strings.TrimSpace(memory)
	if memory == "" {
		return nil
	}
	if strings.TrimSpace(category) == "" {
		category = "episodic"
	}
	key := normalizeMemoryKey(memory)
	now := formatTS(time.Now().UTC())
	expires := sql.NullString{}
	if expiresAt != nil {
		expires = sql.NullString{String: formatTS(expiresAt.UTC()), Valid: true}
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO memories(user_id, memory, memory_key, category, source_conversation_id, created_at, expires_at, confidence)
		VALUES(?, ?, ?, ?, ?, ?, ?, 1.0)
		ON CONFLICT(user_id, memory_key) DO UPDATE SET
			memory = excluded.memory,
			source_conversation_id = excluded.source_conversation_id,
			confidence = 1.0
	`, userID, memory, key, category, sourceConversationID, now, expires)
	return err
}

func (s *Store) StoreMemoryDecision(ctx context.Context, userID, decision, category, source string, sourceConversationID int64) error {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	decision = strings.TrimSpace(decision)
	if decision == "" {
		return nil
	}
	if strings.TrimSpace(category) == "" {
		category = "preference"
	}
	if strings.TrimSpace(source) == "" {
		source = "extracted"
	}
	key := normalizeMemoryKey(decision)
	now := formatTS(time.Now().UTC())
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO decisions(user_id, decision, decision_key, category, source, source_conversation_id, active, created_at, updated_at, confidence)
		VALUES(?, ?, ?, ?, ?, ?, 1, ?, ?, 1.0)
		ON CONFLICT(user_id, decision_key) DO UPDATE SET
			decision = excluded.decision,
			source = excluded.source,
			source_conversation_id = excluded.source_conversation_id,
			active = 1,
			updated_at = excluded.updated_at,
			confidence = 1.0
	`, userID, decision, key, category, source, sourceConversationID, now, now)
	return err
}

func (s *Store) ListMemoryConversations(ctx context.Context, userID string, limit int) ([]MemoryConversation, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 2000 {
		limit = 20
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, session_id, user_message, user_content_json, assistant_message, timestamp
		FROM conversations
		WHERE user_id = ?
		ORDER BY timestamp DESC
		LIMIT ?
	`, userID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []MemoryConversation{}
	for rows.Next() {
		var rec MemoryConversation
		var userContentJSON string
		var ts string
		if err := rows.Scan(&rec.ID, &rec.UserID, &rec.SessionID, &rec.UserMessage, &userContentJSON, &rec.AssistantMessage, &ts); err != nil {
			return nil, err
		}
		if strings.TrimSpace(userContentJSON) != "" {
			var raw any
			if err := json.Unmarshal([]byte(userContentJSON), &raw); err == nil {
				rec.UserContent = raw
			}
		}
		v, err := parseTS(ts)
		if err != nil {
			return nil, err
		}
		rec.Timestamp = v
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) GetMemoryFacts(ctx context.Context, userID string) ([]string, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	rows, err := s.db.QueryContext(ctx, `SELECT fact FROM facts WHERE user_id = ? ORDER BY updated_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []string{}
	for rows.Next() {
		var fact string
		if err := rows.Scan(&fact); err != nil {
			return nil, err
		}
		out = append(out, fact)
	}
	return out, rows.Err()
}

func (s *Store) ListMemorySessions(ctx context.Context, userID string, limit int) ([]MemorySession, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 2000 {
		limit = 100
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT session_id, summary, started_at, ended_at, turns, tools_used
		FROM sessions
		WHERE user_id = ?
		ORDER BY ended_at DESC
		LIMIT ?
	`, userID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []MemorySession{}
	for rows.Next() {
		var rec MemorySession
		var started, ended, toolsJSON string
		if err := rows.Scan(&rec.SessionID, &rec.Summary, &started, &ended, &rec.Turns, &toolsJSON); err != nil {
			return nil, err
		}
		startTS, err := parseTS(started)
		if err != nil {
			return nil, err
		}
		endTS, err := parseTS(ended)
		if err != nil {
			return nil, err
		}
		rec.StartedAt = startTS
		rec.EndedAt = endTS
		rec.ToolsUsed = []string{}
		if strings.TrimSpace(toolsJSON) != "" {
			_ = json.Unmarshal([]byte(toolsJSON), &rec.ToolsUsed)
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) ListMemories(ctx context.Context, userID, category string, limit int) ([]MemoryRecord, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 2000 {
		limit = 100
	}
	query := `
		SELECT id, memory, category, confidence, created_at, expires_at
		FROM memories
		WHERE user_id = ?
	`
	args := []any{userID}
	if strings.TrimSpace(category) != "" {
		query += ` AND category = ?`
		args = append(args, category)
	}
	query += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []MemoryRecord{}
	for rows.Next() {
		var rec MemoryRecord
		var created string
		var expires sql.NullString
		if err := rows.Scan(&rec.ID, &rec.Memory, &rec.Category, &rec.Confidence, &created, &expires); err != nil {
			return nil, err
		}
		createdTS, err := parseTS(created)
		if err != nil {
			return nil, err
		}
		rec.CreatedAt = createdTS
		if expires.Valid {
			exp, err := parseTS(expires.String)
			if err == nil {
				rec.ExpiresAt = &exp
			}
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) ListDecisions(ctx context.Context, userID, category string, activeOnly bool) ([]DecisionRecord, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	query := `
		SELECT id, decision, category, source, active, confidence, created_at, updated_at
		FROM decisions
		WHERE user_id = ?
	`
	args := []any{userID}
	if activeOnly {
		query += ` AND active = 1`
	}
	if strings.TrimSpace(category) != "" {
		query += ` AND category = ?`
		args = append(args, category)
	}
	query += ` ORDER BY updated_at DESC`
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []DecisionRecord{}
	for rows.Next() {
		var rec DecisionRecord
		var active int
		var created, updated string
		if err := rows.Scan(&rec.ID, &rec.Decision, &rec.Category, &rec.Source, &active, &rec.Confidence, &created, &updated); err != nil {
			return nil, err
		}
		createdTS, err := parseTS(created)
		if err != nil {
			return nil, err
		}
		updatedTS, err := parseTS(updated)
		if err != nil {
			return nil, err
		}
		rec.Active = active == 1
		rec.CreatedAt = createdTS
		rec.UpdatedAt = updatedTS
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) QueueMemoryNotification(ctx context.Context, userID, text, deliveryType, source string, deliverAt *time.Time, metadata map[string]any) (int64, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(deliveryType) == "" {
		deliveryType = "surface"
	}
	if strings.TrimSpace(source) == "" {
		source = "proactive"
	}
	now := time.Now().UTC()
	meta := map[string]any{}
	if metadata != nil {
		meta = metadata
	}
	b, _ := json.Marshal(meta)
	var deliver sql.NullString
	if deliverAt != nil {
		deliver = sql.NullString{String: formatTS(deliverAt.UTC()), Valid: true}
	}
	expires := formatTS(now.Add(4 * time.Hour))
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO notifications(user_id, text, delivery_type, status, source, deliver_at, expires_at, created_at, metadata)
		VALUES(?, ?, ?, 'pending', ?, ?, ?, ?, ?)
	`, userID, text, deliveryType, source, deliver, expires, formatTS(now), string(b))
	if err != nil {
		return 0, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	return id, nil
}

func (s *Store) ListMemoryNotifications(ctx context.Context, userID string, limit int) ([]MemoryNotification, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 10000 {
		limit = 200
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, text, delivery_type, status, source, deliver_at, delivered_at,
			delivery_attempts, last_attempt_at, next_retry_at, last_error, expires_at, created_at, metadata
		FROM notifications
		WHERE user_id = ?
		ORDER BY created_at DESC
		LIMIT ?
	`, userID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []MemoryNotification{}
	for rows.Next() {
		var rec MemoryNotification
		var deliverAt, deliveredAt, lastAttemptAt, nextRetryAt, expiresAt sql.NullString
		var createdAt, metadataJSON string
		if err := rows.Scan(&rec.ID, &rec.Text, &rec.DeliveryType, &rec.Status, &rec.Source, &deliverAt, &deliveredAt, &rec.DeliveryAttempts, &lastAttemptAt, &nextRetryAt, &rec.LastError, &expiresAt, &createdAt, &metadataJSON); err != nil {
			return nil, err
		}
		createdTS, err := parseTS(createdAt)
		if err != nil {
			return nil, err
		}
		rec.CreatedAt = createdTS
		if deliverAt.Valid {
			if ts, err := parseTS(deliverAt.String); err == nil {
				rec.DeliverAt = &ts
			}
		}
		if deliveredAt.Valid {
			if ts, err := parseTS(deliveredAt.String); err == nil {
				rec.DeliveredAt = &ts
			}
		}
		if lastAttemptAt.Valid {
			if ts, err := parseTS(lastAttemptAt.String); err == nil {
				rec.LastAttemptAt = &ts
			}
		}
		if nextRetryAt.Valid {
			if ts, err := parseTS(nextRetryAt.String); err == nil {
				rec.NextRetryAt = &ts
			}
		}
		if expiresAt.Valid {
			if ts, err := parseTS(expiresAt.String); err == nil {
				rec.ExpiresAt = &ts
			}
		}
		rec.Metadata = map[string]any{}
		if strings.TrimSpace(metadataJSON) != "" {
			_ = json.Unmarshal([]byte(metadataJSON), &rec.Metadata)
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) GetMemoryReliabilitySnapshot(ctx context.Context, userID string) (map[string]any, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	pending := 0
	retry := 0
	oldestAge := 0.0
	now := time.Now().UTC()

	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM notifications WHERE user_id = ? AND status = 'pending'`, userID).Scan(&pending); err != nil {
		return nil, err
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM notifications WHERE user_id = ? AND status = 'pending' AND next_retry_at IS NOT NULL`, userID).Scan(&retry); err != nil {
		return nil, err
	}
	var oldest sql.NullString
	if err := s.db.QueryRowContext(ctx, `SELECT MIN(created_at) FROM notifications WHERE user_id = ? AND status = 'pending'`, userID).Scan(&oldest); err != nil {
		return nil, err
	}
	if oldest.Valid {
		if ts, err := parseTS(oldest.String); err == nil {
			oldestAge = math.Max(0, now.Sub(ts).Seconds())
		}
	}

	rows, err := s.db.QueryContext(ctx, `SELECT status, COUNT(*) FROM proactive_events WHERE user_id = ? GROUP BY status`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	events := map[string]int{}
	for rows.Next() {
		var status string
		var count int
		if err := rows.Scan(&status, &count); err != nil {
			return nil, err
		}
		events[status] = count
	}

	return map[string]any{
		"pending_notifications": pending,
		"pending_with_retry":    retry,
		"oldest_pending_age_s":  oldestAge,
		"proactive_events":      events,
	}, nil
}

func (s *Store) ExportMemorySnapshot(ctx context.Context, userID string) (map[string]any, error) {
	facts, err := s.GetMemoryFacts(ctx, userID)
	if err != nil {
		return nil, err
	}
	memories, err := s.ListMemories(ctx, userID, "", 10000)
	if err != nil {
		return nil, err
	}
	decisions, err := s.ListDecisions(ctx, userID, "", false)
	if err != nil {
		return nil, err
	}
	conversations, err := s.ListMemoryConversations(ctx, userID, 10000)
	if err != nil {
		return nil, err
	}
	sessions, err := s.ListMemorySessions(ctx, userID, 10000)
	if err != nil {
		return nil, err
	}
	notifications, err := s.ListMemoryNotifications(ctx, userID, 10000)
	if err != nil {
		return nil, err
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, event_id, plugin, event_type, status, decision, delivery_type, notification_id, error, payload, created_at, updated_at
		FROM proactive_events
		WHERE user_id = ?
		ORDER BY created_at DESC
		LIMIT 10000
	`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	proactive := []map[string]any{}
	for rows.Next() {
		var id int64
		var eventID, plugin, eventType, status, decision, deliveryType, errText, payloadJSON, createdAt, updatedAt string
		var notifID sql.NullInt64
		if err := rows.Scan(&id, &eventID, &plugin, &eventType, &status, &decision, &deliveryType, &notifID, &errText, &payloadJSON, &createdAt, &updatedAt); err != nil {
			return nil, err
		}
		payload := map[string]any{}
		_ = json.Unmarshal([]byte(payloadJSON), &payload)
		proactive = append(proactive, map[string]any{
			"id":            id,
			"event_id":      eventID,
			"plugin":        plugin,
			"event_type":    eventType,
			"status":        status,
			"decision":      decision,
			"delivery_type": deliveryType,
			"notification_id": func() any {
				if notifID.Valid {
					return notifID.Int64
				}
				return nil
			}(),
			"error":      errText,
			"payload":    payload,
			"created_at": createdAt,
			"updated_at": updatedAt,
		})
	}

	return map[string]any{
		"facts":            facts,
		"memories":         memories,
		"decisions":        decisions,
		"conversations":    conversations,
		"sessions":         sessions,
		"notifications":    notifications,
		"proactive_events": proactive,
	}, nil
}

func (s *Store) SearchMemory(ctx context.Context, userID, query string, limit int) ([]MemorySearchResult, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 100 {
		limit = 5
	}
	qTokens := tokenizeQuery(query)
	if len(qTokens) == 0 {
		return []MemorySearchResult{}, nil
	}
	results := []MemorySearchResult{}

	appendIfMatch := func(kind, text string, boost float64, ts time.Time, enrich func(*MemorySearchResult)) {
		score := overlapScore(qTokens, tokenizeQuery(text))
		if score <= 0 {
			return
		}
		item := MemorySearchResult{Type: kind, Similarity: score + boost, Timestamp: ts}
		if enrich != nil {
			enrich(&item)
		}
		results = append(results, item)
	}

	convRows, err := s.ListMemoryConversations(ctx, userID, 200)
	if err != nil {
		return nil, err
	}
	for _, row := range convRows {
		text := row.UserMessage + " " + row.AssistantMessage
		appendIfMatch("conversation", text, 0, row.Timestamp, func(r *MemorySearchResult) {
			r.UserMessage = row.UserMessage
			r.AssistantMessage = row.AssistantMessage
		})
	}

	facts, err := s.GetMemoryFacts(ctx, userID)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	for _, fact := range facts {
		f := fact
		appendIfMatch("fact", f, 0.1, now, func(r *MemorySearchResult) {
			r.Fact = f
		})
	}

	memoryRows, err := s.ListMemories(ctx, userID, "", 200)
	if err != nil {
		return nil, err
	}
	for _, row := range memoryRows {
		if row.ExpiresAt != nil && row.ExpiresAt.Before(now) {
			continue
		}
		m := row.Memory
		appendIfMatch("memory", m, 0.05, row.CreatedAt, func(r *MemorySearchResult) {
			r.Memory = m
			r.Category = row.Category
			r.Confidence = row.Confidence
		})
	}

	decisionRows, err := s.ListDecisions(ctx, userID, "", true)
	if err != nil {
		return nil, err
	}
	for _, row := range decisionRows {
		d := row.Decision
		appendIfMatch("decision", d, 0.15, row.UpdatedAt, func(r *MemorySearchResult) {
			r.Decision = d
			r.Category = row.Category
			r.Source = row.Source
			r.Confidence = row.Confidence
		})
	}

	actionsRows, err := s.db.QueryContext(ctx, `
		SELECT tool_name, arguments, output, error, timestamp
		FROM actions
		WHERE user_id = ?
		ORDER BY timestamp DESC
		LIMIT 200
	`, userID)
	if err != nil {
		return nil, err
	}
	for actionsRows.Next() {
		var toolName, args, output, ts string
		var errInt int
		if err := actionsRows.Scan(&toolName, &args, &output, &errInt, &ts); err != nil {
			actionsRows.Close()
			return nil, err
		}
		timeVal, err := parseTS(ts)
		if err != nil {
			actionsRows.Close()
			return nil, err
		}
		text := toolName + " " + args + " " + output
		appendIfMatch("action", text, 0.05, timeVal, func(r *MemorySearchResult) {
			r.ToolName = toolName
			r.Arguments = args
			r.Output = output
			r.Error = errInt == 1
		})
	}
	actionsRows.Close()

	sessions, err := s.ListMemorySessions(ctx, userID, 200)
	if err != nil {
		return nil, err
	}
	for _, row := range sessions {
		summary := row.Summary
		appendIfMatch("session", summary, 0.05, row.EndedAt, func(r *MemorySearchResult) {
			r.Summary = summary
		})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Similarity == results[j].Similarity {
			return results[i].Timestamp.After(results[j].Timestamp)
		}
		return results[i].Similarity > results[j].Similarity
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func normalizeMemoryKey(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	v = strings.ReplaceAll(v, "\u2019", "'")
	re := regexp.MustCompile(`[a-z0-9]+`)
	parts := re.FindAllString(v, -1)
	return strings.Join(parts, " ")
}

func tokenizeQuery(v string) map[string]struct{} {
	v = strings.TrimSpace(strings.ToLower(v))
	re := regexp.MustCompile(`[a-z0-9]+`)
	parts := re.FindAllString(v, -1)
	out := map[string]struct{}{}
	for _, p := range parts {
		if p == "" {
			continue
		}
		out[p] = struct{}{}
	}
	return out
}

func overlapScore(a, b map[string]struct{}) float64 {
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
	denom := math.Sqrt(float64(len(a) * len(b)))
	if denom <= 0 {
		return 0
	}
	return float64(common) / denom
}

func (s *Store) RecordProactiveEvent(ctx context.Context, userID, eventID, plugin, eventType, status string, payload map[string]any) (int64, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if payload == nil {
		payload = map[string]any{}
	}
	b, _ := json.Marshal(payload)
	now := formatTS(time.Now().UTC())
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO proactive_events(user_id, event_id, plugin, event_type, status, payload, created_at, updated_at)
		VALUES(?, ?, ?, ?, ?, ?, ?, ?)
	`, userID, eventID, plugin, eventType, status, string(b), now, now)
	if err != nil {
		return 0, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	return id, nil
}

func (s *Store) UpdateProactiveEvent(ctx context.Context, rowID int64, status, decision, deliveryType string, notificationID *int64, errText string) error {
	notification := sql.NullInt64{}
	if notificationID != nil {
		notification = sql.NullInt64{Int64: *notificationID, Valid: true}
	}
	_, err := s.db.ExecContext(ctx, `
		UPDATE proactive_events
		SET status = ?, decision = ?, delivery_type = ?, notification_id = ?, error = ?, updated_at = ?
		WHERE id = ?
	`, status, decision, deliveryType, notification, errText, formatTS(time.Now().UTC()), rowID)
	return err
}

func (s *Store) MarkMemoryNotificationDelivered(ctx context.Context, notificationID int64) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE notifications
		SET status = 'delivered', delivered_at = ?, next_retry_at = NULL, last_error = ''
		WHERE id = ?
	`, formatTS(time.Now().UTC()), notificationID)
	return err
}

func (s *Store) MarkMemoryNotificationAttempt(ctx context.Context, notificationID int64) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE notifications
		SET delivery_attempts = COALESCE(delivery_attempts, 0) + 1, last_attempt_at = ?
		WHERE id = ?
	`, formatTS(time.Now().UTC()), notificationID)
	return err
}

func (s *Store) MarkMemoryNotificationError(ctx context.Context, notificationID int64, errText string) error {
	var attempts int
	if err := s.db.QueryRowContext(ctx, `SELECT COALESCE(delivery_attempts, 1) FROM notifications WHERE id = ?`, notificationID).Scan(&attempts); err != nil {
		return err
	}
	if attempts < 1 {
		attempts = 1
	}
	backoff := time.Duration(1<<max(0, attempts-1)) * time.Second
	if backoff > 5*time.Minute {
		backoff = 5 * time.Minute
	}
	_, err := s.db.ExecContext(ctx, `
		UPDATE notifications
		SET status = 'pending', last_error = ?, next_retry_at = ?
		WHERE id = ?
	`, truncateString(errText, 500), formatTS(time.Now().UTC().Add(backoff)), notificationID)
	return err
}

func truncateString(v string, maxLen int) string {
	if maxLen <= 0 || len(v) <= maxLen {
		return v
	}
	return v[:maxLen]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (s *Store) EnsureMemoryReady(ctx context.Context) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store is unavailable")
	}
	_, err := s.db.ExecContext(ctx, `SELECT 1`)
	return err
}
