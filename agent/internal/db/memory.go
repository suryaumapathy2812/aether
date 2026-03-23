package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
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
	EntityName       string
	EntityType       string
	EntitySummary    string
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

// MergeEntities merges the source entity into the target entity.
// It moves all observations, interactions, and relations from source to target,
// adds the source name as an alias on the target, then deletes the source.
func (s *Store) MergeEntities(ctx context.Context, targetID, sourceID string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if targetID == sourceID {
		return nil
	}

	// Move observations: update entity_id, handle key conflicts by deleting dupes.
	_, _ = s.db.ExecContext(ctx, `
		DELETE FROM entity_observations
		WHERE entity_id = ? AND observation_key IN (
			SELECT observation_key FROM entity_observations WHERE entity_id = ?
		)
	`, sourceID, targetID)
	_, _ = s.db.ExecContext(ctx, `
		UPDATE entity_observations SET entity_id = ? WHERE entity_id = ?
	`, targetID, sourceID)

	// Move interactions.
	_, _ = s.db.ExecContext(ctx, `
		UPDATE entity_interactions SET entity_id = ? WHERE entity_id = ?
	`, targetID, sourceID)

	// Move relations: update source_entity_id and target_entity_id references.
	// Delete any that would create self-referencing or duplicate relations.
	_, _ = s.db.ExecContext(ctx, `
		DELETE FROM entity_relations
		WHERE source_entity_id = ? AND target_entity_id = ?
	`, sourceID, targetID)
	_, _ = s.db.ExecContext(ctx, `
		DELETE FROM entity_relations
		WHERE source_entity_id = ? AND target_entity_id = ?
	`, targetID, sourceID)
	_, _ = s.db.ExecContext(ctx, `
		UPDATE OR IGNORE entity_relations SET source_entity_id = ? WHERE source_entity_id = ?
	`, targetID, sourceID)
	_, _ = s.db.ExecContext(ctx, `
		UPDATE OR IGNORE entity_relations SET target_entity_id = ? WHERE target_entity_id = ?
	`, targetID, sourceID)
	// Clean up any remaining orphaned relations from the source.
	_, _ = s.db.ExecContext(ctx, `
		DELETE FROM entity_relations WHERE source_entity_id = ? OR target_entity_id = ?
	`, sourceID, sourceID)

	// Add source name as alias on target.
	source, err := s.GetEntity(ctx, sourceID)
	if err == nil && source != nil {
		_ = s.AddEntityAlias(ctx, targetID, source.Name)
		for _, alias := range source.Aliases {
			_ = s.AddEntityAlias(ctx, targetID, alias)
		}
	}

	// Accumulate interaction count.
	_, _ = s.db.ExecContext(ctx, `
		UPDATE entities SET
			interaction_count = interaction_count + COALESCE((SELECT interaction_count FROM entities WHERE id = ?), 0),
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, sourceID, targetID)

	// Delete source entity (cascades observations, interactions, relations).
	return s.DeleteEntity(ctx, sourceID)
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
	conversations, err := s.ListMemoryConversations(ctx, userID, 10000)
	if err != nil {
		return nil, err
	}
	sessions, err := s.ListMemorySessions(ctx, userID, 10000)
	if err != nil {
		return nil, err
	}
	canonicalItems, err := s.listMemoryItemsByKind(ctx, userID, []string{"fact", "memory", "decision", "summary", "entity_observation"}, "", false, 10000)
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

	entities, err := s.ListEntities(ctx, userID, "", 10000)
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"memory_items":     canonicalItems,
		"conversations":    conversations,
		"sessions":         sessions,
		"notifications":    notifications,
		"proactive_events": proactive,
		"entities":         entities,
	}, nil
}

func dedupeMemoryResults(in []MemorySearchResult) []MemorySearchResult {
	if len(in) == 0 {
		return in
	}
	best := make(map[string]MemorySearchResult, len(in))
	for _, item := range in {
		key := memoryResultKey(item)
		if prev, ok := best[key]; !ok || item.Similarity > prev.Similarity {
			best[key] = item
		}
	}
	out := make([]MemorySearchResult, 0, len(best))
	for _, v := range best {
		out = append(out, v)
	}
	return out
}

func memoryResultKey(item MemorySearchResult) string {
	base := strings.TrimSpace(strings.ToLower(item.Fact + "|" + item.Memory + "|" + item.Decision + "|" + item.UserMessage + "|" + item.EntityName + "|" + item.EntitySummary))
	if base == "" {
		base = strings.TrimSpace(strings.ToLower(item.Type))
	}
	return item.Type + "|" + base
}

func normalizeMemoryKey(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	v = strings.ReplaceAll(v, "\u2019", "'")
	re := regexp.MustCompile(`[a-z0-9]+`)
	parts := re.FindAllString(v, -1)
	return strings.Join(parts, " ")
}

// Stopwords to filter out (common English words that add noise)
var stopwords = map[string]struct{}{
	"the": {}, "a": {}, "an": {}, "is": {}, "are": {}, "was": {}, "were": {},
	"be": {}, "been": {}, "being": {}, "have": {}, "has": {}, "had": {},
	"do": {}, "does": {}, "did": {}, "will": {}, "would": {}, "could": {},
	"should": {}, "may": {}, "might": {}, "must": {}, "shall": {},
	"can": {}, "need": {}, "dare": {}, "ought": {}, "used": {},
	"to": {}, "of": {}, "in": {}, "for": {}, "on": {}, "with": {}, "at": {},
	"by": {}, "from": {}, "as": {}, "into": {}, "through": {}, "during": {},
	"before": {}, "after": {}, "above": {}, "below": {}, "between": {},
	"under": {}, "again": {}, "further": {}, "then": {}, "once": {},
	"here": {}, "there": {}, "when": {}, "where": {}, "why": {}, "how": {},
	"all": {}, "each": {}, "few": {}, "more": {}, "most": {}, "other": {},
	"some": {}, "such": {}, "no": {}, "nor": {}, "not": {}, "only": {},
	"own": {}, "same": {}, "so": {}, "than": {}, "too": {}, "very": {},
	"just": {}, "also": {}, "now": {}, "i": {}, "me": {}, "my": {},
	"myself": {}, "we": {}, "our": {}, "ours": {}, "ourselves": {},
	"you": {}, "your": {}, "yours": {}, "yourself": {}, "yourselves": {},
	"he": {}, "him": {}, "his": {}, "himself": {}, "she": {}, "her": {},
	"hers": {}, "herself": {}, "it": {}, "its": {}, "itself": {},
	"they": {}, "them": {}, "their": {}, "theirs": {}, "themselves": {},
	"what": {}, "which": {}, "who": {}, "whom": {}, "this": {},
	"that": {}, "these": {}, "those": {}, "am": {}, "and": {}, "but": {},
	"if": {}, "or": {}, "because": {}, "until": {}, "while": {},
	"about": {}, "against": {}, "out": {}, "up": {}, "down": {},
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
		// Filter stopwords
		if _, isStop := stopwords[p]; isStop {
			continue
		}
		// Apply simple stemming
		p = simpleStem(p)
		if p == "" {
			continue
		}
		out[p] = struct{}{}
	}
	return out
}

// simpleStem applies basic suffix stripping for common English endings
func simpleStem(word string) string {
	// Handle common suffixes
	if len(word) > 4 && strings.HasSuffix(word, "ing") {
		return word[:len(word)-3]
	}
	if len(word) > 3 && strings.HasSuffix(word, "ed") {
		return word[:len(word)-2]
	}
	if len(word) > 3 && strings.HasSuffix(word, "es") {
		return word[:len(word)-2]
	}
	if len(word) > 3 && strings.HasSuffix(word, "s") && !strings.HasSuffix(word, "ss") {
		return word[:len(word)-1]
	}
	if len(word) > 4 && strings.HasSuffix(word, "ly") {
		return word[:len(word)-2]
	}
	if len(word) > 4 && strings.HasSuffix(word, "er") {
		return word[:len(word)-2]
	}
	if len(word) > 5 && strings.HasSuffix(word, "ness") {
		return word[:len(word)-4]
	}
	return word
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
