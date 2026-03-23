package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

type MemoryItemRecord struct {
	ID            int64      `json:"id"`
	UserID        string     `json:"user_id"`
	Kind          string     `json:"kind"`
	Category      string     `json:"category"`
	Content       string     `json:"content"`
	NormalizedKey string     `json:"normalized_key"`
	Status        string     `json:"status"`
	Confidence    float64    `json:"confidence"`
	Importance    float64    `json:"importance"`
	EvidenceCount int        `json:"evidence_count"`
	FirstSeenAt   time.Time  `json:"first_seen_at"`
	LastSeenAt    time.Time  `json:"last_seen_at"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
	SourceType    string     `json:"source_type"`
	SourceID      string     `json:"source_id"`
	SessionID     string     `json:"session_id"`
	MetadataJSON  string     `json:"metadata_json"`
}

type AddMemoryInput struct {
	UserID     string
	Kind       string
	Category   string
	Content    string
	Status     string
	Confidence float64
	Importance float64
	ExpiresAt  *time.Time
	SourceType string
	SourceID   string
	SessionID  string
	Metadata   map[string]any
	Embedding  []float32
	ObservedAt time.Time
}

type MemoryListQuery struct {
	UserID   string
	Kinds    []string
	Category string
	Status   string
	Limit    int
}

type MemorySearchQuery struct {
	UserID         string
	Text           string
	Kinds          []string
	Category       string
	Status         string
	Limit          int
	QueryEmbedding []float32
}

func (s *Store) AddMemory(ctx context.Context, input AddMemoryInput) (int64, error) {
	if s == nil || s.db == nil {
		return 0, fmt.Errorf("store not initialized")
	}
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = "default"
	}
	kind := strings.TrimSpace(input.Kind)
	content := strings.TrimSpace(input.Content)
	if kind == "" || content == "" {
		return 0, nil
	}
	category := strings.TrimSpace(input.Category)
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	confidence := input.Confidence
	if confidence <= 0 {
		confidence = 0.8
	}
	importance := input.Importance
	if importance <= 0 {
		importance = defaultImportanceForKind(kind)
	}
	now := input.ObservedAt.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	normalizedKey := normalizeMemoryKey(content)
	metadataJSON, err := marshalJSONMap(input.Metadata)
	if err != nil {
		return 0, err
	}
	vectorJSON, hasEmbedding, err := encodeVectorJSON(input.Embedding)
	if err != nil {
		return 0, err
	}
	if !s.vectorEnabled {
		hasEmbedding = false
		vectorJSON = ""
	}

	existing, err := s.getMemoryItemByKey(ctx, userID, kind, normalizedKey)
	if err != nil && err != ErrNotFound {
		return 0, err
	}
	if existing == nil {
		existing, err = s.findMemoryItemMergeCandidate(ctx, userID, kind, content)
		if err != nil {
			return 0, err
		}
	}

	if existing != nil {
		content = preferRicherContent(existing.Content, content)
		if category == "" {
			category = existing.Category
		}
		if status == "" {
			status = existing.Status
		}
		if input.SourceType == "" {
			input.SourceType = existing.SourceType
		}
		if input.SourceID == "" {
			input.SourceID = existing.SourceID
		}
		if input.SessionID == "" {
			input.SessionID = existing.SessionID
		}
		if confidence < existing.Confidence {
			confidence = existing.Confidence
		}
		if importance < existing.Importance {
			importance = existing.Importance
		}
		query := `
			UPDATE memory_items
			SET category = ?, content = ?, status = ?, confidence = ?, importance = ?,
				evidence_count = evidence_count + 1, last_seen_at = ?, updated_at = ?,
				expires_at = ?, source_type = ?, source_id = ?, session_id = ?, metadata_json = ?`
		args := []any{category, content, status, confidence, importance, formatTS(now), formatTS(now), nullableTS(input.ExpiresAt), strings.TrimSpace(input.SourceType), strings.TrimSpace(input.SourceID), strings.TrimSpace(input.SessionID), metadataJSON}
		if hasEmbedding {
			query += `, embedding = vector32(?)`
			args = append(args, vectorJSON)
		}
		query += ` WHERE id = ?`
		args = append(args, existing.ID)
		if _, err := s.db.ExecContext(ctx, query, args...); err != nil {
			return 0, err
		}
		if err := s.upsertMemoryItemFTS(ctx, existing.ID, userID, kind, category, content, status); err != nil {
			return 0, err
		}
		return existing.ID, nil
	}

	query := `
		INSERT INTO memory_items(
			user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at, expires_at,
			source_type, source_id, session_id, metadata_json`
	args := []any{userID, kind, category, content, normalizedKey, status, confidence, importance, 1, formatTS(now), formatTS(now), formatTS(now), formatTS(now), nullableTS(input.ExpiresAt), strings.TrimSpace(input.SourceType), strings.TrimSpace(input.SourceID), strings.TrimSpace(input.SessionID), metadataJSON}
	if hasEmbedding {
		query += `, embedding`
	}
	query += `) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?`
	if hasEmbedding {
		query += `, vector32(?)`
		args = append(args, vectorJSON)
	}
	query += `)`
	res, err := s.db.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	if err := s.upsertMemoryItemFTS(ctx, id, userID, kind, category, content, status); err != nil {
		return 0, err
	}
	return id, nil
}

func (s *Store) SearchMemory(ctx context.Context, query MemorySearchQuery) ([]MemorySearchResult, error) {
	userID := strings.TrimSpace(query.UserID)
	if userID == "" {
		userID = "default"
	}
	limit := query.Limit
	if limit <= 0 || limit > 100 {
		limit = 12
	}
	text := strings.TrimSpace(query.Text)
	if text == "" {
		return []MemorySearchResult{}, nil
	}
	results := make([]MemorySearchResult, 0, limit*2)

	ftsResults, err := s.searchMemoryItemsFTS(ctx, userID, text, limit*3)
	if err != nil {
		return nil, err
	}
	results = append(results, filterMemorySearchResults(ftsResults, query)...)

	if len(query.QueryEmbedding) > 0 {
		vectorResults, err := s.searchMemoryItemsVector(ctx, userID, query.QueryEmbedding, limit*6)
		if err != nil {
			return nil, err
		}
		results = append(results, filterMemorySearchResults(vectorResults, query)...)
	}

	results = dedupeMemoryResults(results)
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

func (s *Store) ListMemoryItems(ctx context.Context, query MemoryListQuery) ([]MemoryItemRecord, error) {
	items, err := s.listMemoryItemsByKind(ctx, query.UserID, normalizeKinds(query.Kinds), query.Category, false, query.Limit)
	if err != nil {
		return nil, err
	}
	status := strings.TrimSpace(strings.ToLower(query.Status))
	if status == "" || status == "all" {
		return items, nil
	}
	filtered := make([]MemoryItemRecord, 0, len(items))
	for _, item := range items {
		if strings.EqualFold(item.Status, status) {
			filtered = append(filtered, item)
		}
	}
	return filtered, nil
}

func (s *Store) DeleteMemoryItem(ctx context.Context, id int64) error {
	if id == 0 {
		return nil
	}
	if _, err := s.db.ExecContext(ctx, `DELETE FROM memory_items WHERE id = ?`, id); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `DELETE FROM memory_items_fts WHERE item_id = ?`, id)
	return err
}

func (s *Store) searchMemoryItemsFTS(ctx context.Context, userID, query string, limit int) ([]MemorySearchResult, error) {
	tokens := tokenizeQuery(query)
	if len(tokens) == 0 {
		return nil, nil
	}
	parts := make([]string, 0, len(tokens))
	for token := range tokens {
		parts = append(parts, token)
	}
	sort.Strings(parts)
	ftsQuery := strings.Join(parts, " OR ")
	rows, err := s.db.QueryContext(ctx, `
		SELECT mi.kind, mi.category, mi.content, mi.confidence, mi.importance, mi.updated_at
		FROM memory_items_fts mf
		JOIN memory_items mi ON mi.id = mf.item_id
		WHERE mf.user_id = ? AND memory_items_fts MATCH ? AND mi.status = 'active'
		LIMIT ?
	`, userID, ftsQuery, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]MemorySearchResult, 0)
	for rows.Next() {
		var kind, category, content, updatedAt string
		var confidence, importance float64
		if err := rows.Scan(&kind, &category, &content, &confidence, &importance, &updatedAt); err != nil {
			return nil, err
		}
		ts, err := parseTS(updatedAt)
		if err != nil {
			return nil, err
		}
		results = append(results, toMemorySearchResult(kind, category, content, 0.5+importance*0.2, confidence, ts))
	}
	return results, rows.Err()
}

func (s *Store) searchMemoryItemsVector(ctx context.Context, userID string, queryEmbedding []float32, limit int) ([]MemorySearchResult, error) {
	vectorJSON, ok, err := encodeVectorJSON(queryEmbedding)
	if err != nil || !ok {
		return nil, err
	}
	if !s.vectorEnabled {
		return nil, nil
	}
	overfetch := limit * 10
	if overfetch < 20 {
		overfetch = 20
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT mi.kind, mi.category, mi.content, mi.confidence, mi.importance, mi.updated_at,
			vector_distance_cos(mi.embedding, vector32(?)) AS distance
		FROM vector_top_k('idx_memory_items_vec', vector32(?), ?)
		JOIN memory_items mi ON mi.id = id
		WHERE mi.user_id = ? AND mi.status = 'active'
		LIMIT ?
	`, vectorJSON, vectorJSON, overfetch, userID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]MemorySearchResult, 0)
	for rows.Next() {
		var kind, category, content, updatedAt string
		var confidence, importance, distance float64
		if err := rows.Scan(&kind, &category, &content, &confidence, &importance, &updatedAt, &distance); err != nil {
			return nil, err
		}
		ts, err := parseTS(updatedAt)
		if err != nil {
			return nil, err
		}
		similarity := math.Max(0, 1.15-distance) + importance*0.15
		results = append(results, toMemorySearchResult(kind, category, content, similarity, confidence, ts))
	}
	return results, rows.Err()
}

func (s *Store) findMemoryItemMergeCandidate(ctx context.Context, userID, kind, content string) (*MemoryItemRecord, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at, expires_at,
			source_type, source_id, session_id, metadata_json
		FROM memory_items
		WHERE user_id = ? AND kind = ? AND status = 'active'
		ORDER BY updated_at DESC
		LIMIT 50
	`, userID, kind)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	target := strings.ToLower(strings.TrimSpace(content))
	targetTokens := tokenizeQuery(target)
	var best *MemoryItemRecord
	bestScore := 0.0
	for rows.Next() {
		rec, err := scanMemoryItem(rows)
		if err != nil {
			return nil, err
		}
		existingLower := strings.ToLower(strings.TrimSpace(rec.Content))
		if existingLower == target {
			return rec, nil
		}
		if strings.Contains(existingLower, target) || strings.Contains(target, existingLower) {
			return rec, nil
		}
		score := overlapScore(targetTokens, tokenizeQuery(existingLower))
		if score > bestScore {
			bestScore = score
			best = rec
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if bestScore >= 0.6 {
		return best, nil
	}
	return nil, nil
}

func (s *Store) getMemoryItemByKey(ctx context.Context, userID, kind, normalizedKey string) (*MemoryItemRecord, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at, expires_at,
			source_type, source_id, session_id, metadata_json
		FROM memory_items
		WHERE user_id = ? AND kind = ? AND normalized_key = ?
	`, userID, kind, normalizedKey)
	return scanMemoryItem(row)
}

func scanMemoryItem(scanner interface{ Scan(dest ...any) error }) (*MemoryItemRecord, error) {
	var rec MemoryItemRecord
	var firstSeenAt, lastSeenAt, createdAt, updatedAt string
	var expiresAt sql.NullString
	if err := scanner.Scan(&rec.ID, &rec.UserID, &rec.Kind, &rec.Category, &rec.Content, &rec.NormalizedKey, &rec.Status, &rec.Confidence, &rec.Importance, &rec.EvidenceCount, &firstSeenAt, &lastSeenAt, &createdAt, &updatedAt, &expiresAt, &rec.SourceType, &rec.SourceID, &rec.SessionID, &rec.MetadataJSON); err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrNotFound
		}
		return nil, err
	}
	var err error
	if rec.FirstSeenAt, err = parseTS(firstSeenAt); err != nil {
		return nil, err
	}
	if rec.LastSeenAt, err = parseTS(lastSeenAt); err != nil {
		return nil, err
	}
	if rec.CreatedAt, err = parseTS(createdAt); err != nil {
		return nil, err
	}
	if rec.UpdatedAt, err = parseTS(updatedAt); err != nil {
		return nil, err
	}
	if expiresAt.Valid && strings.TrimSpace(expiresAt.String) != "" {
		ts, err := parseTS(expiresAt.String)
		if err != nil {
			return nil, err
		}
		rec.ExpiresAt = &ts
	}
	return &rec, nil
}

func (s *Store) upsertMemoryItemFTS(ctx context.Context, itemID int64, userID, kind, category, content, status string) error {
	if status != "active" {
		_, err := s.db.ExecContext(ctx, `DELETE FROM memory_items_fts WHERE item_id = ?`, itemID)
		return err
	}
	if _, err := s.db.ExecContext(ctx, `DELETE FROM memory_items_fts WHERE item_id = ?`, itemID); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO memory_items_fts(user_id, item_id, kind, category, content)
		VALUES(?, ?, ?, ?, ?)
	`, userID, itemID, kind, category, content)
	return err
}

func encodeVectorJSON(embedding []float32) (string, bool, error) {
	if len(embedding) == 0 {
		return "", false, nil
	}
	if len(embedding) != defaultMemoryEmbeddingDimensions {
		return "", false, nil
	}
	b, err := json.Marshal(embedding)
	if err != nil {
		return "", false, err
	}
	return string(b), true, nil
}

func marshalJSONMap(v map[string]any) (string, error) {
	if len(v) == 0 {
		return "{}", nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func nullableTS(v *time.Time) any {
	if v == nil {
		return nil
	}
	return formatTS(v.UTC())
}

func preferRicherContent(existing, incoming string) string {
	existing = strings.TrimSpace(existing)
	incoming = strings.TrimSpace(incoming)
	if existing == "" {
		return incoming
	}
	if incoming == "" {
		return existing
	}
	if len(incoming) > len(existing) {
		return incoming
	}
	return existing
}

func defaultImportanceForKind(kind string) float64 {
	switch kind {
	case "decision":
		return 0.95
	case "fact":
		return 0.85
	case "summary":
		return 0.8
	case "entity_observation":
		return 0.7
	default:
		return 0.65
	}
}

func normalizeKinds(kinds []string) []string {
	if len(kinds) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(kinds))
	out := make([]string, 0, len(kinds))
	for _, kind := range kinds {
		kind = strings.TrimSpace(strings.ToLower(kind))
		if kind == "" {
			continue
		}
		if _, ok := seen[kind]; ok {
			continue
		}
		seen[kind] = struct{}{}
		out = append(out, kind)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func filterMemorySearchResults(results []MemorySearchResult, query MemorySearchQuery) []MemorySearchResult {
	if len(results) == 0 {
		return results
	}
	kinds := normalizeKinds(query.Kinds)
	kindSet := make(map[string]struct{}, len(kinds))
	for _, kind := range kinds {
		kindSet[kind] = struct{}{}
	}
	category := strings.TrimSpace(strings.ToLower(query.Category))
	status := strings.TrimSpace(strings.ToLower(query.Status))
	out := make([]MemorySearchResult, 0, len(results))
	for _, item := range results {
		if len(kindSet) > 0 {
			if _, ok := kindSet[strings.ToLower(item.Type)]; !ok {
				continue
			}
		}
		if category != "" && !strings.EqualFold(item.Category, category) {
			continue
		}
		if status != "" && status != "all" && !strings.EqualFold(status, "active") {
			continue
		}
		out = append(out, item)
	}
	return out
}

func toMemorySearchResult(kind, category, content string, similarity, confidence float64, ts time.Time) MemorySearchResult {
	item := MemorySearchResult{Type: kind, Category: category, Similarity: similarity, Confidence: confidence, Timestamp: ts}
	switch kind {
	case "fact":
		item.Fact = content
	case "decision":
		item.Decision = content
	case "summary":
		item.Summary = content
	case "entity_observation":
		item.EntitySummary = content
	default:
		item.Type = "memory"
		item.Memory = content
	}
	return item
}

func (s *Store) ListMemoryUsers(ctx context.Context) ([]string, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `SELECT DISTINCT user_id FROM memory_items ORDER BY user_id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	users := make([]string, 0)
	for rows.Next() {
		var userID string
		if err := rows.Scan(&userID); err != nil {
			return nil, err
		}
		userID = strings.TrimSpace(userID)
		if userID == "" {
			continue
		}
		users = append(users, userID)
	}
	if len(users) == 0 {
		users = append(users, "default")
	}
	return users, rows.Err()
}

func (s *Store) ArchiveStaleMemoryItems(ctx context.Context, userID string, before time.Time, keepDecisions bool) (int64, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	query := `
		UPDATE memory_items
		SET status = 'archived', updated_at = ?
		WHERE user_id = ? AND status = 'active' AND last_seen_at < ?
		  AND kind IN ('fact', 'memory', 'summary', 'entity_observation')
		  AND evidence_count <= 1 AND confidence < 0.9
	`
	if !keepDecisions {
		query = `
			UPDATE memory_items
			SET status = 'archived', updated_at = ?
			WHERE user_id = ? AND status = 'active' AND last_seen_at < ?
			  AND kind IN ('fact', 'memory', 'summary', 'entity_observation', 'decision')
			  AND evidence_count <= 1 AND confidence < 0.9
		`
	}
	res, err := s.db.ExecContext(ctx, query, formatTS(time.Now().UTC()), userID, formatTS(before.UTC()))
	if err != nil {
		return 0, err
	}
	count, err := res.RowsAffected()
	if err != nil {
		return 0, err
	}
	if count > 0 {
		if err := s.rebuildMemoryItemsFTS(ctx); err != nil {
			return count, err
		}
	}
	return count, nil
}

func (s *Store) listMemoryItemsByKind(ctx context.Context, userID string, kinds []string, category string, activeOnly bool, limit int) ([]MemoryItemRecord, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 10000 {
		limit = 100
	}
	query := `
		SELECT id, user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at, expires_at,
			source_type, source_id, session_id, metadata_json
		FROM memory_items
		WHERE user_id = ?
	`
	args := []any{userID}
	if len(kinds) > 0 {
		placeholders := strings.TrimRight(strings.Repeat("?,", len(kinds)), ",")
		query += ` AND kind IN (` + placeholders + `)`
		for _, kind := range kinds {
			args = append(args, kind)
		}
	}
	if activeOnly {
		query += ` AND status = 'active'`
	}
	if strings.TrimSpace(category) != "" {
		query += ` AND category = ?`
		args = append(args, category)
	}
	query += ` ORDER BY updated_at DESC LIMIT ?`
	args = append(args, limit)
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := make([]MemoryItemRecord, 0)
	for rows.Next() {
		rec, err := scanMemoryItem(rows)
		if err != nil {
			return nil, err
		}
		items = append(items, *rec)
	}
	return items, rows.Err()
}

func (s *Store) backfillLegacyMemoryItems(ctx context.Context) error {
	exists, err := s.tableExists(ctx, "facts")
	if err != nil {
		return err
	}
	if exists {
		if err := s.backfillFacts(ctx); err != nil {
			return err
		}
	}
	exists, err = s.tableExists(ctx, "memories")
	if err != nil {
		return err
	}
	if exists {
		if err := s.backfillMemories(ctx); err != nil {
			return err
		}
	}
	exists, err = s.tableExists(ctx, "decisions")
	if err != nil {
		return err
	}
	if exists {
		if err := s.backfillDecisions(ctx); err != nil {
			return err
		}
	}
	exists, err = s.tableExists(ctx, "sessions")
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	if err := s.backfillSessionSummaries(ctx); err != nil {
		return err
	}
	return s.rebuildMemoryItemsFTS(ctx)
}

func (s *Store) backfillFacts(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO memory_items(
			user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at, source_type, source_id, session_id, metadata_json
		)
		SELECT user_id, 'fact', 'profile', fact, fact_key, 'active', 1.0, 0.85,
			1, created_at, updated_at, created_at, updated_at,
			'legacy_fact', COALESCE(CAST(source_conversation_id AS TEXT), ''), '', '{}'
		FROM facts
	`)
	return err
}

func (s *Store) backfillMemories(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO memory_items(
			user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at, expires_at,
			source_type, source_id, session_id, metadata_json
		)
		SELECT user_id, 'memory', category, memory, memory_key, 'active', confidence, 0.65,
			1, created_at, created_at, created_at, created_at, expires_at,
			'legacy_memory', COALESCE(CAST(source_conversation_id AS TEXT), ''), '', '{}'
		FROM memories
	`)
	return err
}

func (s *Store) backfillDecisions(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO memory_items(
			user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at,
			source_type, source_id, session_id, metadata_json
		)
		SELECT user_id, 'decision', category, decision, decision_key,
			CASE WHEN active = 1 THEN 'active' ELSE 'superseded' END,
			confidence, 0.95, 1, created_at, updated_at, created_at, updated_at,
			source, COALESCE(CAST(source_conversation_id AS TEXT), ''), '', '{}'
		FROM decisions
	`)
	return err
}

func (s *Store) backfillSessionSummaries(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO memory_items(
			user_id, kind, category, content, normalized_key, status, confidence, importance,
			evidence_count, first_seen_at, last_seen_at, created_at, updated_at,
			source_type, source_id, session_id, metadata_json
		)
		SELECT user_id, 'summary', 'session', summary, 'session:' || session_id || ':' || started_at,
			'active', 0.9, 0.8, 1, started_at, ended_at, started_at, ended_at,
			'session', session_id, session_id, json_object('turns', turns, 'tools_used', tools_used)
		FROM sessions
		WHERE trim(summary) <> ''
	`)
	return err
}
