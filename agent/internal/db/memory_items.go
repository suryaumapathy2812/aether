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
	ID            int64
	UserID        string
	Kind          string
	Category      string
	Content       string
	NormalizedKey string
	Status        string
	Confidence    float64
	Importance    float64
	EvidenceCount int
	FirstSeenAt   time.Time
	LastSeenAt    time.Time
	CreatedAt     time.Time
	UpdatedAt     time.Time
	ExpiresAt     *time.Time
	SourceType    string
	SourceID      string
	SessionID     string
	MetadataJSON  string
}

type MemoryItemUpsert struct {
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

func (s *Store) UpsertMemoryItem(ctx context.Context, input MemoryItemUpsert) (int64, error) {
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

func (s *Store) SearchMemoryHybrid(ctx context.Context, userID, query string, queryEmbedding []float32, limit int) ([]MemorySearchResult, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 100 {
		limit = 12
	}
	results := make([]MemorySearchResult, 0, limit*2)
	query = strings.TrimSpace(query)
	if query == "" {
		return results, nil
	}

	ftsResults, err := s.searchMemoryItemsFTS(ctx, userID, query, limit*3)
	if err != nil {
		return nil, err
	}
	results = append(results, ftsResults...)

	if len(queryEmbedding) > 0 {
		vectorResults, err := s.searchMemoryItemsVector(ctx, userID, queryEmbedding, limit*6)
		if err != nil {
			return nil, err
		}
		results = append(results, vectorResults...)
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
	rows, err := s.db.QueryContext(ctx, `
		SELECT DISTINCT user_id FROM (
			SELECT user_id FROM memory_items
			UNION
			SELECT user_id FROM facts
			UNION
			SELECT user_id FROM memories
			UNION
			SELECT user_id FROM decisions
		)
		ORDER BY user_id
	`)
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
