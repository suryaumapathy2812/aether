package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// EntityRecord represents a known entity (person, project, organization, etc.).
type EntityRecord struct {
	ID               string
	UserID           string
	EntityType       string
	Name             string
	Aliases          []string
	Summary          string
	Properties       map[string]any
	FirstSeenAt      time.Time
	LastSeenAt       time.Time
	InteractionCount int
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

// EntityObservationRecord represents an observation or trait about an entity.
type EntityObservationRecord struct {
	ID          int64
	EntityID    string
	Observation string
	Category    string
	Confidence  float64
	Source      string
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// EntityInteractionRecord represents a recorded interaction with an entity.
type EntityInteractionRecord struct {
	ID            int64
	EntityID      string
	Summary       string
	Source        string
	SourceRef     string
	InteractionAt time.Time
	CreatedAt     time.Time
}

// EntityRelationRecord represents a relationship between two entities.
type EntityRelationRecord struct {
	ID             int64
	SourceEntityID string
	Relation       string
	TargetEntityID string
	Context        string
	Confidence     float64
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

// UpsertEntity creates or updates an entity. If an entity with the same name
// (case-insensitive) and user_id exists, it merges aliases and properties.
// Returns the entity ID.
func (s *Store) UpsertEntity(ctx context.Context, userID, entityType, name string, aliases []string, properties map[string]any) (string, error) {
	if s == nil || s.db == nil {
		return "", fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return "", fmt.Errorf("entity name is required")
	}
	entityType = strings.TrimSpace(entityType)
	if entityType == "" {
		return "", fmt.Errorf("entity type is required")
	}
	if aliases == nil {
		aliases = []string{}
	}
	if properties == nil {
		properties = map[string]any{}
	}

	// Try to find existing entity by name (case-insensitive) and user_id.
	var existingID, existingAliasesJSON, existingPropsJSON string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, aliases, properties FROM entities
		WHERE user_id = ? AND LOWER(name) = LOWER(?)
		LIMIT 1
	`, userID, name).Scan(&existingID, &existingAliasesJSON, &existingPropsJSON)

	if err == nil {
		// Entity exists — merge aliases and properties, then update.
		existingAliases := []string{}
		if strings.TrimSpace(existingAliasesJSON) != "" {
			_ = json.Unmarshal([]byte(existingAliasesJSON), &existingAliases)
		}
		mergedAliases := mergeStringSlice(existingAliases, aliases)
		aliasesBlob, _ := json.Marshal(mergedAliases)

		existingProps := map[string]any{}
		if strings.TrimSpace(existingPropsJSON) != "" {
			_ = json.Unmarshal([]byte(existingPropsJSON), &existingProps)
		}
		for k, v := range properties {
			existingProps[k] = v
		}
		propsBlob, _ := json.Marshal(existingProps)

		_, err = s.db.ExecContext(ctx, `
			UPDATE entities
			SET aliases = ?, properties = ?,
				last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
				updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
			WHERE id = ?
		`, string(aliasesBlob), string(propsBlob), existingID)
		if err != nil {
			return "", err
		}
		return existingID, nil
	}

	if !errors.Is(err, sql.ErrNoRows) {
		return "", err
	}

	// Entity does not exist — insert new.
	id, err := newID()
	if err != nil {
		return "", err
	}
	aliasesBlob, _ := json.Marshal(aliases)
	propsBlob, _ := json.Marshal(properties)

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO entities(id, user_id, entity_type, name, aliases, properties)
		VALUES(?, ?, ?, ?, ?, ?)
	`, id, userID, entityType, name, string(aliasesBlob), string(propsBlob))
	if err != nil {
		return "", err
	}
	return id, nil
}

// GetEntity retrieves an entity by ID.
func (s *Store) GetEntity(ctx context.Context, entityID string) (*EntityRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	var rec EntityRecord
	var aliasesJSON, propsJSON string
	var firstSeen, lastSeen, created, updated string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, entity_type, name, aliases, summary, properties,
			first_seen_at, last_seen_at, interaction_count, created_at, updated_at
		FROM entities WHERE id = ?
	`, entityID).Scan(
		&rec.ID, &rec.UserID, &rec.EntityType, &rec.Name, &aliasesJSON, &rec.Summary, &propsJSON,
		&firstSeen, &lastSeen, &rec.InteractionCount, &created, &updated,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if err := hydrateEntityRecord(&rec, aliasesJSON, propsJSON, firstSeen, lastSeen, created, updated); err != nil {
		return nil, err
	}
	return &rec, nil
}

// FindEntityByAlias searches entities where the name or aliases JSON contains
// the alias (case-insensitive). Returns the first match or ErrNotFound.
func (s *Store) FindEntityByAlias(ctx context.Context, userID, alias string) (*EntityRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	alias = strings.TrimSpace(alias)
	if alias == "" {
		return nil, ErrNotFound
	}
	var rec EntityRecord
	var aliasesJSON, propsJSON string
	var firstSeen, lastSeen, created, updated string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, entity_type, name, aliases, summary, properties,
			first_seen_at, last_seen_at, interaction_count, created_at, updated_at
		FROM entities
		WHERE user_id = ? AND (LOWER(name) = LOWER(?) OR LOWER(aliases) LIKE '%' || LOWER(?) || '%')
		LIMIT 1
	`, userID, alias, alias).Scan(
		&rec.ID, &rec.UserID, &rec.EntityType, &rec.Name, &aliasesJSON, &rec.Summary, &propsJSON,
		&firstSeen, &lastSeen, &rec.InteractionCount, &created, &updated,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if err := hydrateEntityRecord(&rec, aliasesJSON, propsJSON, firstSeen, lastSeen, created, updated); err != nil {
		return nil, err
	}
	return &rec, nil
}

// ListEntities lists entities for a user, optionally filtered by type.
// Ordered by last_seen_at DESC. Default limit 50.
func (s *Store) ListEntities(ctx context.Context, userID, entityType string, limit int) ([]EntityRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 10000 {
		limit = 50
	}
	query := `
		SELECT id, user_id, entity_type, name, aliases, summary, properties,
			first_seen_at, last_seen_at, interaction_count, created_at, updated_at
		FROM entities
		WHERE user_id = ?
	`
	args := []any{userID}
	if strings.TrimSpace(entityType) != "" {
		query += ` AND entity_type = ?`
		args = append(args, entityType)
	}
	query += ` ORDER BY last_seen_at DESC LIMIT ?`
	args = append(args, limit)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []EntityRecord{}
	for rows.Next() {
		var rec EntityRecord
		var aliasesJSON, propsJSON string
		var firstSeen, lastSeen, created, updated string
		if err := rows.Scan(
			&rec.ID, &rec.UserID, &rec.EntityType, &rec.Name, &aliasesJSON, &rec.Summary, &propsJSON,
			&firstSeen, &lastSeen, &rec.InteractionCount, &created, &updated,
		); err != nil {
			return nil, err
		}
		if err := hydrateEntityRecord(&rec, aliasesJSON, propsJSON, firstSeen, lastSeen, created, updated); err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

// SearchEntities searches entities for a user using tokenized query matching
// against name, summary, and aliases text. Returns top matches sorted by score.
func (s *Store) SearchEntities(ctx context.Context, userID, query string, limit int) ([]EntityRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 100 {
		limit = 5
	}
	qTokens := tokenizeQuery(query)
	if len(qTokens) == 0 {
		return []EntityRecord{}, nil
	}

	// Load all entities for user and score them.
	entities, err := s.ListEntities(ctx, userID, "", 1000)
	if err != nil {
		return nil, err
	}

	type scored struct {
		entity EntityRecord
		score  float64
	}
	var matches []scored
	for _, ent := range entities {
		text := ent.Name + " " + ent.Summary + " " + strings.Join(ent.Aliases, " ")
		score := overlapScore(qTokens, tokenizeQuery(text))
		if score > 0 {
			matches = append(matches, scored{entity: ent, score: score})
		}
	}

	// Sort by score descending.
	for i := 0; i < len(matches); i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[j].score > matches[i].score {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}
	if len(matches) > limit {
		matches = matches[:limit]
	}
	out := make([]EntityRecord, 0, len(matches))
	for _, m := range matches {
		out = append(out, m.entity)
	}
	return out, nil
}

// UpdateEntitySummary updates the summary field of an entity.
func (s *Store) UpdateEntitySummary(ctx context.Context, entityID, summary string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE entities
		SET summary = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, summary, entityID)
	if err != nil {
		return err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if affected == 0 {
		return ErrNotFound
	}
	return nil
}

// AddEntityObservation upserts an observation for an entity. Uses
// normalizeMemoryKey(observation) as the observation_key for de-duplication.
func (s *Store) AddEntityObservation(ctx context.Context, entityID, userID, observation, category, source string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	observation = strings.TrimSpace(observation)
	if observation == "" {
		return nil
	}
	if strings.TrimSpace(category) == "" {
		category = "trait"
	}
	if strings.TrimSpace(source) == "" {
		source = "extracted"
	}
	key := normalizeMemoryKey(observation)
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO entity_observations(entity_id, user_id, observation, observation_key, category, source)
		VALUES(?, ?, ?, ?, ?, ?)
		ON CONFLICT(entity_id, observation_key) DO UPDATE SET
			observation = excluded.observation,
			category = excluded.category,
			source = excluded.source,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
	`, entityID, userID, observation, key, category, source)
	return err
}

// ListEntityObservations lists observations for an entity, ordered by
// updated_at DESC. Default limit 50.
func (s *Store) ListEntityObservations(ctx context.Context, entityID string, limit int) ([]EntityObservationRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if limit <= 0 || limit > 2000 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, entity_id, observation, category, confidence, source, created_at, updated_at
		FROM entity_observations
		WHERE entity_id = ?
		ORDER BY updated_at DESC
		LIMIT ?
	`, entityID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []EntityObservationRecord{}
	for rows.Next() {
		var rec EntityObservationRecord
		var created, updated string
		if err := rows.Scan(&rec.ID, &rec.EntityID, &rec.Observation, &rec.Category, &rec.Confidence, &rec.Source, &created, &updated); err != nil {
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
		rec.CreatedAt = createdTS
		rec.UpdatedAt = updatedTS
		out = append(out, rec)
	}
	return out, rows.Err()
}

// AddEntityInteraction records an interaction with an entity and touches the entity.
func (s *Store) AddEntityInteraction(ctx context.Context, entityID, userID, summary, source, sourceRef string, interactionAt time.Time) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	var srcRef sql.NullString
	if strings.TrimSpace(sourceRef) != "" {
		srcRef = sql.NullString{String: sourceRef, Valid: true}
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO entity_interactions(entity_id, user_id, summary, source, source_ref, interaction_at)
		VALUES(?, ?, ?, ?, ?, ?)
	`, entityID, userID, summary, source, srcRef, formatTS(interactionAt.UTC()))
	if err != nil {
		return err
	}
	return s.TouchEntity(ctx, entityID)
}

// ListEntityInteractions lists interactions for an entity, ordered by
// interaction_at DESC. Default limit 50.
func (s *Store) ListEntityInteractions(ctx context.Context, entityID string, limit int) ([]EntityInteractionRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if limit <= 0 || limit > 2000 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, entity_id, summary, source, source_ref, interaction_at, created_at
		FROM entity_interactions
		WHERE entity_id = ?
		ORDER BY interaction_at DESC
		LIMIT ?
	`, entityID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []EntityInteractionRecord{}
	for rows.Next() {
		var rec EntityInteractionRecord
		var srcRef sql.NullString
		var interactionAt, created string
		if err := rows.Scan(&rec.ID, &rec.EntityID, &rec.Summary, &rec.Source, &srcRef, &interactionAt, &created); err != nil {
			return nil, err
		}
		if srcRef.Valid {
			rec.SourceRef = srcRef.String
		}
		interactionTS, err := parseTS(interactionAt)
		if err != nil {
			return nil, err
		}
		createdTS, err := parseTS(created)
		if err != nil {
			return nil, err
		}
		rec.InteractionAt = interactionTS
		rec.CreatedAt = createdTS
		out = append(out, rec)
	}
	return out, rows.Err()
}

// AddEntityRelation upserts a relationship between two entities.
func (s *Store) AddEntityRelation(ctx context.Context, userID, sourceEntityID, relation, targetEntityID, relContext string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO entity_relations(user_id, source_entity_id, relation, target_entity_id, context)
		VALUES(?, ?, ?, ?, ?)
		ON CONFLICT(source_entity_id, relation, target_entity_id) DO UPDATE SET
			context = excluded.context,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
	`, userID, sourceEntityID, relation, targetEntityID, relContext)
	return err
}

// ListEntityRelations returns all relations where the entity is source OR target.
func (s *Store) ListEntityRelations(ctx context.Context, entityID string) ([]EntityRelationRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, source_entity_id, relation, target_entity_id, context, confidence, created_at, updated_at
		FROM entity_relations
		WHERE source_entity_id = ? OR target_entity_id = ?
		ORDER BY updated_at DESC
	`, entityID, entityID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []EntityRelationRecord{}
	for rows.Next() {
		var rec EntityRelationRecord
		var created, updated string
		if err := rows.Scan(&rec.ID, &rec.SourceEntityID, &rec.Relation, &rec.TargetEntityID, &rec.Context, &rec.Confidence, &created, &updated); err != nil {
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
		rec.CreatedAt = createdTS
		rec.UpdatedAt = updatedTS
		out = append(out, rec)
	}
	return out, rows.Err()
}

// DeleteEntity deletes an entity by ID. Foreign keys cascade to observations,
// interactions, and relations.
func (s *Store) DeleteEntity(ctx context.Context, entityID string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	res, err := s.db.ExecContext(ctx, `DELETE FROM entities WHERE id = ?`, entityID)
	if err != nil {
		return err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if affected == 0 {
		return ErrNotFound
	}
	return nil
}

// TouchEntity updates last_seen_at, increments interaction_count, and sets updated_at.
func (s *Store) TouchEntity(ctx context.Context, entityID string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	_, err := s.db.ExecContext(ctx, `
		UPDATE entities
		SET last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
			interaction_count = interaction_count + 1,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, entityID)
	return err
}

// hydrateEntityRecord parses JSON and timestamp fields into an EntityRecord.
func hydrateEntityRecord(rec *EntityRecord, aliasesJSON, propsJSON, firstSeen, lastSeen, created, updated string) error {
	rec.Aliases = []string{}
	if strings.TrimSpace(aliasesJSON) != "" {
		_ = json.Unmarshal([]byte(aliasesJSON), &rec.Aliases)
	}
	rec.Properties = map[string]any{}
	if strings.TrimSpace(propsJSON) != "" {
		_ = json.Unmarshal([]byte(propsJSON), &rec.Properties)
	}
	firstSeenTS, err := parseTS(firstSeen)
	if err != nil {
		return err
	}
	lastSeenTS, err := parseTS(lastSeen)
	if err != nil {
		return err
	}
	createdTS, err := parseTS(created)
	if err != nil {
		return err
	}
	updatedTS, err := parseTS(updated)
	if err != nil {
		return err
	}
	rec.FirstSeenAt = firstSeenTS
	rec.LastSeenAt = lastSeenTS
	rec.CreatedAt = createdTS
	rec.UpdatedAt = updatedTS
	return nil
}

// mergeStringSlice merges two string slices, de-duplicating case-insensitively.
func mergeStringSlice(existing, incoming []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(existing)+len(incoming))
	for _, v := range existing {
		lower := strings.ToLower(strings.TrimSpace(v))
		if lower != "" && !seen[lower] {
			seen[lower] = true
			out = append(out, v)
		}
	}
	for _, v := range incoming {
		lower := strings.ToLower(strings.TrimSpace(v))
		if lower != "" && !seen[lower] {
			seen[lower] = true
			out = append(out, v)
		}
	}
	return out
}
