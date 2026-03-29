package db

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

type ChatSession struct {
	ID              string     `json:"id"`
	UserID          string     `json:"user_id"`
	Title           string     `json:"title"`
	Archived        bool       `json:"archived"`
	LastActivityAt  *time.Time `json:"last_activity_at,omitempty"`
	LatestSummaryID *int64     `json:"latest_summary_id,omitempty"`
	SummaryPreview  string     `json:"summary_preview"`
	SummaryCount    int        `json:"summary_count"`
	TitleSource     string     `json:"title_source"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
}

type ChatSessionSummary struct {
	ID              int64     `json:"id"`
	SessionID       string    `json:"session_id"`
	UserID          string    `json:"user_id"`
	Revision        int       `json:"revision"`
	SummaryText     string    `json:"summary_text"`
	TitleSuggestion string    `json:"title_suggestion"`
	MessageCount    int       `json:"message_count"`
	CreatedAt       time.Time `json:"created_at"`
}

func generateSessionID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func (s *Store) CreateChatSession(ctx context.Context, userID, title string) (ChatSession, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	id := generateSessionID()
	now := formatTS(time.Now().UTC())
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO chat_sessions(id, user_id, title, last_activity_at, created_at, updated_at, title_source)
		VALUES(?, ?, ?, ?, ?, ?, 'seed')
	`, id, userID, title, now, now, now)
	if err != nil {
		return ChatSession{}, fmt.Errorf("create chat session: %w", err)
	}
	return s.GetChatSession(ctx, id)
}

func (s *Store) GetChatSession(ctx context.Context, sessionID string) (ChatSession, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, title, archived, last_activity_at, latest_summary_id, summary_preview, summary_count, title_source, created_at, updated_at
		FROM chat_sessions WHERE id = ?
	`, sessionID)
	rec, err := scanChatSession(row)
	if err != nil {
		return ChatSession{}, fmt.Errorf("get chat session: %w", err)
	}
	return *rec, nil
}

func (s *Store) ListChatSessions(ctx context.Context, userID string, limit int) ([]ChatSession, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, title, archived, last_activity_at, latest_summary_id, summary_preview, summary_count, title_source, created_at, updated_at
		FROM chat_sessions
		WHERE user_id = ? AND archived = 0
		ORDER BY updated_at DESC
		LIMIT ?
	`, userID, limit)
	if err != nil {
		return nil, fmt.Errorf("list chat sessions: %w", err)
	}
	defer rows.Close()
	out := []ChatSession{}
	for rows.Next() {
		rec, err := scanChatSession(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *rec)
	}
	return out, rows.Err()
}

func (s *Store) UpdateChatSessionTitle(ctx context.Context, sessionID, title string) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions SET title = ?, title_source = 'manual', updated_at = ? WHERE id = ?
	`, title, formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) TouchChatSession(ctx context.Context, sessionID string) error {
	return s.TouchChatSessionActivity(ctx, sessionID)
}

func (s *Store) TouchChatSessionActivity(ctx context.Context, sessionID string) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?
	`, formatTS(time.Now().UTC()), formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) SetChatSessionTitleAuto(ctx context.Context, sessionID, title string) error {
	title = strings.TrimSpace(title)
	if title == "" {
		return nil
	}
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions
		SET title = ?, title_source = 'auto', updated_at = ?
		WHERE id = ? AND title_source != 'manual'
	`, title, formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) UpdateChatSessionSummaryMeta(ctx context.Context, sessionID string, summaryID int64, preview string, count int) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions
		SET latest_summary_id = ?, summary_preview = ?, summary_count = ?, updated_at = ?
		WHERE id = ?
	`, nullableSummaryID(summaryID), strings.TrimSpace(preview), count, formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) AddChatSessionSummary(ctx context.Context, sessionID, userID, summaryText, titleSuggestion string, messageCount int) (int64, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" || strings.TrimSpace(summaryText) == "" {
		return 0, fmt.Errorf("session id and summary text are required")
	}
	revision := 1
	row := s.db.QueryRowContext(ctx, `SELECT COALESCE(MAX(revision), 0) + 1 FROM chat_session_summaries WHERE session_id = ?`, sessionID)
	if err := row.Scan(&revision); err != nil {
		return 0, err
	}
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO chat_session_summaries(session_id, user_id, revision, summary_text, title_suggestion, message_count, created_at)
		VALUES(?, ?, ?, ?, ?, ?, ?)
	`, sessionID, userID, revision, strings.TrimSpace(summaryText), strings.TrimSpace(titleSuggestion), messageCount, formatTS(time.Now().UTC()))
	if err != nil {
		return 0, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	if err := s.UpdateChatSessionSummaryMeta(ctx, sessionID, id, deriveSummaryPreview(summaryText), revision); err != nil {
		return 0, err
	}
	return id, nil
}

func (s *Store) GetLatestSessionSummary(ctx context.Context, sessionID string) (*ChatSessionSummary, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, session_id, user_id, revision, summary_text, title_suggestion, message_count, created_at
		FROM chat_session_summaries
		WHERE session_id = ?
		ORDER BY revision DESC
		LIMIT 1
	`, sessionID)
	return scanChatSessionSummary(row)
}

func (s *Store) ListSessionSummaries(ctx context.Context, sessionID string, limit int) ([]ChatSessionSummary, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, session_id, user_id, revision, summary_text, title_suggestion, message_count, created_at
		FROM chat_session_summaries
		WHERE session_id = ?
		ORDER BY revision DESC
		LIMIT ?
	`, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	summaries := make([]ChatSessionSummary, 0)
	for rows.Next() {
		item, err := scanChatSessionSummary(rows)
		if err != nil {
			return nil, err
		}
		summaries = append(summaries, *item)
	}
	return summaries, rows.Err()
}

func (s *Store) ListIdleChatSessions(ctx context.Context, idleSince time.Time, limit int) ([]ChatSession, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, title, archived, last_activity_at, latest_summary_id, summary_preview, summary_count, title_source, created_at, updated_at
		FROM chat_sessions
		WHERE archived = 0 AND COALESCE(last_activity_at, updated_at) < ?
		ORDER BY COALESCE(last_activity_at, updated_at) ASC
		LIMIT ?
	`, formatTS(idleSince.UTC()), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]ChatSession, 0)
	for rows.Next() {
		rec, err := scanChatSession(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *rec)
	}
	return out, rows.Err()
}

func (s *Store) ArchiveChatSession(ctx context.Context, sessionID string) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions SET archived = 1, updated_at = ? WHERE id = ?
	`, formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) DeleteChatSession(ctx context.Context, sessionID string) error {
	// Delete messages first, then session.
	_, _ = s.db.ExecContext(ctx, `DELETE FROM chat_messages WHERE session_id = ?`, sessionID)
	_, err := s.db.ExecContext(ctx, `DELETE FROM chat_sessions WHERE id = ?`, sessionID)
	return err
}

func scanChatSession(scanner interface{ Scan(dest ...any) error }) (*ChatSession, error) {
	var rec ChatSession
	var archived int
	var createdAt, updatedAt string
	var lastActivityAt sql.NullString
	var latestSummaryID sql.NullInt64
	if err := scanner.Scan(&rec.ID, &rec.UserID, &rec.Title, &archived, &lastActivityAt, &latestSummaryID, &rec.SummaryPreview, &rec.SummaryCount, &rec.TitleSource, &createdAt, &updatedAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrNotFound
		}
		return nil, err
	}
	rec.Archived = archived != 0
	var err error
	rec.CreatedAt, err = parseTS(createdAt)
	if err != nil {
		return nil, err
	}
	rec.UpdatedAt, err = parseTS(updatedAt)
	if err != nil {
		return nil, err
	}
	if lastActivityAt.Valid && strings.TrimSpace(lastActivityAt.String) != "" {
		ts, err := parseTS(lastActivityAt.String)
		if err != nil {
			return nil, err
		}
		rec.LastActivityAt = &ts
	}
	if latestSummaryID.Valid {
		id := latestSummaryID.Int64
		rec.LatestSummaryID = &id
	}
	return &rec, nil
}

func scanChatSessionSummary(scanner interface{ Scan(dest ...any) error }) (*ChatSessionSummary, error) {
	var rec ChatSessionSummary
	var createdAt string
	if err := scanner.Scan(&rec.ID, &rec.SessionID, &rec.UserID, &rec.Revision, &rec.SummaryText, &rec.TitleSuggestion, &rec.MessageCount, &createdAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrNotFound
		}
		return nil, err
	}
	ts, err := parseTS(createdAt)
	if err != nil {
		return nil, err
	}
	rec.CreatedAt = ts
	return &rec, nil
}

func deriveSummaryPreview(summary string) string {
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return ""
	}
	lines := strings.Split(summary, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(strings.ToLower(line), "current state:") {
			return truncateSessionPreview(strings.TrimSpace(strings.TrimPrefix(line, "Current State:")))
		}
		if !strings.HasPrefix(line, "#") && !strings.HasPrefix(line, "- ") {
			return truncateSessionPreview(line)
		}
	}
	return truncateSessionPreview(summary)
}

func truncateSessionPreview(summary string) string {
	summary = strings.TrimSpace(summary)
	if len(summary) <= 180 {
		return summary
	}
	return strings.TrimSpace(summary[:180]) + "..."
}

func nullableSummaryID(v int64) any {
	if v <= 0 {
		return nil
	}
	return v
}
