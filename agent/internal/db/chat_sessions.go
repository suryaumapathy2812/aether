package db

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

type ChatSession struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
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
		INSERT INTO chat_sessions(id, user_id, title, created_at, updated_at)
		VALUES(?, ?, ?, ?, ?)
	`, id, userID, title, now, now)
	if err != nil {
		return ChatSession{}, fmt.Errorf("create chat session: %w", err)
	}
	return s.GetChatSession(ctx, id)
}

func (s *Store) GetChatSession(ctx context.Context, sessionID string) (ChatSession, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, title, created_at, updated_at
		FROM chat_sessions WHERE id = ?
	`, sessionID)
	var rec ChatSession
	var createdAt, updatedAt string
	if err := row.Scan(&rec.ID, &rec.UserID, &rec.Title, &createdAt, &updatedAt); err != nil {
		return ChatSession{}, fmt.Errorf("get chat session: %w", err)
	}
	var err error
	rec.CreatedAt, err = parseTS(createdAt)
	if err != nil {
		return ChatSession{}, err
	}
	rec.UpdatedAt, err = parseTS(updatedAt)
	if err != nil {
		return ChatSession{}, err
	}
	return rec, nil
}

func (s *Store) ListChatSessions(ctx context.Context, userID string, limit int) ([]ChatSession, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, title, created_at, updated_at
		FROM chat_sessions
		WHERE user_id = ?
		ORDER BY updated_at DESC
		LIMIT ?
	`, userID, limit)
	if err != nil {
		return nil, fmt.Errorf("list chat sessions: %w", err)
	}
	defer rows.Close()
	out := []ChatSession{}
	for rows.Next() {
		var rec ChatSession
		var createdAt, updatedAt string
		if err := rows.Scan(&rec.ID, &rec.UserID, &rec.Title, &createdAt, &updatedAt); err != nil {
			return nil, err
		}
		rec.CreatedAt, _ = parseTS(createdAt)
		rec.UpdatedAt, _ = parseTS(updatedAt)
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) UpdateChatSessionTitle(ctx context.Context, sessionID, title string) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?
	`, title, formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) TouchChatSession(ctx context.Context, sessionID string) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions SET updated_at = ? WHERE id = ?
	`, formatTS(time.Now().UTC()), sessionID)
	return err
}

func (s *Store) DeleteChatSession(ctx context.Context, sessionID string) error {
	// Delete messages first, then session.
	_, _ = s.db.ExecContext(ctx, `DELETE FROM chat_messages WHERE session_id = ?`, sessionID)
	_, err := s.db.ExecContext(ctx, `DELETE FROM chat_sessions WHERE id = ?`, sessionID)
	return err
}
