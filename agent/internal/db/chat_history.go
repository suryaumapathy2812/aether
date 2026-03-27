package db

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type ChatMessageRecord struct {
	ID        int64          `json:"id"`
	UserID    string         `json:"user_id"`
	SessionID string         `json:"session_id"`
	Role      string         `json:"role"`
	Content   map[string]any `json:"content"`
	CreatedAt time.Time      `json:"created_at"`
}

func (s *Store) AppendChatMessage(ctx context.Context, userID, sessionID string, message map[string]any) error {
	_, err := s.AppendChatMessageReturningID(ctx, userID, sessionID, message)
	return err
}

func (s *Store) AppendChatMessageReturningID(ctx context.Context, userID, sessionID string, message map[string]any) (int64, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	if message == nil {
		return 0, fmt.Errorf("message is required")
	}
	role, _ := message["role"].(string)
	role = strings.TrimSpace(strings.ToLower(role))
	if role == "" {
		return 0, fmt.Errorf("message.role is required")
	}
	b, err := json.Marshal(message)
	if err != nil {
		return 0, err
	}
	result, err := s.db.ExecContext(ctx, `
		INSERT INTO chat_messages(user_id, session_id, role, content_json, created_at)
		VALUES(?, ?, ?, ?, ?)
	`, userID, sessionID, role, string(b), formatTS(time.Now().UTC()))
	if err != nil {
		return 0, err
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, err
	}
	return id, nil
}

func (s *Store) DeleteChatMessageByID(ctx context.Context, id int64) error {
	if id <= 0 {
		return nil
	}
	_, err := s.db.ExecContext(ctx, `DELETE FROM chat_messages WHERE id = ?`, id)
	return err
}

func (s *Store) ListChatMessages(ctx context.Context, userID, sessionID string, limit int) ([]ChatMessageRecord, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	if limit <= 0 || limit > 4000 {
		limit = 500
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, session_id, role, content_json, created_at
		FROM chat_messages
		WHERE user_id = ? AND session_id = ?
		ORDER BY id ASC
		LIMIT ?
	`, userID, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []ChatMessageRecord{}
	for rows.Next() {
		var rec ChatMessageRecord
		var contentJSON string
		var createdAt string
		if err := rows.Scan(&rec.ID, &rec.UserID, &rec.SessionID, &rec.Role, &contentJSON, &createdAt); err != nil {
			return nil, err
		}
		rec.Content = map[string]any{}
		if strings.TrimSpace(contentJSON) != "" {
			_ = json.Unmarshal([]byte(contentJSON), &rec.Content)
		}
		ts, err := parseTS(createdAt)
		if err != nil {
			return nil, err
		}
		rec.CreatedAt = ts
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) ListChatMessagesSince(ctx context.Context, userID, sessionID string, since time.Time, limit int) ([]ChatMessageRecord, error) {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	if limit <= 0 || limit > 4000 {
		limit = 500
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, session_id, role, content_json, created_at
		FROM chat_messages
		WHERE user_id = ? AND session_id = ? AND created_at >= ?
		ORDER BY id ASC
		LIMIT ?
	`, userID, sessionID, formatTS(since), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []ChatMessageRecord{}
	for rows.Next() {
		var rec ChatMessageRecord
		var contentJSON string
		var createdAt string
		if err := rows.Scan(&rec.ID, &rec.UserID, &rec.SessionID, &rec.Role, &contentJSON, &createdAt); err != nil {
			return nil, err
		}
		rec.Content = map[string]any{}
		if strings.TrimSpace(contentJSON) != "" {
			_ = json.Unmarshal([]byte(contentJSON), &rec.Content)
		}
		ts, err := parseTS(createdAt)
		if err != nil {
			return nil, err
		}
		rec.CreatedAt = ts
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) DeleteChatMessagesBySession(ctx context.Context, userID, sessionID string) error {
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "chat"
	}
	_, err := s.db.ExecContext(ctx, `DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?`, userID, sessionID)
	return err
}
