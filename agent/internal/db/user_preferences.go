package db

import (
	"context"
	"fmt"
	"strings"
)

type UserPreferenceRecord struct {
	ID        int64  `json:"id"`
	UserID    string `json:"user_id"`
	PrefKey   string `json:"pref_key"`
	PrefValue string `json:"pref_value"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

func (s *Store) SaveUserPreference(ctx context.Context, userID, key, value string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(key) == "" {
		return fmt.Errorf("pref_key is required")
	}
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("pref_value is required")
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO user_preferences(user_id, pref_key, pref_value)
		VALUES(?, ?, ?)
		ON CONFLICT(user_id, pref_key) DO UPDATE SET
			pref_value = excluded.pref_value,
			updated_at = strftime('%Y-%m-%dT%H:%M:%S','now')
	`, userID, key, value)
	return err
}

func (s *Store) GetUserPreference(ctx context.Context, userID, key string) (string, error) {
	if s == nil || s.db == nil {
		return "", fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(key) == "" {
		return "", fmt.Errorf("pref_key is required")
	}
	var value string
	err := s.db.QueryRowContext(ctx, `
		SELECT pref_value FROM user_preferences WHERE user_id = ? AND pref_key = ?
	`, userID, key).Scan(&value)
	if err != nil {
		return "", err
	}
	return value, nil
}

func (s *Store) DeleteUserPreference(ctx context.Context, userID, key string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	if strings.TrimSpace(key) == "" {
		return fmt.Errorf("pref_key is required")
	}
	_, err := s.db.ExecContext(ctx, `
		DELETE FROM user_preferences WHERE user_id = ? AND pref_key = ?
	`, userID, key)
	return err
}
