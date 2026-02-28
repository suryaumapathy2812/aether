package db

import (
	"context"
	"fmt"
	"strings"
)

// PushSubscriptionRecord represents a stored Web Push subscription.
type PushSubscriptionRecord struct {
	ID        int64  `json:"id"`
	UserID    string `json:"user_id"`
	Endpoint  string `json:"endpoint"`
	KeyP256dh string `json:"key_p256dh"`
	KeyAuth   string `json:"key_auth"`
	CreatedAt string `json:"created_at"`
}

// SavePushSubscription upserts a push subscription for a user.
func (s *Store) SavePushSubscription(ctx context.Context, userID string, rec PushSubscriptionRecord) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		return fmt.Errorf("user_id is required")
	}
	if strings.TrimSpace(rec.Endpoint) == "" {
		return fmt.Errorf("endpoint is required")
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO push_subscriptions(user_id, endpoint, key_p256dh, key_auth)
		VALUES(?, ?, ?, ?)
		ON CONFLICT(user_id, endpoint) DO UPDATE SET
			key_p256dh = excluded.key_p256dh,
			key_auth = excluded.key_auth
	`, userID, rec.Endpoint, rec.KeyP256dh, rec.KeyAuth)
	return err
}

// DeletePushSubscription removes a push subscription by user and endpoint.
func (s *Store) DeletePushSubscription(ctx context.Context, userID, endpoint string) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("store unavailable")
	}
	_, err := s.db.ExecContext(ctx, `
		DELETE FROM push_subscriptions WHERE user_id = ? AND endpoint = ?
	`, userID, endpoint)
	return err
}

// GetPushSubscriptions returns all push subscriptions for a user.
func (s *Store) GetPushSubscriptions(ctx context.Context, userID string) ([]PushSubscriptionRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, endpoint, key_p256dh, key_auth, created_at
		FROM push_subscriptions WHERE user_id = ?
		ORDER BY created_at DESC
	`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []PushSubscriptionRecord
	for rows.Next() {
		var r PushSubscriptionRecord
		if err := rows.Scan(&r.ID, &r.UserID, &r.Endpoint, &r.KeyP256dh, &r.KeyAuth, &r.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
