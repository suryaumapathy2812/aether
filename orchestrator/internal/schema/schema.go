package schema

import (
	"context"

	"github.com/jackc/pgx/v5/pgxpool"
)

func Bootstrap(ctx context.Context, db *pgxpool.Pool) error {
	_, err := db.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS "user" (
			id TEXT PRIMARY KEY,
			email TEXT,
			name TEXT,
			created_at TIMESTAMPTZ DEFAULT now()
		);

		CREATE TABLE IF NOT EXISTS session (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL,
			token TEXT UNIQUE NOT NULL,
			expires_at TIMESTAMPTZ NOT NULL,
			created_at TIMESTAMPTZ DEFAULT now()
		);

		CREATE TABLE IF NOT EXISTS devices (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL,
			token TEXT UNIQUE NOT NULL,
			name TEXT DEFAULT 'Unknown Device',
			device_type TEXT DEFAULT 'ios',
			paired_at TIMESTAMPTZ DEFAULT now(),
			last_seen TIMESTAMPTZ
		);

		CREATE TABLE IF NOT EXISTS agents (
			id TEXT PRIMARY KEY,
			user_id TEXT UNIQUE,
			container_id TEXT,
			host TEXT NOT NULL,
			port INTEGER NOT NULL,
			status TEXT DEFAULT 'starting',
			registered_at TIMESTAMPTZ DEFAULT now(),
			last_health TIMESTAMPTZ,
			stopped_at TIMESTAMPTZ
		);

		ALTER TABLE agents ADD COLUMN IF NOT EXISTS stopped_at TIMESTAMPTZ;

		CREATE INDEX IF NOT EXISTS idx_agents_user_status ON agents(user_id, status, last_health DESC);
		CREATE INDEX IF NOT EXISTS idx_session_token_expires ON session(token, expires_at);
		CREATE INDEX IF NOT EXISTS idx_devices_token ON devices(token);
	`)
	return err
}
