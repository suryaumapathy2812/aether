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
			plugin_name TEXT DEFAULT '',
			config_json TEXT DEFAULT '{}',
			paired_at TIMESTAMPTZ DEFAULT now(),
			last_seen TIMESTAMPTZ
		);

		ALTER TABLE devices ADD COLUMN IF NOT EXISTS plugin_name TEXT DEFAULT '';
		ALTER TABLE devices ADD COLUMN IF NOT EXISTS config_json TEXT DEFAULT '{}';

		CREATE TABLE IF NOT EXISTS pair_requests (
			code TEXT PRIMARY KEY,
			device_type TEXT DEFAULT 'ios',
			device_name TEXT DEFAULT 'Unknown Device',
			created_at TIMESTAMPTZ DEFAULT now(),
			expires_at TIMESTAMPTZ DEFAULT now() + interval '10 minutes',
			claimed_by TEXT,
			issued_channel_id TEXT,
			issued_token TEXT,
			issued_at TIMESTAMPTZ
		);

		ALTER TABLE pair_requests ADD COLUMN IF NOT EXISTS issued_channel_id TEXT;
		ALTER TABLE pair_requests ADD COLUMN IF NOT EXISTS issued_token TEXT;
		ALTER TABLE pair_requests ADD COLUMN IF NOT EXISTS issued_at TIMESTAMPTZ;

		CREATE TABLE IF NOT EXISTS agents (
			id TEXT PRIMARY KEY,
			user_id TEXT UNIQUE,
			container_id TEXT,
			subdomain_prefix TEXT,
			host TEXT NOT NULL,
			port INTEGER NOT NULL,
			status TEXT DEFAULT 'starting',
			registered_at TIMESTAMPTZ DEFAULT now(),
			last_health TIMESTAMPTZ,
			stopped_at TIMESTAMPTZ
		);

		ALTER TABLE agents ADD COLUMN IF NOT EXISTS stopped_at TIMESTAMPTZ;
		ALTER TABLE agents ADD COLUMN IF NOT EXISTS subdomain_prefix TEXT;

		CREATE INDEX IF NOT EXISTS idx_agents_user_status ON agents(user_id, status, last_health DESC);
		CREATE UNIQUE INDEX IF NOT EXISTS idx_agents_subdomain_prefix ON agents(subdomain_prefix) WHERE subdomain_prefix IS NOT NULL;
		CREATE INDEX IF NOT EXISTS idx_session_token_expires ON session(token, expires_at);
		CREATE INDEX IF NOT EXISTS idx_devices_token ON devices(token);
		CREATE INDEX IF NOT EXISTS idx_devices_user_type ON devices(user_id, device_type, plugin_name);
		CREATE INDEX IF NOT EXISTS idx_pair_requests_expires_at ON pair_requests(expires_at);

		CREATE TABLE IF NOT EXISTS email_mappings (
			email TEXT NOT NULL,
			user_id TEXT NOT NULL,
			plugin_name TEXT NOT NULL DEFAULT 'google-workspace',
			created_at TIMESTAMPTZ DEFAULT now(),
			PRIMARY KEY (email, user_id)
		);

		CREATE INDEX IF NOT EXISTS idx_email_mappings_email ON email_mappings(email);
	`)
	return err
}
