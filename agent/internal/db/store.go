package db

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/tursodatabase/go-libsql"
)

var (
	ErrNotFound          = errors.New("not found")
	ErrCryptoUnavailable = errors.New("encryption key is not configured")
	ErrInvalidCiphertext = errors.New("invalid ciphertext")
)

type Store struct {
	db            *instrumentedDB
	aead          cipher.AEAD
	path          string
	vectorEnabled bool
}

const defaultMemoryEmbeddingDimensions = 1536

type SkillRecord struct {
	Name        string
	Description string
	Location    string
	Source      string
}

type PluginRecord struct {
	Name        string
	DisplayName string
	Description string
	Version     string
	PluginType  string
	Location    string
	Source      string
	HasSkill    bool
	Enabled     bool
	Config      map[string]string
}

// ChannelRecord represents a communication channel (Telegram, WhatsApp, etc.)
type ChannelRecord struct {
	ID          string            `json:"id"`
	UserID      string            `json:"user_id"`
	ChannelType string            `json:"channel_type"`
	ChannelID   string            `json:"channel_id"`
	BotToken    string            `json:"bot_token,omitempty"`
	DisplayName string            `json:"display_name"`
	Config      map[string]string `json:"config"`
	Enabled     bool              `json:"enabled"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// ChannelMessageRecord represents a message in a channel
type ChannelMessageRecord struct {
	ID           string    `json:"id"`
	ChannelID    string    `json:"channel_id"`
	MessageID    string    `json:"message_id,omitempty"`
	Direction    string    `json:"direction"` // "inbound" or "outbound"
	Content      string    `json:"content"`
	MetadataJSON string    `json:"metadata_json,omitempty"`
	Timestamp    time.Time `json:"timestamp"`
}

type CronStatus string

const (
	CronStatusScheduled CronStatus = "scheduled"
	CronStatusRunning   CronStatus = "running"
	CronStatusRetry     CronStatus = "retry"
	CronStatusDone      CronStatus = "done"
	CronStatusCancelled CronStatus = "cancelled"
	CronStatusFailed    CronStatus = "failed"
	CronStatusPaused    CronStatus = "paused"
)

type CronJobCreate struct {
	ID          string
	Module      string
	JobType     string
	Payload     any
	RunAt       time.Time
	IntervalS   *int64
	MaxAttempts int
}

type CronJobRecord struct {
	ID           string
	Module       string
	JobType      string
	PayloadJSON  string
	RunAt        time.Time
	IntervalS    *int64
	Status       CronStatus
	Enabled      bool
	AttemptCount int
	MaxAttempts  int
	LastError    string
	LockedUntil  *time.Time
	LockToken    string
	LastRunAt    *time.Time
	NextRunAt    time.Time
	CreatedAt    time.Time
	UpdatedAt    time.Time
}

// Open opens the SQLite database at path and configures encryption using
// stateKey (pass "" to disable encryption).
func Open(path, stateKey string) (*Store, error) {
	if path == "" {
		return nil, fmt.Errorf("db path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("libsql", libsqlFilePath(path))
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	if err := applyPragma(db, `PRAGMA journal_mode = WAL;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := applyPragma(db, `PRAGMA busy_timeout = 5000;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := applyPragma(db, `PRAGMA synchronous = NORMAL;`); err != nil {
		_ = db.Close()
		return nil, err
	}

	store := &Store{db: newInstrumentedDB(db), path: path}
	if err := store.ConfigureCrypto(stateKey); err != nil {
		_ = db.Close()
		return nil, err
	}
	store.vectorEnabled = detectVectorSupport(db)
	if err := store.migrate(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

// OpenInAssets opens the SQLite database in the given assets directory.
func OpenInAssets(assetsDir, stateKey string) (*Store, error) {
	if assetsDir == "" {
		return nil, fmt.Errorf("assets directory is required")
	}
	return Open(filepath.Join(assetsDir, "db", "state.db"), stateKey)
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func libsqlFilePath(path string) string {
	trimmed := strings.TrimSpace(path)
	if strings.HasPrefix(trimmed, "file:") {
		return trimmed
	}
	return "file:" + trimmed
}

func applyPragma(db *sql.DB, query string) error {
	rows, err := db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		// Drain any rows returned by PRAGMA statements.
	}
	return rows.Err()
}

func detectVectorSupport(db *sql.DB) bool {
	row := db.QueryRow(`SELECT vector_distance_cos(vector32('[0,1]'), vector32('[0,1]'))`)
	var distance float64
	return row.Scan(&distance) == nil
}

func (s *Store) migrate(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS skills (
			name TEXT PRIMARY KEY,
			description TEXT NOT NULL DEFAULT '',
			location TEXT NOT NULL DEFAULT '',
			source TEXT NOT NULL DEFAULT '',
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE TABLE IF NOT EXISTS plugins (
			name TEXT PRIMARY KEY,
			display_name TEXT NOT NULL DEFAULT '',
			description TEXT NOT NULL DEFAULT '',
			version TEXT NOT NULL DEFAULT '0.1.0',
			plugin_type TEXT NOT NULL DEFAULT 'sensor',
			location TEXT NOT NULL DEFAULT '',
			source TEXT NOT NULL DEFAULT '',
			has_skill INTEGER NOT NULL DEFAULT 0,
			enabled INTEGER NOT NULL DEFAULT 0,
			config_json TEXT NOT NULL DEFAULT '{}',
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE TABLE IF NOT EXISTS cron_jobs (
			id TEXT PRIMARY KEY,
			module TEXT NOT NULL,
			job_type TEXT NOT NULL,
			payload_json TEXT NOT NULL DEFAULT '{}',
			run_at TEXT NOT NULL,
			interval_s INTEGER,
			status TEXT NOT NULL DEFAULT 'scheduled',
			enabled INTEGER NOT NULL DEFAULT 1,
			attempt_count INTEGER NOT NULL DEFAULT 0,
			max_attempts INTEGER NOT NULL DEFAULT 5,
			last_error TEXT NOT NULL DEFAULT '',
			locked_until TEXT,
			lock_token TEXT NOT NULL DEFAULT '',
			last_run_at TEXT,
			next_run_at TEXT NOT NULL,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE INDEX IF NOT EXISTS idx_cron_jobs_due ON cron_jobs(enabled, status, next_run_at);`,
		`CREATE INDEX IF NOT EXISTS idx_cron_jobs_module_type ON cron_jobs(module, job_type);`,
		`CREATE INDEX IF NOT EXISTS idx_cron_jobs_lock ON cron_jobs(locked_until);`,
		`CREATE TABLE IF NOT EXISTS conversations (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			session_id TEXT NOT NULL DEFAULT '',
			user_message TEXT NOT NULL,
			user_content_json TEXT NOT NULL DEFAULT '',
			assistant_message TEXT NOT NULL,
			timestamp TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_conversations_user_timestamp ON conversations(user_id, timestamp DESC);`,
		`CREATE TABLE IF NOT EXISTS chat_sessions (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL DEFAULT 'default',
			title TEXT NOT NULL DEFAULT '',
			archived INTEGER NOT NULL DEFAULT 0,
			last_activity_at TEXT,
			latest_summary_id INTEGER,
			summary_preview TEXT NOT NULL DEFAULT '',
			summary_count INTEGER NOT NULL DEFAULT 0,
			title_source TEXT NOT NULL DEFAULT 'seed',
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id, updated_at DESC);`,
		`CREATE INDEX IF NOT EXISTS idx_chat_sessions_idle ON chat_sessions(user_id, last_activity_at DESC);`,
		`CREATE TABLE IF NOT EXISTS chat_messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			session_id TEXT NOT NULL DEFAULT 'chat',
			role TEXT NOT NULL,
			content_json TEXT NOT NULL,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(user_id, session_id, id);`,
		`CREATE TABLE IF NOT EXISTS actions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			session_id TEXT NOT NULL DEFAULT '',
			tool_name TEXT NOT NULL,
			arguments TEXT NOT NULL DEFAULT '{}',
			output TEXT NOT NULL DEFAULT '',
			error INTEGER NOT NULL DEFAULT 0,
			timestamp TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_actions_user_timestamp ON actions(user_id, timestamp DESC);`,
		`CREATE TABLE IF NOT EXISTS sessions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			session_id TEXT NOT NULL,
			summary TEXT NOT NULL,
			started_at TEXT NOT NULL,
			ended_at TEXT NOT NULL,
			turns INTEGER NOT NULL DEFAULT 0,
			tools_used TEXT NOT NULL DEFAULT '[]'
		);`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user_ended_at ON sessions(user_id, ended_at DESC);`,
		`CREATE TABLE IF NOT EXISTS memory_items (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			kind TEXT NOT NULL,
			category TEXT NOT NULL DEFAULT '',
			content TEXT NOT NULL,
			normalized_key TEXT NOT NULL,
			status TEXT NOT NULL DEFAULT 'active',
			confidence REAL NOT NULL DEFAULT 1.0,
			importance REAL NOT NULL DEFAULT 0.5,
			evidence_count INTEGER NOT NULL DEFAULT 1,
			first_seen_at TEXT NOT NULL,
			last_seen_at TEXT NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			expires_at TEXT,
			source_type TEXT NOT NULL DEFAULT 'system',
			source_id TEXT NOT NULL DEFAULT '',
			session_id TEXT NOT NULL DEFAULT '',
			metadata_json TEXT NOT NULL DEFAULT '{}',
			scope TEXT NOT NULL DEFAULT 'contextual',
			recall_count INTEGER NOT NULL DEFAULT 0,
			last_recalled_at TEXT,
			embedding F32_BLOB(` + fmt.Sprintf("%d", defaultMemoryEmbeddingDimensions) + `),
			UNIQUE(user_id, kind, normalized_key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_user_kind_status ON memory_items(user_id, kind, status, updated_at DESC);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_user_last_seen ON memory_items(user_id, last_seen_at DESC);`,
		`CREATE TABLE IF NOT EXISTS notifications (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			text TEXT NOT NULL,
			delivery_type TEXT NOT NULL DEFAULT 'surface',
			status TEXT NOT NULL DEFAULT 'pending',
			source TEXT NOT NULL DEFAULT 'proactive',
			deliver_at TEXT,
			delivered_at TEXT,
			delivery_attempts INTEGER NOT NULL DEFAULT 0,
			last_attempt_at TEXT,
			next_retry_at TEXT,
			last_error TEXT NOT NULL DEFAULT '',
			expires_at TEXT,
			created_at TEXT NOT NULL,
			metadata TEXT NOT NULL DEFAULT '{}'
		);`,
		`CREATE INDEX IF NOT EXISTS idx_notifications_user_status ON notifications(user_id, status);`,
		`CREATE INDEX IF NOT EXISTS idx_notifications_user_deliver_at ON notifications(user_id, deliver_at);`,
		`CREATE INDEX IF NOT EXISTS idx_notifications_user_next_retry_at ON notifications(user_id, next_retry_at);`,
		`CREATE TABLE IF NOT EXISTS proactive_events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			event_id TEXT NOT NULL DEFAULT '',
			plugin TEXT NOT NULL,
			event_type TEXT NOT NULL,
			status TEXT NOT NULL,
			decision TEXT NOT NULL DEFAULT '',
			delivery_type TEXT NOT NULL DEFAULT '',
			notification_id INTEGER,
			error TEXT NOT NULL DEFAULT '',
			payload TEXT NOT NULL DEFAULT '{}',
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_proactive_events_user_created_at ON proactive_events(user_id, created_at DESC);`,
		`CREATE INDEX IF NOT EXISTS idx_proactive_events_user_status ON proactive_events(user_id, status);`,
		`CREATE TABLE IF NOT EXISTS push_subscriptions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL,
			endpoint TEXT NOT NULL,
			key_p256dh TEXT NOT NULL DEFAULT '',
			key_auth TEXT NOT NULL DEFAULT '',
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			UNIQUE(user_id, endpoint)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user ON push_subscriptions(user_id);`,
		// --- User preferences ---
		`CREATE TABLE IF NOT EXISTS user_preferences (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			pref_key TEXT NOT NULL,
			pref_value TEXT NOT NULL,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
			UNIQUE(user_id, pref_key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_user_preferences_user_key ON user_preferences(user_id, pref_key);`,
		// --- Entity memory tables ---
		`CREATE TABLE IF NOT EXISTS entities (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL DEFAULT 'default',
			entity_type TEXT NOT NULL,
			name TEXT NOT NULL,
			aliases TEXT NOT NULL DEFAULT '[]',
			summary TEXT NOT NULL DEFAULT '',
			properties TEXT NOT NULL DEFAULT '{}',
			first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			interaction_count INTEGER NOT NULL DEFAULT 0,
			archived INTEGER NOT NULL DEFAULT 0,
			last_summary_at TEXT,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE INDEX IF NOT EXISTS idx_entities_user_type ON entities(user_id, entity_type);`,
		`CREATE INDEX IF NOT EXISTS idx_entities_user_name ON entities(user_id, name);`,
		`CREATE INDEX IF NOT EXISTS idx_entities_user_last_seen ON entities(user_id, last_seen_at DESC);`,
		`CREATE TABLE IF NOT EXISTS entity_observations (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			entity_id TEXT NOT NULL,
			user_id TEXT NOT NULL DEFAULT 'default',
			observation TEXT NOT NULL,
			observation_key TEXT NOT NULL,
			category TEXT NOT NULL DEFAULT 'trait',
			confidence REAL NOT NULL DEFAULT 1.0,
			source TEXT NOT NULL DEFAULT 'extracted',
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			UNIQUE(entity_id, observation_key),
			FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_entity_observations_entity ON entity_observations(entity_id);`,
		`CREATE TABLE IF NOT EXISTS entity_interactions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			entity_id TEXT NOT NULL,
			user_id TEXT NOT NULL DEFAULT 'default',
			summary TEXT NOT NULL,
			source TEXT NOT NULL,
			source_ref TEXT,
			interaction_at TEXT NOT NULL,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_entity_interactions_entity ON entity_interactions(entity_id, interaction_at DESC);`,
		`CREATE INDEX IF NOT EXISTS idx_entity_interactions_time ON entity_interactions(user_id, interaction_at DESC);`,
		`CREATE TABLE IF NOT EXISTS entity_relations (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			source_entity_id TEXT NOT NULL,
			relation TEXT NOT NULL,
			target_entity_id TEXT NOT NULL,
			context TEXT NOT NULL DEFAULT '',
			confidence REAL NOT NULL DEFAULT 1.0,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			UNIQUE(source_entity_id, relation, target_entity_id),
			FOREIGN KEY(source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
			FOREIGN KEY(target_entity_id) REFERENCES entities(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_entity_relations_source ON entity_relations(source_entity_id);`,
		`CREATE INDEX IF NOT EXISTS idx_entity_relations_target ON entity_relations(target_entity_id);`,
		// Channels table - for managing user communication channels (Telegram, WhatsApp, etc.)
		`CREATE TABLE IF NOT EXISTS channels (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL DEFAULT 'default',
			channel_type TEXT NOT NULL,
			channel_id TEXT NOT NULL,
			bot_token TEXT,
			display_name TEXT,
			config_json TEXT,
			enabled INTEGER NOT NULL DEFAULT 1,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			UNIQUE(user_id, channel_type, channel_id)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_channels_user ON channels(user_id);`,
		`CREATE INDEX IF NOT EXISTS idx_channels_type ON channels(channel_type);`,
		// Channel messages table - for storing message history
		`CREATE TABLE IF NOT EXISTS channel_messages (
			id TEXT PRIMARY KEY,
			channel_id TEXT NOT NULL,
			message_id TEXT,
			direction TEXT NOT NULL,
			content TEXT NOT NULL,
			metadata_json TEXT,
			timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_channel_messages_channel ON channel_messages(channel_id, timestamp DESC);`,
		`CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
			user_id UNINDEXED,
			item_id UNINDEXED,
			kind UNINDEXED,
			category UNINDEXED,
			content,
			tokenize='porter unicode61'
		);`,
		// --- Session summary revision history ---
		`CREATE TABLE IF NOT EXISTS chat_session_summaries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			user_id TEXT NOT NULL DEFAULT 'default',
			revision INTEGER NOT NULL DEFAULT 1,
			summary_text TEXT NOT NULL DEFAULT '',
			title_suggestion TEXT NOT NULL DEFAULT '',
			message_count INTEGER NOT NULL DEFAULT 0,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			UNIQUE(session_id, revision)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_css_session ON chat_session_summaries(session_id, revision DESC);`,
		// --- Memory item session linkage (for promotion tracking) ---
		`CREATE TABLE IF NOT EXISTS memory_item_sessions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			memory_item_id INTEGER NOT NULL,
			session_id TEXT NOT NULL,
			user_id TEXT NOT NULL DEFAULT 'default',
			reinforced_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			UNIQUE(memory_item_id, session_id),
			FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_mis_item ON memory_item_sessions(memory_item_id);`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return err
		}
	}
	if s.vectorEnabled {
		if _, err := s.db.ExecContext(ctx, `CREATE INDEX IF NOT EXISTS idx_memory_items_vec ON memory_items(libsql_vector_idx(embedding)) WHERE embedding IS NOT NULL AND status = 'active';`); err != nil {
			return err
		}
	}
	if err := s.ensureColumn(ctx, "conversations", "user_content_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		return err
	}
	// --- Memory lifecycle columns ---
	if err := s.ensureColumn(ctx, "memory_items", "scope", "TEXT NOT NULL DEFAULT 'contextual'"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "memory_items", "recall_count", "INTEGER NOT NULL DEFAULT 0"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "memory_items", "last_recalled_at", "TEXT"); err != nil {
		return err
	}
	// --- Chat session summary metadata columns ---
	if err := s.ensureColumn(ctx, "chat_sessions", "last_activity_at", "TEXT"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "chat_sessions", "latest_summary_id", "INTEGER"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "chat_sessions", "summary_preview", "TEXT NOT NULL DEFAULT ''"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "chat_sessions", "summary_count", "INTEGER NOT NULL DEFAULT 0"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "chat_sessions", "title_source", "TEXT NOT NULL DEFAULT 'seed'"); err != nil {
		return err
	}
	// --- Entity lifecycle columns ---
	if err := s.ensureColumn(ctx, "entities", "archived", "INTEGER NOT NULL DEFAULT 0"); err != nil {
		return err
	}
	if err := s.ensureColumn(ctx, "entities", "last_summary_at", "TEXT"); err != nil {
		return err
	}
	// --- Backfill scope for existing memory items ---
	if err := s.backfillMemoryScopes(ctx); err != nil {
		return err
	}
	if err := s.backfillLegacyMemoryItems(ctx); err != nil {
		return err
	}
	if err := s.backfillMemoryItemSessions(ctx); err != nil {
		return err
	}
	if err := s.rebuildMemoryItemsFTS(ctx); err != nil {
		return err
	}
	if err := s.dropLegacyMemoryTables(ctx); err != nil {
		return err
	}
	return nil
}

func (s *Store) rebuildMemoryItemsFTS(ctx context.Context) error {
	if s == nil || s.db == nil {
		return nil
	}
	if _, err := s.db.ExecContext(ctx, `DELETE FROM memory_items_fts`); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO memory_items_fts(user_id, item_id, kind, category, content)
		SELECT user_id, id, kind, category, content
		FROM memory_items
		WHERE status = 'active'
	`)
	return err
}

func (s *Store) backfillMemoryScopes(ctx context.Context) error {
	if s == nil || s.db == nil {
		return nil
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, kind, category, content, scope, evidence_count, last_seen_at
		FROM memory_items
	`)
	if err != nil {
		return err
	}
	defer rows.Close()

	type scopeUpdate struct {
		id    int64
		scope string
	}
	updates := make([]scopeUpdate, 0)
	cutoff := time.Now().UTC().Add(-30 * 24 * time.Hour)

	for rows.Next() {
		var (
			id            int64
			kind          string
			category      string
			content       string
			scope         string
			evidenceCount int
			lastSeenRaw   string
		)
		if err := rows.Scan(&id, &kind, &category, &content, &scope, &evidenceCount, &lastSeenRaw); err != nil {
			return err
		}

		nextScope := string(InferMemoryScope(kind, category, content))
		if evidenceCount >= 3 && (strings.EqualFold(kind, "decision") || strings.EqualFold(kind, "fact")) {
			nextScope = string(ScopeGlobal)
		}
		if evidenceCount <= 1 {
			if lastSeen, err := parseTS(lastSeenRaw); err == nil && lastSeen.Before(cutoff) && nextScope == string(ScopeContextual) {
				nextScope = string(ScopeVolatile)
			}
		}
		if strings.TrimSpace(scope) == strings.TrimSpace(nextScope) {
			continue
		}
		updates = append(updates, scopeUpdate{id: id, scope: nextScope})
	}
	if err := rows.Err(); err != nil {
		return err
	}

	for _, item := range updates {
		if _, err := s.db.ExecContext(ctx, `UPDATE memory_items SET scope = ? WHERE id = ?`, item.scope, item.id); err != nil {
			return err
		}
	}

	if _, err := s.db.ExecContext(ctx, `
		UPDATE chat_sessions
		SET last_activity_at = COALESCE((
			SELECT MAX(created_at) FROM chat_messages
			WHERE chat_messages.session_id = chat_sessions.id
		), updated_at)
		WHERE last_activity_at IS NULL OR trim(last_activity_at) = ''
	`); err != nil {
		return err
	}

	return nil
}

func (s *Store) backfillMemoryItemSessions(ctx context.Context) error {
	if s == nil || s.db == nil {
		return nil
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO memory_item_sessions(memory_item_id, session_id, user_id, reinforced_at)
		SELECT id, session_id, user_id, COALESCE(last_seen_at, created_at)
		FROM memory_items
		WHERE trim(session_id) <> ''
	`)
	return err
}

func (s *Store) ensureColumn(ctx context.Context, table, column, ddl string) error {
	rows, err := s.db.QueryContext(ctx, `PRAGMA table_info(`+table+`)`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name, ctype string
		var notnull int
		var dflt sql.NullString
		var pk int
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			return err
		}
		if strings.EqualFold(strings.TrimSpace(name), strings.TrimSpace(column)) {
			return nil
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `ALTER TABLE `+table+` ADD COLUMN `+column+` `+ddl)
	return err
}

func (s *Store) tableExists(ctx context.Context, table string) (bool, error) {
	row := s.db.QueryRowContext(ctx, `SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1`, table)
	var one int
	if err := row.Scan(&one); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func (s *Store) dropLegacyMemoryTables(ctx context.Context) error {
	for _, table := range []string{"facts", "memories", "decisions", "memory_fts"} {
		exists, err := s.tableExists(ctx, table)
		if err != nil {
			return err
		}
		if !exists {
			continue
		}
		if _, err := s.db.ExecContext(ctx, `DROP TABLE IF EXISTS `+table); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) UpsertSkill(ctx context.Context, r SkillRecord) error {
	if r.Name == "" {
		return fmt.Errorf("skill name is required")
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO skills(name, description, location, source, updated_at)
		VALUES(?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		ON CONFLICT(name) DO UPDATE SET
			description = excluded.description,
			location = excluded.location,
			source = excluded.source,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
	`, r.Name, r.Description, r.Location, r.Source)
	return err
}

func (s *Store) GetSkill(ctx context.Context, name string) (SkillRecord, error) {
	var r SkillRecord
	err := s.db.QueryRowContext(ctx, `SELECT name, description, location, source FROM skills WHERE name = ?`, name).
		Scan(&r.Name, &r.Description, &r.Location, &r.Source)
	if errors.Is(err, sql.ErrNoRows) {
		return SkillRecord{}, ErrNotFound
	}
	return r, err
}

func (s *Store) ListSkills(ctx context.Context) ([]SkillRecord, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT name, description, location, source FROM skills ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []SkillRecord
	for rows.Next() {
		var r SkillRecord
		if err := rows.Scan(&r.Name, &r.Description, &r.Location, &r.Source); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) DeleteSkill(ctx context.Context, name string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM skills WHERE name = ?`, strings.TrimSpace(name))
	return err
}

func (s *Store) UpsertPlugin(ctx context.Context, r PluginRecord) error {
	if r.Name == "" {
		return fmt.Errorf("plugin name is required")
	}
	if r.Config == nil {
		r.Config = map[string]string{}
	}
	blob, err := json.Marshal(r.Config)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO plugins(
			name, display_name, description, version, plugin_type, location, source, has_skill, enabled, config_json, updated_at
		)
		VALUES(?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT enabled FROM plugins WHERE name = ?), 0), COALESCE((SELECT config_json FROM plugins WHERE name = ?), ?), strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		ON CONFLICT(name) DO UPDATE SET
			display_name = excluded.display_name,
			description = excluded.description,
			version = excluded.version,
			plugin_type = excluded.plugin_type,
			location = excluded.location,
			source = excluded.source,
			has_skill = excluded.has_skill,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
	`, r.Name, r.DisplayName, r.Description, r.Version, r.PluginType, r.Location, r.Source, boolToInt(r.HasSkill), r.Name, r.Name, string(blob))
	return err
}

func (s *Store) GetPlugin(ctx context.Context, name string) (PluginRecord, error) {
	var r PluginRecord
	var hasSkill int
	var enabled int
	var configJSON string
	err := s.db.QueryRowContext(ctx, `
		SELECT name, display_name, description, version, plugin_type, location, source, has_skill, enabled, config_json
		FROM plugins WHERE name = ?
	`, name).Scan(
		&r.Name, &r.DisplayName, &r.Description, &r.Version, &r.PluginType, &r.Location, &r.Source, &hasSkill, &enabled, &configJSON,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return PluginRecord{}, ErrNotFound
	}
	if err != nil {
		return PluginRecord{}, err
	}
	r.HasSkill = hasSkill == 1
	r.Enabled = enabled == 1
	r.Config = map[string]string{}
	if configJSON != "" {
		_ = json.Unmarshal([]byte(configJSON), &r.Config)
	}
	return r, nil
}

func (s *Store) ListPlugins(ctx context.Context) ([]PluginRecord, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT name, display_name, description, version, plugin_type, location, source, has_skill, enabled, config_json
		FROM plugins ORDER BY name
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []PluginRecord{}
	for rows.Next() {
		var r PluginRecord
		var hasSkill int
		var enabled int
		var configJSON string
		if err := rows.Scan(&r.Name, &r.DisplayName, &r.Description, &r.Version, &r.PluginType, &r.Location, &r.Source, &hasSkill, &enabled, &configJSON); err != nil {
			return nil, err
		}
		r.HasSkill = hasSkill == 1
		r.Enabled = enabled == 1
		r.Config = map[string]string{}
		if configJSON != "" {
			_ = json.Unmarshal([]byte(configJSON), &r.Config)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) SetPluginEnabled(ctx context.Context, name string, enabled bool) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE plugins SET enabled = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE name = ?
	`, boolToInt(enabled), name)
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

func (s *Store) SetPluginConfig(ctx context.Context, name string, cfg map[string]string) error {
	if cfg == nil {
		cfg = map[string]string{}
	}
	blob, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE plugins SET config_json = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE name = ?
	`, string(blob), name)
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

type PluginScope struct {
	store      *Store
	pluginName string
}

func (s *Store) ScopePlugin(pluginName string) *PluginScope {
	return &PluginScope{store: s, pluginName: pluginName}
}

func (p *PluginScope) Ensure(ctx context.Context, base PluginRecord) error {
	base.Name = p.pluginName
	return p.store.UpsertPlugin(ctx, base)
}

func (p *PluginScope) Get(ctx context.Context) (PluginRecord, error) {
	return p.store.GetPlugin(ctx, p.pluginName)
}

func (p *PluginScope) IsEnabled(ctx context.Context) (bool, error) {
	r, err := p.store.GetPlugin(ctx, p.pluginName)
	if err != nil {
		return false, err
	}
	return r.Enabled, nil
}

func (p *PluginScope) SetEnabled(ctx context.Context, enabled bool) error {
	return p.store.SetPluginEnabled(ctx, p.pluginName, enabled)
}

func (p *PluginScope) GetConfig(ctx context.Context) (map[string]string, error) {
	r, err := p.store.GetPlugin(ctx, p.pluginName)
	if err != nil {
		return nil, err
	}
	return r.Config, nil
}

func (p *PluginScope) SetConfig(ctx context.Context, cfg map[string]string) error {
	return p.store.SetPluginConfig(ctx, p.pluginName, cfg)
}

func (p *PluginScope) EncryptString(plaintext string) (string, error) {
	return p.store.EncryptString(plaintext)
}

func (p *PluginScope) DecryptString(ciphertext string) (string, error) {
	return p.store.DecryptString(ciphertext)
}

// ─────────────────────────────────────────────────────────────────────
// Channel Operations
// ─────────────────────────────────────────────────────────────────────

// UpsertChannel creates or updates a channel record
func (s *Store) UpsertChannel(ctx context.Context, in ChannelRecord) (ChannelRecord, error) {
	if s == nil || s.db == nil {
		return ChannelRecord{}, fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(in.UserID) == "" {
		in.UserID = "default"
	}
	if strings.TrimSpace(in.ChannelType) == "" {
		return ChannelRecord{}, fmt.Errorf("channel_type is required")
	}
	if strings.TrimSpace(in.ChannelID) == "" {
		return ChannelRecord{}, fmt.Errorf("channel_id is required")
	}

	id := strings.TrimSpace(in.ID)
	if id == "" {
		generated, err := newID()
		if err != nil {
			return ChannelRecord{}, err
		}
		id = generated
	}

	configJSON := "{}"
	if in.Config != nil {
		blob, _ := json.Marshal(in.Config)
		configJSON = string(blob)
	}

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO channels(id, user_id, channel_type, channel_id, bot_token, display_name, config_json, enabled)
		VALUES(?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(user_id, channel_type, channel_id) DO UPDATE SET
			bot_token = COALESCE(excluded.bot_token, channels.bot_token),
			display_name = COALESCE(excluded.display_name, channels.display_name),
			config_json = COALESCE(excluded.config_json, channels.config_json),
			enabled = COALESCE(excluded.enabled, channels.enabled),
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
	`, id, in.UserID, in.ChannelType, in.ChannelID, in.BotToken, in.DisplayName, configJSON, boolToInt(in.Enabled))
	if err != nil {
		return ChannelRecord{}, err
	}

	return s.GetChannel(ctx, id)
}

// GetChannel retrieves a channel by ID
func (s *Store) GetChannel(ctx context.Context, id string) (ChannelRecord, error) {
	if s == nil || s.db == nil {
		return ChannelRecord{}, fmt.Errorf("store unavailable")
	}
	var r ChannelRecord
	var configJSON string
	var created, updated string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, channel_type, channel_id, bot_token, display_name, config_json, enabled, created_at, updated_at
		FROM channels WHERE id = ?
	`, id).Scan(&r.ID, &r.UserID, &r.ChannelType, &r.ChannelID, &r.BotToken, &r.DisplayName, &configJSON, &r.Enabled, &created, &updated)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelRecord{}, ErrNotFound
	}
	if err != nil {
		return ChannelRecord{}, err
	}
	r.Config = map[string]string{}
	if configJSON != "" {
		_ = json.Unmarshal([]byte(configJSON), &r.Config)
	}
	r.CreatedAt, _ = parseTS(created)
	r.UpdatedAt, _ = parseTS(updated)
	return r, nil
}

// GetChannelByUserAndType retrieves a channel by user_id and channel_type
func (s *Store) GetChannelByUserAndType(ctx context.Context, userID, channelType, channelID string) (ChannelRecord, error) {
	if s == nil || s.db == nil {
		return ChannelRecord{}, fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	var r ChannelRecord
	var id string
	var configJSON string
	var created, updated string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, channel_type, channel_id, bot_token, display_name, config_json, enabled, created_at, updated_at
		FROM channels WHERE user_id = ? AND channel_type = ? AND channel_id = ?
	`, userID, channelType, channelID).Scan(&id, &r.UserID, &r.ChannelType, &r.ChannelID, &r.BotToken, &r.DisplayName, &configJSON, &r.Enabled, &created, &updated)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelRecord{}, ErrNotFound
	}
	if err != nil {
		return ChannelRecord{}, err
	}
	r.ID = id
	r.Config = map[string]string{}
	if configJSON != "" {
		_ = json.Unmarshal([]byte(configJSON), &r.Config)
	}
	r.CreatedAt, _ = parseTS(created)
	r.UpdatedAt, _ = parseTS(updated)
	return r, nil
}

// ListChannels lists all channels for a user
func (s *Store) ListChannels(ctx context.Context, userID string) ([]ChannelRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if strings.TrimSpace(userID) == "" {
		userID = "default"
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, channel_type, channel_id, bot_token, display_name, config_json, enabled, created_at, updated_at
		FROM channels WHERE user_id = ?
		ORDER BY created_at DESC
	`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ChannelRecord
	for rows.Next() {
		var r ChannelRecord
		var configJSON string
		var created, updated string
		if err := rows.Scan(&r.ID, &r.UserID, &r.ChannelType, &r.ChannelID, &r.BotToken, &r.DisplayName, &configJSON, &r.Enabled, &created, &updated); err != nil {
			return nil, err
		}
		r.Config = map[string]string{}
		if configJSON != "" {
			_ = json.Unmarshal([]byte(configJSON), &r.Config)
		}
		r.CreatedAt, _ = parseTS(created)
		r.UpdatedAt, _ = parseTS(updated)
		out = append(out, r)
	}
	return out, rows.Err()
}

// GetChannelByTypeAndChatID looks up a channel by channel_type and channel_id (e.g. Telegram chat_id).
// This is used by webhook handlers to map an incoming message to a user.
func (s *Store) GetChannelByTypeAndChatID(ctx context.Context, channelType, chatID string) (ChannelRecord, error) {
	if s == nil || s.db == nil {
		return ChannelRecord{}, fmt.Errorf("store unavailable")
	}
	var r ChannelRecord
	var configJSON string
	var created, updated string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, channel_type, channel_id, bot_token, display_name, config_json, enabled, created_at, updated_at
		FROM channels WHERE channel_type = ? AND channel_id = ?
	`, channelType, chatID).Scan(&r.ID, &r.UserID, &r.ChannelType, &r.ChannelID, &r.BotToken, &r.DisplayName, &configJSON, &r.Enabled, &created, &updated)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelRecord{}, ErrNotFound
	}
	if err != nil {
		return ChannelRecord{}, err
	}
	r.Config = map[string]string{}
	if configJSON != "" {
		_ = json.Unmarshal([]byte(configJSON), &r.Config)
	}
	r.CreatedAt, _ = parseTS(created)
	r.UpdatedAt, _ = parseTS(updated)
	return r, nil
}

// SetChannelEnabled enables or disables a channel
func (s *Store) SetChannelEnabled(ctx context.Context, channelID string, enabled bool) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE channels SET enabled = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?
	`, boolToInt(enabled), channelID)
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

// DeleteChannel deletes a channel by ID
func (s *Store) DeleteChannel(ctx context.Context, channelID string) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM channels WHERE id = ?`, channelID)
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

// AddChannelMessage adds a message to a channel's history
func (s *Store) AddChannelMessage(ctx context.Context, in ChannelMessageRecord) (ChannelMessageRecord, error) {
	if s == nil || s.db == nil {
		return ChannelMessageRecord{}, fmt.Errorf("store unavailable")
	}
	id := strings.TrimSpace(in.ID)
	if id == "" {
		generated, err := newID()
		if err != nil {
			return ChannelMessageRecord{}, err
		}
		id = generated
	}
	if strings.TrimSpace(in.MetadataJSON) == "" {
		in.MetadataJSON = "{}"
	}

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO channel_messages(id, channel_id, message_id, direction, content, metadata_json)
		VALUES(?, ?, ?, ?, ?, ?)
	`, id, in.ChannelID, in.MessageID, in.Direction, in.Content, in.MetadataJSON)
	if err != nil {
		return ChannelMessageRecord{}, err
	}

	in.ID = id
	return in, nil
}

// ListChannelMessages lists messages for a channel
func (s *Store) ListChannelMessages(ctx context.Context, channelID string, limit int) ([]ChannelMessageRecord, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("store unavailable")
	}
	if limit <= 0 || limit > 1000 {
		limit = 100
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, channel_id, message_id, direction, content, metadata_json, timestamp
		FROM channel_messages WHERE channel_id = ?
		ORDER BY timestamp DESC LIMIT ?
	`, channelID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ChannelMessageRecord
	for rows.Next() {
		var r ChannelMessageRecord
		var ts string
		if err := rows.Scan(&r.ID, &r.ChannelID, &r.MessageID, &r.Direction, &r.Content, &r.MetadataJSON, &ts); err != nil {
			return nil, err
		}
		r.Timestamp, _ = parseTS(ts)
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) ScheduleCronJob(ctx context.Context, in CronJobCreate) (CronJobRecord, error) {
	if strings.TrimSpace(in.Module) == "" {
		return CronJobRecord{}, fmt.Errorf("module is required")
	}
	if strings.TrimSpace(in.JobType) == "" {
		return CronJobRecord{}, fmt.Errorf("job type is required")
	}
	if in.RunAt.IsZero() {
		return CronJobRecord{}, fmt.Errorf("run_at is required")
	}
	jobID := strings.TrimSpace(in.ID)
	if jobID == "" {
		generated, err := newID()
		if err != nil {
			return CronJobRecord{}, err
		}
		jobID = generated
	}
	maxAttempts := in.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 5
	}
	payloadJSON := "{}"
	if in.Payload != nil {
		b, err := json.Marshal(in.Payload)
		if err != nil {
			return CronJobRecord{}, err
		}
		payloadJSON = string(b)
	}
	runAt := in.RunAt.UTC()
	nextRunAt := runAt
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO cron_jobs(
			id, module, job_type, payload_json, run_at, interval_s, status, enabled, attempt_count,
			max_attempts, last_error, lock_token, next_run_at, updated_at
		)
		VALUES(?, ?, ?, ?, ?, ?, ?, 1, 0, ?, '', '', ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
	`, jobID, in.Module, in.JobType, payloadJSON, formatTS(runAt), nullableInt64(in.IntervalS), CronStatusScheduled, maxAttempts, formatTS(nextRunAt))
	if err != nil {
		return CronJobRecord{}, err
	}
	return s.GetCronJob(ctx, jobID)
}

func (s *Store) GetCronJob(ctx context.Context, id string) (CronJobRecord, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, module, job_type, payload_json, run_at, interval_s, status, enabled, attempt_count,
			max_attempts, last_error, locked_until, lock_token, last_run_at, next_run_at, created_at, updated_at
		FROM cron_jobs WHERE id = ?
	`, id)
	return scanCronJobRow(row)
}

func (s *Store) ListCronJobsByModule(ctx context.Context, module string) ([]CronJobRecord, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, module, job_type, payload_json, run_at, interval_s, status, enabled, attempt_count,
			max_attempts, last_error, locked_until, lock_token, last_run_at, next_run_at, created_at, updated_at
		FROM cron_jobs WHERE module = ? ORDER BY next_run_at
	`, module)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []CronJobRecord{}
	for rows.Next() {
		rec, err := scanCronJobRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) ListCronJobs(ctx context.Context) ([]CronJobRecord, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, module, job_type, payload_json, run_at, interval_s, status, enabled, attempt_count,
			max_attempts, last_error, locked_until, lock_token, last_run_at, next_run_at, created_at, updated_at
		FROM cron_jobs ORDER BY next_run_at
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []CronJobRecord{}
	for rows.Next() {
		rec, err := scanCronJobRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) CancelCronJob(ctx context.Context, id string) error {
	return s.updateCronStatus(ctx, id, CronStatusCancelled, false)
}

func (s *Store) PauseCronJob(ctx context.Context, id string) error {
	return s.updateCronStatus(ctx, id, CronStatusPaused, false)
}

func (s *Store) ResumeCronJob(ctx context.Context, id string) error {
	return s.updateCronStatus(ctx, id, CronStatusScheduled, true)
}

func (s *Store) updateCronStatus(ctx context.Context, id string, status CronStatus, enabled bool) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE cron_jobs
		SET status = ?, enabled = ?, lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, string(status), boolToInt(enabled), id)
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

func (s *Store) AcquireDueCronJobs(ctx context.Context, now time.Time, limit int, leaseFor time.Duration) ([]CronJobRecord, error) {
	if limit <= 0 {
		limit = 10
	}
	if leaseFor <= 0 {
		leaseFor = 30 * time.Second
	}
	now = now.UTC()
	leaseUntil := now.Add(leaseFor)
	lockToken, err := newID()
	if err != nil {
		return nil, err
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	_, err = tx.ExecContext(ctx, `
		UPDATE cron_jobs
		SET status = ?, lock_token = ?, locked_until = ?, attempt_count = attempt_count + 1,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id IN (
			SELECT id FROM cron_jobs
			WHERE enabled = 1
				AND status IN (?, ?)
				AND next_run_at <= ?
				AND (locked_until IS NULL OR locked_until <= ?)
			ORDER BY next_run_at
			LIMIT ?
		)
	`, string(CronStatusRunning), lockToken, formatTS(leaseUntil), string(CronStatusScheduled), string(CronStatusRetry), formatTS(now), formatTS(now), limit)
	if err != nil {
		return nil, err
	}

	rows, err := tx.QueryContext(ctx, `
		SELECT id, module, job_type, payload_json, run_at, interval_s, status, enabled, attempt_count,
			max_attempts, last_error, locked_until, lock_token, last_run_at, next_run_at, created_at, updated_at
		FROM cron_jobs WHERE lock_token = ?
		ORDER BY next_run_at
	`, lockToken)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	jobs := []CronJobRecord{}
	for rows.Next() {
		rec, scanErr := scanCronJobRows(rows)
		if scanErr != nil {
			err = scanErr
			return nil, err
		}
		jobs = append(jobs, rec)
	}
	if err = rows.Err(); err != nil {
		return nil, err
	}

	if err = tx.Commit(); err != nil {
		return nil, err
	}
	return jobs, nil
}

func (s *Store) MarkCronJobSuccess(ctx context.Context, id string, now time.Time, nextRunAt *time.Time) error {
	now = now.UTC()
	status := CronStatusDone
	enabled := 0
	next := sql.NullString{}
	if nextRunAt != nil {
		n := nextRunAt.UTC()
		next = sql.NullString{String: formatTS(n), Valid: true}
		status = CronStatusScheduled
		enabled = 1
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE cron_jobs
		SET status = ?, enabled = ?, lock_token = '', locked_until = NULL,
			last_error = '', last_run_at = ?, next_run_at = COALESCE(?, next_run_at),
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, string(status), enabled, formatTS(now), next, id)
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

func (s *Store) MarkCronJobFailure(ctx context.Context, id string, now time.Time, errMsg string, nextRetryAt *time.Time, terminal bool) error {
	now = now.UTC()
	status := CronStatusRetry
	enabled := 1
	next := sql.NullString{}
	if nextRetryAt != nil {
		n := nextRetryAt.UTC()
		next = sql.NullString{String: formatTS(n), Valid: true}
	}
	if terminal {
		status = CronStatusFailed
		enabled = 0
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE cron_jobs
		SET status = ?, enabled = ?, lock_token = '', locked_until = NULL,
			last_error = ?, last_run_at = ?, next_run_at = COALESCE(?, next_run_at),
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, string(status), enabled, errMsg, formatTS(now), next, id)
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

type CronScope struct {
	store  *Store
	module string
}

func (s *Store) ScopeCronModule(module string) *CronScope {
	return &CronScope{store: s, module: module}
}

func (c *CronScope) ScheduleOnce(ctx context.Context, jobType string, payload any, runAt time.Time, maxAttempts int) (CronJobRecord, error) {
	return c.store.ScheduleCronJob(ctx, CronJobCreate{Module: c.module, JobType: jobType, Payload: payload, RunAt: runAt, MaxAttempts: maxAttempts})
}

func (c *CronScope) ScheduleRecurring(ctx context.Context, jobType string, payload any, runAt time.Time, intervalS int64, maxAttempts int) (CronJobRecord, error) {
	interval := intervalS
	return c.store.ScheduleCronJob(ctx, CronJobCreate{Module: c.module, JobType: jobType, Payload: payload, RunAt: runAt, IntervalS: &interval, MaxAttempts: maxAttempts})
}

func (c *CronScope) List(ctx context.Context) ([]CronJobRecord, error) {
	return c.store.ListCronJobsByModule(ctx, c.module)
}

func (c *CronScope) Get(ctx context.Context, id string) (CronJobRecord, error) {
	rec, err := c.store.GetCronJob(ctx, id)
	if err != nil {
		return CronJobRecord{}, err
	}
	if rec.Module != c.module {
		return CronJobRecord{}, ErrNotFound
	}
	return rec, nil
}

func (c *CronScope) Cancel(ctx context.Context, id string) error {
	res, err := c.store.db.ExecContext(ctx, `
		UPDATE cron_jobs
		SET status = ?, enabled = 0, lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND module = ?
	`, string(CronStatusCancelled), id, c.module)
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

// ── Removed: agent_tasks CRUD (runtime was removed) ──

// agent task code removed — see git history

func scanCronJobRow(row *sql.Row) (CronJobRecord, error) {
	var rec CronJobRecord
	var runAt, nextRunAt, createdAt, updatedAt string
	var interval sql.NullInt64
	var lockedUntil sql.NullString
	var lastRunAt sql.NullString
	var enabled int
	err := row.Scan(
		&rec.ID, &rec.Module, &rec.JobType, &rec.PayloadJSON, &runAt, &interval, &rec.Status, &enabled,
		&rec.AttemptCount, &rec.MaxAttempts, &rec.LastError, &lockedUntil, &rec.LockToken, &lastRunAt,
		&nextRunAt, &createdAt, &updatedAt,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return CronJobRecord{}, ErrNotFound
	}
	if err != nil {
		return CronJobRecord{}, err
	}
	return hydrateCronJob(rec, runAt, nextRunAt, createdAt, updatedAt, interval, lockedUntil, lastRunAt, enabled)
}

type cronScanner interface {
	Scan(dest ...any) error
}

func scanCronJobRows(scanner cronScanner) (CronJobRecord, error) {
	var rec CronJobRecord
	var runAt, nextRunAt, createdAt, updatedAt string
	var interval sql.NullInt64
	var lockedUntil sql.NullString
	var lastRunAt sql.NullString
	var enabled int
	if err := scanner.Scan(
		&rec.ID, &rec.Module, &rec.JobType, &rec.PayloadJSON, &runAt, &interval, &rec.Status, &enabled,
		&rec.AttemptCount, &rec.MaxAttempts, &rec.LastError, &lockedUntil, &rec.LockToken, &lastRunAt,
		&nextRunAt, &createdAt, &updatedAt,
	); err != nil {
		return CronJobRecord{}, err
	}
	return hydrateCronJob(rec, runAt, nextRunAt, createdAt, updatedAt, interval, lockedUntil, lastRunAt, enabled)
}

func hydrateCronJob(rec CronJobRecord, runAt, nextRunAt, createdAt, updatedAt string, interval sql.NullInt64, lockedUntil sql.NullString, lastRunAt sql.NullString, enabled int) (CronJobRecord, error) {
	parsedRunAt, err := parseTS(runAt)
	if err != nil {
		return CronJobRecord{}, err
	}
	parsedNext, err := parseTS(nextRunAt)
	if err != nil {
		return CronJobRecord{}, err
	}
	parsedCreated, err := parseTS(createdAt)
	if err != nil {
		return CronJobRecord{}, err
	}
	parsedUpdated, err := parseTS(updatedAt)
	if err != nil {
		return CronJobRecord{}, err
	}
	rec.RunAt = parsedRunAt
	rec.NextRunAt = parsedNext
	rec.CreatedAt = parsedCreated
	rec.UpdatedAt = parsedUpdated
	rec.Enabled = enabled == 1
	if interval.Valid {
		v := interval.Int64
		rec.IntervalS = &v
	}
	if lockedUntil.Valid {
		v, e := parseTS(lockedUntil.String)
		if e != nil {
			return CronJobRecord{}, e
		}
		rec.LockedUntil = &v
	}
	if lastRunAt.Valid {
		v, e := parseTS(lastRunAt.String)
		if e != nil {
			return CronJobRecord{}, e
		}
		rec.LastRunAt = &v
	}
	return rec, nil
}

func parseTS(v string) (time.Time, error) {
	t, err := time.Parse(time.RFC3339Nano, v)
	if err == nil {
		return t.UTC(), nil
	}
	t, err = time.Parse(time.RFC3339, v)
	if err != nil {
		return time.Time{}, err
	}
	return t.UTC(), nil
}

func formatTS(t time.Time) string {
	return t.UTC().Format(time.RFC3339Nano)
}

func nullableInt64(v *int64) sql.NullInt64 {
	if v == nil {
		return sql.NullInt64{}
	}
	return sql.NullInt64{Int64: *v, Valid: true}
}

func newID() (string, error) {
	b := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func (s *Store) Encrypt(plaintext []byte) (string, error) {
	if s.aead == nil {
		return "", ErrCryptoUnavailable
	}
	nonce := make([]byte, s.aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}
	ciphertext := s.aead.Seal(nil, nonce, plaintext, nil)
	blob := append(nonce, ciphertext...)
	return "enc:v1:" + base64.StdEncoding.EncodeToString(blob), nil
}

func (s *Store) Decrypt(ciphertext string) ([]byte, error) {
	if s.aead == nil {
		return nil, ErrCryptoUnavailable
	}
	prefix := "enc:v1:"
	if !strings.HasPrefix(ciphertext, prefix) {
		return nil, ErrInvalidCiphertext
	}
	encoded := strings.TrimPrefix(ciphertext, prefix)
	blob, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, ErrInvalidCiphertext
	}
	ns := s.aead.NonceSize()
	if len(blob) <= ns {
		return nil, ErrInvalidCiphertext
	}
	nonce := blob[:ns]
	data := blob[ns:]
	plain, err := s.aead.Open(nil, nonce, data, nil)
	if err != nil {
		return nil, ErrInvalidCiphertext
	}
	return plain, nil
}

func (s *Store) EncryptString(plaintext string) (string, error) {
	return s.Encrypt([]byte(plaintext))
}

func (s *Store) DecryptString(ciphertext string) (string, error) {
	plain, err := s.Decrypt(ciphertext)
	if err != nil {
		return "", err
	}
	return string(plain), nil
}

// ConfigureCrypto sets up AES-256-GCM encryption for secrets.
// Pass an empty key to disable encryption.
func (s *Store) ConfigureCrypto(stateKey string) error {
	raw := strings.TrimSpace(stateKey)
	if raw == "" {
		s.aead = nil
		return nil
	}
	key, err := parseAES256Key(raw)
	if err != nil {
		return fmt.Errorf("invalid AGENT_STATE_KEY: %w", err)
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return err
	}
	s.aead = aead
	return nil
}

func parseAES256Key(raw string) ([]byte, error) {
	if decoded, err := base64.StdEncoding.DecodeString(raw); err == nil {
		if len(decoded) == 32 {
			return decoded, nil
		}
	}
	if len(raw) == 32 {
		return []byte(raw), nil
	}
	return nil, fmt.Errorf("expected 32-byte raw key or base64-encoded 32-byte key")
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
