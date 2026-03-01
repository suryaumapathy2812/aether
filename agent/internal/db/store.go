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

	_ "modernc.org/sqlite"
)

var (
	ErrNotFound          = errors.New("not found")
	ErrCryptoUnavailable = errors.New("encryption key is not configured")
	ErrInvalidCiphertext = errors.New("invalid ciphertext")
)

type Store struct {
	db   *instrumentedDB
	aead cipher.AEAD
}

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

type AgentTaskStatus string

const (
	AgentTaskQueued       AgentTaskStatus = "queued"
	AgentTaskRunning      AgentTaskStatus = "running"
	AgentTaskWaitingInput AgentTaskStatus = "waiting_input"
	AgentTaskCompleted    AgentTaskStatus = "completed"
	AgentTaskFailed       AgentTaskStatus = "failed"
	AgentTaskCancelled    AgentTaskStatus = "cancelled"
	AgentTaskTimedOut     AgentTaskStatus = "timed_out"
)

type AgentTaskCreate struct {
	ID        string
	UserID    string
	SessionID string
	Title     string
	Goal      string
	Priority  int
	MaxSteps  int
	Deadline  *time.Time
	Metadata  map[string]any
}

type AgentTaskRecord struct {
	ID              string
	UserID          string
	SessionID       string
	Title           string
	Goal            string
	Status          AgentTaskStatus
	Priority        int
	MaxSteps        int
	StepCount       int
	CancelRequested bool
	Deadline        *time.Time
	StartedAt       *time.Time
	FinishedAt      *time.Time
	LastError       string
	ResultSummary   string
	ResultJSON      string
	MetadataJSON    string
	LockToken       string
	LockedUntil     *time.Time
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

type AgentTaskMessage struct {
	ID          int64
	TaskID      string
	Role        string
	ContentJSON string
	CreatedAt   time.Time
}

type AgentTaskEvent struct {
	ID          int64
	TaskID      string
	Kind        string
	PayloadJSON string
	CreatedAt   time.Time
}

func Open(path string) (*Store, error) {
	if path == "" {
		return nil, fmt.Errorf("db path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	if _, err := db.Exec(`PRAGMA journal_mode = WAL;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	if _, err := db.Exec(`PRAGMA busy_timeout = 5000;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	if _, err := db.Exec(`PRAGMA synchronous = NORMAL;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	store := &Store{db: newInstrumentedDB(db)}
	if err := store.configureCryptoFromEnv(); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := store.migrate(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func OpenInAssets(assetsDir string) (*Store, error) {
	if assetsDir == "" {
		return nil, fmt.Errorf("assets directory is required")
	}
	return Open(filepath.Join(assetsDir, "state.db"))
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
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
		`CREATE TABLE IF NOT EXISTS agent_tasks (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL DEFAULT '',
			session_id TEXT NOT NULL DEFAULT '',
			title TEXT NOT NULL DEFAULT '',
			goal TEXT NOT NULL DEFAULT '',
			status TEXT NOT NULL DEFAULT 'queued',
			priority INTEGER NOT NULL DEFAULT 0,
			max_steps INTEGER NOT NULL DEFAULT 10,
			step_count INTEGER NOT NULL DEFAULT 0,
			cancel_requested INTEGER NOT NULL DEFAULT 0,
			deadline_at TEXT,
			started_at TEXT,
			finished_at TEXT,
			last_error TEXT NOT NULL DEFAULT '',
			result_summary TEXT NOT NULL DEFAULT '',
			result_json TEXT NOT NULL DEFAULT '{}',
			metadata_json TEXT NOT NULL DEFAULT '{}',
			lock_token TEXT NOT NULL DEFAULT '',
			locked_until TEXT,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
		);`,
		`CREATE INDEX IF NOT EXISTS idx_agent_tasks_queue ON agent_tasks(status, priority, created_at);`,
		`CREATE INDEX IF NOT EXISTS idx_agent_tasks_owner ON agent_tasks(user_id, session_id, created_at);`,
		`CREATE INDEX IF NOT EXISTS idx_agent_tasks_lock ON agent_tasks(locked_until);`,
		`CREATE TABLE IF NOT EXISTS agent_task_messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			task_id TEXT NOT NULL,
			role TEXT NOT NULL,
			content_json TEXT NOT NULL,
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_agent_task_messages_task ON agent_task_messages(task_id, id);`,
		`CREATE TABLE IF NOT EXISTS agent_task_events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			task_id TEXT NOT NULL,
			kind TEXT NOT NULL,
			payload_json TEXT NOT NULL DEFAULT '{}',
			created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
			FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_agent_task_events_task ON agent_task_events(task_id, id);`,
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
		`CREATE TABLE IF NOT EXISTS facts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			fact TEXT NOT NULL,
			fact_key TEXT NOT NULL,
			source_conversation_id INTEGER,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			UNIQUE(user_id, fact_key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_facts_user_updated_at ON facts(user_id, updated_at DESC);`,
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
		`CREATE TABLE IF NOT EXISTS memories (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			memory TEXT NOT NULL,
			memory_key TEXT NOT NULL,
			category TEXT NOT NULL DEFAULT 'episodic',
			source_conversation_id INTEGER,
			created_at TEXT NOT NULL,
			expires_at TEXT,
			confidence REAL NOT NULL DEFAULT 1.0,
			UNIQUE(user_id, memory_key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_memories_user_created_at ON memories(user_id, created_at DESC);`,
		`CREATE TABLE IF NOT EXISTS decisions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT NOT NULL DEFAULT 'default',
			decision TEXT NOT NULL,
			decision_key TEXT NOT NULL,
			category TEXT NOT NULL DEFAULT 'preference',
			source TEXT NOT NULL DEFAULT 'extracted',
			source_conversation_id INTEGER,
			active INTEGER NOT NULL DEFAULT 1,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			confidence REAL NOT NULL DEFAULT 1.0,
			UNIQUE(user_id, decision_key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_decisions_user_updated_at ON decisions(user_id, updated_at DESC);`,
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
	}
	for _, stmt := range stmts {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return err
		}
	}
	if err := s.ensureColumn(ctx, "conversations", "user_content_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		return err
	}
	return nil
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

func (s *Store) CreateAgentTask(ctx context.Context, in AgentTaskCreate) (AgentTaskRecord, error) {
	if strings.TrimSpace(in.UserID) == "" {
		return AgentTaskRecord{}, fmt.Errorf("user_id is required")
	}
	if strings.TrimSpace(in.Title) == "" {
		return AgentTaskRecord{}, fmt.Errorf("title is required")
	}
	if strings.TrimSpace(in.Goal) == "" {
		return AgentTaskRecord{}, fmt.Errorf("goal is required")
	}
	id := strings.TrimSpace(in.ID)
	if id == "" {
		generated, err := newID()
		if err != nil {
			return AgentTaskRecord{}, err
		}
		id = generated
	}
	priority := in.Priority
	maxSteps := in.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 10
	}
	metadataJSON := "{}"
	if in.Metadata != nil {
		b, err := json.Marshal(in.Metadata)
		if err != nil {
			return AgentTaskRecord{}, err
		}
		metadataJSON = string(b)
	}
	deadline := sql.NullString{}
	if in.Deadline != nil {
		d := in.Deadline.UTC()
		deadline = sql.NullString{String: formatTS(d), Valid: true}
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO agent_tasks(
			id, user_id, session_id, title, goal, status, priority, max_steps, step_count,
			cancel_requested, deadline_at, result_json, metadata_json, lock_token, locked_until, updated_at
		)
		VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, '{}', ?, '', NULL, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
	`, id, in.UserID, in.SessionID, in.Title, in.Goal, string(AgentTaskQueued), priority, maxSteps, deadline, metadataJSON)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if err := s.AppendAgentTaskMessage(ctx, id, "user", map[string]any{"role": "user", "content": in.Goal}); err != nil {
		return AgentTaskRecord{}, err
	}
	return s.GetAgentTask(ctx, id)
}

func (s *Store) GetAgentTask(ctx context.Context, id string) (AgentTaskRecord, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, session_id, title, goal, status, priority, max_steps, step_count,
			cancel_requested, deadline_at, started_at, finished_at, last_error, result_summary, result_json,
			metadata_json, lock_token, locked_until, created_at, updated_at
		FROM agent_tasks WHERE id = ?
	`, id)
	return scanAgentTaskRow(row)
}

func (s *Store) ListAgentTasksByUser(ctx context.Context, userID string, limit int) ([]AgentTaskRecord, error) {
	return s.ListAgentTasksByUserWithStatus(ctx, userID, "", limit)
}

func (s *Store) ListAgentTasksByUserWithStatus(ctx context.Context, userID string, status string, limit int) ([]AgentTaskRecord, error) {
	if strings.TrimSpace(userID) == "" {
		return nil, fmt.Errorf("user_id is required")
	}
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	query := `
		SELECT id, user_id, session_id, title, goal, status, priority, max_steps, step_count,
			cancel_requested, deadline_at, started_at, finished_at, last_error, result_summary, result_json,
			metadata_json, lock_token, locked_until, created_at, updated_at
		FROM agent_tasks WHERE user_id = ?`
	args := []any{userID}
	if strings.TrimSpace(status) != "" {
		query += ` AND status = ?`
		args = append(args, status)
	}
	query += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []AgentTaskRecord{}
	for rows.Next() {
		rec, err := scanAgentTaskRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) RequestCancelAgentTask(ctx context.Context, id string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET cancel_requested = 1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, id)
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

func (s *Store) ClaimNextAgentTask(ctx context.Context, now time.Time, leaseFor time.Duration) (AgentTaskRecord, error) {
	now = now.UTC()
	if leaseFor <= 0 {
		leaseFor = 45 * time.Second
	}
	leaseUntil := now.Add(leaseFor)
	lockToken, err := newID()
	if err != nil {
		return AgentTaskRecord{}, err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	defer func() { _ = tx.Rollback() }()

	res, err := tx.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, lock_token = ?, locked_until = ?, started_at = COALESCE(started_at, ?),
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = (
			SELECT id FROM agent_tasks
			WHERE status IN (?, ?) AND cancel_requested = 0
				AND (locked_until IS NULL OR locked_until <= ?)
			ORDER BY priority DESC, created_at ASC
			LIMIT 1
		)
	`, string(AgentTaskRunning), lockToken, formatTS(leaseUntil), formatTS(now), string(AgentTaskQueued), string(AgentTaskRunning), formatTS(now))
	if err != nil {
		return AgentTaskRecord{}, err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if affected == 0 {
		return AgentTaskRecord{}, ErrNotFound
	}
	row := tx.QueryRowContext(ctx, `
		SELECT id, user_id, session_id, title, goal, status, priority, max_steps, step_count,
			cancel_requested, deadline_at, started_at, finished_at, last_error, result_summary, result_json,
			metadata_json, lock_token, locked_until, created_at, updated_at
		FROM agent_tasks WHERE lock_token = ?
		ORDER BY updated_at DESC LIMIT 1
	`, lockToken)
	rec, err := scanAgentTaskRow(row)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if err := tx.Commit(); err != nil {
		return AgentTaskRecord{}, err
	}
	return rec, nil
}

func (s *Store) RenewAgentTaskLease(ctx context.Context, id, lockToken string, leaseUntil time.Time) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET locked_until = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND lock_token = ?
	`, formatTS(leaseUntil.UTC()), id, lockToken)
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

func (s *Store) IncrementAgentTaskStep(ctx context.Context, id, lockToken string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET step_count = step_count + 1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND lock_token = ?
	`, id, lockToken)
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

func (s *Store) CompleteAgentTask(ctx context.Context, id, lockToken, summary string, result any) error {
	resultJSON := "{}"
	if result != nil {
		b, err := json.Marshal(result)
		if err != nil {
			return err
		}
		resultJSON = string(b)
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, finished_at = ?, result_summary = ?, result_json = ?,
			last_error = '', lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND lock_token = ?
	`, string(AgentTaskCompleted), formatTS(time.Now().UTC()), summary, resultJSON, id, lockToken)
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

func (s *Store) FailAgentTask(ctx context.Context, id, lockToken, errMsg string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, finished_at = ?, last_error = ?, lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND lock_token = ?
	`, string(AgentTaskFailed), formatTS(time.Now().UTC()), errMsg, id, lockToken)
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

func (s *Store) CancelAgentTask(ctx context.Context, id, lockToken string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, finished_at = ?, lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND lock_token = ?
	`, string(AgentTaskCancelled), formatTS(time.Now().UTC()), id, lockToken)
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

func (s *Store) AppendAgentTaskMessage(ctx context.Context, taskID, role string, content any) error {
	if strings.TrimSpace(taskID) == "" {
		return fmt.Errorf("task_id is required")
	}
	if strings.TrimSpace(role) == "" {
		return fmt.Errorf("role is required")
	}
	b, err := json.Marshal(content)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO agent_task_messages(task_id, role, content_json)
		VALUES(?, ?, ?)
	`, taskID, role, string(b))
	return err
}

func (s *Store) ListAgentTaskMessages(ctx context.Context, taskID string, limit int) ([]AgentTaskMessage, error) {
	if limit <= 0 || limit > 2000 {
		limit = 500
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, task_id, role, content_json, created_at
		FROM agent_task_messages
		WHERE task_id = ?
		ORDER BY id ASC
		LIMIT ?
	`, taskID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []AgentTaskMessage{}
	for rows.Next() {
		var rec AgentTaskMessage
		var createdAt string
		if err := rows.Scan(&rec.ID, &rec.TaskID, &rec.Role, &rec.ContentJSON, &createdAt); err != nil {
			return nil, err
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

func (s *Store) AppendAgentTaskEvent(ctx context.Context, taskID, kind string, payload any) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO agent_task_events(task_id, kind, payload_json)
		VALUES(?, ?, ?)
	`, taskID, kind, string(b))
	return err
}

func (s *Store) ListAgentTaskEvents(ctx context.Context, taskID string, limit int) ([]AgentTaskEvent, error) {
	if limit <= 0 || limit > 2000 {
		limit = 200
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, task_id, kind, payload_json, created_at
		FROM agent_task_events
		WHERE task_id = ?
		ORDER BY id ASC
		LIMIT ?
	`, taskID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []AgentTaskEvent{}
	for rows.Next() {
		var rec AgentTaskEvent
		var createdAt string
		if err := rows.Scan(&rec.ID, &rec.TaskID, &rec.Kind, &rec.PayloadJSON, &createdAt); err != nil {
			return nil, err
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

func (s *Store) CountAgentTaskEventsAfter(ctx context.Context, taskID string, afterID int64) (int, error) {
	var n int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(1) FROM agent_task_events WHERE task_id = ? AND id > ?`, taskID, afterID).Scan(&n)
	return n, err
}

func (s *Store) IsAgentTaskCancelRequested(ctx context.Context, id string) (bool, error) {
	var cancelRequested int
	err := s.db.QueryRowContext(ctx, `SELECT cancel_requested FROM agent_tasks WHERE id = ?`, id).Scan(&cancelRequested)
	if errors.Is(err, sql.ErrNoRows) {
		return false, ErrNotFound
	}
	if err != nil {
		return false, err
	}
	return cancelRequested == 1, nil
}

func (s *Store) SetAgentTaskWaitingInput(ctx context.Context, id, lockToken, summary string, payload any) error {
	if strings.TrimSpace(summary) == "" {
		summary = "Awaiting human confirmation"
	}
	if err := s.AppendAgentTaskEvent(ctx, id, "status", map[string]any{
		"status":  string(AgentTaskWaitingInput),
		"message": summary,
		"payload": payload,
	}); err != nil {
		return err
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, result_summary = ?, lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ? AND lock_token = ?
	`, string(AgentTaskWaitingInput), summary, id, lockToken)
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

func (s *Store) ResumeAgentTask(ctx context.Context, id, userID, humanMessage string) (AgentTaskRecord, error) {
	task, err := s.GetAgentTask(ctx, id)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if strings.TrimSpace(userID) != "" && task.UserID != userID {
		return AgentTaskRecord{}, ErrNotFound
	}
	if task.Status != AgentTaskWaitingInput {
		return AgentTaskRecord{}, fmt.Errorf("task is not waiting for input")
	}
	msg := strings.TrimSpace(humanMessage)
	if msg == "" {
		msg = "Human approved. Continue execution."
	}
	if q := s.latestWaitingQuestion(ctx, id); strings.TrimSpace(q) != "" {
		msg = fmt.Sprintf(
			"Human answered the pending question.\nQuestion: %s\nAnswer: %s\nUse this answer directly and continue. Do not ask the same question again unless the answer is empty or invalid.",
			q,
			msg,
		)
	}
	if err := s.AppendAgentTaskMessage(ctx, id, "user", map[string]any{"role": "user", "content": msg}); err != nil {
		return AgentTaskRecord{}, err
	}
	if err := s.AppendAgentTaskEvent(ctx, id, "status", map[string]any{"status": string(AgentTaskQueued), "message": "Task resumed by human"}); err != nil {
		return AgentTaskRecord{}, err
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, cancel_requested = 0, finished_at = NULL,
			last_error = '', lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, string(AgentTaskQueued), id)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if affected == 0 {
		return AgentTaskRecord{}, ErrNotFound
	}
	return s.GetAgentTask(ctx, id)
}

func (s *Store) latestWaitingQuestion(ctx context.Context, taskID string) string {
	events, err := s.ListAgentTaskEvents(ctx, taskID, 200)
	if err != nil {
		return ""
	}
	for i := len(events) - 1; i >= 0; i-- {
		var payload map[string]any
		if err := json.Unmarshal([]byte(events[i].PayloadJSON), &payload); err != nil {
			continue
		}
		status, _ := payload["status"].(string)
		if status != string(AgentTaskWaitingInput) {
			continue
		}
		if q, ok := payload["question"].(string); ok && strings.TrimSpace(q) != "" {
			return strings.TrimSpace(q)
		}
		if nested, ok := payload["payload"].(map[string]any); ok {
			if q, ok := nested["question"].(string); ok && strings.TrimSpace(q) != "" {
				return strings.TrimSpace(q)
			}
		}
		if m, ok := payload["message"].(string); ok && strings.TrimSpace(m) != "" && strings.Contains(strings.ToLower(m), "?") {
			return strings.TrimSpace(m)
		}
	}
	return ""
}

func (s *Store) RejectAgentTask(ctx context.Context, id, userID, reason string) (AgentTaskRecord, error) {
	task, err := s.GetAgentTask(ctx, id)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if strings.TrimSpace(userID) != "" && task.UserID != userID {
		return AgentTaskRecord{}, ErrNotFound
	}
	if task.Status != AgentTaskWaitingInput {
		return AgentTaskRecord{}, fmt.Errorf("task is not waiting for input")
	}
	msg := strings.TrimSpace(reason)
	if msg == "" {
		msg = "Human rejected request. Stop execution."
	}
	if err := s.AppendAgentTaskMessage(ctx, id, "user", map[string]any{"role": "user", "content": msg}); err != nil {
		return AgentTaskRecord{}, err
	}
	if err := s.AppendAgentTaskEvent(ctx, id, "decision", map[string]any{
		"decision": "rejected",
		"reason":   msg,
	}); err != nil {
		return AgentTaskRecord{}, err
	}
	res, err := s.db.ExecContext(ctx, `
		UPDATE agent_tasks
		SET status = ?, finished_at = ?, last_error = ?, lock_token = '', locked_until = NULL,
			updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE id = ?
	`, string(AgentTaskCancelled), formatTS(time.Now().UTC()), "Cancelled by human: "+msg, id)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return AgentTaskRecord{}, err
	}
	if affected == 0 {
		return AgentTaskRecord{}, ErrNotFound
	}
	return s.GetAgentTask(ctx, id)
}

type agentTaskScanner interface {
	Scan(dest ...any) error
}

func scanAgentTaskRow(row *sql.Row) (AgentTaskRecord, error) {
	return scanAgentTaskRows(row)
}

func scanAgentTaskRows(scanner agentTaskScanner) (AgentTaskRecord, error) {
	var rec AgentTaskRecord
	var cancelRequested int
	var deadline sql.NullString
	var started sql.NullString
	var finished sql.NullString
	var lockedUntil sql.NullString
	var createdAt string
	var updatedAt string
	if err := scanner.Scan(
		&rec.ID, &rec.UserID, &rec.SessionID, &rec.Title, &rec.Goal, &rec.Status, &rec.Priority,
		&rec.MaxSteps, &rec.StepCount, &cancelRequested, &deadline, &started, &finished,
		&rec.LastError, &rec.ResultSummary, &rec.ResultJSON, &rec.MetadataJSON, &rec.LockToken,
		&lockedUntil, &createdAt, &updatedAt,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return AgentTaskRecord{}, ErrNotFound
		}
		return AgentTaskRecord{}, err
	}
	rec.CancelRequested = cancelRequested == 1
	if deadline.Valid {
		ts, err := parseTS(deadline.String)
		if err != nil {
			return AgentTaskRecord{}, err
		}
		rec.Deadline = &ts
	}
	if started.Valid {
		ts, err := parseTS(started.String)
		if err != nil {
			return AgentTaskRecord{}, err
		}
		rec.StartedAt = &ts
	}
	if finished.Valid {
		ts, err := parseTS(finished.String)
		if err != nil {
			return AgentTaskRecord{}, err
		}
		rec.FinishedAt = &ts
	}
	if lockedUntil.Valid {
		ts, err := parseTS(lockedUntil.String)
		if err != nil {
			return AgentTaskRecord{}, err
		}
		rec.LockedUntil = &ts
	}
	createdTS, err := parseTS(createdAt)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	updatedTS, err := parseTS(updatedAt)
	if err != nil {
		return AgentTaskRecord{}, err
	}
	rec.CreatedAt = createdTS
	rec.UpdatedAt = updatedTS
	return rec, nil
}

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

func (s *Store) configureCryptoFromEnv() error {
	raw := strings.TrimSpace(os.Getenv("AGENT_STATE_KEY"))
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
