package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	"github.com/docker/docker/errdefs"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/caddy"
)

type Target struct {
	Host string
	Port int
}

type ManagerConfig struct {
	Image             string
	Network           string
	IdleTimeout       time.Duration
	AgentPort         int
	HealthTimeout     time.Duration
	AdminToken        string
	DirectTokenSecret string

	OpenAIAPIKey  string
	OpenAIBaseURL string
	OpenAIModel   string
	AgentStateKey string
	AssetsRoot    string
	VapidPublic   string
	VapidPrivate  string
	VapidSubject  string
	S3Bucket      string
	S3Template    string
	S3Region      string
	S3AccessKey   string
	S3SecretKey   string
	S3Endpoint    string
	S3PublicBase  string
	S3ForcePath   string
	S3PutTTL      string
	S3GetTTL      string
	UpdateRepo    string
	UpdateToken   string

	// PublicBaseURL is the public base URL (e.g. "https://aether.example.com")
	// from AETHER_PUBLIC_BASE_URL. Passed to agent containers so the agent
	// can construct webhook callback URLs for Telegram, WhatsApp, etc.
	PublicBaseURL     string
	DirectAgentDomain string
	RouteManager      *caddy.RouteManager

	// OAuthEnvVars holds "KEY=VALUE" pairs for OAuth provider credentials
	// (e.g. GOOGLE_CLIENT_ID=…, SPOTIFY_CLIENT_SECRET=…) that are forwarded
	// to agent containers so plugins can use env-backed credentials.
	OAuthEnvVars []string
}

type Manager struct {
	docker     *client.Client
	db         *pgxpool.Pool
	httpClient *http.Client
	cfg        ManagerConfig

	mu         sync.Mutex
	userLocks  map[string]*sync.Mutex
	lastActive map[string]time.Time
}

type record struct {
	ID              string
	UserID          string
	ContainerID     string
	SubdomainPrefix string
	Host            string
	Port            int
	Status          string
}

func NewManager(ctx context.Context, db *pgxpool.Pool, cfg ManagerConfig) (*Manager, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	m := &Manager{
		docker:     cli,
		db:         db,
		httpClient: &http.Client{Timeout: 3 * time.Second},
		cfg:        cfg,
		userLocks:  map[string]*sync.Mutex{},
		lastActive: map[string]time.Time{},
	}

	if strings.TrimSpace(m.cfg.Network) == "" {
		if n, e := m.detectCurrentNetwork(ctx); e == nil {
			m.cfg.Network = n
		}
	}
	if strings.TrimSpace(m.cfg.Network) == "" {
		return nil, errors.New("AGENT_NETWORK is required when network auto-detection fails")
	}
	m.ensureAssetsRoot()
	if m.cfg.AgentPort <= 0 {
		m.cfg.AgentPort = 8000
	}
	if m.cfg.HealthTimeout <= 0 {
		m.cfg.HealthTimeout = 30 * time.Second
	}

	if err := m.ensureImage(ctx, m.cfg.Image); err != nil {
		return nil, fmt.Errorf("failed to initialize agent image %q: %w", m.cfg.Image, err)
	}
	return m, nil
}

func (m *Manager) Close() error {
	if m == nil || m.docker == nil {
		return nil
	}
	return m.docker.Close()
}

func (m *Manager) Provision(ctx context.Context, userID string) (Target, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return Target{}, errors.New("user id is required")
	}
	lock := m.lockForUser(userID)
	lock.Lock()
	defer lock.Unlock()
	return m.provisionLocked(ctx, userID)
}

func (m *Manager) provisionLocked(ctx context.Context, userID string) (Target, error) {
	rec, err := m.loadRecord(ctx, userID)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return Target{}, err
	}
	if err == nil {
		prefix, err := m.ensureSubdomainPrefix(ctx, userID, rec.SubdomainPrefix)
		if err != nil {
			return Target{}, err
		}
		rec.SubdomainPrefix = prefix
		target, ok, e := m.ensureRecordRunning(ctx, rec)
		if e != nil {
			return Target{}, e
		}
		if ok {
			m.RecordActivity(userID)
			return target, nil
		}
	}

	containerName := containerNameForUser(userID)
	if err := m.ensureImage(ctx, m.cfg.Image); err != nil {
		return Target{}, err
	}

	ctrID, err := m.ensureContainer(ctx, userID, containerName, "")
	if err != nil {
		return Target{}, err
	}
	if err := m.ensureContainerStarted(ctx, ctrID); err != nil {
		return Target{}, err
	}

	inspect, err := m.docker.ContainerInspect(ctx, ctrID)
	if err != nil {
		return Target{}, err
	}
	host := m.agentHostFromInspect(inspect, containerName)
	target := Target{Host: host, Port: m.cfg.AgentPort}
	if err := m.waitHealthy(ctx, ctrID, target.Host, target.Port); err != nil {
		return Target{}, err
	}
	if err := m.ensureUserMediaBucket(ctx, userID, target); err != nil {
		log.Printf("media bucket ensure warning (new provision): user=%s err=%v", userID, err)
	}

	agentID := agentIDForUser(userID)
	prefix, err := m.ensureSubdomainPrefix(ctx, userID, "")
	if err != nil {
		return Target{}, err
	}
	_, err = m.db.Exec(ctx, `
		INSERT INTO agents (id, user_id, container_id, subdomain_prefix, host, port, status, registered_at, last_health, stopped_at)
		VALUES ($1, $2, $3, $4, $5, $6, 'running', now(), now(), NULL)
		ON CONFLICT (user_id) DO UPDATE SET
			id = EXCLUDED.id,
			container_id = EXCLUDED.container_id,
			subdomain_prefix = COALESCE(NULLIF(agents.subdomain_prefix, ''), EXCLUDED.subdomain_prefix),
			host = EXCLUDED.host,
			port = EXCLUDED.port,
			status = 'running',
			last_health = now(),
			stopped_at = NULL
	`, agentID, userID, ctrID, prefix, target.Host, target.Port)
	if err != nil {
		return Target{}, err
	}
	if err := m.syncDirectRoute(prefix, target); err != nil {
		log.Printf("direct route sync warning (new provision): user=%s prefix=%s err=%v", userID, prefix, err)
	}

	m.RecordActivity(userID)
	return target, nil
}

func (m *Manager) RunAvailabilityReconciler(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	m.reconcileOnce(ctx)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.reconcileOnce(ctx)
		}
	}
}

func (m *Manager) reconcileOnce(ctx context.Context) {
	recs, err := m.listManagedRecords(ctx)
	if err != nil {
		log.Printf("agent reconciler: list managed records failed: %v", err)
		return
	}
	for _, rec := range recs {
		m.reconcileRecord(ctx, rec)
	}
}

func (m *Manager) reconcileRecord(ctx context.Context, rec record) {
	lock := m.lockForUser(rec.UserID)
	lock.Lock()
	defer lock.Unlock()

	target, err := m.provisionLocked(ctx, rec.UserID)
	if err != nil {
		log.Printf("agent reconciler: provision check failed: user=%s err=%v", rec.UserID, err)
		return
	}

	current, err := m.loadRecord(ctx, rec.UserID)
	if err != nil {
		log.Printf("agent reconciler: reload record failed: user=%s err=%v", rec.UserID, err)
		return
	}

	if err := m.ensureDirectRouteHealthy(ctx, current.SubdomainPrefix, target); err != nil {
		log.Printf("agent reconciler: direct health failed: user=%s prefix=%s err=%v", rec.UserID, current.SubdomainPrefix, err)
	}
}

func (m *Manager) ensureDirectRouteHealthy(ctx context.Context, prefix string, target Target) error {
	pfx := strings.TrimSpace(prefix)
	domain := strings.Trim(strings.TrimSpace(m.cfg.DirectAgentDomain), ".")
	if pfx == "" || domain == "" {
		return nil
	}
	url := directHealthURL(pfx, domain)
	if err := m.checkHealthURL(ctx, url); err == nil {
		return nil
	}
	if err := m.syncDirectRoute(pfx, target); err != nil {
		return fmt.Errorf("sync direct route: %w", err)
	}
	if err := m.checkHealthURL(ctx, url); err != nil {
		return fmt.Errorf("direct health check failed after route sync: %w", err)
	}
	return nil
}

func (m *Manager) checkHealthURL(ctx context.Context, healthURL string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	return nil
}

func directHealthURL(prefix, domain string) string {
	return fmt.Sprintf("https://%s.%s/health", strings.TrimSpace(prefix), strings.Trim(strings.TrimSpace(domain), "."))
}

func (m *Manager) RecordActivity(userID string) {
	m.mu.Lock()
	m.lastActive[userID] = time.Now()
	m.mu.Unlock()
}

func (m *Manager) RunIdleReaper(ctx context.Context) {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.reapOnce(ctx)
		}
	}
}

func (m *Manager) reapOnce(ctx context.Context) {
	if m.cfg.IdleTimeout <= 0 {
		return
	}
	now := time.Now()
	users := make([]string, 0)

	m.mu.Lock()
	for userID, ts := range m.lastActive {
		if now.Sub(ts) >= m.cfg.IdleTimeout {
			users = append(users, userID)
		}
	}
	m.mu.Unlock()

	for _, userID := range users {
		_ = m.StopForUser(ctx, userID)
	}
}

func (m *Manager) StopForUser(ctx context.Context, userID string) error {
	lock := m.lockForUser(userID)
	lock.Lock()
	defer lock.Unlock()

	rec, err := m.loadRecord(ctx, userID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		return err
	}
	if strings.TrimSpace(rec.ContainerID) == "" {
		_, _ = m.db.Exec(ctx, `UPDATE agents SET status = 'stopped', stopped_at = now() WHERE user_id = $1`, userID)
		if err := m.removeDirectRoute(rec.SubdomainPrefix); err != nil {
			log.Printf("direct route remove warning (no container): user=%s prefix=%s err=%v", userID, rec.SubdomainPrefix, err)
		}
		return nil
	}

	if err := m.stopAndRemoveContainer(ctx, rec.ContainerID); err != nil {
		return err
	}

	_, err = m.db.Exec(ctx, `UPDATE agents SET status = 'stopped', last_health = now(), stopped_at = now() WHERE user_id = $1`, userID)
	if routeErr := m.removeDirectRoute(rec.SubdomainPrefix); routeErr != nil {
		log.Printf("direct route remove warning: user=%s prefix=%s err=%v", userID, rec.SubdomainPrefix, routeErr)
	}
	m.mu.Lock()
	delete(m.lastActive, userID)
	m.mu.Unlock()
	return err
}

func (m *Manager) ensureRecordRunning(ctx context.Context, rec record) (Target, bool, error) {
	if strings.TrimSpace(rec.ContainerID) == "" {
		t := Target{Host: rec.Host, Port: rec.Port}
		if t.Host == "" || t.Port <= 0 {
			return Target{}, false, nil
		}
		if err := m.waitHealthy(ctx, "", t.Host, t.Port); err != nil {
			return Target{}, false, nil
		}
		return t, true, nil
	}

	inspect, err := m.docker.ContainerInspect(ctx, rec.ContainerID)
	if err != nil {
		if errdefs.IsNotFound(err) {
			return Target{}, false, nil
		}
		return Target{}, false, err
	}

	started := false
	if inspect.State == nil || !inspect.State.Running {
		if err := m.ensureContainerStarted(ctx, rec.ContainerID); err != nil {
			return Target{}, false, err
		}
		started = true
	}

	host := m.agentHostFromInspect(inspect, containerNameForUser(rec.UserID))
	if host == "" {
		host = strings.TrimSpace(rec.Host)
	}
	port := rec.Port
	if port <= 0 {
		port = m.cfg.AgentPort
	}

	if err := m.waitHealthy(ctx, rec.ContainerID, host, port); err != nil {
		return Target{}, false, err
	}
	if started {
		if err := m.ensureUserMediaBucket(ctx, rec.UserID, Target{Host: host, Port: port}); err != nil {
			log.Printf("media bucket ensure warning (container restart): user=%s err=%v", rec.UserID, err)
		}
	}

	_, _ = m.db.Exec(ctx, `
		UPDATE agents
		SET host = $2, port = $3, status = 'running', last_health = now(), stopped_at = NULL
		WHERE user_id = $1
	`, rec.UserID, host, port)
	if err := m.syncDirectRoute(rec.SubdomainPrefix, Target{Host: host, Port: port}); err != nil {
		log.Printf("direct route sync warning (existing agent): user=%s prefix=%s err=%v", rec.UserID, rec.SubdomainPrefix, err)
	}

	return Target{Host: host, Port: port}, true, nil
}

func (m *Manager) ensureImage(ctx context.Context, name string) error {
	if strings.TrimSpace(name) == "" {
		return errors.New("agent image is required")
	}
	// If the image already exists locally, skip pulling from the registry.
	// This avoids overwriting locally-built dev images with stale registry versions.
	if _, _, err := m.docker.ImageInspectWithRaw(ctx, name); err == nil {
		return nil
	}
	rc, err := m.docker.ImagePull(ctx, name, image.PullOptions{})
	if err != nil {
		return err
	}
	defer rc.Close()
	_, _ = io.Copy(io.Discard, rc)
	return nil
}

func (m *Manager) ensureAssetsRoot() {
	if strings.TrimSpace(m.cfg.AssetsRoot) == "" {
		m.cfg.AssetsRoot = "/var/lib/aether/agents"
	}
}

func (m *Manager) ensureContainer(ctx context.Context, userID, containerName, _ string) (string, error) {
	if inspect, err := m.docker.ContainerInspect(ctx, containerName); err == nil {
		return inspect.ID, nil
	} else if err != nil && !errdefs.IsNotFound(err) {
		return "", err
	}

	agentID := agentIDForUser(userID)
	env := []string{
		"PORT=" + strconv.Itoa(m.cfg.AgentPort),
		"AETHER_AGENT_ID=" + agentID,
	}
	if strings.TrimSpace(m.cfg.PublicBaseURL) != "" {
		env = append(env, "AETHER_PUBLIC_BASE_URL="+strings.TrimSpace(m.cfg.PublicBaseURL))
	}
	if strings.TrimSpace(m.cfg.AdminToken) != "" {
		env = append(env, "AGENT_ADMIN_TOKEN="+strings.TrimSpace(m.cfg.AdminToken))
	}
	if strings.TrimSpace(m.cfg.DirectTokenSecret) != "" {
		env = append(env, "AGENT_DIRECT_TOKEN_SECRET="+strings.TrimSpace(m.cfg.DirectTokenSecret))
	}
	if m.cfg.OpenAIAPIKey != "" {
		env = append(env, "OPENAI_API_KEY="+m.cfg.OpenAIAPIKey)
	}
	if m.cfg.OpenAIBaseURL != "" {
		env = append(env, "OPENAI_BASE_URL="+m.cfg.OpenAIBaseURL)
	}
	if m.cfg.OpenAIModel != "" {
		env = append(env, "OPENAI_MODEL="+m.cfg.OpenAIModel)
	}
	if strings.TrimSpace(m.cfg.AgentStateKey) != "" {
		env = append(env, "AGENT_STATE_KEY="+strings.TrimSpace(m.cfg.AgentStateKey))
	}
	m.ensureAssetsRoot()
	userRoot := filepath.Join(m.cfg.AssetsRoot, userID)
	dbHostPath := filepath.Join(userRoot, "db")
	workspaceHostPath := filepath.Join(userRoot, "workspace")
	skillsUserHostPath := filepath.Join(userRoot, "skills", "user")
	skillsExtHostPath := filepath.Join(userRoot, "skills", "external")
	for _, d := range []string{dbHostPath, workspaceHostPath, skillsUserHostPath, skillsExtHostPath} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return "", err
		}
	}
	env = append(env, "AGENT_ASSETS_DIR=/app/assets")
	if strings.TrimSpace(m.cfg.VapidPublic) != "" {
		env = append(env, "VAPID_PUBLIC_KEY="+strings.TrimSpace(m.cfg.VapidPublic))
	}
	if strings.TrimSpace(m.cfg.VapidPrivate) != "" {
		env = append(env, "VAPID_PRIVATE_KEY="+strings.TrimSpace(m.cfg.VapidPrivate))
	}
	if strings.TrimSpace(m.cfg.VapidSubject) != "" {
		env = append(env, "VAPID_SUBJECT="+strings.TrimSpace(m.cfg.VapidSubject))
	}
	if strings.TrimSpace(m.cfg.S3Bucket) != "" {
		env = append(env, "S3_BUCKET="+strings.TrimSpace(m.cfg.S3Bucket))
	}
	if strings.TrimSpace(m.cfg.S3Template) != "" {
		env = append(env, "S3_BUCKET_TEMPLATE="+strings.TrimSpace(m.cfg.S3Template))
	}
	if strings.TrimSpace(m.cfg.S3Region) != "" {
		env = append(env, "S3_REGION="+strings.TrimSpace(m.cfg.S3Region))
	}
	if strings.TrimSpace(m.cfg.S3AccessKey) != "" {
		env = append(env, "S3_ACCESS_KEY_ID="+strings.TrimSpace(m.cfg.S3AccessKey))
	}
	if strings.TrimSpace(m.cfg.S3SecretKey) != "" {
		env = append(env, "S3_SECRET_ACCESS_KEY="+strings.TrimSpace(m.cfg.S3SecretKey))
	}
	if strings.TrimSpace(m.cfg.S3Endpoint) != "" {
		env = append(env, "S3_ENDPOINT="+strings.TrimSpace(m.cfg.S3Endpoint))
	}
	if strings.TrimSpace(m.cfg.S3PublicBase) != "" {
		env = append(env, "S3_PUBLIC_BASE_URL="+strings.TrimSpace(m.cfg.S3PublicBase))
	}
	if strings.TrimSpace(m.cfg.S3ForcePath) != "" {
		env = append(env, "S3_FORCE_PATH_STYLE="+strings.TrimSpace(m.cfg.S3ForcePath))
	}
	if strings.TrimSpace(m.cfg.S3PutTTL) != "" {
		env = append(env, "S3_PUT_URL_TTL_SECONDS="+strings.TrimSpace(m.cfg.S3PutTTL))
	}
	if strings.TrimSpace(m.cfg.S3GetTTL) != "" {
		env = append(env, "S3_GET_URL_TTL_SECONDS="+strings.TrimSpace(m.cfg.S3GetTTL))
	}
	if strings.TrimSpace(m.cfg.UpdateRepo) != "" {
		env = append(env, "AGENT_UPDATE_REPO="+strings.TrimSpace(m.cfg.UpdateRepo))
	}
	if strings.TrimSpace(m.cfg.UpdateToken) != "" {
		env = append(env, "AGENT_UPDATE_TOKEN="+strings.TrimSpace(m.cfg.UpdateToken))
	}

	// Forward OAuth provider credentials (*_CLIENT_ID / *_CLIENT_SECRET) to the agent
	// so plugins can use env-backed credentials without manual user configuration.
	for _, kv := range m.cfg.OAuthEnvVars {
		env = append(env, kv)
	}

	resp, err := m.docker.ContainerCreate(
		ctx,
		&container.Config{
			Image:  m.cfg.Image,
			Env:    env,
			Labels: map[string]string{"managed-by": "aether-orchestrator", "user-id": userID},
		},
		&container.HostConfig{
			// Mount only what needs to persist across container restarts.
			// Builtins (plugins/builtin, skills/builtin, PROMPT.md) come from the image.
			Mounts: []mount.Mount{
				{Type: mount.TypeBind, Source: dbHostPath, Target: "/app/assets/db"},
				{Type: mount.TypeBind, Source: workspaceHostPath, Target: "/app/assets/workspace"},
				{Type: mount.TypeBind, Source: skillsUserHostPath, Target: "/app/assets/skills/user"},
				{Type: mount.TypeBind, Source: skillsExtHostPath, Target: "/app/assets/skills/external"},
			},
		},
		&network.NetworkingConfig{EndpointsConfig: map[string]*network.EndpointSettings{m.cfg.Network: {}}},
		nil,
		containerName,
	)
	if err != nil {
		if !strings.Contains(strings.ToLower(err.Error()), "conflict") {
			return "", err
		}
		inspect, e := m.docker.ContainerInspect(ctx, containerName)
		if e != nil {
			return "", err
		}
		return inspect.ID, nil
	}
	return resp.ID, nil
}

func (m *Manager) ensureContainerStarted(ctx context.Context, id string) error {
	inspect, err := m.docker.ContainerInspect(ctx, id)
	if err != nil {
		return err
	}
	if inspect.State != nil && inspect.State.Running {
		return nil
	}
	if err := m.docker.ContainerStart(ctx, id, container.StartOptions{}); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "already started") {
			return nil
		}
		return err
	}
	return nil
}

func (m *Manager) waitHealthy(ctx context.Context, containerID string, host string, port int) error {
	timeout := m.cfg.HealthTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	healthURL := fmt.Sprintf("http://%s:%d/health", host, port)

	for time.Now().Before(deadline) {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
		resp, err := m.httpClient.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	detail := ""
	if strings.TrimSpace(containerID) != "" {
		detail = m.containerDebugState(ctx, containerID)
	}
	if detail != "" {
		return fmt.Errorf("agent failed health check before timeout: %s (%s)", healthURL, detail)
	}
	return fmt.Errorf("agent failed health check before timeout: %s", healthURL)
}

func (m *Manager) ensureUserMediaBucket(ctx context.Context, userID string, target Target) error {
	uid := strings.TrimSpace(userID)
	if uid == "" {
		return nil
	}
	if strings.TrimSpace(m.cfg.S3Bucket) == "" && strings.TrimSpace(m.cfg.S3Template) == "" {
		return nil
	}
	payload, err := json.Marshal(map[string]any{"user_id": uid})
	if err != nil {
		return err
	}
	url := fmt.Sprintf("http://%s:%d/api/media/ensure-bucket", target.Host, target.Port)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("ensure media bucket request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		if msg := strings.TrimSpace(string(b)); msg != "" {
			return fmt.Errorf("ensure media bucket failed: status=%d body=%s", resp.StatusCode, msg)
		}
		return fmt.Errorf("ensure media bucket failed: status=%d", resp.StatusCode)
	}
	return nil
}

func (m *Manager) agentHostFromInspect(inspect container.InspectResponse, fallback string) string {
	if inspect.NetworkSettings != nil && inspect.NetworkSettings.Networks != nil {
		if netCfg, ok := inspect.NetworkSettings.Networks[m.cfg.Network]; ok {
			if ip := strings.TrimSpace(netCfg.IPAddress); ip != "" {
				return ip
			}
		}
		for _, netCfg := range inspect.NetworkSettings.Networks {
			if ip := strings.TrimSpace(netCfg.IPAddress); ip != "" {
				return ip
			}
		}
	}
	if name := strings.TrimPrefix(strings.TrimSpace(inspect.Name), "/"); name != "" {
		return name
	}
	return strings.TrimSpace(fallback)
}

func (m *Manager) containerDebugState(ctx context.Context, containerID string) string {
	inspect, err := m.docker.ContainerInspect(ctx, containerID)
	if err != nil {
		return ""
	}
	state := "unknown"
	if inspect.State != nil {
		state = inspect.State.Status
		if inspect.State.Error != "" {
			state += ", error=" + inspect.State.Error
		}
		exit := inspect.State.ExitCode
		if exit != 0 {
			state += fmt.Sprintf(", exit_code=%d", exit)
		}
	}
	logs, err := m.docker.ContainerLogs(ctx, containerID, container.LogsOptions{ShowStdout: true, ShowStderr: true, Tail: "20"})
	if err != nil {
		return state
	}
	defer logs.Close()
	b, err := io.ReadAll(io.LimitReader(logs, 8192))
	if err != nil || len(b) == 0 {
		return state
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) > 0 {
		state += ", last_log=" + strings.TrimSpace(lines[len(lines)-1])
	}
	return state
}

func (m *Manager) stopAndRemoveContainer(ctx context.Context, id string) error {
	t := 10
	err := m.docker.ContainerStop(ctx, id, container.StopOptions{Timeout: &t})
	if err != nil && !errdefs.IsNotFound(err) {
		return err
	}
	err = m.docker.ContainerRemove(ctx, id, container.RemoveOptions{Force: true})
	if err != nil && !errdefs.IsNotFound(err) {
		return err
	}
	return nil
}

func (m *Manager) detectCurrentNetwork(ctx context.Context) (string, error) {
	selfID, err := os.Hostname()
	if err != nil || strings.TrimSpace(selfID) == "" {
		return "", errors.New("failed to get container hostname")
	}
	inspect, err := m.docker.ContainerInspect(ctx, selfID)
	if err != nil {
		return "", err
	}
	for name := range inspect.NetworkSettings.Networks {
		if strings.TrimSpace(name) != "" {
			return name, nil
		}
	}
	return "", errors.New("no docker network detected")
}

func (m *Manager) loadRecord(ctx context.Context, userID string) (record, error) {
	var rec record
	err := m.db.QueryRow(ctx, `
		SELECT id, user_id, COALESCE(container_id,''), COALESCE(subdomain_prefix,''), host, port, COALESCE(status,'')
		FROM agents
		WHERE user_id = $1
		LIMIT 1
	`, userID).Scan(&rec.ID, &rec.UserID, &rec.ContainerID, &rec.SubdomainPrefix, &rec.Host, &rec.Port, &rec.Status)
	return rec, err
}

func (m *Manager) listManagedRecords(ctx context.Context) ([]record, error) {
	rows, err := m.db.Query(ctx, `
		SELECT id, user_id, COALESCE(container_id,''), COALESCE(subdomain_prefix,''), host, port, COALESCE(status,'')
		FROM agents
		WHERE COALESCE(status, '') <> 'stopped'
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []record
	for rows.Next() {
		var rec record
		if err := rows.Scan(&rec.ID, &rec.UserID, &rec.ContainerID, &rec.SubdomainPrefix, &rec.Host, &rec.Port, &rec.Status); err != nil {
			return nil, err
		}
		records = append(records, rec)
	}
	return records, rows.Err()
}

func (m *Manager) lockForUser(userID string) *sync.Mutex {
	m.mu.Lock()
	defer m.mu.Unlock()
	l, ok := m.userLocks[userID]
	if ok {
		return l
	}
	l = &sync.Mutex{}
	m.userLocks[userID] = l
	return l
}

func shortHash(in string) string {
	h := sha256.Sum256([]byte(in))
	return hex.EncodeToString(h[:])[:12]
}

func containerNameForUser(userID string) string {
	return "aether-agent-" + shortHash(userID)
}

func agentIDForUser(userID string) string {
	return "managed-" + shortHash(userID)
}

func subdomainPrefixForUser(userID string) string {
	hash := shortHash(userID)
	if len(hash) >= 8 {
		return hash[:8]
	}
	return hash
}

func (m *Manager) ensureSubdomainPrefix(ctx context.Context, userID, existing string) (string, error) {
	prefix := strings.TrimSpace(existing)
	if prefix == "" {
		prefix = subdomainPrefixForUser(userID)
	}
	_, err := m.db.Exec(ctx, `
		UPDATE agents
		SET subdomain_prefix = $2
		WHERE user_id = $1 AND (subdomain_prefix IS NULL OR subdomain_prefix = '')
	`, userID, prefix)
	if err != nil {
		return "", err
	}
	return prefix, nil
}

func (m *Manager) syncDirectRoute(prefix string, target Target) error {
	if m.cfg.RouteManager == nil {
		return nil
	}
	return m.cfg.RouteManager.AddRoute(prefix, target.Host, target.Port)
}

func (m *Manager) removeDirectRoute(prefix string) error {
	if m.cfg.RouteManager == nil {
		return nil
	}
	return m.cfg.RouteManager.RemoveRoute(prefix)
}
