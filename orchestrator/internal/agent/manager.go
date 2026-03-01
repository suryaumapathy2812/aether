package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/volume"
	"github.com/docker/docker/client"
	"github.com/docker/docker/errdefs"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Target struct {
	Host string
	Port int
}

type ManagerConfig struct {
	Image         string
	Network       string
	IdleTimeout   time.Duration
	AgentPort     int
	HealthTimeout time.Duration
	AdminToken    string

	OpenAIAPIKey  string
	OpenAIBaseURL string
	OpenAIModel   string
	AgentStateKey string
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
	ID          string
	UserID      string
	ContainerID string
	Host        string
	Port        int
	Status      string
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

	rec, err := m.loadRecord(ctx, userID)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return Target{}, err
	}
	if err == nil {
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
	volumeName := volumeNameForUser(userID)
	if err := m.ensureVolume(ctx, volumeName, userID); err != nil {
		return Target{}, err
	}
	if err := m.ensureImage(ctx, m.cfg.Image); err != nil {
		return Target{}, err
	}

	ctrID, err := m.ensureContainer(ctx, userID, containerName, volumeName)
	if err != nil {
		return Target{}, err
	}
	if err := m.ensureContainerStarted(ctx, ctrID); err != nil {
		return Target{}, err
	}

	target := Target{Host: containerName, Port: m.cfg.AgentPort}
	if err := m.waitHealthy(ctx, target.Host, target.Port); err != nil {
		return Target{}, err
	}

	agentID := agentIDForUser(userID)
	_, err = m.db.Exec(ctx, `
		INSERT INTO agents (id, user_id, container_id, host, port, status, registered_at, last_health, stopped_at)
		VALUES ($1, $2, $3, $4, $5, 'running', now(), now(), NULL)
		ON CONFLICT (user_id) DO UPDATE SET
			id = EXCLUDED.id,
			container_id = EXCLUDED.container_id,
			host = EXCLUDED.host,
			port = EXCLUDED.port,
			status = 'running',
			last_health = now(),
			stopped_at = NULL
	`, agentID, userID, ctrID, target.Host, target.Port)
	if err != nil {
		return Target{}, err
	}

	m.RecordActivity(userID)
	return target, nil
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
		return nil
	}

	if err := m.stopAndRemoveContainer(ctx, rec.ContainerID); err != nil {
		return err
	}

	_, err = m.db.Exec(ctx, `UPDATE agents SET status = 'stopped', last_health = now(), stopped_at = now() WHERE user_id = $1`, userID)
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
		if err := m.waitHealthy(ctx, t.Host, t.Port); err != nil {
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

	if inspect.State == nil || !inspect.State.Running {
		if err := m.ensureContainerStarted(ctx, rec.ContainerID); err != nil {
			return Target{}, false, err
		}
	}

	host := strings.TrimSpace(rec.Host)
	if host == "" {
		host = strings.TrimSpace(inspect.Name)
		host = strings.TrimPrefix(host, "/")
	}
	if host == "" {
		host = containerNameForUser(rec.UserID)
	}
	port := rec.Port
	if port <= 0 {
		port = m.cfg.AgentPort
	}

	if err := m.waitHealthy(ctx, host, port); err != nil {
		return Target{}, false, err
	}

	_, _ = m.db.Exec(ctx, `
		UPDATE agents
		SET host = $2, port = $3, status = 'running', last_health = now(), stopped_at = NULL
		WHERE user_id = $1
	`, rec.UserID, host, port)

	return Target{Host: host, Port: port}, true, nil
}

func (m *Manager) ensureImage(ctx context.Context, name string) error {
	if strings.TrimSpace(name) == "" {
		return errors.New("agent image is required")
	}
	rc, err := m.docker.ImagePull(ctx, name, image.PullOptions{})
	if err != nil {
		return err
	}
	defer rc.Close()
	_, _ = io.Copy(io.Discard, rc)
	return nil
}

func (m *Manager) ensureVolume(ctx context.Context, name, userID string) error {
	_, err := m.docker.VolumeCreate(ctx, volume.CreateOptions{
		Name: name,
		Labels: map[string]string{
			"managed-by": "aether-orchestrator",
			"user-id":    userID,
		},
	})
	return err
}

func (m *Manager) ensureContainer(ctx context.Context, userID, containerName, volumeName string) (string, error) {
	if inspect, err := m.docker.ContainerInspect(ctx, containerName); err == nil {
		return inspect.ID, nil
	} else if err != nil && !errdefs.IsNotFound(err) {
		return "", err
	}

	env := []string{"PORT=" + strconv.Itoa(m.cfg.AgentPort)}
	if strings.TrimSpace(m.cfg.AdminToken) != "" {
		env = append(env, "AGENT_ADMIN_TOKEN="+strings.TrimSpace(m.cfg.AdminToken))
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

	resp, err := m.docker.ContainerCreate(
		ctx,
		&container.Config{
			Image:  m.cfg.Image,
			Env:    env,
			Labels: map[string]string{"managed-by": "aether-orchestrator", "user-id": userID},
		},
		&container.HostConfig{
			// Each user gets a dedicated persistent SQLite volume at /app/assets.
			// This keeps state.db across container restarts/reprovisioning.
			Mounts: []mount.Mount{{Type: mount.TypeVolume, Source: volumeName, Target: "/app/assets"}},
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

func (m *Manager) waitHealthy(ctx context.Context, host string, port int) error {
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
	return fmt.Errorf("agent failed health check before timeout: %s", healthURL)
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
		SELECT id, user_id, COALESCE(container_id,''), host, port, COALESCE(status,'')
		FROM agents
		WHERE user_id = $1
		LIMIT 1
	`, userID).Scan(&rec.ID, &rec.UserID, &rec.ContainerID, &rec.Host, &rec.Port, &rec.Status)
	return rec, err
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

func volumeNameForUser(userID string) string {
	return "aether-agent-assets-" + shortHash(userID)
}

func agentIDForUser(userID string) string {
	return "managed-" + shortHash(userID)
}
