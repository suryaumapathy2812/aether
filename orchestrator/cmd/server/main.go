package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type config struct {
	Port                 int
	DatabaseURL          string
	AgentSecret          string
	LocalAgentURL        string
	ProxyTimeout         time.Duration
	DefaultAgentID       string
	AutoAssignFirstAgent bool
}

type server struct {
	cfg        config
	db         *pgxpool.Pool
	httpClient *http.Client
	wsUpgrader websocket.Upgrader
}

type identity struct {
	UserID   string
	DeviceID string
	Token    string
}

type agentTarget struct {
	Host string
	Port int
}

func main() {
	cfg := loadConfig()
	pool, err := pgxpool.New(context.Background(), cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("failed to open postgres pool: %v", err)
	}
	defer pool.Close()

	if err := bootstrapSchema(context.Background(), pool); err != nil {
		log.Fatalf("failed to bootstrap schema: %v", err)
	}

	s := &server{
		cfg: cfg,
		db:  pool,
		httpClient: &http.Client{
			Timeout: cfg.ProxyTimeout,
		},
		wsUpgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin:     func(r *http.Request) bool { return true },
		},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/health", s.handleHealth)
	mux.HandleFunc("/auth/me", s.requireIdentity(s.handleAuthMe))
	mux.HandleFunc("/api/auth/me", s.requireIdentity(s.handleAuthMe))

	mux.HandleFunc("/api/agents/register", s.handleRegisterAgent)
	mux.HandleFunc("/api/agents/health", s.handleListAgents)
	mux.HandleFunc("/api/agents/", s.handleAgentsByID)

	mux.HandleFunc("/api/agent/ready", s.requireIdentity(s.handleAgentReady))
	mux.HandleFunc("/api/metrics/latency", s.requireIdentity(s.handleLatency))
	mux.HandleFunc("/api/webrtc/offer", s.requireIdentity(s.handleWebRTCOffer))
	mux.HandleFunc("/api/webrtc/ice", s.requireIdentity(s.handleWebRTCIce))
	mux.HandleFunc("/api/ws/notifications", s.requireIdentity(s.handleNotificationsWS))
	mux.HandleFunc("/api/ws", s.requireIdentity(s.handleNotificationsWS))

	// Dashboard call paths through /api/go proxy.
	mux.HandleFunc("/v1/", s.requireIdentity(s.handleV1Proxy))
	mux.HandleFunc("/api/memory/", s.requireIdentity(s.handleMemoryProxy))
	mux.HandleFunc("/api/plugins", s.requireIdentity(s.handlePluginsProxy))
	mux.HandleFunc("/api/plugins/", s.requireIdentity(s.handlePluginsProxy))
	mux.HandleFunc("/api/push/vapid-key", s.requireIdentity(s.handlePushProxy))
	mux.HandleFunc("/api/push/subscribe", s.requireIdentity(s.handlePushProxy))

	addr := ":" + strconv.Itoa(cfg.Port)
	log.Printf("orchestrator listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server stopped: %v", err)
	}
}

func loadConfig() config {
	port := envInt("PORT", 9000)
	proxyTimeout := time.Duration(envInt("AGENT_PROXY_TIMEOUT_SECONDS", 120)) * time.Second
	databaseURL := strings.TrimSpace(os.Getenv("DATABASE_URL"))
	if databaseURL == "" {
		log.Fatal("DATABASE_URL is required")
	}

	return config{
		Port:                 port,
		DatabaseURL:          databaseURL,
		AgentSecret:          strings.TrimSpace(os.Getenv("AGENT_SECRET")),
		LocalAgentURL:        strings.TrimSpace(os.Getenv("AETHER_LOCAL_AGENT_URL")),
		ProxyTimeout:         proxyTimeout,
		DefaultAgentID:       strings.TrimSpace(os.Getenv("AETHER_DEFAULT_AGENT_ID")),
		AutoAssignFirstAgent: strings.EqualFold(strings.TrimSpace(os.Getenv("AETHER_AUTO_ASSIGN_FIRST_AGENT")), "true"),
	}
}

func bootstrapSchema(ctx context.Context, db *pgxpool.Pool) error {
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
			last_health TIMESTAMPTZ
		);

		CREATE INDEX IF NOT EXISTS idx_agents_user_status ON agents(user_id, status, last_health DESC);
		CREATE INDEX IF NOT EXISTS idx_session_token_expires ON session(token, expires_at);
		CREATE INDEX IF NOT EXISTS idx_devices_token ON devices(token);
	`)
	return err
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	var ok int
	err := s.db.QueryRow(ctx, "SELECT 1").Scan(&ok)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"status": "degraded", "db": "disconnected"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "db": "connected"})
}

func (s *server) handleAuthMe(w http.ResponseWriter, r *http.Request, id identity) {
	var email, name string
	err := s.db.QueryRow(r.Context(), `SELECT COALESCE(email,''), COALESCE(name,'') FROM "user" WHERE id = $1`, id.UserID).Scan(&email, &name)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id.UserID, "email": email, "name": name})
}

func (s *server) handleRegisterAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := s.verifyAgentSecret(r); err != nil {
		writeError(w, err.status, err.msg)
		return
	}

	var req struct {
		AgentID     string `json:"agent_id"`
		Host        string `json:"host"`
		Port        int    `json:"port"`
		ContainerID string `json:"container_id"`
		UserID      string `json:"user_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	req.AgentID = strings.TrimSpace(req.AgentID)
	req.Host = strings.TrimSpace(req.Host)
	if req.AgentID == "" || req.Host == "" || req.Port <= 0 {
		writeError(w, http.StatusBadRequest, "agent_id, host and port are required")
		return
	}

	_, err := s.db.Exec(r.Context(), `
		INSERT INTO agents (id, host, port, container_id, user_id, status, registered_at, last_health)
		VALUES ($1, $2, $3, $4, NULLIF($5,''), 'running', now(), now())
		ON CONFLICT (id) DO UPDATE SET
			host = EXCLUDED.host,
			port = EXCLUDED.port,
			container_id = EXCLUDED.container_id,
			user_id = COALESCE(NULLIF(EXCLUDED.user_id,''), agents.user_id),
			status = 'running',
			last_health = now()
	`, req.AgentID, req.Host, req.Port, req.ContainerID, req.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "registered"})
}

func (s *server) handleListAgents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	rows, err := s.db.Query(r.Context(), `
		SELECT id, user_id, host, port, status, last_health
		FROM agents
		ORDER BY registered_at DESC
	`)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	type rec struct {
		ID         string     `json:"id"`
		UserID     *string    `json:"user_id"`
		Host       string     `json:"host"`
		Port       int        `json:"port"`
		Status     string     `json:"status"`
		LastHealth *time.Time `json:"last_health"`
	}
	out := make([]rec, 0)
	for rows.Next() {
		var v rec
		if err := rows.Scan(&v.ID, &v.UserID, &v.Host, &v.Port, &v.Status, &v.LastHealth); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		out = append(out, v)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *server) handleAgentsByID(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/agents/")
	parts := strings.Split(path, "/")
	if len(parts) != 2 {
		writeError(w, http.StatusNotFound, "not found")
		return
	}
	agentID := strings.TrimSpace(parts[0])
	action := strings.TrimSpace(parts[1])
	if agentID == "" {
		writeError(w, http.StatusBadRequest, "agent id is required")
		return
	}
	if err := s.verifyAgentSecret(r); err != nil {
		writeError(w, err.status, err.msg)
		return
	}

	switch {
	case action == "heartbeat" && r.Method == http.MethodPost:
		_, err := s.db.Exec(r.Context(), "UPDATE agents SET last_health = now(), status = 'running' WHERE id = $1", agentID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
		return
	case action == "assign" && r.Method == http.MethodPost:
		userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
		if userID == "" {
			writeError(w, http.StatusBadRequest, "user_id is required")
			return
		}
		_, err := s.db.Exec(r.Context(), "UPDATE agents SET user_id = $1 WHERE id = $2", userID, agentID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "assigned"})
		return
	default:
		writeError(w, http.StatusNotFound, "not found")
		return
	}
}

func (s *server) handleLatency(w http.ResponseWriter, r *http.Request, id identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err == nil {
		if ok := s.proxyHTTPStream(w, r, id, target, "/metrics/latency", true); ok {
			return
		}
	}
	writeJSON(w, http.StatusServiceUnavailable, map[string]any{
		"status":   "degraded",
		"chat":     map[string]any{},
		"voice":    map[string]any{},
		"kernel":   map[string]any{},
		"services": map[string]any{},
	})
}

func (s *server) handleAgentReady(w http.ResponseWriter, r *http.Request, id identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"ready": false})
		return
	}
	req, _ := http.NewRequestWithContext(r.Context(), http.MethodGet, fmt.Sprintf("http://%s:%d/health", target.Host, target.Port), nil)
	resp, err := s.httpClient.Do(req)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"ready": false})
		return
	}
	defer resp.Body.Close()
	writeJSON(w, http.StatusOK, map[string]any{"ready": resp.StatusCode == http.StatusOK})
}

func (s *server) handleWebRTCOffer(w http.ResponseWriter, r *http.Request, id identity) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !s.proxyHTTPStream(w, r, id, target, "/webrtc/offer", true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *server) handleWebRTCIce(w http.ResponseWriter, r *http.Request, id identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !s.proxyHTTPStream(w, r, id, target, "/webrtc/ice", true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *server) handleV1Proxy(w http.ResponseWriter, r *http.Request, id identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !s.proxyHTTPStream(w, r, id, target, r.URL.Path, true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *server) handleMemoryProxy(w http.ResponseWriter, r *http.Request, id identity) {
	s.proxyToAgentSamePath(w, r, id)
}

func (s *server) handlePluginsProxy(w http.ResponseWriter, r *http.Request, id identity) {
	s.proxyToAgentSamePath(w, r, id)
}

func (s *server) handlePushProxy(w http.ResponseWriter, r *http.Request, id identity) {
	s.proxyToAgentSamePath(w, r, id)
}

func (s *server) proxyToAgentSamePath(w http.ResponseWriter, r *http.Request, id identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !s.proxyHTTPStream(w, r, id, target, r.URL.Path, true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *server) handleNotificationsWS(w http.ResponseWriter, r *http.Request, id identity) {
	if !websocket.IsWebSocketUpgrade(r) {
		writeError(w, http.StatusBadRequest, "websocket upgrade required")
		return
	}
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	clientConn, err := s.wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer clientConn.Close()

	upstreamURL := url.URL{
		Scheme: "ws",
		Host:   fmt.Sprintf("%s:%d", target.Host, target.Port),
		Path:   "/ws/notifications",
	}
	q := upstreamURL.Query()
	q.Set("user_id", id.UserID)
	if id.Token != "" {
		q.Set("token", id.Token)
	}
	upstreamURL.RawQuery = q.Encode()

	agentConn, _, err := websocket.DefaultDialer.Dial(upstreamURL.String(), nil)
	if err != nil {
		return
	}
	defer agentConn.Close()

	var wg sync.WaitGroup
	forward := func(dst, src *websocket.Conn) {
		defer wg.Done()
		for {
			mt, msg, err := src.ReadMessage()
			if err != nil {
				_ = dst.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(2*time.Second))
				return
			}
			if err := dst.WriteMessage(mt, msg); err != nil {
				return
			}
		}
	}
	wg.Add(2)
	go forward(agentConn, clientConn)
	go forward(clientConn, agentConn)
	wg.Wait()
}

func (s *server) proxyHTTPStream(w http.ResponseWriter, incoming *http.Request, id identity, target agentTarget, path string, enforceUser bool) bool {
	ctx := incoming.Context()
	upstreamReq, err := s.buildUpstreamRequest(ctx, incoming, id, target, path, enforceUser)
	if err != nil {
		return false
	}
	resp, err := s.httpClient.Do(upstreamReq)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
	return true
}

func (s *server) buildUpstreamRequest(ctx context.Context, incoming *http.Request, id identity, target agentTarget, path string, enforceUser bool) (*http.Request, error) {
	u := &url.URL{
		Scheme:   "http",
		Host:     fmt.Sprintf("%s:%d", target.Host, target.Port),
		Path:     path,
		RawQuery: incoming.URL.RawQuery,
	}
	if enforceUser {
		u.RawQuery = rewriteQueryUserID(path, u.Query(), id.UserID).Encode()
	}

	bodyBytes := []byte(nil)
	if incoming.Body != nil {
		b, err := io.ReadAll(incoming.Body)
		if err != nil {
			return nil, err
		}
		bodyBytes = b
	}
	if enforceUser {
		bodyBytes = rewriteBodyUserID(path, incoming.Header.Get("Content-Type"), bodyBytes, id.UserID)
	}

	var bodyReader io.Reader
	if len(bodyBytes) > 0 {
		bodyReader = bytes.NewReader(bodyBytes)
	}
	req, err := http.NewRequestWithContext(ctx, incoming.Method, u.String(), bodyReader)
	if err != nil {
		return nil, err
	}
	copyRequestHeaders(req.Header, incoming.Header)
	return req, nil
}

func rewriteQueryUserID(path string, q url.Values, userID string) url.Values {
	copyQ := url.Values{}
	for k, vals := range q {
		copyQ[k] = append([]string(nil), vals...)
	}
	if strings.HasPrefix(path, "/api/memory/") || strings.HasPrefix(path, "/v1/agent/tasks") {
		copyQ.Set("user_id", userID)
	}
	return copyQ
}

func rewriteBodyUserID(path, contentType string, body []byte, userID string) []byte {
	if len(body) == 0 {
		return body
	}
	if !strings.Contains(strings.ToLower(contentType), "application/json") {
		return body
	}
	if !strings.HasPrefix(path, "/v1/") && !strings.HasPrefix(path, "/api/") {
		return body
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return body
	}
	payload["user_id"] = userID
	b, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return b
}

func copyRequestHeaders(dst, src http.Header) {
	for k, vals := range src {
		kl := strings.ToLower(k)
		if kl == "host" || kl == "content-length" || kl == "connection" || kl == "upgrade" {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for k, vals := range src {
		kl := strings.ToLower(k)
		if kl == "connection" || kl == "transfer-encoding" || kl == "keep-alive" || kl == "upgrade" {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func (s *server) resolveAgent(ctx context.Context, userID string) (agentTarget, error) {
	if local := strings.TrimSpace(s.cfg.LocalAgentURL); local != "" {
		u, err := url.Parse(local)
		if err != nil || u.Hostname() == "" {
			return agentTarget{}, fmt.Errorf("invalid AETHER_LOCAL_AGENT_URL")
		}
		port := u.Port()
		if port == "" {
			if strings.EqualFold(u.Scheme, "https") {
				port = "443"
			} else {
				port = "80"
			}
		}
		n, _ := strconv.Atoi(port)
		return agentTarget{Host: u.Hostname(), Port: n}, nil
	}

	var t agentTarget
	err := s.db.QueryRow(ctx, `
		SELECT host, port
		FROM agents
		WHERE user_id = $1 AND status = 'running'
		ORDER BY last_health DESC NULLS LAST
		LIMIT 1
	`, userID).Scan(&t.Host, &t.Port)
	if err == nil {
		return t, nil
	}

	if strings.TrimSpace(s.cfg.DefaultAgentID) != "" {
		err = s.db.QueryRow(ctx, `SELECT host, port FROM agents WHERE id = $1 AND status = 'running' LIMIT 1`, s.cfg.DefaultAgentID).Scan(&t.Host, &t.Port)
		if err == nil {
			_, _ = s.db.Exec(ctx, `UPDATE agents SET user_id = $1 WHERE id = $2`, userID, s.cfg.DefaultAgentID)
			return t, nil
		}
	}

	if s.cfg.AutoAssignFirstAgent {
		var agentID string
		err = s.db.QueryRow(ctx, `
			SELECT id, host, port FROM agents
			WHERE status = 'running'
			ORDER BY last_health DESC NULLS LAST
			LIMIT 1
		`).Scan(&agentID, &t.Host, &t.Port)
		if err == nil {
			_, _ = s.db.Exec(ctx, `UPDATE agents SET user_id = $1 WHERE id = $2`, userID, agentID)
			return t, nil
		}
	}

	return agentTarget{}, err
}

func (s *server) requireIdentity(next func(http.ResponseWriter, *http.Request, identity)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := s.identityFromRequest(r)
		if err != nil {
			writeError(w, http.StatusUnauthorized, err.Error())
			return
		}
		next(w, r, id)
	}
}

func (s *server) identityFromRequest(r *http.Request) (identity, error) {
	tokens := extractTokens(r)
	if len(tokens) == 0 {
		return identity{}, errors.New("Missing authorization")
	}

	for _, token := range tokens {
		if token == "" {
			continue
		}
		var userID string
		err := s.db.QueryRow(r.Context(), `
			SELECT user_id
			FROM session
			WHERE token = $1 AND expires_at > now()
			LIMIT 1
		`, token).Scan(&userID)
		if err == nil && strings.TrimSpace(userID) != "" {
			return identity{UserID: userID, Token: token}, nil
		}

		var deviceID string
		err = s.db.QueryRow(r.Context(), `
			SELECT id, user_id
			FROM devices
			WHERE token = $1
			LIMIT 1
		`, token).Scan(&deviceID, &userID)
		if err == nil && strings.TrimSpace(userID) != "" {
			_, _ = s.db.Exec(r.Context(), "UPDATE devices SET last_seen = now() WHERE token = $1", token)
			return identity{UserID: userID, DeviceID: deviceID, Token: token}, nil
		}
	}

	return identity{}, errors.New("Invalid or expired session")
}

func extractTokens(r *http.Request) []string {
	out := make([]string, 0, 3)
	if t := bearerToken(r); t != "" {
		out = append(out, t)
	}
	for _, cookieName := range []string{"__Secure-better-auth.session_token", "better-auth.session_token"} {
		if c, err := r.Cookie(cookieName); err == nil {
			v := strings.TrimSpace(c.Value)
			if v != "" {
				if i := strings.Index(v, "."); i > 0 {
					v = v[:i]
				}
				out = append(out, v)
			}
		}
	}
	if q := strings.TrimSpace(r.URL.Query().Get("token")); q != "" {
		out = append(out, q)
	}
	seen := map[string]bool{}
	uniq := make([]string, 0, len(out))
	for _, v := range out {
		if seen[v] {
			continue
		}
		seen[v] = true
		uniq = append(uniq, v)
	}
	return uniq
}

func bearerToken(r *http.Request) string {
	h := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(h), "bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}

type authErr struct {
	status int
	msg    string
}

func (s *server) verifyAgentSecret(r *http.Request) *authErr {
	if s.cfg.AgentSecret == "" {
		return nil
	}
	auth := bearerToken(r)
	if auth == "" {
		return &authErr{status: http.StatusUnauthorized, msg: "missing agent authorization"}
	}
	if auth != s.cfg.AgentSecret {
		return &authErr{status: http.StatusForbidden, msg: "invalid agent secret"}
	}
	return nil
}

func envInt(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}
