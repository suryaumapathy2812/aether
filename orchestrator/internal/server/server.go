package server

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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/agent"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/auth"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/config"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/proxy"
)

type Server struct {
	cfg        config.Config
	db         *pgxpool.Pool
	httpClient *http.Client
	wsUpgrader websocket.Upgrader
	auth       *auth.Authenticator
	agentMgr   *agent.Manager
}

type authErr struct {
	status int
	msg    string
}

const (
	wsPongWait   = 60 * time.Second
	wsPingPeriod = 25 * time.Second
	wsWriteWait  = 5 * time.Second

	orchestratorAPIPrefix = "/go/v1"
	agentAPIPrefix        = "/agent/v1"
)

func registerRouteAliases(mux *http.ServeMux, paths []string, handler http.HandlerFunc) {
	for _, path := range paths {
		mux.HandleFunc(path, handler)
	}
}

func trimAnyPrefix(path string, prefixes ...string) string {
	for _, prefix := range prefixes {
		if strings.HasPrefix(path, prefix) {
			return strings.TrimPrefix(path, prefix)
		}
	}
	return path
}

func New(cfg config.Config, db *pgxpool.Pool, mgr *agent.Manager) *Server {
	return &Server{
		cfg:      cfg,
		db:       db,
		auth:     auth.New(db, cfg.AgentSecret, cfg.AgentDirectTokenSecret),
		agentMgr: mgr,
		httpClient: &http.Client{
			Timeout: cfg.ProxyTimeout,
		},
		wsUpgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin:     func(r *http.Request) bool { return true },
		},
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	registerRouteAliases(mux, []string{"/health", "/api/health", orchestratorAPIPrefix + "/health"}, s.handleHealth)
	registerRouteAliases(mux, []string{"/auth/me", "/api/auth/me", orchestratorAPIPrefix + "/auth/me"}, s.requireIdentity(s.handleAuthMe))

	registerRouteAliases(mux, []string{"/api/agents/register", orchestratorAPIPrefix + "/agents/register"}, s.handleRegisterAgent)
	registerRouteAliases(mux, []string{"/api/agents/health", orchestratorAPIPrefix + "/agents/health"}, s.handleListAgents)
	registerRouteAliases(mux, []string{"/api/agents/version", orchestratorAPIPrefix + "/agents/version"}, s.handleAgentVersionAdmin)
	registerRouteAliases(mux, []string{"/api/agents/reload", orchestratorAPIPrefix + "/agents/reload"}, s.handleAgentReloadAdmin)
	registerRouteAliases(mux, []string{"/api/agents/upgrade", orchestratorAPIPrefix + "/agents/upgrade"}, s.handleAgentUpgradeAdmin)
	registerRouteAliases(mux, []string{"/api/agents/", orchestratorAPIPrefix + "/agents/"}, s.handleAgentsByID)

	registerRouteAliases(mux, []string{"/api/agent/ready", orchestratorAPIPrefix + "/agent/ready"}, s.requireIdentity(s.handleAgentReady))
	registerRouteAliases(mux, []string{"/api/agent/subdomain", orchestratorAPIPrefix + "/agent/subdomain"}, s.requireIdentity(s.handleAgentSubdomain))
	registerRouteAliases(mux, []string{"/api/metrics/latency", orchestratorAPIPrefix + "/metrics/latency"}, s.requireIdentity(s.handleLatency))
	registerRouteAliases(mux, []string{"/api/ws/notifications", "/api/ws", agentAPIPrefix + "/ws/notifications"}, s.requireIdentity(s.handleNotificationsWS))
	registerRouteAliases(mux, []string{"/api/ws/conversation", agentAPIPrefix + "/ws/conversation"}, s.requireIdentity(s.handleConversationWS))
	registerRouteAliases(mux, []string{"/api/pair/request", orchestratorAPIPrefix + "/pair/request"}, s.handlePairRequest)
	registerRouteAliases(mux, []string{"/api/pair/status/", orchestratorAPIPrefix + "/pair/status/"}, s.handlePairStatus)
	registerRouteAliases(mux, []string{"/api/pair/claim", orchestratorAPIPrefix + "/pair/claim"}, s.requireIdentity(s.handlePairClaim))
	registerRouteAliases(mux, []string{"/api/devices", orchestratorAPIPrefix + "/devices"}, s.requireIdentity(s.handleDevices))
	registerRouteAliases(mux, []string{"/api/devices/telegram", orchestratorAPIPrefix + "/devices/telegram"}, s.requireIdentity(s.handleTelegramDevice))
	registerRouteAliases(mux, []string{"/api/devices/", orchestratorAPIPrefix + "/devices/"}, s.requireIdentity(s.handleDeviceByID))
	registerRouteAliases(mux, []string{"/api/hooks/pubsub/", orchestratorAPIPrefix + "/hooks/pubsub/"}, s.handlePubsubWebhookIngress)
	registerRouteAliases(mux, []string{"/api/hooks/", orchestratorAPIPrefix + "/hooks/"}, s.handlePluginWebhookIngress)
	registerRouteAliases(mux, []string{"/api/email-mappings", orchestratorAPIPrefix + "/email-mappings"}, s.handleEmailMappings)

	registerRouteAliases(mux, []string{"/v1/", agentAPIPrefix + "/"}, s.requireIdentity(s.handleV1Proxy))
	registerRouteAliases(mux, []string{"/api/memory/", "/api/preferences", "/api/preferences/", "/api/plugins", "/api/plugins/", "/api/skills", "/api/skills/", "/api/push/vapid-key", "/api/push/subscribe", "/api/push/test", "/api/channels", "/api/channels/"}, s.requireIdentity(s.proxyToAgentSamePath))

	// Channel webhook — unauthenticated, called by Telegram/WhatsApp/etc.
	// URL: POST /go/v1/{user_id}/channels/{channel_type}/webhook/{agent_id}
	//
	// NOTE: We register this on /go/v1/ (prefix) and parse the path manually to
	// avoid net/http ServeMux wildcard conflicts with existing /api/* routes.
	registerRouteAliases(mux, []string{"/api/", orchestratorAPIPrefix + "/"}, s.handleChannelWebhookProxy)

	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
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

func (s *Server) handleAuthMe(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	var email, name string
	err := s.db.QueryRow(r.Context(), `SELECT COALESCE(email,''), COALESCE(name,'') FROM "user" WHERE id = $1`, id.UserID).Scan(&email, &name)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id.UserID, "email": email, "name": name})
}

func (s *Server) handleRegisterAgent(w http.ResponseWriter, r *http.Request) {
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
		INSERT INTO agents (id, host, port, container_id, user_id, status, registered_at, last_health, stopped_at)
		VALUES ($1, $2, $3, $4, NULLIF($5,''), 'running', now(), now(), NULL)
		ON CONFLICT (id) DO UPDATE SET
			host = EXCLUDED.host,
			port = EXCLUDED.port,
			container_id = EXCLUDED.container_id,
			user_id = COALESCE(NULLIF(EXCLUDED.user_id,''), agents.user_id),
			status = 'running',
			last_health = now(),
			stopped_at = NULL
	`, req.AgentID, req.Host, req.Port, req.ContainerID, req.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "registered"})
}

func (s *Server) handleListAgents(w http.ResponseWriter, r *http.Request) {
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

func (s *Server) handleAgentsByID(w http.ResponseWriter, r *http.Request) {
	path := trimAnyPrefix(r.URL.Path, orchestratorAPIPrefix+"/agents/", "/api/agents/")
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
		_, err := s.db.Exec(r.Context(), "UPDATE agents SET last_health = now(), status = 'running', stopped_at = NULL WHERE id = $1", agentID)
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

func (s *Server) handleAgentVersionAdmin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := s.verifyAgentSecret(r); err != nil {
		writeError(w, err.status, err.msg)
		return
	}
	targets, err := s.resolveAdminTargets(r.Context(), strings.TrimSpace(r.URL.Query().Get("user_id")))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	results := make([]map[string]any, 0, len(targets))
	for _, t := range targets {
		status, payload, callErr := s.callAgentAdmin(r.Context(), t.Host, t.Port, http.MethodGet, "/admin/version", nil)
		results = append(results, map[string]any{
			"user_id": t.UserID,
			"host":    t.Host,
			"port":    t.Port,
			"status":  status,
			"result":  payload,
			"error":   errString(callErr),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"agents": results})
}

func (s *Server) handleAgentReloadAdmin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := s.verifyAgentSecret(r); err != nil {
		writeError(w, err.status, err.msg)
		return
	}
	targets, err := s.resolveAdminTargets(r.Context(), strings.TrimSpace(r.URL.Query().Get("user_id")))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	results := make([]map[string]any, 0, len(targets))
	for _, t := range targets {
		status, payload, callErr := s.callAgentAdmin(r.Context(), t.Host, t.Port, http.MethodPost, "/admin/reload", map[string]any{})
		results = append(results, map[string]any{
			"user_id": t.UserID,
			"host":    t.Host,
			"port":    t.Port,
			"status":  status,
			"result":  payload,
			"error":   errString(callErr),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"agents": results})
}

func (s *Server) handleAgentUpgradeAdmin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := s.verifyAgentSecret(r); err != nil {
		writeError(w, err.status, err.msg)
		return
	}
	targets, err := s.resolveAdminTargets(r.Context(), strings.TrimSpace(r.URL.Query().Get("user_id")))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	results := make([]map[string]any, 0, len(targets))
	for _, t := range targets {
		status, payload, callErr := s.callAgentAdmin(r.Context(), t.Host, t.Port, http.MethodPost, "/admin/update/apply", map[string]any{})
		results = append(results, map[string]any{
			"user_id": t.UserID,
			"host":    t.Host,
			"port":    t.Port,
			"status":  status,
			"result":  payload,
			"error":   errString(callErr),
		})
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"agents": results})
}

func (s *Server) handleLatency(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err == nil {
		if ok := proxy.HTTPStream(s.httpClient, w, r, target.Host, target.Port, "/metrics/latency", id.UserID, true); ok {
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

func (s *Server) handleAgentReady(w http.ResponseWriter, r *http.Request, id auth.Identity) {
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

func (s *Server) handleAgentSubdomain(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	_, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	prefix, agentID, err := s.lookupAgentDirectInfo(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	baseURL, wsURL, err := s.directAgentURLs(prefix)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	token, claims, err := s.auth.MintDirectToken(id.UserID, prefix, agentID, time.Hour)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"prefix":       prefix,
		"base_url":     baseURL,
		"ws_url":       wsURL,
		"direct_token": token,
		"expires_at":   claims.ExpiresAt,
		"agent_id":     agentID,
	})
}

func (s *Server) handleV1Proxy(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !proxy.HTTPStream(s.httpClient, w, r, target.Host, target.Port, r.URL.Path, id.UserID, true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *Server) handleMemoryProxy(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	s.proxyToAgentSamePath(w, r, id)
}

func (s *Server) handlePluginsProxy(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	s.proxyToAgentSamePath(w, r, id)
}

func (s *Server) handlePushProxy(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	s.proxyToAgentSamePath(w, r, id)
}

// handleChannelWebhookProxy proxies inbound channel webhooks (e.g. Telegram)
// to the correct agent without requiring user authentication.
//
// URL: POST /go/v1/{user_id}/channels/{channel_type}/webhook/{agent_id}
//
// The user_id in the URL allows the orchestrator to resolve the correct agent
// via the standard resolveAgent() path. The agent_id is used for direct DB
// lookup as a fast path; user_id is the fallback.
// The upstream path forwarded to the agent strips the user_id and agent_id:
// /agent/v1/channels/{channel_type}/webhook
func (s *Server) handleChannelWebhookProxy(w http.ResponseWriter, r *http.Request) {
	// Match only: /go/v1/{user_id}/channels/{channel_type}/webhook/{agent_id}
	trimmed := strings.Trim(trimAnyPrefix(r.URL.Path, orchestratorAPIPrefix+"/", "/api/"), "/")
	parts := strings.Split(trimmed, "/")
	if len(parts) != 5 || parts[1] != "channels" || parts[3] != "webhook" {
		// Not a channel webhook path; let this fallback handler return 404.
		writeError(w, http.StatusNotFound, "not found")
		return
	}
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	userID := strings.TrimSpace(parts[0])
	channelType := strings.TrimSpace(parts[2])
	agentID := strings.TrimSpace(parts[4])
	if userID == "" || channelType == "" {
		writeError(w, http.StatusNotFound, "not found")
		return
	}

	// The agent receives the standard webhook path (no user_id / agent_id).
	upstreamPath := fmt.Sprintf("%s/channels/%s/webhook", agentAPIPrefix, channelType)

	// In local dev mode, proxy directly to the configured agent URL.
	if local := strings.TrimSpace(s.cfg.LocalAgentURL); local != "" {
		u, err := url.Parse(local)
		if err != nil || u.Hostname() == "" {
			log.Printf("channel webhook: invalid local agent URL")
			w.WriteHeader(http.StatusOK)
			return
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
		if !proxy.HTTPStream(s.httpClient, w, r, u.Hostname(), n, upstreamPath, "", false) {
			w.WriteHeader(http.StatusOK) // 200 to avoid Telegram retries
		}
		return
	}

	host, port, found := s.resolveChannelWebhookAgent(r.Context(), userID, agentID)
	if !found {
		w.WriteHeader(http.StatusOK)
		return
	}
	if !proxy.HTTPStream(s.httpClient, w, r, host, port, upstreamPath, "", false) {
		w.WriteHeader(http.StatusOK) // 200 to Telegram even on failure
	}
}

// resolveChannelWebhookAgent finds the agent host/port for an inbound channel webhook.
// It tries: agentID direct lookup → user_id resolution → any running agent.
func (s *Server) resolveChannelWebhookAgent(ctx context.Context, userID, agentID string) (host string, port int, ok bool) {
	// Fast path: by agentID.
	if agentID != "" {
		err := s.db.QueryRow(ctx, `
			SELECT host, port FROM agents
			WHERE id = $1 AND status = 'running'
			LIMIT 1
		`, agentID).Scan(&host, &port)
		if err == nil {
			return host, port, true
		}
		log.Printf("channel webhook: agent_id=%s not found, falling back to user_id=%s", agentID, userID)
	}

	// Resolve via user_id (same logic as authenticated requests).
	if userID != "" {
		target, err := s.resolveAgent(ctx, userID)
		if err == nil {
			return target.Host, target.Port, true
		}
		log.Printf("channel webhook: resolveAgent failed for user_id=%s: %v", userID, err)
	}

	// Last resort: any running agent.
	err := s.db.QueryRow(ctx, `
		SELECT host, port FROM agents
		WHERE status = 'running'
		ORDER BY last_health DESC NULLS LAST
		LIMIT 1
	`).Scan(&host, &port)
	if err != nil {
		log.Printf("channel webhook: no running agent found: %v", err)
		return "", 0, false
	}
	return host, port, true
}

func (s *Server) proxyToAgentSamePath(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !proxy.HTTPStream(s.httpClient, w, r, target.Host, target.Port, r.URL.Path, id.UserID, true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *Server) handleNotificationsWS(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	s.handleAgentWSProxy(w, r, id, agentAPIPrefix+"/ws/notifications")
}

func (s *Server) handleConversationWS(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	s.handleAgentWSProxy(w, r, id, agentAPIPrefix+"/ws/conversation")
}

func (s *Server) handleAgentWSProxy(w http.ResponseWriter, r *http.Request, id auth.Identity, upstreamPath string) {
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

	agentConn, _, err := websocket.DefaultDialer.Dial(buildAgentWSURL(target, upstreamPath, r.URL.Query(), id), nil)
	if err != nil {
		_ = clientConn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseTryAgainLater, "agent unavailable"), time.Now().Add(wsWriteWait))
		return
	}
	defer agentConn.Close()

	prepareWSConn(clientConn)
	prepareWSConn(agentConn)

	done := make(chan struct{})
	var doneOnce sync.Once
	closeDone := func() {
		doneOnce.Do(func() { close(done) })
	}

	go heartbeatWS(clientConn, done)
	go heartbeatWS(agentConn, done)

	var wg sync.WaitGroup
	forward := func(dst, src *websocket.Conn) {
		defer wg.Done()
		defer closeDone()
		for {
			mt, msg, err := src.ReadMessage()
			if err != nil {
				closeWithMappedCode(dst, err)
				return
			}
			if err := dst.WriteMessage(mt, msg); err != nil {
				closeWithMappedCode(src, err)
				return
			}
		}
	}
	wg.Add(2)
	go forward(agentConn, clientConn)
	go forward(clientConn, agentConn)
	wg.Wait()
}

func buildAgentWSURL(target agent.Target, upstreamPath string, incomingQuery url.Values, id auth.Identity) string {
	u := url.URL{
		Scheme: "ws",
		Host:   fmt.Sprintf("%s:%d", target.Host, target.Port),
		Path:   upstreamPath,
	}
	q := cloneQuery(incomingQuery)
	q.Set("user_id", id.UserID)
	if id.Token != "" {
		q.Set("token", id.Token)
	}
	u.RawQuery = q.Encode()
	return u.String()
}

func cloneQuery(in url.Values) url.Values {
	out := make(url.Values, len(in))
	for k, values := range in {
		copied := make([]string, len(values))
		copy(copied, values)
		out[k] = copied
	}
	return out
}

func prepareWSConn(conn *websocket.Conn) {
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(wsPongWait))
	})
}

func heartbeatWS(conn *websocket.Conn, done <-chan struct{}) {
	ticker := time.NewTicker(wsPingPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			if err := conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(wsWriteWait)); err != nil {
				return
			}
		}
	}
}

func closeWithMappedCode(conn *websocket.Conn, readErr error) {
	code := websocket.CloseNormalClosure
	text := ""
	if ce, ok := readErr.(*websocket.CloseError); ok {
		code = ce.Code
		text = ce.Text
	}
	_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(code, text), time.Now().Add(wsWriteWait))
}

// resolveAgent finds the agent Target for a given user, trying strategies in order:
// local override → manager provision → DB by user_id → default agent → auto-assign first.
func (s *Server) resolveAgent(ctx context.Context, userID string) (agent.Target, error) {
	if t, ok := s.resolveLocalAgent(); ok {
		return t, nil
	}
	if t, ok := s.resolveViaManager(ctx, userID); ok {
		return t, nil
	}
	if t, ok := s.resolveByUserDB(ctx, userID); ok {
		return t, nil
	}
	if t, ok := s.resolveByDefaultID(ctx, userID); ok {
		return t, nil
	}
	if t, ok := s.resolveAutoAssignFirst(ctx, userID); ok {
		return t, nil
	}
	return agent.Target{}, fmt.Errorf("no running agent found for user %s", userID)
}

func (s *Server) resolveLocalAgent() (agent.Target, bool) {
	local := strings.TrimSpace(s.cfg.LocalAgentURL)
	if local == "" {
		return agent.Target{}, false
	}
	u, err := url.Parse(local)
	if err != nil || u.Hostname() == "" {
		return agent.Target{}, false
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
	return agent.Target{Host: u.Hostname(), Port: n}, true
}

func (s *Server) resolveViaManager(ctx context.Context, userID string) (agent.Target, bool) {
	if s.agentMgr == nil {
		return agent.Target{}, false
	}
	t, err := s.agentMgr.Provision(ctx, userID)
	if err != nil {
		log.Printf("agent provision failed for user %s: %v", userID, err)
		return agent.Target{}, false
	}
	s.agentMgr.RecordActivity(userID)
	return t, true
}

func (s *Server) resolveByUserDB(ctx context.Context, userID string) (agent.Target, bool) {
	var t agent.Target
	err := s.db.QueryRow(ctx, `
		SELECT host, port FROM agents
		WHERE user_id = $1 AND status = 'running'
		ORDER BY last_health DESC NULLS LAST
		LIMIT 1
	`, userID).Scan(&t.Host, &t.Port)
	return t, err == nil
}

func (s *Server) resolveByDefaultID(ctx context.Context, userID string) (agent.Target, bool) {
	if strings.TrimSpace(s.cfg.DefaultAgentID) == "" {
		return agent.Target{}, false
	}
	var t agent.Target
	err := s.db.QueryRow(ctx, `SELECT host, port FROM agents WHERE id = $1 AND status = 'running' LIMIT 1`, s.cfg.DefaultAgentID).Scan(&t.Host, &t.Port)
	if err != nil {
		return agent.Target{}, false
	}
	_, _ = s.db.Exec(ctx, `UPDATE agents SET user_id = $1 WHERE id = $2`, userID, s.cfg.DefaultAgentID)
	return t, true
}

func (s *Server) resolveAutoAssignFirst(ctx context.Context, userID string) (agent.Target, bool) {
	if !s.cfg.AutoAssignFirstAgent {
		return agent.Target{}, false
	}
	var agentID string
	var t agent.Target
	err := s.db.QueryRow(ctx, `
		SELECT id, host, port FROM agents
		WHERE status = 'running'
		ORDER BY last_health DESC NULLS LAST
		LIMIT 1
	`).Scan(&agentID, &t.Host, &t.Port)
	if err != nil {
		return agent.Target{}, false
	}
	_, _ = s.db.Exec(ctx, `UPDATE agents SET user_id = $1 WHERE id = $2`, userID, agentID)
	return t, true
}

func (s *Server) requireIdentity(next func(http.ResponseWriter, *http.Request, auth.Identity)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := s.auth.IdentityFromRequest(r)
		if err != nil {
			writeError(w, http.StatusUnauthorized, err.Error())
			return
		}
		next(w, r, id)
	}
}

func (s *Server) verifyAgentSecret(r *http.Request) *authErr {
	if s.cfg.AgentSecret == "" {
		return nil
	}
	a := auth.BearerToken(r)
	if a == "" {
		return &authErr{status: http.StatusUnauthorized, msg: "missing agent authorization"}
	}
	if a != s.cfg.AgentSecret {
		return &authErr{status: http.StatusForbidden, msg: "invalid agent secret"}
	}
	return nil
}

type adminTarget struct {
	UserID string
	Host   string
	Port   int
}

func (s *Server) resolveAdminTargets(ctx context.Context, userID string) ([]adminTarget, error) {
	if strings.TrimSpace(userID) != "" {
		target, err := s.resolveAgent(ctx, userID)
		if err != nil {
			return nil, err
		}
		return []adminTarget{{UserID: userID, Host: target.Host, Port: target.Port}}, nil
	}

	rows, err := s.db.Query(ctx, `
		SELECT COALESCE(user_id,''), host, port
		FROM agents
		WHERE status = 'running'
		ORDER BY last_health DESC NULLS LAST
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]adminTarget, 0)
	for rows.Next() {
		var t adminTarget
		if err := rows.Scan(&t.UserID, &t.Host, &t.Port); err != nil {
			return nil, err
		}
		if strings.TrimSpace(t.Host) == "" || t.Port <= 0 {
			continue
		}
		out = append(out, t)
	}
	if len(out) == 0 {
		return nil, errors.New("no running agents found")
	}
	return out, nil
}

func (s *Server) callAgentAdmin(ctx context.Context, host string, port int, method string, path string, payload map[string]any) (int, map[string]any, error) {
	urlStr := fmt.Sprintf("http://%s:%d%s", host, port, path)
	var body io.Reader
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return 0, nil, err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, urlStr, body)
	if err != nil {
		return 0, nil, err
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if strings.TrimSpace(s.cfg.AgentSecret) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(s.cfg.AgentSecret))
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()

	b, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return resp.StatusCode, nil, err
	}
	out := map[string]any{}
	if len(b) > 0 {
		if err := json.Unmarshal(b, &out); err != nil {
			out["raw"] = strings.TrimSpace(string(b))
		}
	}
	if resp.StatusCode >= 400 {
		return resp.StatusCode, out, fmt.Errorf("agent admin call failed: %s", resp.Status)
	}
	return resp.StatusCode, out, nil
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func (s *Server) lookupAgentDirectInfo(ctx context.Context, userID string) (string, string, error) {
	var prefix string
	var agentID string
	err := s.db.QueryRow(ctx, `
		SELECT COALESCE(subdomain_prefix, ''), COALESCE(id, '')
		FROM agents
		WHERE user_id = $1
		LIMIT 1
	`, userID).Scan(&prefix, &agentID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			if strings.TrimSpace(s.cfg.LocalAgentURL) != "" {
				return "local", "", nil
			}
			return "", "", fmt.Errorf("agent not found")
		}
		return "", "", err
	}
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return "", "", fmt.Errorf("agent subdomain is not ready")
	}
	return prefix, strings.TrimSpace(agentID), nil
}

func (s *Server) directAgentURLs(prefix string) (string, string, error) {
	if local := strings.TrimSpace(s.cfg.LocalAgentURL); local != "" {
		u, err := url.Parse(local)
		if err != nil {
			return "", "", fmt.Errorf("invalid local agent url")
		}
		base := strings.TrimRight(u.String(), "/")
		wsScheme := "ws"
		if strings.EqualFold(u.Scheme, "https") {
			wsScheme = "wss"
		}
		return base, wsScheme + "://" + u.Host, nil
	}
	domain := strings.Trim(strings.TrimSpace(s.cfg.DirectAgentDomain), ".")
	if domain == "" {
		return "", "", fmt.Errorf("direct agent domain is not configured")
	}
	host := strings.TrimSpace(prefix) + "." + domain
	return "https://" + host, "wss://" + host, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}
