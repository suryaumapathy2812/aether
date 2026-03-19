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
)

func New(cfg config.Config, db *pgxpool.Pool, mgr *agent.Manager) *Server {
	return &Server{
		cfg:      cfg,
		db:       db,
		auth:     auth.New(db, cfg.AgentSecret),
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
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/health", s.handleHealth)
	mux.HandleFunc("/auth/me", s.requireIdentity(s.handleAuthMe))
	mux.HandleFunc("/api/auth/me", s.requireIdentity(s.handleAuthMe))

	mux.HandleFunc("/api/agents/register", s.handleRegisterAgent)
	mux.HandleFunc("/api/agents/health", s.handleListAgents)
	mux.HandleFunc("/api/agents/version", s.handleAgentVersionAdmin)
	mux.HandleFunc("/api/agents/reload", s.handleAgentReloadAdmin)
	mux.HandleFunc("/api/agents/upgrade", s.handleAgentUpgradeAdmin)
	mux.HandleFunc("/api/agents/", s.handleAgentsByID)

	mux.HandleFunc("/api/agent/ready", s.requireIdentity(s.handleAgentReady))
	mux.HandleFunc("/api/metrics/latency", s.requireIdentity(s.handleLatency))
	mux.HandleFunc("/api/webrtc/offer", s.requireIdentity(s.handleWebRTCOffer))
	mux.HandleFunc("/api/webrtc/ice", s.requireIdentity(s.handleWebRTCIce))
	mux.HandleFunc("/api/ws/notifications", s.requireIdentity(s.handleNotificationsWS))
	mux.HandleFunc("/api/ws/conversation", s.requireIdentity(s.handleConversationWS))
	mux.HandleFunc("/api/ws", s.requireIdentity(s.handleNotificationsWS))
	mux.HandleFunc("/api/pair/request", s.handlePairRequest)
	mux.HandleFunc("/api/pair/status/", s.handlePairStatus)
	mux.HandleFunc("/api/pair/claim", s.requireIdentity(s.handlePairClaim))
	mux.HandleFunc("/api/devices", s.requireIdentity(s.handleDevices))
	mux.HandleFunc("/api/devices/telegram", s.requireIdentity(s.handleTelegramDevice))
	mux.HandleFunc("/api/devices/", s.requireIdentity(s.handleDeviceByID))
	mux.HandleFunc("/api/hooks/", s.handlePluginWebhookIngress)
	mux.HandleFunc("/api/hooks/pubsub/", s.handlePubsubWebhookIngress)

	mux.HandleFunc("/v1/", s.requireIdentity(s.handleV1Proxy))
	mux.HandleFunc("/api/memory/", s.requireIdentity(s.handleMemoryProxy))
	mux.HandleFunc("/api/preferences", s.requireIdentity(s.proxyToAgentSamePath))
	mux.HandleFunc("/api/preferences/", s.requireIdentity(s.proxyToAgentSamePath))
	mux.HandleFunc("/api/plugins", s.requireIdentity(s.handlePluginsProxy))
	mux.HandleFunc("/api/plugins/", s.requireIdentity(s.handlePluginsProxy))
	mux.HandleFunc("/api/push/vapid-key", s.requireIdentity(s.handlePushProxy))
	mux.HandleFunc("/api/push/subscribe", s.requireIdentity(s.handlePushProxy))
	mux.HandleFunc("/api/push/test", s.requireIdentity(s.handlePushProxy))

	// Channel webhook — unauthenticated, called by Telegram/WhatsApp/etc.
	// URL: POST /api/{user_id}/channels/{channel_type}/webhook/{agent_id}
	//
	// NOTE: We register this on /api/ (prefix) and parse the path manually to
	// avoid net/http ServeMux wildcard conflicts with existing /api/* routes.
	mux.HandleFunc("/api/", s.handleChannelWebhookProxy)

	// Channel management — authenticated, proxy to agent for Telegram/WhatsApp/etc.
	mux.HandleFunc("/api/channels", s.requireIdentity(s.proxyToAgentSamePath))
	mux.HandleFunc("/api/channels/", s.requireIdentity(s.proxyToAgentSamePath))

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

func (s *Server) handleWebRTCOffer(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !proxy.HTTPStream(s.httpClient, w, r, target.Host, target.Port, "/webrtc/offer", id.UserID, true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
}

func (s *Server) handleWebRTCIce(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	target, err := s.resolveAgent(r.Context(), id.UserID)
	if err != nil {
		writeError(w, http.StatusNotFound, "No agent assigned")
		return
	}
	if !proxy.HTTPStream(s.httpClient, w, r, target.Host, target.Port, "/webrtc/ice", id.UserID, true) {
		writeError(w, http.StatusBadGateway, "agent unavailable")
	}
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
// URL: POST /api/{user_id}/channels/{channel_type}/webhook/{agent_id}
//
// The user_id in the URL allows the orchestrator to resolve the correct agent
// via the standard resolveAgent() path. The agent_id is used for direct DB
// lookup as a fast path; user_id is the fallback.
// The upstream path forwarded to the agent strips the user_id and agent_id:
// /api/channels/{channel_type}/webhook
func (s *Server) handleChannelWebhookProxy(w http.ResponseWriter, r *http.Request) {
	// Match only: /api/{user_id}/channels/{channel_type}/webhook/{agent_id}
	trimmed := strings.Trim(r.URL.Path, "/")
	parts := strings.Split(trimmed, "/")
	if len(parts) != 6 || parts[0] != "api" || parts[2] != "channels" || parts[4] != "webhook" {
		// Not a channel webhook path; let this fallback handler return 404.
		writeError(w, http.StatusNotFound, "not found")
		return
	}
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	userID := strings.TrimSpace(parts[1])
	channelType := strings.TrimSpace(parts[3])
	agentID := strings.TrimSpace(parts[5])
	if userID == "" || channelType == "" {
		writeError(w, http.StatusNotFound, "not found")
		return
	}

	// The agent receives the standard webhook path (no user_id / agent_id).
	upstreamPath := fmt.Sprintf("/api/channels/%s/webhook", channelType)

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

	// Production: try agent_id first (fast path), then fall back to user_id resolution.
	var host string
	var port int

	if agentID != "" {
		err := s.db.QueryRow(r.Context(), `
			SELECT host, port FROM agents
			WHERE id = $1 AND status = 'running'
			LIMIT 1
		`, agentID).Scan(&host, &port)
		if err == nil {
			goto forward
		}
		log.Printf("channel webhook: agent_id=%s not found, falling back to user_id=%s", agentID, userID)
	}

	// Fallback: resolve agent by user_id (same path as authenticated requests).
	if userID != "" {
		target, err := s.resolveAgent(r.Context(), userID)
		if err == nil {
			host = target.Host
			port = target.Port
			goto forward
		}
		log.Printf("channel webhook: resolveAgent failed for user_id=%s: %v", userID, err)
	}

	// Last resort: any running agent.
	{
		err := s.db.QueryRow(r.Context(), `
			SELECT host, port FROM agents
			WHERE status = 'running'
			ORDER BY last_health DESC NULLS LAST
			LIMIT 1
		`).Scan(&host, &port)
		if err != nil {
			log.Printf("channel webhook: no running agent found: %v", err)
			w.WriteHeader(http.StatusOK)
			return
		}
	}

forward:
	if !proxy.HTTPStream(s.httpClient, w, r, host, port, upstreamPath, "", false) {
		w.WriteHeader(http.StatusOK) // 200 to Telegram even on failure
	}
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
	s.handleAgentWSProxy(w, r, id, "/ws/notifications")
}

func (s *Server) handleConversationWS(w http.ResponseWriter, r *http.Request, id auth.Identity) {
	s.handleAgentWSProxy(w, r, id, "/ws/conversation")
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

func (s *Server) resolveAgent(ctx context.Context, userID string) (agent.Target, error) {
	if local := strings.TrimSpace(s.cfg.LocalAgentURL); local != "" {
		u, err := url.Parse(local)
		if err != nil || u.Hostname() == "" {
			return agent.Target{}, fmt.Errorf("invalid AETHER_LOCAL_AGENT_URL")
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
		return agent.Target{Host: u.Hostname(), Port: n}, nil
	}

	if s.agentMgr != nil {
		t, err := s.agentMgr.Provision(ctx, userID)
		if err == nil {
			s.agentMgr.RecordActivity(userID)
			return t, nil
		}
		log.Printf("agent provision failed for user %s: %v", userID, err)
	}

	var t agent.Target
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

	return agent.Target{}, err
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}
