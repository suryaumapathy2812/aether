package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
)

const maskedSecretValue = "__configured__"

type Handler struct {
	registry     *tools.Registry
	orchestrator *tools.Orchestrator
	plugins      *plugins.Manager
	store        *db.Store
}

type Options struct {
	Registry     *tools.Registry
	Orchestrator *tools.Orchestrator
	Plugins      *plugins.Manager
	Store        *db.Store
}

func New(opts Options) *Handler {
	return &Handler{registry: opts.Registry, orchestrator: opts.Orchestrator, plugins: opts.Plugins, store: opts.Store}
}

func (h *Handler) EnsurePluginCronJobs(ctx context.Context) error {
	if h == nil || h.plugins == nil {
		return nil
	}
	recs, err := h.plugins.ListEnabled(ctx)
	if err != nil {
		return err
	}
	for _, rec := range recs {
		manifest, err := h.plugins.ReadManifest(rec.Name)
		if err != nil {
			continue
		}
		_ = h.ensurePluginCronJobs(ctx, manifest, rec.Config)
	}
	return nil
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/internal/tools", h.handleTools)
	mux.HandleFunc("/internal/tools/schemas/openai", h.handleOpenAISchemas)
	mux.HandleFunc("/internal/tools/execute", h.handleExecute)
	mux.HandleFunc("/internal/hooks/", h.handleInternalHooks)
	mux.HandleFunc("/internal/plugins/status", h.handlePluginsStatus)
	mux.HandleFunc("/api/plugins", h.handlePluginsAPI)
	mux.HandleFunc("/api/plugins/", h.handlePluginsAPI)
}

func (h *Handler) handleInternalHooks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "store unavailable")
		return
	}

	pluginName := strings.Trim(strings.TrimPrefix(r.URL.Path, "/internal/hooks/"), "/")
	if pluginName == "" {
		httputil.WriteError(w, http.StatusBadRequest, "plugin name is required")
		return
	}
	userID := strings.TrimSpace(r.Header.Get("X-Aether-User-ID"))
	if userID == "" {
		userID = strings.TrimSpace(r.URL.Query().Get("user_id"))
	}
	if userID == "" {
		httputil.WriteError(w, http.StatusBadRequest, "missing user id")
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 2*1024*1024))
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	if rec, recErr := h.store.GetPlugin(r.Context(), pluginName); recErr == nil {
		if !rec.Enabled {
			httputil.WriteError(w, http.StatusConflict, "plugin is disabled")
			return
		}
	}

	webhookReq := plugins.WebhookRequest{
		Plugin:   pluginName,
		UserID:   userID,
		DeviceID: strings.TrimSpace(r.Header.Get("X-Aether-Device-ID")),
		Body:     body,
		Header:   r.Header,
	}
	task, err := plugins.DefaultWebhookRegistry().BuildTask(r.Context(), webhookReq)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	if task == nil {
		httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "ignored"})
		return
	}

	if strings.TrimSpace(task.SessionID) == "" {
		task.SessionID = pluginName + "-webhook"
	}
	if strings.TrimSpace(task.Title) == "" {
		task.Title = "Process " + pluginName + " webhook"
	}
	if strings.TrimSpace(task.Goal) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "webhook task goal is empty")
		return
	}

	// Webhook events are acknowledged and logged. When the agent system
	// is rebuilt, these will be routed to the LLM for processing.
	log.Printf("webhook received: plugin=%s title=%s user=%s", pluginName, task.Title, userID)
	httputil.WriteJSON(w, http.StatusAccepted, map[string]any{
		"status": "acknowledged",
		"plugin": pluginName,
		"title":  task.Title,
	})
}

func cloneAnyMap(src map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range src {
		out[k] = v
	}
	return out
}

func firstNonEmptyString(v any, fallback string) string {
	s, _ := v.(string)
	if strings.TrimSpace(s) != "" {
		return strings.TrimSpace(s)
	}
	return fallback
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (h *Handler) handleTools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.registry == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "tool registry unavailable")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"tools": h.registry.Definitions()})
}

func (h *Handler) handleOpenAISchemas(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.registry == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "tool registry unavailable")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"tools": h.registry.OpenAISchemas()})
}

func (h *Handler) handleExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.orchestrator == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "tool orchestrator unavailable")
		return
	}
	var req struct {
		Name   string         `json:"name"`
		Args   map[string]any `json:"args"`
		CallID string         `json:"call_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if req.Name == "" {
		httputil.WriteError(w, http.StatusBadRequest, "name is required")
		return
	}
	result := h.orchestrator.Execute(r.Context(), req.Name, req.Args, req.CallID)
	status := http.StatusOK
	if result.Error {
		status = http.StatusBadRequest
	}
	httputil.WriteJSON(w, status, map[string]any{"result": result})
}

func (h *Handler) handlePluginsStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.plugins == nil || h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "plugins status unavailable")
		return
	}
	ctx := r.Context()
	_, _ = h.plugins.Discover(ctx)

	pluginMetas := h.plugins.List()
	storeRecords, err := h.store.ListPlugins(ctx)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	byName := map[string]db.PluginRecord{}
	for _, rec := range storeRecords {
		byName[rec.Name] = rec
	}

	type item struct {
		Name              string   `json:"name"`
		DisplayName       string   `json:"display_name"`
		Source            string   `json:"source"`
		Enabled           bool     `json:"enabled"`
		MissingConfig     []string `json:"missing_config_keys"`
		LastRefreshAt     string   `json:"last_refresh_at,omitempty"`
		LastRefreshStatus string   `json:"last_refresh_status,omitempty"`
		LastRefreshError  string   `json:"last_refresh_error,omitempty"`
	}
	out := make([]item, 0, len(pluginMetas))
	for _, meta := range pluginMetas {
		rec := byName[meta.Name]
		requiredKeys, _ := h.plugins.RequiredConfigKeys(meta.Name)
		missing := []string{}
		for _, key := range requiredKeys {
			if strings.TrimSpace(rec.Config[key]) == "" {
				missing = append(missing, key)
			}
		}
		sort.Strings(missing)
		out = append(out, item{
			Name:              meta.Name,
			DisplayName:       meta.DisplayName,
			Source:            string(meta.Source),
			Enabled:           rec.Enabled,
			MissingConfig:     missing,
			LastRefreshAt:     rec.Config["last_refresh_at"],
			LastRefreshStatus: rec.Config["last_refresh_status"],
			LastRefreshError:  rec.Config["last_refresh_error"],
		})
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"plugins": out, "count": len(out)})
}

func (h *Handler) handlePluginsAPI(w http.ResponseWriter, r *http.Request) {
	if h.plugins == nil || h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "plugins api unavailable")
		return
	}
	ctx := r.Context()
	_, _ = h.plugins.Discover(ctx)

	if r.URL.Path == "/api/plugins" {
		if r.Method != http.MethodGet {
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		h.listPluginsCompat(w, r)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/plugins/")
	path = strings.Trim(path, "/")
	parts := strings.Split(path, "/")
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		httputil.WriteError(w, http.StatusNotFound, "not found")
		return
	}
	name := parts[0]

	if len(parts) == 1 {
		if r.Method == http.MethodDelete {
			if err := h.store.SetPluginEnabled(ctx, name, false); err != nil {
				if err == db.ErrNotFound {
					httputil.WriteError(w, http.StatusNotFound, "plugin not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "disabled"})
			return
		}
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	action := parts[1]
	switch action {
	case "install":
		if r.Method != http.MethodPost {
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		h.handlePluginInstallCompat(w, r, name)
		return
	case "enable":
		if r.Method != http.MethodPost {
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		manifest, manifestErr := h.plugins.ReadManifest(name)
		if manifestErr != nil {
			httputil.WriteError(w, http.StatusNotFound, "plugin not found")
			return
		}
		if err := h.store.SetPluginEnabled(ctx, name, true); err != nil {
			if err == db.ErrNotFound {
				httputil.WriteError(w, http.StatusNotFound, "plugin not found")
				return
			}
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if rec, err := h.store.GetPlugin(ctx, name); err == nil {
			_ = h.ensurePluginCronJobs(ctx, manifest, rec.Config)
		}
		httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "enabled"})
		return
	case "disable":
		if r.Method != http.MethodPost {
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if err := h.store.SetPluginEnabled(ctx, name, false); err != nil {
			if err == db.ErrNotFound {
				httputil.WriteError(w, http.StatusNotFound, "plugin not found")
				return
			}
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
		httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "disabled"})
		return
	case "config":
		switch r.Method {
		case http.MethodGet:
			rec, err := h.store.GetPlugin(ctx, name)
			if err != nil {
				if err == db.ErrNotFound {
					httputil.WriteError(w, http.StatusNotFound, "plugin not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			manifest, err := h.plugins.ReadManifest(name)
			if err != nil {
				httputil.WriteError(w, http.StatusNotFound, "plugin not found")
				return
			}
			httputil.WriteJSON(w, http.StatusOK, scrubSecretConfig(rec.Config, secretConfigKeys(manifest)))
			return
		case http.MethodPost:
			var req struct {
				Config map[string]string `json:"config"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
				return
			}
			if req.Config == nil {
				req.Config = map[string]string{}
			}
			rec, err := h.store.GetPlugin(ctx, name)
			if err != nil {
				if err == db.ErrNotFound {
					httputil.WriteError(w, http.StatusNotFound, "plugin not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			manifest, err := h.plugins.ReadManifest(name)
			if err != nil {
				httputil.WriteError(w, http.StatusNotFound, "plugin not found")
				return
			}
			merged := mergePluginConfig(rec.Config, req.Config, secretConfigKeys(manifest))
			if err := h.store.SetPluginConfig(ctx, name, encryptSecretConfig(merged, h.store, secretConfigKeys(manifest))); err != nil {
				if err == db.ErrNotFound {
					httputil.WriteError(w, http.StatusNotFound, "plugin not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}

			autoEnabled := false
			authType, _, _ := pluginAuthDetails(manifest)
			required, reqErr := h.plugins.RequiredConfigKeys(name)
			if authType == "api_key" && reqErr == nil && len(required) > 0 {
				complete := true
				for _, key := range required {
					if strings.TrimSpace(merged[key]) == "" {
						complete = false
						break
					}
				}
				if complete {
					if err := h.store.SetPluginEnabled(ctx, name, true); err == nil {
						autoEnabled = true
					}
				}
			}
			httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "updated", "auto_enabled": autoEnabled})
			return
		default:
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
	case "oauth":
		if len(parts) == 3 && parts[2] == "start" && r.Method == http.MethodGet {
			h.handlePluginOAuthStart(w, r, name)
			return
		}
		if len(parts) == 3 && parts[2] == "callback" && r.Method == http.MethodPost {
			h.handlePluginOAuthCallback(w, r, name)
			return
		}
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	default:
		httputil.WriteError(w, http.StatusNotFound, "not found")
		return
	}
}

func (h *Handler) listPluginsCompat(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	plugs := h.plugins.List()
	recs, err := h.store.ListPlugins(ctx)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	byName := map[string]db.PluginRecord{}
	for _, rec := range recs {
		byName[rec.Name] = rec
	}

	out := make([]map[string]any, 0, len(plugs))
	for _, p := range plugs {
		manifest, err := h.plugins.ReadManifest(p.Name)
		if err != nil {
			continue
		}
		rec := byName[p.Name]
		authType, authProvider, fields := pluginAuthDetails(manifest)
		fields = filterEnvBackedOAuthFields(fields, authProvider)
		requiredKeys := []string{}
		for _, f := range fields {
			required, _ := f["required"].(bool)
			if required {
				if key, _ := f["key"].(string); strings.TrimSpace(key) != "" {
					requiredKeys = append(requiredKeys, key)
				}
			}
		}
		hasRequired := true
		for _, key := range requiredKeys {
			if strings.TrimSpace(rec.Config[key]) == "" {
				hasRequired = false
				break
			}
		}
		connected := authType == "none" || hasRequired
		if authType == "oauth2" {
			if strings.TrimSpace(rec.Config["access_token"]) == "" && strings.TrimSpace(rec.Config["refresh_token"]) == "" {
				connected = false
			}
		}
		needsReconnect, _ := strconv.ParseBool(strings.TrimSpace(rec.Config["needs_reconnect"]))

		out = append(out, map[string]any{
			"name":            p.Name,
			"display_name":    p.DisplayName,
			"description":     p.Description,
			"auth_type":       authType,
			"auth_provider":   authProvider,
			"config_fields":   fields,
			"installed":       true,
			"plugin_id":       p.Name,
			"enabled":         rec.Enabled,
			"connected":       connected,
			"needs_reconnect": needsReconnect,
		})
	}

	httputil.WriteJSON(w, http.StatusOK, out)
}

func (h *Handler) handlePluginInstallCompat(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	manifest, err := h.plugins.ReadManifest(name)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "plugin not found")
		return
	}
	base := db.PluginRecord{
		Name:        manifest.Name,
		DisplayName: manifest.DisplayName,
		Description: manifest.Description,
		Version:     manifest.Version,
		PluginType:  manifest.PluginType,
		Location:    manifest.Location,
		Source:      string(manifest.Source),
		HasSkill:    strings.TrimSpace(manifest.SkillPath) != "",
		Config:      map[string]string{},
	}
	if err := h.store.UpsertPlugin(ctx, base); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"plugin_id": name, "status": "installed"})
}

func (h *Handler) handlePluginOAuthStart(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	manifest, err := h.plugins.ReadManifest(name)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "plugin not found")
		return
	}
	authType, provider, _ := pluginAuthDetails(manifest)
	if authType != "oauth2" {
		httputil.WriteError(w, http.StatusBadRequest, "plugin does not use oauth2")
		return
	}

	if err := h.ensurePluginInstalled(ctx, manifest); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	rec, err := h.store.GetPlugin(ctx, name)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	cfg := cloneConfig(rec.Config)

	clientID := strings.TrimSpace(cfg["client_id"])
	clientSecret, err := maybeDecryptStoredSecret(cfg["client_secret"], h.store)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid oauth client_secret")
		return
	}
	envClientID, envClientSecret := oauthProviderEnvCredentials(provider)
	updated := false
	if clientID == "" && strings.TrimSpace(envClientID) != "" {
		clientID = strings.TrimSpace(envClientID)
		cfg["client_id"] = clientID
		updated = true
	}
	if strings.TrimSpace(clientSecret) == "" && strings.TrimSpace(envClientSecret) != "" {
		clientSecret = strings.TrimSpace(envClientSecret)
		cfg["client_secret"] = encryptIfPossible(clientSecret, h.store)
		updated = true
	}
	if updated {
		if err := h.store.SetPluginConfig(ctx, name, cfg); err != nil {
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
	}

	if clientID == "" {
		httputil.WriteError(w, http.StatusBadRequest, "missing oauth client_id in plugin config")
		return
	}
	if strings.TrimSpace(clientSecret) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "missing oauth client_secret in plugin config")
		return
	}

	state, err := generateOAuthState()
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, "failed to create oauth state")
		return
	}
	redirectURI := oauthRedirectURI(r, name)
	oauthCfg := oauthProviderConfig(provider)
	if oauthCfg.AuthURL == "" || oauthCfg.TokenURL == "" {
		httputil.WriteError(w, http.StatusBadRequest, "unsupported oauth provider")
		return
	}

	cfg["oauth_provider"] = provider
	cfg["oauth_state"] = state
	cfg["oauth_state_expires_at"] = strconv.FormatInt(time.Now().UTC().Add(10*time.Minute).Unix(), 10)
	cfg["oauth_redirect_uri"] = redirectURI
	cfg["oauth_token_url"] = oauthCfg.TokenURL
	if err := h.store.SetPluginConfig(ctx, name, cfg); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	scopes := oauthScopes(manifest)
	v := url.Values{}
	v.Set("client_id", clientID)
	v.Set("redirect_uri", redirectURI)
	v.Set("response_type", "code")
	v.Set("state", state)
	if len(scopes) > 0 {
		v.Set("scope", strings.Join(scopes, " "))
	}
	for k, val := range oauthCfg.AuthParams {
		v.Set(k, val)
	}
	authURL := oauthCfg.AuthURL + "?" + v.Encode()
	http.Redirect(w, r, authURL, http.StatusFound)
}

func (h *Handler) handlePluginOAuthCallback(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	manifest, err := h.plugins.ReadManifest(name)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "plugin not found")
		return
	}
	authType, provider, _ := pluginAuthDetails(manifest)
	if authType != "oauth2" {
		httputil.WriteError(w, http.StatusBadRequest, "plugin does not use oauth2")
		return
	}

	var req struct {
		Code  string `json:"code"`
		State string `json:"state"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if strings.TrimSpace(req.Code) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "missing oauth code")
		return
	}

	rec, err := h.store.GetPlugin(ctx, name)
	if err != nil {
		if err == db.ErrNotFound {
			httputil.WriteError(w, http.StatusNotFound, "plugin not found")
			return
		}
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	cfg := cloneConfig(rec.Config)
	if !validOAuthState(cfg, req.State) {
		httputil.WriteError(w, http.StatusBadRequest, "invalid or expired oauth state")
		return
	}

	oauthCfg := oauthProviderConfig(provider)
	tokenURL := firstNonEmpty(strings.TrimSpace(cfg["oauth_token_url"]), oauthCfg.TokenURL)
	redirectURI := firstNonEmpty(strings.TrimSpace(cfg["oauth_redirect_uri"]), oauthRedirectURI(r, name))
	clientID := strings.TrimSpace(cfg["client_id"])
	clientSecret, err := maybeDecryptStoredSecret(cfg["client_secret"], h.store)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid oauth client_secret")
		return
	}
	if tokenURL == "" || clientID == "" || strings.TrimSpace(clientSecret) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "oauth plugin config missing token endpoint or client credentials")
		return
	}

	tokenResp, err := exchangeOAuthCode(ctx, tokenURL, clientID, clientSecret, redirectURI, strings.TrimSpace(req.Code), provider == "spotify")
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	now := time.Now().UTC()
	expiresIn := tokenResp.ExpiresIn
	if expiresIn <= 0 {
		expiresIn = 3600
	}
	expiresAt := now.Add(time.Duration(expiresIn) * time.Second)

	cfg["access_token"] = encryptIfPossible(tokenResp.AccessToken, h.store)
	if strings.TrimSpace(tokenResp.RefreshToken) != "" {
		cfg["refresh_token"] = encryptIfPossible(tokenResp.RefreshToken, h.store)
	}
	if strings.TrimSpace(tokenResp.TokenType) != "" {
		cfg["token_type"] = tokenResp.TokenType
	}
	if strings.TrimSpace(tokenResp.Scope) != "" {
		cfg["scope"] = tokenResp.Scope
	}
	cfg["expires_at"] = strconv.FormatInt(expiresAt.Unix(), 10)
	cfg["last_refresh_at"] = now.Format(time.RFC3339)
	cfg["last_refresh_status"] = "ok"
	cfg["last_refresh_error"] = ""
	cfg["refresh_fail_count"] = "0"
	cfg["next_refresh_at"] = expiresAt.Add(-5 * time.Minute).Format(time.RFC3339)
	cfg["needs_reconnect"] = "false"
	cfg["oauth_state"] = ""
	cfg["oauth_state_expires_at"] = ""

	if email, err := lookupOAuthAccountEmail(ctx, provider, tokenResp.AccessToken); err == nil && strings.TrimSpace(email) != "" {
		cfg["account_email"] = email
	}

	if err := h.store.SetPluginConfig(ctx, name, cfg); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := h.store.SetPluginEnabled(ctx, name, true); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	_ = h.ensurePluginCronJobs(ctx, manifest, cfg)

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "connected", "plugin": name})
}

type oauthProvider struct {
	AuthURL    string
	TokenURL   string
	AuthParams map[string]string
}

type oauthTokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
	Scope        string `json:"scope"`
	Error        string `json:"error"`
	Description  string `json:"error_description"`
}

func oauthProviderConfig(provider string) oauthProvider {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "google":
		return oauthProvider{
			AuthURL:  "https://accounts.google.com/o/oauth2/v2/auth",
			TokenURL: "https://oauth2.googleapis.com/token",
			AuthParams: map[string]string{
				"access_type":            "offline",
				"prompt":                 "consent",
				"include_granted_scopes": "true",
			},
		}
	case "spotify":
		return oauthProvider{
			AuthURL:  "https://accounts.spotify.com/authorize",
			TokenURL: "https://accounts.spotify.com/api/token",
			AuthParams: map[string]string{
				"show_dialog": "true",
			},
		}
	default:
		return oauthProvider{}
	}
}

func oauthProviderEnvCredentials(provider string) (string, string) {
	p := strings.ToLower(strings.TrimSpace(provider))
	// Explicit mappings for known providers.
	switch p {
	case "google":
		return strings.TrimSpace(os.Getenv("GOOGLE_CLIENT_ID")), strings.TrimSpace(os.Getenv("GOOGLE_CLIENT_SECRET"))
	case "spotify":
		return strings.TrimSpace(os.Getenv("SPOTIFY_CLIENT_ID")), strings.TrimSpace(os.Getenv("SPOTIFY_CLIENT_SECRET"))
	}
	// Generic fallback: check {PROVIDER}_CLIENT_ID / {PROVIDER}_CLIENT_SECRET.
	// This allows new OAuth2 plugins to use env vars without code changes —
	// just set e.g. GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env.
	if p != "" {
		prefix := strings.ToUpper(strings.ReplaceAll(p, "-", "_"))
		return strings.TrimSpace(os.Getenv(prefix + "_CLIENT_ID")),
			strings.TrimSpace(os.Getenv(prefix + "_CLIENT_SECRET"))
	}
	return "", ""
}

func filterEnvBackedOAuthFields(fields []map[string]any, provider string) []map[string]any {
	envClientID, envClientSecret := oauthProviderEnvCredentials(provider)
	if strings.TrimSpace(envClientID) == "" || strings.TrimSpace(envClientSecret) == "" {
		return fields
	}
	filtered := make([]map[string]any, 0, len(fields))
	for _, field := range fields {
		key, _ := field["key"].(string)
		key = strings.TrimSpace(key)
		if key == "client_id" || key == "client_secret" {
			continue
		}
		filtered = append(filtered, field)
	}
	return filtered
}

func oauthScopes(manifest plugins.PluginManifest) []string {
	return manifest.Auth.Scopes
}

func oauthRedirectURI(r *http.Request, pluginName string) string {
	host := strings.TrimSpace(r.Header.Get("X-Forwarded-Host"))
	proto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto"))
	if host != "" {
		if proto == "" {
			proto = "https"
		}
		return fmt.Sprintf("%s://%s/plugins/%s/oauth/callback", proto, host, url.PathEscape(pluginName))
	}
	if origin := strings.TrimSpace(r.Header.Get("Origin")); origin != "" {
		return strings.TrimRight(origin, "/") + "/plugins/" + url.PathEscape(pluginName) + "/oauth/callback"
	}
	if ref := strings.TrimSpace(r.Header.Get("Referer")); ref != "" {
		if u, err := url.Parse(ref); err == nil && u.Scheme != "" && u.Host != "" {
			return u.Scheme + "://" + u.Host + "/plugins/" + url.PathEscape(pluginName) + "/oauth/callback"
		}
	}
	return "http://localhost:3000/plugins/" + url.PathEscape(pluginName) + "/oauth/callback"
}

func generateOAuthState() (string, error) {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func validOAuthState(cfg map[string]string, state string) bool {
	if strings.TrimSpace(state) == "" {
		return false
	}
	expected := strings.TrimSpace(cfg["oauth_state"])
	if expected == "" || state != expected {
		return false
	}
	expiresRaw := strings.TrimSpace(cfg["oauth_state_expires_at"])
	if expiresRaw == "" {
		return true
	}
	expiresUnix, err := strconv.ParseInt(expiresRaw, 10, 64)
	if err != nil {
		return false
	}
	return time.Now().UTC().Unix() <= expiresUnix
}

func exchangeOAuthCode(ctx context.Context, tokenURL, clientID, clientSecret, redirectURI, code string, useBasicAuth bool) (oauthTokenResponse, error) {
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("redirect_uri", redirectURI)
	if !useBasicAuth {
		form.Set("client_id", clientID)
		form.Set("client_secret", clientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return oauthTokenResponse{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if useBasicAuth {
		req.SetBasicAuth(clientID, clientSecret)
	}

	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return oauthTokenResponse{}, err
	}
	defer resp.Body.Close()

	var out oauthTokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return oauthTokenResponse{}, fmt.Errorf("failed to decode oauth token response")
	}
	if resp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(out.Error)
		if strings.TrimSpace(out.Description) != "" {
			msg = strings.TrimSpace(out.Description)
		}
		if msg == "" {
			msg = fmt.Sprintf("token endpoint status %d", resp.StatusCode)
		}
		return oauthTokenResponse{}, fmt.Errorf("oauth exchange failed: %s", msg)
	}
	if strings.TrimSpace(out.AccessToken) == "" {
		return oauthTokenResponse{}, fmt.Errorf("oauth exchange failed: missing access_token")
	}
	return out, nil
}

func lookupOAuthAccountEmail(ctx context.Context, provider, accessToken string) (string, error) {
	accessToken = strings.TrimSpace(accessToken)
	if accessToken == "" {
		return "", fmt.Errorf("missing access token")
	}
	provider = strings.ToLower(strings.TrimSpace(provider))
	endpoint := ""
	switch provider {
	case "google":
		endpoint = "https://www.googleapis.com/oauth2/v2/userinfo"
	case "spotify":
		endpoint = "https://api.spotify.com/v1/me"
	default:
		return "", fmt.Errorf("unsupported provider")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)
	resp, err := (&http.Client{Timeout: 15 * time.Second}).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("userinfo status %d", resp.StatusCode)
	}
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return "", err
	}
	email, _ := obj["email"].(string)
	return strings.TrimSpace(email), nil
}

func (h *Handler) ensurePluginInstalled(ctx context.Context, manifest plugins.PluginManifest) error {
	if _, err := h.store.GetPlugin(ctx, manifest.Name); err == nil {
		return nil
	} else if err != db.ErrNotFound {
		return err
	}
	return h.store.UpsertPlugin(ctx, db.PluginRecord{
		Name:        manifest.Name,
		DisplayName: manifest.DisplayName,
		Description: manifest.Description,
		Version:     manifest.Version,
		PluginType:  manifest.PluginType,
		Location:    manifest.Location,
		Source:      string(manifest.Source),
		HasSkill:    strings.TrimSpace(manifest.SkillPath) != "",
		Config:      map[string]string{},
	})
}

func (h *Handler) ensurePluginCronJobs(ctx context.Context, manifest plugins.PluginManifest, cfg map[string]string) error {
	scope := h.store.ScopeCronModule(plugins.CronModulePlugins)
	jobs, err := scope.List(ctx)
	if err != nil {
		return err
	}

	rotateInterval := int64(manifest.Auth.RefreshInterval)
	if rotateInterval > 0 {
		runAt := time.Now().UTC().Add(time.Duration(rotateInterval) * time.Second)
		if next := strings.TrimSpace(cfg["next_refresh_at"]); next != "" {
			if t, err := time.Parse(time.RFC3339, next); err == nil && t.After(time.Now().UTC()) {
				runAt = t
			}
		}
		if !hasActivePluginCronJob(jobs, manifest.Name, plugins.CronJobTypeRotate) {
			_, _ = scope.ScheduleRecurring(ctx, plugins.CronJobTypeRotate, map[string]any{"plugin": manifest.Name}, runAt, rotateInterval, 5)
		}
	}

	renewInterval := int64FromAny(manifest.Webhook["renew_interval"])
	if renewInterval > 0 {
		runAt := time.Now().UTC().Add(time.Duration(renewInterval) * time.Second)
		if !hasActivePluginCronJob(jobs, manifest.Name, plugins.CronJobTypeRenewWatch) {
			_, _ = scope.ScheduleRecurring(ctx, plugins.CronJobTypeRenewWatch, map[string]any{"plugin": manifest.Name}, runAt, renewInterval, 5)
		}
	}
	return nil
}

func hasActivePluginCronJob(jobs []db.CronJobRecord, pluginName, jobType string) bool {
	for _, job := range jobs {
		if job.JobType != jobType || !job.Enabled {
			continue
		}
		if !jobPayloadHasPlugin(job.PayloadJSON, pluginName) {
			continue
		}
		if job.Status == db.CronStatusCancelled || job.Status == db.CronStatusDone || job.Status == db.CronStatusFailed {
			continue
		}
		return true
	}
	return false
}

func jobPayloadHasPlugin(payloadJSON, pluginName string) bool {
	var payload map[string]any
	if err := json.Unmarshal([]byte(payloadJSON), &payload); err != nil {
		return false
	}
	v, ok := payload["plugin"].(string)
	if !ok || strings.TrimSpace(v) == "" {
		v, _ = payload["plugin_name"].(string)
	}
	return strings.EqualFold(strings.TrimSpace(v), strings.TrimSpace(pluginName))
}

func int64FromAny(v any) int64 {
	switch n := v.(type) {
	case int:
		return int64(n)
	case int64:
		return n
	case float64:
		return int64(n)
	case string:
		i, err := strconv.ParseInt(strings.TrimSpace(n), 10, 64)
		if err == nil {
			return i
		}
	}
	return 0
}

func cloneConfig(src map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range src {
		out[k] = v
	}
	return out
}

func secretConfigKeys(manifest plugins.PluginManifest) map[string]bool {
	keys := map[string]bool{}
	for _, f := range manifest.Auth.ConfigFields {
		if strings.EqualFold(strings.TrimSpace(f.Type), "password") && strings.TrimSpace(f.Key) != "" {
			keys[f.Key] = true
		}
	}
	for _, key := range []string{"access_token", "refresh_token", "oauth_refresh_token", "client_secret", "google_client_secret", "spotify_client_secret", "auth_token", "bot_token", "secret_token", "api_key", "wolfram_app_id", "exchangerate_api_key", "google_api_key", "google_places_api_key"} {
		keys[key] = true
	}
	return keys
}

func scrubSecretConfig(cfg map[string]string, secretKeys map[string]bool) map[string]string {
	out := cloneConfig(cfg)
	for key := range secretKeys {
		if strings.TrimSpace(out[key]) != "" {
			out[key] = maskedSecretValue
		}
	}
	return out
}

func mergePluginConfig(existing, incoming map[string]string, secretKeys map[string]bool) map[string]string {
	merged := cloneConfig(existing)
	for k, v := range incoming {
		if secretKeys[k] && v == maskedSecretValue {
			continue
		}
		merged[k] = v
	}
	return merged
}

func encryptSecretConfig(cfg map[string]string, store *db.Store, secretKeys map[string]bool) map[string]string {
	out := cloneConfig(cfg)
	for key := range secretKeys {
		value := strings.TrimSpace(out[key])
		if value == "" || strings.HasPrefix(value, "enc:v1:") {
			continue
		}
		if enc, err := store.EncryptString(value); err == nil {
			out[key] = enc
		}
	}
	return out
}

func maybeDecryptStoredSecret(value string, store *db.Store) (string, error) {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "enc:v1:") {
		return value, nil
	}
	return store.DecryptString(value)
}

func encryptIfPossible(value string, store *db.Store) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if strings.HasPrefix(value, "enc:v1:") {
		return value
	}
	if enc, err := store.EncryptString(value); err == nil {
		return enc
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func pluginAuthDetails(manifest plugins.PluginManifest) (string, string, []map[string]any) {
	authType := manifest.Auth.Type
	if strings.TrimSpace(authType) == "" {
		authType = "none"
	}
	authProvider := manifest.Auth.Provider
	fields := make([]map[string]any, 0, len(manifest.Auth.ConfigFields))
	for _, f := range manifest.Auth.ConfigFields {
		field := map[string]any{
			"key":      f.Key,
			"label":    f.Label,
			"type":     f.Type,
			"required": f.Required,
		}
		if f.Description != "" {
			field["description"] = f.Description
		}
		fields = append(fields, field)
	}
	return authType, authProvider, fields
}


