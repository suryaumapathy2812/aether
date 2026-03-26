package httpapi

import (
	"bytes"
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

	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
	"github.com/suryaumapathy2812/core-ai/agent/internal/integrations"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

const maskedSecretValue = "__configured__"

type Handler struct {
	registry     *tools.Registry
	orchestrator *tools.Orchestrator
	integrations *integrations.Manager
	store        *db.Store
	validator    *agentauth.Validator
}

type Options struct {
	Registry     *tools.Registry
	Orchestrator *tools.Orchestrator
	Integrations *integrations.Manager
	Store        *db.Store
	Validator    *agentauth.Validator
}

func New(opts Options) *Handler {
	return &Handler{registry: opts.Registry, orchestrator: opts.Orchestrator, integrations: opts.Integrations, store: opts.Store, validator: opts.Validator}
}

func (h *Handler) EnsurePluginCronJobs(ctx context.Context) error {
	if h == nil || h.integrations == nil {
		return nil
	}
	recs, err := h.integrations.ListEnabled(ctx)
	if err != nil {
		return err
	}
	for _, rec := range recs {
		manifest, err := h.integrations.ReadManifest(rec.Name)
		if err != nil {
			continue
		}
		_ = h.ensureIntegrationCronJobs(ctx, manifest, rec.Config)
	}
	return nil
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/internal/tools", h.handleTools)
	mux.HandleFunc("/internal/tools/schemas/openai", h.handleOpenAISchemas)
	mux.HandleFunc("/internal/tools/execute", h.handleExecute)
	mux.HandleFunc("/internal/hooks/", h.handleInternalHooks)
	mux.HandleFunc("/internal/integrations/status", h.handlePluginsStatus)
	for _, path := range []string{"/agent/v1/integrations", "/api/integrations"} {
		mux.HandleFunc(path, h.handlePluginsAPI)
	}
	for _, path := range []string{"/agent/v1/integrations/", "/api/integrations/"} {
		mux.HandleFunc(path, h.handlePluginsAPI)
	}
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
		httputil.WriteError(w, http.StatusBadRequest, "integration name is required")
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
			httputil.WriteError(w, http.StatusConflict, "integration is disabled")
			return
		}
	}

	webhookReq := integrations.WebhookRequest{
		Plugin:   pluginName,
		UserID:   userID,
		DeviceID: strings.TrimSpace(r.Header.Get("X-Aether-Device-ID")),
		Body:     body,
		Header:   r.Header,
	}
	task, err := integrations.DefaultWebhookRegistry().BuildTask(r.Context(), webhookReq)
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
	log.Printf("webhook received: integration=%s title=%s user=%s", pluginName, task.Title, userID)
	httputil.WriteJSON(w, http.StatusAccepted, map[string]any{
		"status":      "acknowledged",
		"integration": pluginName,
		"title":       task.Title,
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
	if h.integrations == nil || h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "integrations status unavailable")
		return
	}
	ctx := r.Context()
	_, _ = h.integrations.Discover(ctx)

	pluginMetas := h.integrations.List()
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
		requiredKeys, _ := h.integrations.RequiredConfigKeys(meta.Name)
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

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"integrations": out, "count": len(out)})
}

func (h *Handler) handlePluginsAPI(w http.ResponseWriter, r *http.Request) {
	if err := agentauth.AuthorizeDirectRequest(r, h.validator); err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	if h.integrations == nil || h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "integrations api unavailable")
		return
	}
	ctx := r.Context()
	_, _ = h.integrations.Discover(ctx)

	if r.URL.Path == "/api/integrations" || r.URL.Path == "/agent/v1/integrations" {
		if r.Method != http.MethodGet {
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		h.listPluginsCompat(w, r)
		return
	}

	path := strings.TrimPrefix(strings.TrimPrefix(r.URL.Path, "/agent/v1/integrations/"), "/api/integrations/")
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
					httputil.WriteError(w, http.StatusNotFound, "integration not found")
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
		manifest, manifestErr := h.integrations.ReadManifest(name)
		if manifestErr != nil {
			httputil.WriteError(w, http.StatusNotFound, "integration not found")
			return
		}
		if err := h.store.SetPluginEnabled(ctx, name, true); err != nil {
			if err == db.ErrNotFound {
				httputil.WriteError(w, http.StatusNotFound, "integration not found")
				return
			}
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if rec, err := h.store.GetPlugin(ctx, name); err == nil {
			_ = h.ensureIntegrationCronJobs(ctx, manifest, rec.Config)
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
				httputil.WriteError(w, http.StatusNotFound, "integration not found")
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
					httputil.WriteError(w, http.StatusNotFound, "integration not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			manifest, err := h.integrations.ReadManifest(name)
			if err != nil {
				httputil.WriteError(w, http.StatusNotFound, "integration not found")
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
					httputil.WriteError(w, http.StatusNotFound, "integration not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			manifest, err := h.integrations.ReadManifest(name)
			if err != nil {
				httputil.WriteError(w, http.StatusNotFound, "integration not found")
				return
			}
			merged := mergeIntegrationConfig(rec.Config, req.Config, secretConfigKeys(manifest))
			if err := h.store.SetPluginConfig(ctx, name, encryptSecretConfig(merged, h.store, secretConfigKeys(manifest))); err != nil {
				if err == db.ErrNotFound {
					httputil.WriteError(w, http.StatusNotFound, "integration not found")
					return
				}
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}

			autoEnabled := false
			authType, _, _ := integrationAuthDetails(manifest)
			required, reqErr := h.integrations.RequiredConfigKeys(name)
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
	case "disconnect":
		if r.Method != http.MethodPost {
			httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		h.handlePluginDisconnect(w, r, name)
		return
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
	plugs := h.integrations.List()
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
		manifest, err := h.integrations.ReadManifest(p.Name)
		if err != nil {
			continue
		}
		rec := byName[p.Name]
		authType, authProvider, fields := integrationAuthDetails(manifest)
		fields = filterEnvBackedOAuthFields(fields, authProvider)
		oauthEnvConfigured := authType == "oauth2" && hasEnvBackedCredentials(authProvider)
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
			"name":                 p.Name,
			"display_name":         p.DisplayName,
			"description":          p.Description,
			"auth_type":            authType,
			"auth_provider":        authProvider,
			"config_fields":        fields,
			"installed":            true,
			"integration_id":       p.Name,
			"enabled":              rec.Enabled,
			"connected":            connected,
			"needs_reconnect":      needsReconnect,
			"oauth_env_configured": oauthEnvConfigured,
		})
	}

	httputil.WriteJSON(w, http.StatusOK, out)
}

func (h *Handler) handlePluginInstallCompat(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	manifest, err := h.integrations.ReadManifest(name)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "integration not found")
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
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"integration_id": name, "status": "installed"})
}

func (h *Handler) handlePluginOAuthStart(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	manifest, err := h.integrations.ReadManifest(name)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "integration not found")
		return
	}
	authType, provider, _ := integrationAuthDetails(manifest)
	if authType != "oauth2" {
		httputil.WriteError(w, http.StatusBadRequest, "integration does not use oauth2")
		return
	}

	if err := h.ensureIntegrationInstalled(ctx, manifest); err != nil {
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
		httputil.WriteError(w, http.StatusBadRequest, "missing oauth client_id in integration config")
		return
	}
	if strings.TrimSpace(clientSecret) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "missing oauth client_secret in integration config")
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
	manifest, err := h.integrations.ReadManifest(name)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "integration not found")
		return
	}
	authType, provider, _ := integrationAuthDetails(manifest)
	if authType != "oauth2" {
		httputil.WriteError(w, http.StatusBadRequest, "integration does not use oauth2")
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
			httputil.WriteError(w, http.StatusNotFound, "integration not found")
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
		httputil.WriteError(w, http.StatusBadRequest, "oauth integration config missing token endpoint or client credentials")
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
		registerEmailWithOrchestrator(email, name)
	}

	if err := h.store.SetPluginConfig(ctx, name, cfg); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := h.store.SetPluginEnabled(ctx, name, true); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	_ = h.ensureIntegrationCronJobs(ctx, manifest, cfg)

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "connected", "integration": name})
}

// oauthAndTokenConfigKeys lists all config keys that store OAuth / token state.
// These are cleared on disconnect to fully revoke the connection.
var oauthAndTokenConfigKeys = []string{
	"access_token", "refresh_token", "oauth_refresh_token",
	"token_type", "scope", "expires_at",
	"oauth_state", "oauth_state_expires_at", "oauth_redirect_uri", "oauth_provider", "oauth_token_url",
	"last_refresh_at", "last_refresh_status", "last_refresh_error",
	"refresh_fail_count", "next_refresh_at",
	"needs_reconnect", "account_email",
	"watch_last_renew_at", "watch_last_renew_status", "watch_last_renew_error",
	"watch_history_id", "watch_expires_at",
}

func (h *Handler) handlePluginDisconnect(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()

	rec, err := h.store.GetPlugin(ctx, name)
	if err != nil {
		if err == db.ErrNotFound {
			httputil.WriteError(w, http.StatusNotFound, "integration not found")
			return
		}
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Remove all OAuth/token keys from config, keep user-provided config like client_id.
	cleaned := cloneConfig(rec.Config)
	for _, key := range oauthAndTokenConfigKeys {
		delete(cleaned, key)
	}
	if err := h.store.SetPluginConfig(ctx, name, cleaned); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Disable the integration.
	if err := h.store.SetPluginEnabled(ctx, name, false); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Cancel any active cron jobs for this integration.
	scope := h.store.ScopeCronModule(integrations.CronModulePlugins)
	jobs, listErr := scope.List(ctx)
	if listErr == nil {
		for _, job := range jobs {
			if !job.Enabled {
				continue
			}
			if jobPayloadHasIntegration(job.PayloadJSON, name) {
				_ = scope.Cancel(ctx, job.ID)
			}
		}
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "disconnected"})
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

func hasEnvBackedCredentials(provider string) bool {
	envClientID, envClientSecret := oauthProviderEnvCredentials(provider)
	return strings.TrimSpace(envClientID) != "" && strings.TrimSpace(envClientSecret) != ""
}

func oauthScopes(manifest integrations.PluginManifest) []string {
	return manifest.Auth.Scopes
}

func oauthRedirectURI(r *http.Request, pluginName string) string {
	host := strings.TrimSpace(r.Header.Get("X-Forwarded-Host"))
	proto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto"))
	if host != "" {
		if proto == "" {
			proto = "https"
		}
		return fmt.Sprintf("%s://%s/integrations/%s/oauth/callback", proto, host, url.PathEscape(pluginName))
	}
	if origin := strings.TrimSpace(r.Header.Get("Origin")); origin != "" {
		return strings.TrimRight(origin, "/") + "/integrations/" + url.PathEscape(pluginName) + "/oauth/callback"
	}
	if ref := strings.TrimSpace(r.Header.Get("Referer")); ref != "" {
		if u, err := url.Parse(ref); err == nil && u.Scheme != "" && u.Host != "" {
			return u.Scheme + "://" + u.Host + "/integrations/" + url.PathEscape(pluginName) + "/oauth/callback"
		}
	}
	return "http://localhost:3000/integrations/" + url.PathEscape(pluginName) + "/oauth/callback"
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

func registerEmailWithOrchestrator(email, pluginName string) {
	orchURL := strings.TrimSpace(os.Getenv("ORCHESTRATOR_URL"))
	if orchURL == "" || strings.TrimSpace(email) == "" {
		return
	}
	userID := strings.TrimSpace(os.Getenv("AETHER_USER_ID"))
	if userID == "" {
		return
	}
	payload, _ := json.Marshal(map[string]string{
		"email":       email,
		"user_id":     userID,
		"plugin_name": pluginName,
	})
	url := strings.TrimRight(orchURL, "/") + "/api/email-mappings"
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			log.Printf("email mapping: failed to create request: %v", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			log.Printf("email mapping: request failed: %v", err)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 300 {
			log.Printf("email mapping: orchestrator responded %d", resp.StatusCode)
		}
	}()
}

func (h *Handler) ensureIntegrationInstalled(ctx context.Context, manifest integrations.PluginManifest) error {
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

func (h *Handler) ensureIntegrationCronJobs(ctx context.Context, manifest integrations.PluginManifest, cfg map[string]string) error {
	scope := h.store.ScopeCronModule(integrations.CronModulePlugins)
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
		if !hasActiveIntegrationCronJob(jobs, manifest.Name, integrations.CronJobTypeRotate) {
			_, _ = scope.ScheduleRecurring(ctx, integrations.CronJobTypeRotate, map[string]any{"integration": manifest.Name}, runAt, rotateInterval, 5)
		}
	}

	renewInterval := int64FromAny(manifest.Webhook["renew_interval"])
	if renewInterval > 0 {
		runAt := time.Now().UTC().Add(time.Duration(renewInterval) * time.Second)
		if !hasActiveIntegrationCronJob(jobs, manifest.Name, integrations.CronJobTypeRenewWatch) {
			_, _ = scope.ScheduleRecurring(ctx, integrations.CronJobTypeRenewWatch, map[string]any{"integration": manifest.Name}, runAt, renewInterval, 5)
		}
	}
	return nil
}

func hasActiveIntegrationCronJob(jobs []db.CronJobRecord, pluginName, jobType string) bool {
	for _, job := range jobs {
		if job.JobType != jobType || !job.Enabled {
			continue
		}
		if !jobPayloadHasIntegration(job.PayloadJSON, pluginName) {
			continue
		}
		if job.Status == db.CronStatusCancelled || job.Status == db.CronStatusDone || job.Status == db.CronStatusFailed {
			continue
		}
		return true
	}
	return false
}

func jobPayloadHasIntegration(payloadJSON, pluginName string) bool {
	var payload map[string]any
	if err := json.Unmarshal([]byte(payloadJSON), &payload); err != nil {
		return false
	}
	v, ok := payload["integration"].(string)
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

func secretConfigKeys(manifest integrations.PluginManifest) map[string]bool {
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

func mergeIntegrationConfig(existing, incoming map[string]string, secretKeys map[string]bool) map[string]string {
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

func integrationAuthDetails(manifest integrations.PluginManifest) (string, string, []map[string]any) {
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
