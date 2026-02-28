package httpapi

import (
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/suryaumapathy/core-ai/agent/internal/db"
	"github.com/suryaumapathy/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

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

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/internal/tools", h.handleTools)
	mux.HandleFunc("/internal/tools/schemas/openai", h.handleOpenAISchemas)
	mux.HandleFunc("/internal/tools/execute", h.handleExecute)
	mux.HandleFunc("/internal/plugins/status", h.handlePluginsStatus)
	mux.HandleFunc("/api/plugins", h.handlePluginsAPI)
	mux.HandleFunc("/api/plugins/", h.handlePluginsAPI)
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (h *Handler) handleTools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.registry == nil {
		writeError(w, http.StatusInternalServerError, "tool registry unavailable")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"tools": h.registry.Definitions()})
}

func (h *Handler) handleOpenAISchemas(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.registry == nil {
		writeError(w, http.StatusInternalServerError, "tool registry unavailable")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"tools": h.registry.OpenAISchemas()})
}

func (h *Handler) handleExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.orchestrator == nil {
		writeError(w, http.StatusInternalServerError, "tool orchestrator unavailable")
		return
	}
	var req struct {
		Name   string         `json:"name"`
		Args   map[string]any `json:"args"`
		CallID string         `json:"call_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	result := h.orchestrator.Execute(r.Context(), req.Name, req.Args, req.CallID)
	status := http.StatusOK
	if result.Error {
		status = http.StatusBadRequest
	}
	writeJSON(w, status, map[string]any{"result": result})
}

func (h *Handler) handlePluginsStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.plugins == nil || h.store == nil {
		writeError(w, http.StatusInternalServerError, "plugins status unavailable")
		return
	}
	ctx := r.Context()
	_, _ = h.plugins.Discover(ctx)

	pluginMetas := h.plugins.List()
	storeRecords, err := h.store.ListPlugins(ctx)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
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

	writeJSON(w, http.StatusOK, map[string]any{"plugins": out, "count": len(out)})
}

func (h *Handler) handlePluginsAPI(w http.ResponseWriter, r *http.Request) {
	if h.plugins == nil || h.store == nil {
		writeError(w, http.StatusInternalServerError, "plugins api unavailable")
		return
	}
	ctx := r.Context()
	_, _ = h.plugins.Discover(ctx)

	if r.URL.Path == "/api/plugins" {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		h.listPluginsCompat(w, r)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/plugins/")
	path = strings.Trim(path, "/")
	parts := strings.Split(path, "/")
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		writeError(w, http.StatusNotFound, "not found")
		return
	}
	name := parts[0]

	if len(parts) == 1 {
		if r.Method == http.MethodDelete {
			if err := h.store.SetPluginEnabled(ctx, name, false); err != nil {
				if err == db.ErrNotFound {
					writeError(w, http.StatusNotFound, "plugin not found")
					return
				}
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"status": "disabled"})
			return
		}
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	action := parts[1]
	switch action {
	case "install":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		h.handlePluginInstallCompat(w, r, name)
		return
	case "enable":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if err := h.store.SetPluginEnabled(ctx, name, true); err != nil {
			if err == db.ErrNotFound {
				writeError(w, http.StatusNotFound, "plugin not found")
				return
			}
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "enabled"})
		return
	case "disable":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if err := h.store.SetPluginEnabled(ctx, name, false); err != nil {
			if err == db.ErrNotFound {
				writeError(w, http.StatusNotFound, "plugin not found")
				return
			}
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "disabled"})
		return
	case "config":
		switch r.Method {
		case http.MethodGet:
			rec, err := h.store.GetPlugin(ctx, name)
			if err != nil {
				if err == db.ErrNotFound {
					writeError(w, http.StatusNotFound, "plugin not found")
					return
				}
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}
			writeJSON(w, http.StatusOK, rec.Config)
			return
		case http.MethodPost:
			var req struct {
				Config map[string]string `json:"config"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeError(w, http.StatusBadRequest, "invalid json body")
				return
			}
			if req.Config == nil {
				req.Config = map[string]string{}
			}
			if err := h.store.SetPluginConfig(ctx, name, req.Config); err != nil {
				if err == db.ErrNotFound {
					writeError(w, http.StatusNotFound, "plugin not found")
					return
				}
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}

			autoEnabled := false
			required, reqErr := h.plugins.RequiredConfigKeys(name)
			if reqErr == nil && len(required) > 0 {
				complete := true
				for _, key := range required {
					if strings.TrimSpace(req.Config[key]) == "" {
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
			writeJSON(w, http.StatusOK, map[string]any{"status": "updated", "auto_enabled": autoEnabled})
			return
		default:
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
	case "oauth":
		if len(parts) == 3 && parts[2] == "start" && r.Method == http.MethodGet {
			writeJSON(w, http.StatusNotImplemented, map[string]any{
				"detail": "OAuth setup is not yet available in the Go runtime. Configure API-key plugins from the dashboard.",
			})
			return
		}
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	default:
		writeError(w, http.StatusNotFound, "not found")
		return
	}
}

func (h *Handler) listPluginsCompat(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	plugs := h.plugins.List()
	recs, err := h.store.ListPlugins(ctx)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
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

	writeJSON(w, http.StatusOK, out)
}

func (h *Handler) handlePluginInstallCompat(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	manifest, err := h.plugins.ReadManifest(name)
	if err != nil {
		writeError(w, http.StatusNotFound, "plugin not found")
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
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"plugin_id": name, "status": "installed"})
}

func pluginAuthDetails(manifest plugins.PluginManifest) (string, string, []map[string]any) {
	authType, _ := manifest.Auth["type"].(string)
	authProvider, _ := manifest.Auth["provider"].(string)
	if strings.TrimSpace(authType) == "" {
		authType = "none"
	}
	rawFields, _ := manifest.Auth["config_fields"].([]any)
	fields := make([]map[string]any, 0, len(rawFields))
	for _, raw := range rawFields {
		entry, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		field := map[string]any{}
		if v, ok := entry["key"].(string); ok {
			field["key"] = v
		}
		if v, ok := entry["label"].(string); ok {
			field["label"] = v
		}
		if v, ok := entry["type"].(string); ok {
			field["type"] = v
		}
		required := false
		switch v := entry["required"].(type) {
		case bool:
			required = v
		case string:
			required = strings.EqualFold(strings.TrimSpace(v), "true")
		}
		field["required"] = required
		if v, ok := entry["description"].(string); ok {
			field["description"] = v
		}
		fields = append(fields, field)
	}
	return authType, authProvider, fields
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}
