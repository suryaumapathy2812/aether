package httpapi

import (
	"net/http"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
	"github.com/suryaumapathy2812/core-ai/agent/internal/integrations"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
	"github.com/suryaumapathy2812/core-ai/agent/internal/updater"
)

type Handler struct {
	updater      *updater.Updater
	builder      *llm.ContextBuilder
	skills       *skills.Manager
	integrations *integrations.Manager
	adminToken   string
}

type Options struct {
	Updater      *updater.Updater
	Builder      *llm.ContextBuilder
	Skills       *skills.Manager
	Integrations *integrations.Manager
	AdminToken   string
}

func New(opts Options) *Handler {
	return &Handler{updater: opts.Updater, builder: opts.Builder, skills: opts.Skills, integrations: opts.Integrations, adminToken: opts.AdminToken}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/admin/version", h.handleVersion)
	mux.HandleFunc("/admin/update/check", h.handleCheckUpdate)
	mux.HandleFunc("/admin/update/apply", h.handleApplyUpdate)
	mux.HandleFunc("/admin/reload", h.handleReload)
}

func (h *Handler) handleVersion(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := h.requireAdminAuth(r); err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"version":    buildinfo.Version,
		"commit":     buildinfo.Commit,
		"build_time": buildinfo.BuildTime,
	})
}

func (h *Handler) handleCheckUpdate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := h.requireAdminAuth(r); err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	if h.updater == nil {
		httputil.WriteError(w, http.StatusServiceUnavailable, "updater unavailable")
		return
	}
	release, available, err := h.updater.Check(r.Context())
	if err != nil {
		httputil.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"current_version": h.updater.CurrentVersion(),
		"available":       available,
		"latest":          release,
	})
}

func (h *Handler) handleApplyUpdate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := h.requireAdminAuth(r); err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	if h.updater == nil {
		httputil.WriteError(w, http.StatusServiceUnavailable, "updater unavailable")
		return
	}

	release, err := h.updater.ApplyLatest(r.Context())
	if err != nil {
		httputil.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}
	if h.builder != nil {
		h.builder.ReloadSystemPrompt()
	}
	if h.skills != nil {
		_, _ = h.skills.Discover(r.Context())
	}
	if h.integrations != nil {
		_, _ = h.integrations.Discover(r.Context())
	}

	httputil.WriteJSON(w, http.StatusAccepted, map[string]any{
		"status":            "updated",
		"version":           release.Version,
		"restart_scheduled": true,
	})

	go func() {
		time.Sleep(500 * time.Millisecond)
		_ = h.updater.RestartSelf()
	}()
}

func (h *Handler) handleReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := h.requireAdminAuth(r); err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}

	promptLen := 0
	if h.builder != nil {
		promptLen = len(h.builder.ReloadSystemPrompt())
	}
	skillCount := 0
	if h.skills != nil {
		if discovered, err := h.skills.Discover(r.Context()); err == nil {
			skillCount = discovered
		}
	}
	pluginCount := 0
	if h.integrations != nil {
		if discovered, err := h.integrations.Discover(r.Context()); err == nil {
			pluginCount = discovered
		}
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"status":        "reloaded",
		"prompt_length": promptLen,
		"skills":        skillCount,
		"plugins":       pluginCount,
	})
}

func (h *Handler) requireAdminAuth(r *http.Request) error {
	secret := strings.TrimSpace(h.adminToken)
	if secret == "" {
		return nil
	}
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if !strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return errString("missing admin authorization")
	}
	token := strings.TrimSpace(auth[7:])
	if token == "" || token != secret {
		return errString("invalid admin token")
	}
	return nil
}

type errString string

func (e errString) Error() string { return string(e) }
