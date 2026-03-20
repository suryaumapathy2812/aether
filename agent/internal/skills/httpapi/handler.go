package httpapi

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
)

type Handler struct {
	manager *skills.Manager
	store   *db.Store
}

type Options struct {
	Manager *skills.Manager
	Store   *db.Store
}

func New(opts Options) *Handler {
	return &Handler{manager: opts.Manager, store: opts.Store}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/skills/marketplace/search", h.handleMarketplaceSearch)
	mux.HandleFunc("/api/skills/installed", h.handleInstalled)
	mux.HandleFunc("/api/skills/install", h.handleInstall)
	mux.HandleFunc("/api/skills/remove", h.handleRemove)
}

func (h *Handler) handleMarketplaceSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.manager == nil {
		writeError(w, http.StatusInternalServerError, "skills manager unavailable")
		return
	}
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" {
		writeError(w, http.StatusBadRequest, "query is required")
		return
	}
	limit := 10
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil {
			limit = n
		}
	}
	result, err := h.manager.SearchMarketplace(r.Context(), query, limit)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (h *Handler) handleInstalled(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.manager == nil {
		writeError(w, http.StatusInternalServerError, "skills manager unavailable")
		return
	}
	items := make([]skills.SkillMeta, 0)
	for _, item := range h.manager.List() {
		if item.Source == skills.SourceExternal {
			items = append(items, item)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"skills": items, "count": len(items)})
}

func (h *Handler) handleInstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.manager == nil {
		writeError(w, http.StatusInternalServerError, "skills manager unavailable")
		return
	}
	var req struct {
		Source    string `json:"source"`
		SkillName string `json:"skill_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	source := strings.TrimSpace(req.Source)
	if source == "" {
		writeError(w, http.StatusBadRequest, "source is required")
		return
	}
	skillName := strings.TrimSpace(req.SkillName)
	if skillName != "" && !strings.Contains(source, "@") {
		source = source + "@" + skillName
	}
	result, err := h.manager.InstallFromSource(r.Context(), source)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.store != nil {
		_ = h.store.UpsertSkill(r.Context(), db.SkillRecord{
			Name:        result.Installed.Name,
			Description: result.Installed.Description,
			Location:    result.Installed.Location,
			Source:      string(result.Installed.Source),
		})
	}
	writeJSON(w, http.StatusOK, result)
}

func (h *Handler) handleRemove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.manager == nil {
		writeError(w, http.StatusInternalServerError, "skills manager unavailable")
		return
	}
	var req struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	if err := h.manager.Remove(name); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.store != nil {
		_ = h.store.DeleteSkill(r.Context(), name)
	}
	writeJSON(w, http.StatusOK, map[string]any{"removed": true, "name": name})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}
