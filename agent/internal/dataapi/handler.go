package dataapi

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
)

type Handler struct {
	store     *db.Store
	media     *media.Service
	validator *agentauth.Validator
}

type Options struct {
	Store     *db.Store
	Media     *media.Service
	Validator *agentauth.Validator
}

func trimDataAPIPathPrefix(path string, prefixes ...string) string {
	for _, prefix := range prefixes {
		if strings.HasPrefix(path, prefix) {
			return strings.TrimPrefix(path, prefix)
		}
	}
	return path
}

func New(opts Options) *Handler {
	return &Handler{store: opts.Store, media: opts.Media, validator: opts.Validator}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	for _, path := range []string{"/agent/v1/media/ensure-bucket", "/api/media/ensure-bucket"} {
		mux.HandleFunc(path, h.handleEnsureMediaBucket)
	}
	for _, path := range []string{"/agent/v1/memory/facts", "/api/memory/facts"} {
		mux.HandleFunc(path, h.handleMemoryFacts)
	}
	for _, path := range []string{"/agent/v1/memory/sessions", "/api/memory/sessions"} {
		mux.HandleFunc(path, h.handleMemorySessions)
	}
	for _, path := range []string{"/agent/v1/memory/conversations", "/api/memory/conversations"} {
		mux.HandleFunc(path, h.handleMemoryConversations)
	}
	for _, path := range []string{"/agent/v1/memory/memories", "/api/memory/memories"} {
		mux.HandleFunc(path, h.handleMemories)
	}
	for _, path := range []string{"/agent/v1/memory/decisions", "/api/memory/decisions"} {
		mux.HandleFunc(path, h.handleMemoryDecisions)
	}
	for _, path := range []string{"/agent/v1/memory/notifications", "/api/memory/notifications"} {
		mux.HandleFunc(path, h.handleMemoryNotifications)
	}
	for _, path := range []string{"/agent/v1/memory/export", "/api/memory/export"} {
		mux.HandleFunc(path, h.handleMemoryExport)
	}
	for _, path := range []string{"/agent/v1/memory/entities", "/api/memory/entities"} {
		mux.HandleFunc(path, h.handleEntities)
	}
	for _, path := range []string{"/agent/v1/memory/entities/", "/api/memory/entities/"} {
		mux.HandleFunc(path, h.handleEntityByID)
	}
	for _, path := range []string{"/agent/v1/agent/jobs", "/v1/agent/jobs"} {
		mux.HandleFunc(path, h.handleJobs)
	}
}

func (h *Handler) handleEnsureMediaBucket(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.media == nil || !h.media.Enabled() {
		httputil.WriteError(w, http.StatusBadRequest, "media storage is not configured")
		return
	}
	var req struct {
		UserID string `json:"user_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	userID, err := agentauth.ResolveDirectUserID(r, h.validator, req.UserID)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	bucket := h.media.BucketForUser(userID)
	if strings.TrimSpace(bucket) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "media bucket is not configured")
		return
	}
	if err := h.media.EnsureBucket(r.Context(), bucket); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"success": true, "bucket": bucket})
}

func (h *Handler) memoryUserID(r *http.Request) (string, error) {
	return agentauth.ResolveDirectUserID(r, h.validator, r.URL.Query().Get("user_id"))
}

func (h *Handler) handleMemoryFacts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	facts, err := h.store.GetMemoryFacts(r.Context(), userID)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"facts": facts})
}

func (h *Handler) handleMemorySessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	limit := 100
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	sessions, err := h.store.ListMemorySessions(r.Context(), userID, limit)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"sessions": sessions})
}

func (h *Handler) handleMemoryConversations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	limit := 20
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	conversations, err := h.store.ListMemoryConversations(r.Context(), userID, limit)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if h.media != nil && h.media.Enabled() {
		for i := range conversations {
			if hydrated, ok := hydrateConversationMedia(r.Context(), h.media, userID, conversations[i].UserContent); ok {
				conversations[i].UserContent = hydrated
			}
		}
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"conversations": conversations})
}

func hydrateConversationMedia(ctx context.Context, mediaSvc *media.Service, userID string, content any) (any, bool) {
	parts, ok := content.([]any)
	if !ok {
		return content, false
	}
	changed := false
	out := make([]any, 0, len(parts))
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			out = append(out, raw)
			continue
		}
		typ, _ := part["type"].(string)
		switch typ {
		case "image_ref":
			mediaObj, _ := part["media"].(map[string]any)
			key, _ := mediaObj["key"].(string)
			bucket, _ := mediaObj["bucket"].(string)
			bucket = firstNonEmpty(strings.TrimSpace(bucket), mediaSvc.BucketForUser(userID))
			if strings.TrimSpace(key) == "" {
				out = append(out, raw)
				continue
			}
			url, err := mediaSvc.PresignGet(ctx, bucket, strings.TrimSpace(key))
			if err != nil {
				out = append(out, raw)
				continue
			}
			mediaObj["bucket"] = bucket
			mediaObj["url"] = url
			changed = true
			out = append(out, map[string]any{"type": "image_ref", "media": mediaObj})
		case "audio_ref":
			mediaObj, _ := part["media"].(map[string]any)
			key, _ := mediaObj["key"].(string)
			bucket, _ := mediaObj["bucket"].(string)
			bucket = firstNonEmpty(strings.TrimSpace(bucket), mediaSvc.BucketForUser(userID))
			if strings.TrimSpace(key) == "" {
				out = append(out, raw)
				continue
			}
			url, err := mediaSvc.PresignGet(ctx, bucket, strings.TrimSpace(key))
			if err != nil {
				out = append(out, raw)
				continue
			}
			mediaObj["bucket"] = bucket
			mediaObj["url"] = url
			changed = true
			out = append(out, map[string]any{"type": "audio_ref", "media": mediaObj})
		default:
			out = append(out, raw)
		}
	}
	if !changed {
		return content, false
	}
	return out, true
}

func (h *Handler) handleMemories(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	limit := 100
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	category := strings.TrimSpace(r.URL.Query().Get("category"))
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	recs, err := h.store.ListMemories(r.Context(), userID, category, limit)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"memories": recs})
}

func (h *Handler) handleMemoryDecisions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	category := strings.TrimSpace(r.URL.Query().Get("category"))
	activeOnly := true
	if raw := strings.TrimSpace(r.URL.Query().Get("active_only")); raw != "" {
		activeOnly = !strings.EqualFold(raw, "false")
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	recs, err := h.store.ListDecisions(r.Context(), userID, category, activeOnly)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"decisions": recs})
}

func (h *Handler) handleMemoryNotifications(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	limit := 200
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	notifications, err := h.store.ListMemoryNotifications(r.Context(), userID, limit)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	reliability, err := h.store.GetMemoryReliabilitySnapshot(r.Context(), userID)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"notifications": notifications, "reliability": reliability})
}

func (h *Handler) handleMemoryExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	export, err := h.store.ExportMemorySnapshot(r.Context(), userID)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"export": export})
}

func (h *Handler) handleEntities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	limit := 50
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	userID, err := h.memoryUserID(r)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	q := strings.TrimSpace(r.URL.Query().Get("q"))

	var entities []db.EntityRecord
	var entityErr error
	if q != "" {
		entities, entityErr = h.store.SearchEntities(r.Context(), userID, q, limit)
	} else {
		entityType := strings.TrimSpace(r.URL.Query().Get("type"))
		entities, entityErr = h.store.ListEntities(r.Context(), userID, entityType, limit)
	}
	if entityErr != nil {
		httputil.WriteError(w, http.StatusInternalServerError, entityErr.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"entities": entities, "count": len(entities)})
}

func (h *Handler) handleEntityByID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	entityID := trimDataAPIPathPrefix(r.URL.Path, "/agent/v1/memory/entities/", "/api/memory/entities/")
	entityID = strings.Trim(entityID, "/")
	if entityID == "" {
		httputil.WriteError(w, http.StatusNotFound, "entity id is required")
		return
	}

	ctx := r.Context()

	entity, err := h.store.GetEntity(ctx, entityID)
	if err != nil {
		status := http.StatusInternalServerError
		if err == db.ErrNotFound {
			status = http.StatusNotFound
		}
		httputil.WriteError(w, status, err.Error())
		return
	}

	observations, err := h.store.ListEntityObservations(ctx, entityID, 50)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	interactions, err := h.store.ListEntityInteractions(ctx, entityID, 50)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	relations, err := h.store.ListEntityRelations(ctx, entityID)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"entity":       entity,
		"observations": observations,
		"interactions": interactions,
		"relations":    relations,
	})
}

func (h *Handler) handleJobs(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "store unavailable")
		return
	}
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	module := strings.TrimSpace(r.URL.Query().Get("module"))
	limit := 50
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	jobs := []db.CronJobRecord{}
	var err error
	if module != "" {
		jobs, err = h.store.ListCronJobsByModule(r.Context(), module)
	} else {
		jobs, err = h.store.ListCronJobs(r.Context())
	}
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if len(jobs) > limit {
		jobs = jobs[:limit]
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"jobs": jobs, "count": len(jobs)})
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
