package dataapi

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
)

type Handler struct {
	store *db.Store
	media *media.Service
}

func New(store *db.Store, mediaSvc *media.Service) *Handler {
	return &Handler{store: store, media: mediaSvc}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/media/ensure-bucket", h.handleEnsureMediaBucket)
	mux.HandleFunc("/api/memory/facts", h.handleMemoryFacts)
	mux.HandleFunc("/api/memory/sessions", h.handleMemorySessions)
	mux.HandleFunc("/api/memory/conversations", h.handleMemoryConversations)
	mux.HandleFunc("/api/memory/memories", h.handleMemories)
	mux.HandleFunc("/api/memory/decisions", h.handleMemoryDecisions)
	mux.HandleFunc("/api/memory/notifications", h.handleMemoryNotifications)
	mux.HandleFunc("/api/memory/export", h.handleMemoryExport)
	mux.HandleFunc("/api/memory/entities", h.handleEntities)
	mux.HandleFunc("/api/memory/entities/", h.handleEntityByID)
	mux.HandleFunc("/v1/agent/jobs", h.handleJobs)
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
	userID := strings.TrimSpace(req.UserID)
	if userID == "" {
		userID = "default"
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

func (h *Handler) memoryUserID(r *http.Request) string {
	if raw := strings.TrimSpace(r.URL.Query().Get("user_id")); raw != "" {
		return raw
	}
	return "default"
}

func (h *Handler) handleMemoryFacts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	facts, err := h.store.GetMemoryFacts(r.Context(), h.memoryUserID(r))
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
	sessions, err := h.store.ListMemorySessions(r.Context(), h.memoryUserID(r), limit)
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
	userID := h.memoryUserID(r)
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
	recs, err := h.store.ListMemories(r.Context(), h.memoryUserID(r), category, limit)
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
	recs, err := h.store.ListDecisions(r.Context(), h.memoryUserID(r), category, activeOnly)
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
	notifications, err := h.store.ListMemoryNotifications(r.Context(), h.memoryUserID(r), limit)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	reliability, err := h.store.GetMemoryReliabilitySnapshot(r.Context(), h.memoryUserID(r))
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
	export, err := h.store.ExportMemorySnapshot(r.Context(), h.memoryUserID(r))
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
	userID := h.memoryUserID(r)
	q := strings.TrimSpace(r.URL.Query().Get("q"))

	var entities []db.EntityRecord
	var err error
	if q != "" {
		entities, err = h.store.SearchEntities(r.Context(), userID, q, limit)
	} else {
		entityType := strings.TrimSpace(r.URL.Query().Get("type"))
		entities, err = h.store.ListEntities(r.Context(), userID, entityType, limit)
	}
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"entities": entities, "count": len(entities)})
}

func (h *Handler) handleEntityByID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	entityID := strings.TrimPrefix(r.URL.Path, "/api/memory/entities/")
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
