package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
)

type Handler struct {
	store *db.Store
	media *media.Service
}

func New(store *db.Store, mediaSvc *media.Service) *Handler {
	return &Handler{store: store, media: mediaSvc}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/agent/tasks", h.handleTasks)
	mux.HandleFunc("/v1/agent/tasks/", h.handleTaskByID)
	mux.HandleFunc("/v1/agent/jobs", h.handleJobs)
	mux.HandleFunc("/api/memory/facts", h.handleMemoryFacts)
	mux.HandleFunc("/api/memory/sessions", h.handleMemorySessions)
	mux.HandleFunc("/api/memory/conversations", h.handleMemoryConversations)
	mux.HandleFunc("/api/memory/memories", h.handleMemories)
	mux.HandleFunc("/api/memory/decisions", h.handleMemoryDecisions)
	mux.HandleFunc("/api/memory/notifications", h.handleMemoryNotifications)
	mux.HandleFunc("/api/memory/export", h.handleMemoryExport)
}

func (h *Handler) memoryUserID(r *http.Request) string {
	if raw := strings.TrimSpace(r.URL.Query().Get("user_id")); raw != "" {
		return raw
	}
	return "default"
}

func (h *Handler) handleMemoryFacts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	facts, err := h.store.GetMemoryFacts(r.Context(), h.memoryUserID(r))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"facts": facts})
}

func (h *Handler) handleMemorySessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
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
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"sessions": sessions})
}

func (h *Handler) handleMemoryConversations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
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
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if h.media != nil && h.media.Enabled() {
		for i := range conversations {
			if hydrated, ok := hydrateConversationMedia(r.Context(), h.media, userID, conversations[i].UserContent); ok {
				conversations[i].UserContent = hydrated
			}
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"conversations": conversations})
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
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
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
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"memories": recs})
}

func (h *Handler) handleMemoryDecisions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	category := strings.TrimSpace(r.URL.Query().Get("category"))
	activeOnly := true
	if raw := strings.TrimSpace(r.URL.Query().Get("active_only")); raw != "" {
		activeOnly = !strings.EqualFold(raw, "false")
	}
	recs, err := h.store.ListDecisions(r.Context(), h.memoryUserID(r), category, activeOnly)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"decisions": recs})
}

func (h *Handler) handleMemoryNotifications(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
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
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	reliability, err := h.store.GetMemoryReliabilitySnapshot(r.Context(), h.memoryUserID(r))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"notifications": notifications, "reliability": reliability})
}

func (h *Handler) handleMemoryExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	export, err := h.store.ExportMemorySnapshot(r.Context(), h.memoryUserID(r))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"export": export})
}

func (h *Handler) handleJobs(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		writeError(w, http.StatusInternalServerError, "agent store unavailable")
		return
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
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
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if len(jobs) > limit {
		jobs = jobs[:limit]
	}
	writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs, "count": len(jobs)})
}

func (h *Handler) handleTasks(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		writeError(w, http.StatusInternalServerError, "agent store unavailable")
		return
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	q := r.URL.Query()
	userID := strings.TrimSpace(q.Get("user_id"))
	if userID == "" {
		userID = "default"
	}
	limit := 50
	if raw := strings.TrimSpace(q.Get("limit")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	status := strings.TrimSpace(q.Get("status"))
	items, err := h.store.ListAgentTasksByUserWithStatus(r.Context(), userID, status, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"tasks": items, "count": len(items)})
}

func (h *Handler) handleTaskByID(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		writeError(w, http.StatusInternalServerError, "agent store unavailable")
		return
	}
	path := strings.TrimPrefix(r.URL.Path, "/v1/agent/tasks/")
	path = strings.Trim(path, "/")
	if path == "" {
		writeError(w, http.StatusNotFound, "task id is required")
		return
	}
	parts := strings.Split(path, "/")
	id := parts[0]
	action := ""
	if len(parts) > 1 {
		action = parts[1]
	}

	switch {
	case action == "cancel" && r.Method == http.MethodPost:
		if err := h.store.RequestCancelAgentTask(r.Context(), id); err != nil {
			status := http.StatusInternalServerError
			if err == db.ErrNotFound {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"task_id": id, "status": "cancel_requested"})
		return
	case action == "resume" && r.Method == http.MethodPost:
		var req struct {
			UserID       string `json:"user_id"`
			Message      string `json:"message"`
			Decision     string `json:"decision"`
			Reason       string `json:"reason"`
			Instructions string `json:"instructions"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if strings.TrimSpace(req.UserID) == "" {
			req.UserID = "default"
		}
		msg := strings.TrimSpace(req.Message)
		if msg == "" {
			decision := strings.TrimSpace(req.Decision)
			if decision == "" {
				decision = "approved"
			}
			msg = "Human decision: " + decision
			if strings.TrimSpace(req.Reason) != "" {
				msg += "\nReason: " + strings.TrimSpace(req.Reason)
			}
			if strings.TrimSpace(req.Instructions) != "" {
				msg += "\nInstructions: " + strings.TrimSpace(req.Instructions)
			}
		}
		task, err := h.store.ResumeAgentTask(r.Context(), id, req.UserID, msg)
		if err != nil {
			status := http.StatusInternalServerError
			if err == db.ErrNotFound {
				status = http.StatusNotFound
			} else if strings.Contains(strings.ToLower(err.Error()), "not waiting") {
				status = http.StatusBadRequest
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"task_id": id, "status": task.Status})
		return
	case action == "approve" && r.Method == http.MethodPost:
		var req struct {
			UserID       string `json:"user_id"`
			Decision     string `json:"decision"`
			Reason       string `json:"reason"`
			Instructions string `json:"instructions"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if strings.TrimSpace(req.UserID) == "" {
			req.UserID = "default"
		}
		decision := strings.TrimSpace(req.Decision)
		if decision == "" {
			decision = "approved"
		}
		msg := strings.TrimSpace(req.Instructions)
		if msg == "" {
			msg = strings.TrimSpace(req.Reason)
		}
		if msg == "" {
			msg = "Approved. Continue with best effort."
		}
		task, err := h.store.ResumeAgentTask(r.Context(), id, req.UserID, msg)
		if err != nil {
			status := http.StatusInternalServerError
			if err == db.ErrNotFound {
				status = http.StatusNotFound
			} else if strings.Contains(strings.ToLower(err.Error()), "not waiting") {
				status = http.StatusBadRequest
			}
			writeError(w, status, err.Error())
			return
		}
		_ = h.store.AppendAgentTaskEvent(r.Context(), id, "decision", map[string]any{"decision": decision, "reason": req.Reason, "instructions": req.Instructions, "approved": true})
		writeJSON(w, http.StatusOK, map[string]any{"task_id": id, "status": task.Status, "decision": decision})
		return
	case action == "reject" && r.Method == http.MethodPost:
		var req struct {
			UserID     string `json:"user_id"`
			Reason     string `json:"reason"`
			NextAction string `json:"next_action"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if strings.TrimSpace(req.UserID) == "" {
			req.UserID = "default"
		}
		reason := strings.TrimSpace(req.Reason)
		if reason == "" {
			reason = "Rejected"
		}
		nextAction := strings.TrimSpace(req.NextAction)
		if nextAction == "" {
			nextAction = "Stop and wait for new instructions."
		}
		msg := "Rejected by human. Reason: " + reason + ". Next action: " + nextAction
		task, err := h.store.RejectAgentTask(r.Context(), id, req.UserID, msg)
		if err != nil {
			status := http.StatusInternalServerError
			if err == db.ErrNotFound {
				status = http.StatusNotFound
			} else if strings.Contains(strings.ToLower(err.Error()), "not waiting") {
				status = http.StatusBadRequest
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"task_id": id, "status": task.Status, "decision": "rejected"})
		return
	case action == "events" && r.Method == http.MethodGet:
		limit := 200
		if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
			if n, err := strconv.Atoi(raw); err == nil && n > 0 {
				limit = n
			}
		}
		events, err := h.store.ListAgentTaskEvents(r.Context(), id, limit)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"task_id": id, "events": events, "count": len(events)})
		return
	case action == "" && r.Method == http.MethodGet:
		task, err := h.store.GetAgentTask(r.Context(), id)
		if err != nil {
			status := http.StatusInternalServerError
			if err == db.ErrNotFound {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"task": task})
		return
	default:
		writeError(w, http.StatusMethodNotAllowed, "unsupported task endpoint")
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
