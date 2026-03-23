package httpapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
	agentcfg "github.com/suryaumapathy2812/core-ai/agent/internal/config"
	"github.com/suryaumapathy2812/core-ai/agent/internal/conversation"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type Handler struct {
	runtime   *conversation.Runtime
	builder   *llm.ContextBuilder
	memory    *memory.Service
	media     *media.Service
	store     *db.Store
	limits    agentcfg.MediaLimitsConfig
	notify    func(userID, eventType string, payload map[string]any)
	status    *sessionStatusTracker
	questions *questionManager
	validator *agentauth.Validator
}

type Options struct {
	Runtime   *conversation.Runtime
	Builder   *llm.ContextBuilder
	Memory    *memory.Service
	Media     *media.Service
	Store     *db.Store
	Limits    agentcfg.MediaLimitsConfig
	Notify    func(userID, eventType string, payload map[string]any)
	Validator *agentauth.Validator
}

const (
	agentPublicPrefix = "/agent/v1"
	legacyV1Prefix    = "/v1"
)

func trimConversationPathPrefix(path string, prefixes ...string) string {
	for _, prefix := range prefixes {
		if strings.HasPrefix(path, prefix) {
			return strings.TrimPrefix(path, prefix)
		}
	}
	return path
}

func New(opts Options) *Handler {
	return &Handler{
		runtime:   opts.Runtime,
		builder:   opts.Builder,
		memory:    opts.Memory,
		media:     opts.Media,
		store:     opts.Store,
		limits:    opts.Limits,
		notify:    opts.Notify,
		status:    newSessionStatusTracker(),
		questions: newQuestionManager(),
		validator: opts.Validator,
	}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	for _, path := range []string{agentPublicPrefix + "/conversations/turn", legacyV1Prefix + "/conversations/turn"} {
		mux.HandleFunc(path, h.handleTurn)
	}
	for _, path := range []string{agentPublicPrefix + "/sessions/status", legacyV1Prefix + "/sessions/status"} {
		mux.HandleFunc(path, h.handleSessionStatus)
	}
	for _, path := range []string{agentPublicPrefix + "/sessions", legacyV1Prefix + "/sessions"} {
		mux.HandleFunc(path, h.handleSessions)
	}
	for _, path := range []string{agentPublicPrefix + "/sessions/", legacyV1Prefix + "/sessions/"} {
		mux.HandleFunc(path, h.handleSessionByID)
	}
	for _, path := range []string{agentPublicPrefix + "/questions/", legacyV1Prefix + "/questions/"} {
		mux.HandleFunc(path, h.handleQuestions)
	}
	for _, path := range []string{agentPublicPrefix + "/questions", legacyV1Prefix + "/questions"} {
		mux.HandleFunc(path, h.handleQuestionsList)
	}
}

// QuestionManager returns the handler's question manager so it can be
// wired into the tools.ExecContext as a QuestionAsker implementation.
func (h *Handler) QuestionManager() *questionManager {
	return h.questions
}

type sessionStatusTracker struct {
	mu    sync.RWMutex
	state map[string]string
}

func newSessionStatusTracker() *sessionStatusTracker {
	return &sessionStatusTracker{state: map[string]string{}}
}

func (s *sessionStatusTracker) set(sessionID, status string) {
	if s == nil || strings.TrimSpace(sessionID) == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[sessionID] = status
}

func (s *sessionStatusTracker) get(sessionID string) string {
	if s == nil || strings.TrimSpace(sessionID) == "" {
		return "idle"
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if status, ok := s.state[sessionID]; ok && strings.TrimSpace(status) != "" {
		return status
	}
	return "idle"
}

func (h *Handler) emit(userID, eventType string, payload map[string]any) {
	if h == nil || h.notify == nil || strings.TrimSpace(userID) == "" || strings.TrimSpace(eventType) == "" {
		return
	}
	h.notify(userID, eventType, payload)
}

func (h *Handler) handleTurn(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.runtime == nil || h.builder == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "conversation runtime unavailable")
		return
	}
	var req struct {
		Messages    []map[string]any `json:"messages"`
		Temperature *float64         `json:"temperature"`
		MaxTokens   *int             `json:"max_tokens"`
		User        string           `json:"user"`
		UserID      string           `json:"user_id"`
		Session     string           `json:"session"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if len(req.Messages) == 0 {
		httputil.WriteError(w, http.StatusBadRequest, "messages is required")
		return
	}

	userID, ok := h.resolveUserID(r, strings.TrimSpace(req.User), strings.TrimSpace(req.UserID))
	if !ok {
		httputil.WriteError(w, http.StatusUnauthorized, "invalid token")
		return
	}
	sessionID := strings.TrimSpace(req.Session)

	// Session management: validate existing or create new.
	newSession := false
	if sessionID != "" && h.store != nil {
		if _, err := h.store.GetChatSession(r.Context(), sessionID); err != nil {
			// Session ID provided but doesn't exist — create it.
			title := llm.TruncateTitle(fmt.Sprintf("%v", llm.LatestUserMessageContent(req.Messages)), 60)
			if sess, err := h.store.CreateChatSession(r.Context(), userID, title); err == nil {
				sessionID = sess.ID
				newSession = true
			}
			// If creation fails, proceed anyway — messages still work without session row.
		}
	} else if h.store != nil {
		// No session ID — auto-create one.
		title := llm.TruncateTitle(fmt.Sprintf("%v", llm.LatestUserMessageContent(req.Messages)), 60)
		sess, err := h.store.CreateChatSession(r.Context(), userID, title)
		if err == nil {
			sessionID = sess.ID
			newSession = true
		} else {
			sessionID = userID // fallback
		}
	} else {
		sessionID = userID
	}

	latestUser, err := latestUserTurnMessage(req.Messages)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	resolvedTurn, err := h.resolveMediaRefs(r.Context(), userID, []map[string]any{latestUser})
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(resolvedTurn) == 0 {
		httputil.WriteError(w, http.StatusBadRequest, "latest user message is required")
		return
	}
	latestUser = resolvedTurn[0]

	history := h.loadConversationHistory(r.Context(), userID, sessionID)
	messages := append(history, latestUser)
	if h.store != nil {
		_ = h.store.AppendChatMessage(r.Context(), userID, sessionID, latestUser)
		h.emit(userID, "message.updated", map[string]any{"sessionID": sessionID, "role": "user", "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
	}

	policy := map[string]any{}
	if h.store != nil {
		if modelPref, err := h.store.GetUserPreference(r.Context(), userID, "model"); err == nil && strings.TrimSpace(modelPref) != "" {
			policy["model"] = strings.TrimSpace(modelPref)
		}
	}
	if req.MaxTokens != nil {
		policy["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		policy["temperature"] = *req.Temperature
	}

	env := h.builder.Build(messages, policy, userID, sessionID)

	rCtx := tools.WithTaskRuntimeContext(r.Context(), tools.TaskRuntimeContext{UserID: userID})
	r = r.WithContext(rCtx)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		httputil.WriteError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}

	// Touch session timestamp + emit session ID in start event metadata.
	if h.store != nil && sessionID != "" && sessionID != userID {
		_ = h.store.TouchChatSession(r.Context(), sessionID)
	}
	h.status.set(sessionID, "streaming")
	h.emit(userID, "session.status", map[string]any{"sessionID": sessionID, "status": map[string]any{"type": "busy"}, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
	if newSession {
		h.emit(userID, "session.created", map[string]any{"sessionID": sessionID, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
	}
	_ = newSession // used for auto-title below

	answerParts := []string{}
	pendingToolCalls := []map[string]any{}
	pendingToolCallIDs := map[string]struct{}{}
	assistantFlushedForToolBatch := false

	for ev := range h.runtime.Run(r.Context(), env, conversation.RunOptions{}) {
		// Write every event as a typed SSE chunk.
		chunk := map[string]any{"type": string(ev.EventType)}
		for k, v := range ev.Payload {
			chunk[k] = v
		}
		// Inject sessionId into start event so the client knows which session this is.
		if ev.EventType == conversation.EventStart {
			chunk["messageMetadata"] = map[string]any{"sessionId": sessionID}
		}
		httputil.WriteSSE(w, chunk)
		flusher.Flush()

		// Side effects: persist to DB and collect answer text.
		switch ev.EventType {
		case conversation.EventTextDelta:
			delta, _ := ev.Payload["delta"].(string)
			answerParts = append(answerParts, delta)
			h.emit(userID, "message.part.delta", map[string]any{"sessionID": sessionID, "messageID": "assistant-current", "partID": "text", "delta": delta, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})

		case conversation.EventToolInputAvailable:
			name, _ := ev.Payload["toolName"].(string)
			callID, _ := ev.Payload["toolCallId"].(string)
			input := ev.Payload["input"]
			argsJSON, _ := json.Marshal(input)
			pendingToolCalls = append(pendingToolCalls, map[string]any{
				"id":   callID,
				"type": "function",
				"function": map[string]any{
					"name":      name,
					"arguments": string(argsJSON),
				},
			})
			if strings.TrimSpace(callID) != "" {
				pendingToolCallIDs[callID] = struct{}{}
			}
			assistantFlushedForToolBatch = false

		case conversation.EventToolOutputAvailable, conversation.EventType("tool-output-error"):
			if h.store != nil && !assistantFlushedForToolBatch && len(pendingToolCalls) > 0 {
				_ = h.store.AppendChatMessage(r.Context(), userID, sessionID, map[string]any{
					"role":       "assistant",
					"tool_calls": pendingToolCalls,
				})
				assistantFlushedForToolBatch = true
			}
			if h.store != nil {
				callID, _ := ev.Payload["toolCallId"].(string)
				output, _ := ev.Payload["output"].(string)
				errorText, _ := ev.Payload["errorText"].(string)
				content := output
				if ev.EventType == conversation.EventType("tool-output-error") {
					content = "[tool_error] " + errorText
				}
				if strings.TrimSpace(content) != "" {
					_ = h.store.AppendChatMessage(r.Context(), userID, sessionID, map[string]any{
						"role":         "tool",
						"tool_call_id": callID,
						"content":      content,
					})
				}
				if strings.TrimSpace(callID) != "" {
					delete(pendingToolCallIDs, callID)
				}
				if len(pendingToolCallIDs) == 0 {
					pendingToolCalls = nil
					assistantFlushedForToolBatch = false
				}
			}
		}
	}

	if h.memory != nil {
		answer := strings.TrimSpace(strings.Join(answerParts, ""))
		if answer != "" {
			h.memory.RecordConversation(context.Background(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(req.Messages), llm.LatestUserMessageContent(req.Messages), answer)
		}
	}
	answer := strings.TrimSpace(strings.Join(answerParts, ""))
	if answer != "" && h.store != nil {
		_ = h.store.AppendChatMessage(r.Context(), userID, sessionID, map[string]any{"role": "assistant", "content": answer})
		h.emit(userID, "message.updated", map[string]any{"sessionID": sessionID, "role": "assistant", "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
	}
	h.status.set(sessionID, "idle")
	h.emit(userID, "session.status", map[string]any{"sessionID": sessionID, "status": map[string]any{"type": "idle"}, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})

	log.Printf("conversation turn: user=%s session=%s", env.UserID, env.SessionID)
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

func (h *Handler) resolveUserID(r *http.Request, candidates ...string) (string, bool) {
	userID, err := agentauth.ResolveDirectUserID(r, h.validator, candidates...)
	if err != nil {
		return "", false
	}
	return userID, true
}

func latestUserTurnMessage(messages []map[string]any) (map[string]any, error) {
	for i := len(messages) - 1; i >= 0; i-- {
		role, _ := messages[i]["role"].(string)
		if strings.TrimSpace(role) != "user" {
			continue
		}
		copied := map[string]any{}
		for k, v := range messages[i] {
			copied[k] = v
		}
		if _, ok := copied["content"]; !ok {
			return nil, fmt.Errorf("latest user message content is required")
		}
		return copied, nil
	}
	return nil, fmt.Errorf("latest user message is required")
}

func (h *Handler) loadConversationHistory(ctx context.Context, userID, sessionID string) []map[string]any {
	if h == nil || h.store == nil {
		return []map[string]any{}
	}
	// Load today's conversation only — history resets at midnight UTC.
	now := time.Now().UTC()
	startOfDay := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	recs, err := h.store.ListChatMessagesSince(ctx, userID, sessionID, startOfDay, 500)
	if err != nil || len(recs) == 0 {
		return []map[string]any{}
	}
	out := make([]map[string]any, 0, len(recs))
	for _, rec := range recs {
		if len(rec.Content) == 0 {
			continue
		}
		out = append(out, rec.Content)
	}
	return sanitizeToolMessageHistory(out)
}

func sanitizeToolMessageHistory(messages []map[string]any) []map[string]any {
	if len(messages) == 0 {
		return messages
	}
	out := make([]map[string]any, 0, len(messages))
	pendingToolIDs := map[string]struct{}{}
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		role = strings.TrimSpace(strings.ToLower(role))
		switch role {
		case "assistant":
			for _, id := range extractToolCallIDs(msg) {
				pendingToolIDs[id] = struct{}{}
			}
			out = append(out, msg)
		case "tool":
			callID, _ := msg["tool_call_id"].(string)
			callID = strings.TrimSpace(callID)
			if callID == "" {
				continue
			}
			if _, ok := pendingToolIDs[callID]; !ok {
				continue
			}
			out = append(out, msg)
			delete(pendingToolIDs, callID)
		default:
			out = append(out, msg)
		}
	}
	return out
}

func extractToolCallIDs(msg map[string]any) []string {
	ids := []string{}
	raw, ok := msg["tool_calls"]
	if !ok {
		return ids
	}
	items, ok := raw.([]any)
	if !ok {
		if maps, okMap := raw.([]map[string]any); okMap {
			for _, item := range maps {
				id, _ := item["id"].(string)
				id = strings.TrimSpace(id)
				if id != "" {
					ids = append(ids, id)
				}
			}
		}
		return ids
	}
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		id, _ := item["id"].(string)
		id = strings.TrimSpace(id)
		if id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func (h *Handler) resolveMediaRefs(ctx context.Context, userID string, messages []map[string]any) ([]map[string]any, error) {
	if len(messages) == 0 {
		return messages, nil
	}
	latestUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		role, _ := messages[i]["role"].(string)
		if role == "user" {
			latestUserIndex = i
			break
		}
	}
	out := make([]map[string]any, 0, len(messages))
	for i, msg := range messages {
		copied := map[string]any{}
		for k, v := range msg {
			copied[k] = v
		}
		parts, ok := msg["content"].([]any)
		if !ok {
			out = append(out, copied)
			continue
		}
		resolved := make([]any, 0, len(parts))
		for j, p := range parts {
			part, ok := p.(map[string]any)
			if !ok {
				resolved = append(resolved, p)
				continue
			}
			typ, _ := part["type"].(string)
			switch typ {
			case "image_ref":
				if i != latestUserIndex {
					resolved = append(resolved, map[string]any{"type": "text", "text": "[image]"})
					continue
				}
				if h.media == nil || !h.media.Enabled() {
					return nil, fmt.Errorf("messages[%d].content[%d]: image_ref requires media storage", i, j)
				}
				mediaObj, _ := part["media"].(map[string]any)
				bucket := firstNonEmpty(strings.TrimSpace(stringValue(mediaObj["bucket"])), h.media.BucketForUser(userID))
				expectedBucket := h.media.BucketForUser(userID)
				if expectedBucket != "" && bucket != expectedBucket {
					return nil, fmt.Errorf("messages[%d].content[%d]: media bucket mismatch", i, j)
				}
				key := strings.TrimSpace(stringValue(mediaObj["key"]))
				if key == "" {
					return nil, fmt.Errorf("messages[%d].content[%d]: media.key is required", i, j)
				}
				bytes, contentType, err := h.media.GetObjectBytes(ctx, bucket, key)
				if err != nil {
					return nil, fmt.Errorf("messages[%d].content[%d]: failed to fetch image", i, j)
				}
				if len(bytes) > h.limits.MaxImageBytes {
					return nil, fmt.Errorf("messages[%d].content[%d]: image exceeds size limit", i, j)
				}
				mime := strings.TrimSpace(stringValue(mediaObj["mime"]))
				if mime == "" {
					mime = firstNonEmpty(strings.TrimSpace(contentType), mimeFromPath(key), "image/png")
				}
				url := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(bytes)
				resolved = append(resolved, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			case "audio_ref":
				if i != latestUserIndex {
					resolved = append(resolved, map[string]any{"type": "text", "text": "[audio]"})
					continue
				}
				if h.media == nil || !h.media.Enabled() {
					return nil, fmt.Errorf("messages[%d].content[%d]: audio_ref requires media storage", i, j)
				}
				mediaObj, _ := part["media"].(map[string]any)
				bucket := firstNonEmpty(strings.TrimSpace(stringValue(mediaObj["bucket"])), h.media.BucketForUser(userID))
				expectedBucket := h.media.BucketForUser(userID)
				if expectedBucket != "" && bucket != expectedBucket {
					return nil, fmt.Errorf("messages[%d].content[%d]: media bucket mismatch", i, j)
				}
				key := strings.TrimSpace(stringValue(mediaObj["key"]))
				if key == "" {
					return nil, fmt.Errorf("messages[%d].content[%d]: media.key is required", i, j)
				}
				bytes, contentType, err := h.media.GetObjectBytes(ctx, bucket, key)
				if err != nil {
					return nil, fmt.Errorf("messages[%d].content[%d]: failed to fetch audio", i, j)
				}
				if len(bytes) > h.limits.MaxAudioBytes {
					return nil, fmt.Errorf("messages[%d].content[%d]: audio exceeds size limit", i, j)
				}
				format := strings.TrimSpace(stringValue(mediaObj["format"]))
				if format == "" {
					format = audioFormatFromMimeOrPath(contentType, key)
				}
				resolved = append(resolved, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": base64.StdEncoding.EncodeToString(bytes), "format": format}})
			default:
				resolved = append(resolved, p)
			}
		}
		copied["content"] = resolved
		out = append(out, copied)
	}
	return out, nil
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

func mimeFromPath(v string) string {
	ext := strings.ToLower(path.Ext(strings.TrimSpace(v)))
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	case ".gif":
		return "image/gif"
	case ".wav":
		return "audio/wav"
	case ".mp3":
		return "audio/mpeg"
	case ".ogg":
		return "audio/ogg"
	case ".flac":
		return "audio/flac"
	case ".m4a":
		return "audio/mp4"
	case ".aac":
		return "audio/aac"
	case ".aiff":
		return "audio/aiff"
	case ".webm":
		return "audio/webm"
	default:
		return ""
	}
}

func audioFormatFromMimeOrPath(contentType, objectKey string) string {
	ct := strings.ToLower(strings.TrimSpace(contentType))
	switch {
	case strings.Contains(ct, "wav"):
		return "wav"
	case strings.Contains(ct, "mpeg") || strings.Contains(ct, "mp3"):
		return "mp3"
	case strings.Contains(ct, "ogg"):
		return "ogg"
	case strings.Contains(ct, "flac"):
		return "flac"
	case strings.Contains(ct, "aac"):
		return "aac"
	case strings.Contains(ct, "aiff"):
		return "aiff"
	case strings.Contains(ct, "mp4") || strings.Contains(ct, "m4a"):
		return "m4a"
	case strings.Contains(ct, "webm"):
		return "webm"
	}
	ext := strings.TrimPrefix(strings.ToLower(path.Ext(objectKey)), ".")
	if isSupportedAudioFormat(ext) {
		return ext
	}
	return "wav"
}

func isSupportedAudioFormat(ext string) bool {
	switch ext {
	case "wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a", "pcm16", "pcm24", "webm":
		return true
	default:
		return false
	}
}

// ── Session endpoints ──────────────────────────────────

func (h *Handler) handleSessionStatus(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "store unavailable")
		return
	}
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
	if userID == "" {
		userID = "default"
	}
	limit := 100
	sessions, err := h.store.ListChatSessions(r.Context(), userID, limit)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := map[string]string{}
	for _, sess := range sessions {
		out[sess.ID] = h.status.get(sess.ID)
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"statuses": out})
}

func (h *Handler) handleSessions(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "store unavailable")
		return
	}
	switch r.Method {
	case http.MethodGet:
		userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
		if userID == "" {
			userID = "default"
		}
		limit := 50
		if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
			if n, err := fmt.Sscanf(raw, "%d", &limit); n == 0 || err != nil {
				limit = 50
			}
		}
		sessions, err := h.store.ListChatSessions(r.Context(), userID, limit)
		if err != nil {
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
		httputil.WriteJSON(w, http.StatusOK, map[string]any{"sessions": sessions})

	case http.MethodPost:
		var req struct {
			UserID string `json:"user_id"`
			Title  string `json:"title"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httputil.WriteError(w, http.StatusBadRequest, "invalid json")
			return
		}
		userID := strings.TrimSpace(req.UserID)
		if userID == "" {
			userID = "default"
		}
		title := strings.TrimSpace(req.Title)
		if title == "" {
			title = "New chat"
		}
		sess, err := h.store.CreateChatSession(r.Context(), userID, title)
		if err != nil {
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.status.set(sess.ID, "idle")
		h.emit(userID, "session.created", map[string]any{"sessionID": sess.ID, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
		httputil.WriteJSON(w, http.StatusCreated, sess)

	default:
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (h *Handler) handleSessionByID(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "store unavailable")
		return
	}
	// Extract session ID from path: /agent/v1/sessions/{id}
	sessionID := trimConversationPathPrefix(r.URL.Path, agentPublicPrefix+"/sessions/", legacyV1Prefix+"/sessions/")
	sessionID = strings.Trim(sessionID, "/")
	if sessionID == "" {
		httputil.WriteError(w, http.StatusBadRequest, "session id required")
		return
	}

	switch r.Method {
	case http.MethodGet:
		sess, err := h.store.GetChatSession(r.Context(), sessionID)
		if err != nil {
			httputil.WriteError(w, http.StatusNotFound, "session not found")
			return
		}
		// Also load messages for this session.
		msgs, _ := h.store.ListChatMessages(r.Context(), sess.UserID, sessionID, 500)
		httputil.WriteJSON(w, http.StatusOK, map[string]any{"session": sess, "messages": msgs})

	case http.MethodPatch:
		var req struct {
			Title   *string `json:"title"`
			Archive *bool   `json:"archive"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httputil.WriteError(w, http.StatusBadRequest, "invalid json")
			return
		}
		if req.Title != nil {
			if err := h.store.UpdateChatSessionTitle(r.Context(), sessionID, strings.TrimSpace(*req.Title)); err != nil {
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			h.emit(sessUserIDFromStore(h, r.Context(), sessionID), "session.updated", map[string]any{"sessionID": sessionID, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
		}
		if req.Archive != nil && *req.Archive {
			if err := h.store.ArchiveChatSession(r.Context(), sessionID); err != nil {
				httputil.WriteError(w, http.StatusInternalServerError, err.Error())
				return
			}
			h.emit(sessUserIDFromStore(h, r.Context(), sessionID), "session.updated", map[string]any{"sessionID": sessionID, "archived": true, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
		}
		sess, _ := h.store.GetChatSession(r.Context(), sessionID)
		httputil.WriteJSON(w, http.StatusOK, sess)

	case http.MethodDelete:
		userID := sessUserIDFromStore(h, r.Context(), sessionID)
		if err := h.store.DeleteChatSession(r.Context(), sessionID); err != nil {
			httputil.WriteError(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.status.set(sessionID, "idle")
		h.emit(userID, "session.deleted", map[string]any{"sessionID": sessionID, "updatedAt": time.Now().UTC().Format(time.RFC3339Nano)})
		httputil.WriteJSON(w, http.StatusOK, map[string]any{"deleted": true})

	default:
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func sessUserIDFromStore(h *Handler, ctx context.Context, sessionID string) string {
	if h == nil || h.store == nil {
		return "default"
	}
	sess, err := h.store.GetChatSession(ctx, sessionID)
	if err != nil {
		return "default"
	}
	if strings.TrimSpace(sess.UserID) == "" {
		return "default"
	}
	return sess.UserID
}
