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
	"time"

	agentcfg "github.com/suryaumapathy2812/core-ai/agent/internal/config"
	"github.com/suryaumapathy2812/core-ai/agent/internal/conversation"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type Handler struct {
	runtime *conversation.Runtime
	builder *llm.ContextBuilder
	memory  *memory.Service
	media   *media.Service
	store   *db.Store
	limits  agentcfg.MediaLimitsConfig
}

type Options struct {
	Runtime *conversation.Runtime
	Builder *llm.ContextBuilder
	Memory  *memory.Service
	Media   *media.Service
	Store   *db.Store
	Limits  agentcfg.MediaLimitsConfig
}

func New(opts Options) *Handler {
	return &Handler{runtime: opts.Runtime, builder: opts.Builder, memory: opts.Memory, media: opts.Media, store: opts.Store, limits: opts.Limits}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/conversations/turn", h.handleTurn)
}

func (h *Handler) handleTurn(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.runtime == nil || h.builder == nil {
		writeError(w, http.StatusInternalServerError, "conversation runtime unavailable")
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
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}

	userID := firstNonEmpty(strings.TrimSpace(req.User), strings.TrimSpace(req.UserID), "default")
	sessionID := firstNonEmpty(strings.TrimSpace(req.Session), userID)

	latestUser, err := latestUserTurnMessage(req.Messages)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	resolvedTurn, err := h.resolveMediaRefs(r.Context(), userID, []map[string]any{latestUser})
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(resolvedTurn) == 0 {
		writeError(w, http.StatusBadRequest, "latest user message is required")
		return
	}
	latestUser = resolvedTurn[0]

	history := h.loadConversationHistory(r.Context(), userID, sessionID)
	messages := append(history, latestUser)
	if h.store != nil {
		_ = h.store.AppendChatMessage(r.Context(), userID, sessionID, latestUser)
	}

	policy := map[string]any{}
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
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}

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
		writeSSE(w, chunk)
		flusher.Flush()

		// Side effects: persist to DB and collect answer text.
		switch ev.EventType {
		case conversation.EventTextDelta:
			delta, _ := ev.Payload["delta"].(string)
			answerParts = append(answerParts, delta)

		case conversation.EventToolCall:
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

		case conversation.EventToolResult:
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
				isErr, _ := ev.Payload["error"].(bool)
				if strings.TrimSpace(output) != "" {
					content := output
					if isErr {
						content = "[tool_error] " + content
					}
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
	}

	log.Printf("conversation turn: user=%s session=%s", env.UserID, env.SessionID)
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
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

func writeSSE(w http.ResponseWriter, payload map[string]any) {
	b, _ := json.Marshal(payload)
	_, _ = w.Write([]byte("data: " + string(b) + "\n\n"))
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
