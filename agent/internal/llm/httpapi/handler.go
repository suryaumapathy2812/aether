package httpapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type Handler struct {
	core    *llm.Core
	builder *llm.ContextBuilder
	memory  *memory.Service
	media   *media.Service
}

const (
	defaultMaxImageBytes      = 5 * 1024 * 1024
	defaultMaxAudioBytes      = 12 * 1024 * 1024
	defaultMaxTotalMediaBytes = 20 * 1024 * 1024
	defaultMaxMediaParts      = 4
)

var (
	allowedImageMIMEs = map[string]bool{
		"image/png":  true,
		"image/jpeg": true,
		"image/jpg":  true,
		"image/webp": true,
		"image/gif":  true,
	}
	allowedAudioFormats = map[string]bool{
		"wav":   true,
		"mp3":   true,
		"aiff":  true,
		"aac":   true,
		"ogg":   true,
		"flac":  true,
		"m4a":   true,
		"pcm16": true,
		"pcm24": true,
		"webm":  true,
	}
)

type Options struct {
	Core    *llm.Core
	Builder *llm.ContextBuilder
	Memory  *memory.Service
	Media   *media.Service
}

func New(opts Options) *Handler {
	return &Handler{core: opts.Core, builder: opts.Builder, memory: opts.Memory, media: opts.Media}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/models", h.handleModels)
	mux.HandleFunc("/v1/chat/completions", h.handleChatCompletions)
	mux.HandleFunc("/v1/media/upload/init", h.handleMediaUploadInit)
	mux.HandleFunc("/v1/media/upload/complete", h.handleMediaUploadComplete)
}

func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	model := firstNonEmpty(strings.TrimSpace(os.Getenv("OPENAI_MODEL")), strings.TrimSpace(os.Getenv("AETHER_LLM_MODEL")), "gpt-4o-mini")
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id":       model,
			"object":   "model",
			"created":  0,
			"owned_by": "aether",
		}},
	})
}

func (h *Handler) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.core == nil || h.builder == nil {
		writeError(w, http.StatusInternalServerError, "llm runtime unavailable")
		return
	}
	var req struct {
		Model       string           `json:"model"`
		Messages    []map[string]any `json:"messages"`
		Stream      bool             `json:"stream"`
		Temperature *float64         `json:"temperature"`
		MaxTokens   *int             `json:"max_tokens"`
		User        string           `json:"user"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}
	if err := validateMediaParts(req.Messages); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	userID := firstNonEmpty(strings.TrimSpace(req.User), "default")
	resolvedMessages, err := h.resolveMediaRefs(r.Context(), userID, req.Messages)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	messages, err := llm.ParseChatMessages(resolvedMessages)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	policy := map[string]any{}
	if req.MaxTokens != nil {
		policy["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		policy["temperature"] = *req.Temperature
	}
	if model := strings.TrimSpace(req.Model); model != "" && !strings.EqualFold(model, "aether") {
		policy["model"] = model
	}
	sessionID := firstNonEmpty(strings.TrimSpace(req.User), "http-anon")
	env := h.builder.Build(messages, policy, userID, sessionID)
	log.Printf("llm request: user=%s session=%s messages=%d stream=%t", env.UserID, env.SessionID, len(messages), req.Stream)

	completionID := "chatcmpl-" + uuid.NewString()[:12]
	created := time.Now().Unix()
	model := firstNonEmpty(policyString(policy, "model"), strings.TrimSpace(os.Getenv("OPENAI_MODEL")), strings.TrimSpace(os.Getenv("AETHER_LLM_MODEL")), "gpt-4o-mini")

	// Inject user ID into context so tools (e.g. delegate_task) can use the
	// real authenticated user instead of relying on LLM-generated arguments.
	rCtx := tools.WithTaskRuntimeContext(r.Context(), tools.TaskRuntimeContext{UserID: userID})
	r = r.WithContext(rCtx)

	if req.Stream {
		h.streamResponse(w, r, env, req.Messages, completionID, created, model)
		return
	}
	h.syncResponse(w, r, env, req.Messages, completionID, created, model)
}

func (h *Handler) streamResponse(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, requestMessages []map[string]any, completionID string, created int64, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}

	writeSSE(w, map[string]any{
		"id":      completionID,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{"index": 0, "delta": map[string]any{"role": "assistant"}, "finish_reason": nil}},
	})
	flusher.Flush()

	finish := "stop"
	parts := []string{}
	toolArgs := map[string]map[string]any{}
	for ev := range h.core.GenerateWithTools(r.Context(), env) {
		switch ev.EventType {
		case llm.EventStatus:
			if msg, ok := ev.Payload["message"].(string); ok && msg != "" {
				log.Printf("llm status: id=%s %s", completionID, msg)
			}
		case llm.EventToolCall:
			name, _ := ev.Payload["tool_name"].(string)
			callID, _ := ev.Payload["call_id"].(string)
			args, _ := ev.Payload["arguments"].(map[string]any)
			if strings.TrimSpace(callID) != "" {
				toolArgs[callID] = args
			}
			log.Printf("llm tool_call: id=%s tool=%s", completionID, name)
		case llm.EventToolResult:
			name, _ := ev.Payload["tool_name"].(string)
			errFlag, _ := ev.Payload["error"].(bool)
			output, _ := ev.Payload["output"].(string)
			callID, _ := ev.Payload["call_id"].(string)
			if h.memory != nil {
				h.memory.RecordAction(context.Background(), env.UserID, env.SessionID, name, toolArgs[callID], output, errFlag)
			}
			log.Printf("llm tool_result: id=%s tool=%s error=%t", completionID, name, errFlag)
		case llm.EventTextChunk:
			chunk, _ := ev.Payload["text"].(string)
			if strings.TrimSpace(chunk) == "" {
				continue
			}
			parts = append(parts, chunk)
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": chunk}, "finish_reason": nil}},
			})
			flusher.Flush()
		case llm.EventError:
			msg, _ := ev.Payload["message"].(string)
			if msg == "" {
				msg = "unknown error"
			}
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": "\n[error] " + msg}, "finish_reason": nil}},
			})
			flusher.Flush()
		case llm.EventStreamEnd:
			fr, _ := ev.Payload["finish_reason"].(string)
			if fr != "" {
				finish = fr
			}
			log.Printf("llm stream_end: id=%s reason=%s", completionID, finish)
		}
	}

	writeSSE(w, map[string]any{
		"id":      completionID,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{"index": 0, "delta": map[string]any{}, "finish_reason": finish}},
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	if h.memory != nil {
		content := strings.TrimSpace(strings.Join(parts, ""))
		if content != "" {
			h.memory.RecordConversation(context.Background(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(requestMessages), llm.LatestUserMessageContent(requestMessages), content)
		}
	}
}

func (h *Handler) syncResponse(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, requestMessages []map[string]any, completionID string, created int64, model string) {
	parts := []string{}
	finish := "stop"
	toolArgs := map[string]map[string]any{}
	for ev := range h.core.GenerateWithTools(r.Context(), env) {
		switch ev.EventType {
		case llm.EventStatus:
			if msg, ok := ev.Payload["message"].(string); ok && msg != "" {
				log.Printf("llm status: id=%s %s", completionID, msg)
			}
		case llm.EventToolCall:
			name, _ := ev.Payload["tool_name"].(string)
			callID, _ := ev.Payload["call_id"].(string)
			args, _ := ev.Payload["arguments"].(map[string]any)
			if strings.TrimSpace(callID) != "" {
				toolArgs[callID] = args
			}
			log.Printf("llm tool_call: id=%s tool=%s", completionID, name)
		case llm.EventToolResult:
			name, _ := ev.Payload["tool_name"].(string)
			errFlag, _ := ev.Payload["error"].(bool)
			output, _ := ev.Payload["output"].(string)
			callID, _ := ev.Payload["call_id"].(string)
			if h.memory != nil {
				h.memory.RecordAction(context.Background(), env.UserID, env.SessionID, name, toolArgs[callID], output, errFlag)
			}
			log.Printf("llm tool_result: id=%s tool=%s error=%t", completionID, name, errFlag)
		case llm.EventTextChunk:
			chunk, _ := ev.Payload["text"].(string)
			parts = append(parts, chunk)
		case llm.EventStreamEnd:
			if v, ok := ev.Payload["finish_reason"].(string); ok && v != "" {
				finish = v
			}
			log.Printf("llm stream_end: id=%s reason=%s", completionID, finish)
		case llm.EventError:
			msg, _ := ev.Payload["message"].(string)
			log.Printf("llm error: id=%s message=%s", completionID, msg)
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"message": msg, "type": "server_error", "code": "internal_error"}})
			return
		}
	}
	content := strings.Join(parts, "")
	if h.memory != nil && strings.TrimSpace(content) != "" {
		h.memory.RecordConversation(context.Background(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(requestMessages), llm.LatestUserMessageContent(requestMessages), content)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      completionID,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": finish}},
		"usage":   map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
	})
}

func (h *Handler) handleMediaUploadInit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.media == nil || !h.media.Enabled() {
		writeError(w, http.StatusBadRequest, "media storage is not configured (set S3_BUCKET or S3_BUCKET_TEMPLATE)")
		return
	}
	var req struct {
		UserID      string `json:"user_id"`
		SessionID   string `json:"session_id"`
		FileName    string `json:"file_name"`
		ContentType string `json:"content_type"`
		Size        int64  `json:"size"`
		Kind        string `json:"kind"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if err := validateUploadIntent(req.Kind, req.ContentType, req.Size); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	userID := firstNonEmpty(strings.TrimSpace(req.UserID), "default")
	bucket := h.media.BucketForUser(userID)
	if strings.TrimSpace(bucket) == "" {
		writeError(w, http.StatusBadRequest, "media bucket is not configured")
		return
	}
	if err := h.media.EnsureBucket(r.Context(), bucket); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to prepare media bucket")
		return
	}
	objectKey := h.media.BuildObjectKey(userID, firstNonEmpty(strings.TrimSpace(req.SessionID), "chat"), req.FileName)
	put, err := h.media.PresignUpload(r.Context(), bucket, objectKey, req.ContentType)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"bucket":     bucket,
		"object_key": put.ObjectKey,
		"upload_url": put.UploadURL,
		"headers":    put.Headers,
		"expires_at": put.ExpiresAt.Unix(),
	})
}

func (h *Handler) handleMediaUploadComplete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.media == nil || !h.media.Enabled() {
		writeError(w, http.StatusBadRequest, "media storage is not configured (set S3_BUCKET or S3_BUCKET_TEMPLATE)")
		return
	}
	var req struct {
		UserID      string `json:"user_id"`
		Bucket      string `json:"bucket"`
		ObjectKey   string `json:"object_key"`
		Kind        string `json:"kind"`
		ContentType string `json:"content_type"`
		Size        int64  `json:"size"`
		FileName    string `json:"file_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	req.ObjectKey = strings.TrimSpace(req.ObjectKey)
	if req.ObjectKey == "" {
		writeError(w, http.StatusBadRequest, "object_key is required")
		return
	}
	userID := firstNonEmpty(strings.TrimSpace(req.UserID), "default")
	bucket := firstNonEmpty(strings.TrimSpace(req.Bucket), h.media.BucketForUser(userID))
	if strings.TrimSpace(bucket) == "" {
		writeError(w, http.StatusBadRequest, "bucket is required")
		return
	}
	expectedBucket := h.media.BucketForUser(userID)
	if expectedBucket != "" && bucket != expectedBucket {
		writeError(w, http.StatusBadRequest, "bucket does not match user")
		return
	}
	info, err := h.media.HeadObject(r.Context(), bucket, req.ObjectKey)
	if err != nil {
		writeError(w, http.StatusBadRequest, "media object not found")
		return
	}
	ct := firstNonEmpty(strings.TrimSpace(req.ContentType), strings.TrimSpace(info.ContentType), mimeFromPath(req.ObjectKey))
	if err := validateUploadIntent(req.Kind, ct, info.Size); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	url, err := h.media.PresignGet(r.Context(), bucket, req.ObjectKey)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"media": map[string]any{
			"bucket":    bucket,
			"key":       req.ObjectKey,
			"url":       url,
			"mime":      ct,
			"size":      info.Size,
			"file_name": req.FileName,
			"format":    audioFormatFromMimeOrPath(ct, req.ObjectKey),
		},
	})
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
				if len(bytes) > envInt("AETHER_MAX_IMAGE_BYTES", defaultMaxImageBytes) {
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
				if len(bytes) > envInt("AETHER_MAX_AUDIO_BYTES", defaultMaxAudioBytes) {
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

func policyString(policy map[string]any, key string) string {
	v, _ := policy[key].(string)
	return strings.TrimSpace(v)
}

func validateMediaParts(messages []map[string]any) error {
	maxImageBytes := envInt("AETHER_MAX_IMAGE_BYTES", defaultMaxImageBytes)
	maxAudioBytes := envInt("AETHER_MAX_AUDIO_BYTES", defaultMaxAudioBytes)
	maxTotalMediaBytes := envInt("AETHER_MAX_TOTAL_MEDIA_BYTES", defaultMaxTotalMediaBytes)
	maxMediaParts := envInt("AETHER_MAX_MEDIA_PARTS", defaultMaxMediaParts)

	for i, msg := range messages {
		totalMediaBytes := 0
		mediaParts := 0
		role, _ := msg["role"].(string)
		if role != "user" {
			continue
		}
		rawContent, exists := msg["content"]
		if !exists {
			continue
		}
		parts, ok := rawContent.([]any)
		if !ok {
			continue
		}
		for j, rawPart := range parts {
			part, ok := rawPart.(map[string]any)
			if !ok {
				continue
			}
			typ, _ := part["type"].(string)
			switch typ {
			case "input_audio":
				mediaParts++
				if mediaParts > maxMediaParts {
					return newValidationError(i, j, "too many media parts")
				}
				inputAudio, _ := part["input_audio"].(map[string]any)
				format := strings.ToLower(strings.TrimSpace(stringValue(inputAudio["format"])))
				if !allowedAudioFormats[format] {
					return newValidationError(i, j, "unsupported audio format")
				}
				data := strings.TrimSpace(stringValue(inputAudio["data"]))
				sz, err := decodedBase64Size(data)
				if err != nil {
					return newValidationError(i, j, "invalid base64 audio")
				}
				if sz > maxAudioBytes {
					return newValidationError(i, j, "audio exceeds size limit")
				}
				totalMediaBytes += sz
				if totalMediaBytes > maxTotalMediaBytes {
					return newValidationError(i, j, "total media payload exceeds size limit")
				}
			case "image_url":
				mediaParts++
				if mediaParts > maxMediaParts {
					return newValidationError(i, j, "too many media parts")
				}
				imageURL, _ := part["image_url"].(map[string]any)
				url := strings.TrimSpace(stringValue(imageURL["url"]))
				mime, base64Data, ok := parseDataURL(url)
				if !ok {
					if isHTTPURL(url) {
						continue
					}
					return newValidationError(i, j, "image must be a base64 data URL or HTTPS URL")
				}
				if !allowedImageMIMEs[mime] {
					return newValidationError(i, j, "unsupported image mime type")
				}
				sz, err := decodedBase64Size(base64Data)
				if err != nil {
					return newValidationError(i, j, "invalid base64 image")
				}
				if sz > maxImageBytes {
					return newValidationError(i, j, "image exceeds size limit")
				}
				totalMediaBytes += sz
				if totalMediaBytes > maxTotalMediaBytes {
					return newValidationError(i, j, "total media payload exceeds size limit")
				}
			case "image_ref", "audio_ref":
				mediaParts++
				if mediaParts > maxMediaParts {
					return newValidationError(i, j, "too many media parts")
				}
				mediaObj, _ := part["media"].(map[string]any)
				key := strings.TrimSpace(stringValue(mediaObj["key"]))
				if key == "" {
					return newValidationError(i, j, "media.key is required")
				}
				size := int(numberValue(mediaObj["size"]))
				if typ == "image_ref" {
					if size > maxImageBytes {
						return newValidationError(i, j, "image exceeds size limit")
					}
				} else {
					if size > maxAudioBytes {
						return newValidationError(i, j, "audio exceeds size limit")
					}
				}
				if size > 0 {
					totalMediaBytes += size
					if totalMediaBytes > maxTotalMediaBytes {
						return newValidationError(i, j, "total media payload exceeds size limit")
					}
				}
			}
		}
	}
	return nil
}

func parseDataURL(v string) (string, string, bool) {
	if !strings.HasPrefix(v, "data:") {
		return "", "", false
	}
	comma := strings.Index(v, ",")
	if comma <= 5 {
		return "", "", false
	}
	meta := v[5:comma]
	if !strings.HasSuffix(meta, ";base64") {
		return "", "", false
	}
	mime := strings.ToLower(strings.TrimSpace(strings.TrimSuffix(meta, ";base64")))
	if mime == "" {
		return "", "", false
	}
	return mime, v[comma+1:], true
}

func decodedBase64Size(data string) (int, error) {
	if data == "" {
		return 0, nil
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return 0, err
	}
	return len(decoded), nil
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

func numberValue(v any) float64 {
	switch n := v.(type) {
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case float64:
		return n
	default:
		return 0
	}
}

func envInt(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func newValidationError(messageIndex, partIndex int, reason string) error {
	return &validationError{messageIndex: messageIndex, partIndex: partIndex, reason: reason}
}

type validationError struct {
	messageIndex int
	partIndex    int
	reason       string
}

func (e *validationError) Error() string {
	return "messages[" + strconv.Itoa(e.messageIndex) + "].content[" + strconv.Itoa(e.partIndex) + "]: " + e.reason
}

func validateUploadIntent(kind, contentType string, size int64) error {
	kind = strings.ToLower(strings.TrimSpace(kind))
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	if kind != "image" && kind != "audio" {
		return fmt.Errorf("kind must be image or audio")
	}
	if size <= 0 {
		return fmt.Errorf("size must be greater than 0")
	}
	if kind == "image" {
		if !allowedImageMIMEs[contentType] {
			return fmt.Errorf("unsupported image mime type")
		}
		if size > int64(envInt("AETHER_MAX_IMAGE_BYTES", defaultMaxImageBytes)) {
			return fmt.Errorf("image exceeds size limit")
		}
		return nil
	}
	if !strings.HasPrefix(contentType, "audio/") {
		return fmt.Errorf("unsupported audio mime type")
	}
	if size > int64(envInt("AETHER_MAX_AUDIO_BYTES", defaultMaxAudioBytes)) {
		return fmt.Errorf("audio exceeds size limit")
	}
	return nil
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
	if allowedAudioFormats[ext] {
		return ext
	}
	return "wav"
}

func isHTTPURL(v string) bool {
	v = strings.ToLower(strings.TrimSpace(v))
	return strings.HasPrefix(v, "https://") || strings.HasPrefix(v, "http://")
}
