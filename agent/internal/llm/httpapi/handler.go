package httpapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	agentcfg "github.com/suryaumapathy2812/core-ai/agent/internal/config"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
)

type Handler struct {
	core        *llm.Core
	builder     *llm.ContextBuilder
	memory      *memory.Service
	media       *media.Service
	model       string
	mediaLimits agentcfg.MediaLimitsConfig
}

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
	Core        *llm.Core
	Builder     *llm.ContextBuilder
	Memory      *memory.Service
	Media       *media.Service
	Model       string
	MediaLimits agentcfg.MediaLimitsConfig
}

func New(opts Options) *Handler {
	return &Handler{
		core:        opts.Core,
		builder:     opts.Builder,
		memory:      opts.Memory,
		media:       opts.Media,
		model:       opts.Model,
		mediaLimits: opts.MediaLimits,
	}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/models", h.handleModels)
	mux.HandleFunc("/v1/chat/completions", h.handleChatCompletions)
	mux.HandleFunc("/v1/responses", h.handleResponses)
	mux.HandleFunc("/v1/completions", h.handleCompletions)
	mux.HandleFunc("/v1/media/upload/init", h.handleMediaUploadInit)
	mux.HandleFunc("/v1/media/upload/complete", h.handleMediaUploadComplete)
}

func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id":       h.model,
			"object":   "model",
			"created":  0,
			"owned_by": "aether",
		}},
	})
}

func (h *Handler) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.core == nil || h.builder == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "llm runtime unavailable")
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
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if len(req.Messages) == 0 {
		httputil.WriteError(w, http.StatusBadRequest, "messages is required")
		return
	}
	if err := h.validateMediaParts(req.Messages); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	userID := firstNonEmpty(strings.TrimSpace(req.User), "default")
	resolvedMessages, err := h.resolveMediaRefs(r.Context(), userID, req.Messages)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	messages, err := llm.ParseChatMessages(resolvedMessages)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
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
	model := firstNonEmpty(policyString(policy, "model"), h.model)

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

// eventSink holds per-event callbacks for drainEvents callers.
// All fields are optional; nil callbacks are ignored.
type eventSink struct {
	// onTextDelta is called for each text chunk emitted by the LLM.
	onTextDelta func(chunk string)
	// onToolInput is called when a tool call begins. argsJSON is args pre-marshalled to JSON.
	onToolInput func(name, callID string, args map[string]any, argsJSON []byte)
	// onToolOutput is called when a tool call completes.
	onToolOutput func(callID, name, output string, errFlag bool)
	// onError is called on EventError. Return true to abort the loop (response already written).
	onError func(msg string) bool
}

// eventResult holds the state accumulated by drainEvents.
type eventResult struct {
	parts            []string
	finishReason     string
	promptTokens     int
	completionTokens int
	totalTokens      int
	aborted          bool // true if onError returned true and the loop was cut short
}

// drainEvents drives a single LLM generation run, centralising tool-arg tracking
// and memory recording. Output formatting is delegated to the sink callbacks so
// callers only need to handle their specific wire format (SSE vs JSON, chat vs
// responses API).
func (h *Handler) drainEvents(r *http.Request, env llm.LLMRequestEnvelope, completionID string, sink eventSink) eventResult {
	toolArgs := map[string]map[string]any{}
	res := eventResult{finishReason: "stop"}
	for ev := range h.core.GenerateWithTools(r.Context(), env) {
		switch ev.EventType {
		case llm.EventStartStep:
			if msg, ok := ev.Payload["errorText"].(string); ok && msg != "" {
				log.Printf("llm status: id=%s %s", completionID, msg)
			}
		case llm.EventToolInputAvailable:
			name, _ := ev.Payload["toolName"].(string)
			callID, _ := ev.Payload["toolCallId"].(string)
			args, _ := ev.Payload["input"].(map[string]any)
			argsJSON, _ := json.Marshal(args)
			if strings.TrimSpace(callID) != "" {
				toolArgs[callID] = args
			}
			log.Printf("llm tool_call: id=%s tool=%s", completionID, name)
			if sink.onToolInput != nil {
				sink.onToolInput(name, callID, args, argsJSON)
			}
		case llm.EventToolOutputAvailable:
			name, _ := ev.Payload["toolName"].(string)
			errFlag, _ := ev.Payload["error"].(bool)
			output, _ := ev.Payload["output"].(string)
			callID, _ := ev.Payload["toolCallId"].(string)
			if h.memory != nil {
				h.memory.RecordAction(r.Context(), env.UserID, env.SessionID, name, toolArgs[callID], output, errFlag)
			}
			log.Printf("llm tool_result: id=%s tool=%s error=%t", completionID, name, errFlag)
			if sink.onToolOutput != nil {
				sink.onToolOutput(callID, name, output, errFlag)
			}
		case llm.EventTextDelta:
			chunk, _ := ev.Payload["delta"].(string)
			res.parts = append(res.parts, chunk)
			if sink.onTextDelta != nil {
				sink.onTextDelta(chunk)
			}
		case llm.EventFinish:
			if fr, ok := ev.Payload["finishReason"].(string); ok && fr != "" {
				res.finishReason = fr
			}
			res.promptTokens, res.completionTokens, res.totalTokens = extractUsage(ev.Payload)
			log.Printf("llm stream_end: id=%s reason=%s", completionID, res.finishReason)
		case llm.EventError:
			msg, _ := ev.Payload["errorText"].(string)
			if sink.onError != nil && sink.onError(msg) {
				res.aborted = true
				return res
			}
		}
	}
	return res
}

func (h *Handler) streamResponse(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, requestMessages []map[string]any, completionID string, created int64, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		httputil.WriteError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	httputil.WriteSSE(w, map[string]any{
		"id": completionID, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []map[string]any{{"index": 0, "delta": map[string]any{"role": "assistant"}, "finish_reason": nil}},
	})
	flusher.Flush()

	res := h.drainEvents(r, env, completionID, eventSink{
		onTextDelta: func(chunk string) {
			if strings.TrimSpace(chunk) == "" {
				return
			}
			httputil.WriteSSE(w, map[string]any{
				"id": completionID, "object": "chat.completion.chunk", "created": created, "model": model,
				"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": chunk}, "finish_reason": nil}},
			})
			flusher.Flush()
		},
		onError: func(msg string) bool {
			if msg == "" {
				msg = "unknown error"
			}
			httputil.WriteSSE(w, map[string]any{
				"id": completionID, "object": "chat.completion.chunk", "created": created, "model": model,
				"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": "\n[error] " + msg}, "finish_reason": nil}},
			})
			flusher.Flush()
			return false
		},
	})

	httputil.WriteSSE(w, map[string]any{
		"id": completionID, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []map[string]any{{"index": 0, "delta": map[string]any{}, "finish_reason": res.finishReason}},
		"usage":   map[string]int{"prompt_tokens": res.promptTokens, "completion_tokens": res.completionTokens, "total_tokens": res.totalTokens},
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()

	if h.memory != nil {
		content := strings.TrimSpace(strings.Join(res.parts, ""))
		if content != "" {
			h.memory.RecordConversation(r.Context(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(requestMessages), llm.LatestUserMessageContent(requestMessages), content)
		}
	}
}

func (h *Handler) syncResponse(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, requestMessages []map[string]any, completionID string, created int64, model string) {
	res := h.drainEvents(r, env, completionID, eventSink{
		onError: func(msg string) bool {
			log.Printf("llm error: id=%s message=%s", completionID, msg)
			httputil.WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"message": msg, "type": "server_error", "code": "internal_error"}})
			return true
		},
	})
	if res.aborted {
		return
	}
	content := strings.Join(res.parts, "")
	if h.memory != nil && strings.TrimSpace(content) != "" {
		h.memory.RecordConversation(r.Context(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(requestMessages), llm.LatestUserMessageContent(requestMessages), content)
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"id":      completionID,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": res.finishReason}},
		"usage":   map[string]int{"prompt_tokens": res.promptTokens, "completion_tokens": res.completionTokens, "total_tokens": res.totalTokens},
	})
}

func (h *Handler) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.core == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "llm runtime unavailable")
		return
	}

	var req struct {
		Model              string           `json:"model"`
		Input              []map[string]any `json:"input"`
		Instructions       string           `json:"instructions"`
		Tools              []map[string]any `json:"tools"`
		Stream             bool             `json:"stream"`
		Store              bool             `json:"store"`
		PreviousResponseID string           `json:"previous_response_id"`
		Temperature        *float64         `json:"temperature"`
		MaxTokens          *int             `json:"max_tokens"`
		User               string           `json:"user"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}

	if len(req.Input) == 0 {
		httputil.WriteError(w, http.StatusBadRequest, "input is required")
		return
	}

	userID := firstNonEmpty(strings.TrimSpace(req.User), "default")
	sessionID := firstNonEmpty(strings.TrimSpace(req.User), "http-responses")
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

	messages := normalizeResponsesInput(req.Input)
	env := h.builder.Build(messages, policy, userID, sessionID)
	log.Printf("responses request: user=%s session=%s input_items=%d stream=%t", userID, sessionID, len(req.Input), req.Stream)

	completionID := "resp-" + uuid.NewString()[:12]
	created := time.Now().Unix()
	model := firstNonEmpty(policyString(policy, "model"), h.model)

	rCtx := tools.WithTaskRuntimeContext(r.Context(), tools.TaskRuntimeContext{UserID: userID})
	r = r.WithContext(rCtx)

	if req.Stream {
		h.streamResponses(w, r, env, req.Input, completionID, created, model)
		return
	}
	h.syncResponses(w, r, env, req.Input, completionID, created, model)
}

func (h *Handler) streamResponses(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, requestInput []map[string]any, completionID string, created int64, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		httputil.WriteError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	httputil.WriteSSE(w, map[string]any{
		"type":     "response.created",
		"response": map[string]any{"id": completionID, "created": created, "model": model},
	})
	flusher.Flush()

	toolCalls := []map[string]any{}
	res := h.drainEvents(r, env, completionID, eventSink{
		onToolInput: func(name, callID string, _ map[string]any, argsJSON []byte) {
			toolCalls = append(toolCalls, map[string]any{
				"type": "function_call", "id": callID, "name": name, "arguments": string(argsJSON),
			})
			httputil.WriteSSE(w, map[string]any{"type": "response.function_call.begin", "id": callID, "name": name})
			flusher.Flush()
		},
		onToolOutput: func(callID, _, output string, errFlag bool) {
			if errFlag {
				output = "[error] " + output
			}
			httputil.WriteSSE(w, map[string]any{"type": "response.function_call_output", "call_id": callID, "output": output})
			flusher.Flush()
		},
		onTextDelta: func(chunk string) {
			if strings.TrimSpace(chunk) == "" {
				return
			}
			httputil.WriteSSE(w, map[string]any{"type": "response.output_text.delta", "delta": chunk})
			flusher.Flush()
		},
		onError: func(msg string) bool {
			if msg == "" {
				msg = "unknown error"
			}
			httputil.WriteSSE(w, map[string]any{"type": "error", "message": msg})
			flusher.Flush()
			return false
		},
	})

	for _, tc := range toolCalls {
		httputil.WriteSSE(w, map[string]any{
			"type": "response.function_call.done", "id": tc["id"], "name": tc["name"], "arguments": tc["arguments"],
		})
		flusher.Flush()
	}
	httputil.WriteSSE(w, map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":            completionID,
			"created":       created,
			"model":         model,
			"finish_reason": res.finishReason,
			"usage":         map[string]int{"input_tokens": res.promptTokens, "output_tokens": res.completionTokens, "total_tokens": res.totalTokens},
		},
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()

	if h.memory != nil {
		content := strings.TrimSpace(strings.Join(res.parts, ""))
		if content != "" {
			chatMessages := normalizeResponsesInput(requestInput)
			h.memory.RecordConversation(r.Context(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(chatMessages), llm.LatestUserMessageContent(chatMessages), content)
		}
	}
}

func (h *Handler) syncResponses(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, requestInput []map[string]any, completionID string, created int64, model string) {
	toolCalls := []map[string]any{}
	toolResults := []map[string]any{}
	res := h.drainEvents(r, env, completionID, eventSink{
		onToolInput: func(name, callID string, _ map[string]any, argsJSON []byte) {
			toolCalls = append(toolCalls, map[string]any{
				"type": "function_call", "id": callID, "name": name, "arguments": string(argsJSON),
			})
		},
		onToolOutput: func(callID, _, output string, errFlag bool) {
			if errFlag {
				output = "[error] " + output
			}
			toolResults = append(toolResults, map[string]any{
				"type": "function_call_output", "call_id": callID, "output": output,
			})
		},
		onError: func(msg string) bool {
			log.Printf("responses error: id=%s message=%s", completionID, msg)
			httputil.WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"message": msg, "type": "server_error"}})
			return true
		},
	})
	if res.aborted {
		return
	}

	// Build output items: function_calls, function_call_outputs, then text
	outputItems := append(append([]map[string]any{}, toolCalls...), toolResults...)
	content := strings.Join(res.parts, "")
	if strings.TrimSpace(content) != "" {
		outputItems = append(outputItems, map[string]any{"type": "output_text", "content": content})
	}

	if h.memory != nil && strings.TrimSpace(content) != "" {
		chatMessages := normalizeResponsesInput(requestInput)
		h.memory.RecordConversation(r.Context(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(chatMessages), llm.LatestUserMessageContent(chatMessages), content)
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"id":            completionID,
		"object":        "response",
		"created":       created,
		"model":         model,
		"output":        outputItems,
		"finish_reason": res.finishReason,
		"usage":         map[string]int{"input_tokens": res.promptTokens, "output_tokens": res.completionTokens, "total_tokens": res.totalTokens},
	})
}

func (h *Handler) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.core == nil {
		httputil.WriteError(w, http.StatusInternalServerError, "llm runtime unavailable")
		return
	}

	var req struct {
		Model       string   `json:"model"`
		Prompt      string   `json:"prompt"`
		Stream      bool     `json:"stream"`
		Temperature *float64 `json:"temperature"`
		MaxTokens   *int     `json:"max_tokens"`
		User        string   `json:"user"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}

	if strings.TrimSpace(req.Prompt) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	userID := firstNonEmpty(strings.TrimSpace(req.User), "default")
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

	messages := []map[string]any{
		{"role": "user", "content": req.Prompt},
	}
	env := h.builder.Build(messages, policy, userID, "completions")

	completionID := "cmpl-" + uuid.NewString()[:12]
	created := time.Now().Unix()
	model := firstNonEmpty(policyString(policy, "model"), h.model)

	rCtx := tools.WithTaskRuntimeContext(r.Context(), tools.TaskRuntimeContext{UserID: userID})
	r = r.WithContext(rCtx)

	if req.Stream {
		h.streamCompletions(w, r, env, completionID, created, model)
		return
	}
	h.syncCompletions(w, r, env, completionID, created, model)
}

func (h *Handler) streamCompletions(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, completionID string, created int64, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		httputil.WriteError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}

	finish := "stop"
	promptTokens, completionTokens, totalTokens := 0, 0, 0
	for ev := range h.core.GenerateWithTools(r.Context(), env) {
		switch ev.EventType {
		case llm.EventTextDelta:
			chunk, _ := ev.Payload["delta"].(string)
			if strings.TrimSpace(chunk) == "" {
				continue
			}
			httputil.WriteSSE(w, map[string]any{
				"id":      completionID,
				"object":  "completion",
				"created": created,
				"model":   model,
				"choices": []map[string]any{{"text": chunk, "index": 0}},
			})
			flusher.Flush()
		case llm.EventFinish:
			fr, _ := ev.Payload["finishReason"].(string)
			if fr != "" {
				finish = fr
			}
			promptTokens, completionTokens, totalTokens = extractUsage(ev.Payload)
		case llm.EventError:
			msg, _ := ev.Payload["errorText"].(string)
			if msg == "" {
				msg = "unknown error"
			}
			httputil.WriteSSE(w, map[string]any{
				"id":      completionID,
				"object":  "completion",
				"created": created,
				"model":   model,
				"choices": []map[string]any{{"text": "\n[error] " + msg, "index": 0}},
			})
			flusher.Flush()
		}
	}

	httputil.WriteSSE(w, map[string]any{
		"id":      completionID,
		"object":  "completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{"text": "", "index": 0, "finish_reason": finish}},
		"usage":   map[string]int{"prompt_tokens": promptTokens, "completion_tokens": completionTokens, "total_tokens": totalTokens},
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

func (h *Handler) syncCompletions(w http.ResponseWriter, r *http.Request, env llm.LLMRequestEnvelope, completionID string, created int64, model string) {
	parts := []string{}
	finish := "stop"
	promptTokens, completionTokens, totalTokens := 0, 0, 0
	for ev := range h.core.GenerateWithTools(r.Context(), env) {
		switch ev.EventType {
		case llm.EventTextDelta:
			chunk, _ := ev.Payload["delta"].(string)
			parts = append(parts, chunk)
		case llm.EventFinish:
			if v, ok := ev.Payload["finishReason"].(string); ok && v != "" {
				finish = v
			}
			promptTokens, completionTokens, totalTokens = extractUsage(ev.Payload)
		case llm.EventError:
			msg, _ := ev.Payload["errorText"].(string)
			log.Printf("completions error: id=%s message=%s", completionID, msg)
			httputil.WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"message": msg, "type": "server_error"}})
			return
		}
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"id":      completionID,
		"object":  "completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{"text": strings.Join(parts, ""), "index": 0, "finish_reason": finish}},
		"usage":   map[string]int{"prompt_tokens": promptTokens, "completion_tokens": completionTokens, "total_tokens": totalTokens},
	})
}

func normalizeResponsesInput(input []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(input))
	for _, item := range input {
		typ, _ := item["type"].(string)
		switch typ {
		case "message":
			role, _ := item["role"].(string)
			content := item["content"]
			out = append(out, map[string]any{"role": role, "content": content})
		case "function_call":
			callID, _ := item["id"].(string)
			name, _ := item["name"].(string)
			args, _ := item["arguments"].(string)
			out = append(out, map[string]any{
				"role": "assistant",
				"tool_calls": []map[string]any{{
					"id":   callID,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": args,
					},
				}},
			})
		case "function_call_output":
			callID, _ := item["call_id"].(string)
			output, _ := item["output"].(string)
			out = append(out, map[string]any{
				"role":         "tool",
				"tool_call_id": callID,
				"content":      output,
			})
		default:
			if content, ok := item["content"].(string); ok {
				out = append(out, map[string]any{"role": "user", "content": content})
			}
		}
	}
	return out
}

func (h *Handler) handleMediaUploadInit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.media == nil || !h.media.Enabled() {
		httputil.WriteError(w, http.StatusBadRequest, "media storage is not configured (set S3_BUCKET or S3_BUCKET_TEMPLATE)")
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
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if err := h.validateUploadIntent(req.Kind, req.ContentType, req.Size); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	userID := firstNonEmpty(strings.TrimSpace(req.UserID), "default")
	bucket := h.media.BucketForUser(userID)
	if strings.TrimSpace(bucket) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "media bucket is not configured")
		return
	}
	if err := h.media.EnsureBucket(r.Context(), bucket); err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, "failed to prepare media bucket")
		return
	}
	objectKey := h.media.BuildObjectKey(userID, firstNonEmpty(strings.TrimSpace(req.SessionID), "chat"), req.FileName)
	put, err := h.media.PresignUpload(r.Context(), bucket, objectKey, req.ContentType)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{
		"bucket":     bucket,
		"object_key": put.ObjectKey,
		"upload_url": put.UploadURL,
		"headers":    put.Headers,
		"expires_at": put.ExpiresAt.Unix(),
	})
}

func (h *Handler) handleMediaUploadComplete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.media == nil || !h.media.Enabled() {
		httputil.WriteError(w, http.StatusBadRequest, "media storage is not configured (set S3_BUCKET or S3_BUCKET_TEMPLATE)")
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
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	req.ObjectKey = strings.TrimSpace(req.ObjectKey)
	if req.ObjectKey == "" {
		httputil.WriteError(w, http.StatusBadRequest, "object_key is required")
		return
	}
	userID := firstNonEmpty(strings.TrimSpace(req.UserID), "default")
	bucket := firstNonEmpty(strings.TrimSpace(req.Bucket), h.media.BucketForUser(userID))
	if strings.TrimSpace(bucket) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "bucket is required")
		return
	}
	expectedBucket := h.media.BucketForUser(userID)
	if expectedBucket != "" && bucket != expectedBucket {
		httputil.WriteError(w, http.StatusBadRequest, "bucket does not match user")
		return
	}
	info, err := h.media.HeadObject(r.Context(), bucket, req.ObjectKey)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "media object not found")
		return
	}
	ct := firstNonEmpty(strings.TrimSpace(req.ContentType), strings.TrimSpace(info.ContentType), mimeFromPath(req.ObjectKey))
	if err := h.validateUploadIntent(req.Kind, ct, info.Size); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	url, err := h.media.PresignGet(r.Context(), bucket, req.ObjectKey)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{
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
				if len(bytes) > h.mediaLimits.MaxImageBytes {
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
				if len(bytes) > h.mediaLimits.MaxAudioBytes {
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

// extractUsage extracts token usage from an EventFinish or EventFinishStep payload.
// Returns (prompt_tokens, completion_tokens, total_tokens).
// Falls back to zeros if usage is not present.
func extractUsage(payload map[string]any) (int, int, int) {
	usageRaw, ok := payload["usage"].(map[string]any)
	if !ok {
		return 0, 0, 0
	}
	prompt := toInt(usageRaw["prompt_tokens"])
	completion := toInt(usageRaw["completion_tokens"])
	total := toInt(usageRaw["total_tokens"])
	return prompt, completion, total
}

// toInt converts a numeric value (int, float64, etc.) to int.
func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
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

func (h *Handler) validateMediaParts(messages []map[string]any) error {
	maxImageBytes := h.mediaLimits.MaxImageBytes
	maxAudioBytes := h.mediaLimits.MaxAudioBytes
	maxTotalMediaBytes := h.mediaLimits.MaxTotalMediaBytes
	maxMediaParts := h.mediaLimits.MaxMediaParts

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

func (h *Handler) validateUploadIntent(kind, contentType string, size int64) error {
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
		if size > int64(h.mediaLimits.MaxImageBytes) {
			return fmt.Errorf("image exceeds size limit")
		}
		return nil
	}
	if !strings.HasPrefix(contentType, "audio/") {
		return fmt.Errorf("unsupported audio mime type")
	}
	if size > int64(h.mediaLimits.MaxAudioBytes) {
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
