package httpapi

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/suryaumapathy2812/core-ai/agent/internal/conversation"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type Handler struct {
	runtime *conversation.Runtime
	builder *llm.ContextBuilder
	memory  *memory.Service
}

type Options struct {
	Runtime *conversation.Runtime
	Builder *llm.ContextBuilder
	Memory  *memory.Service
}

func New(opts Options) *Handler {
	return &Handler{runtime: opts.Runtime, builder: opts.Builder, memory: opts.Memory}
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

	userID := firstNonEmpty(strings.TrimSpace(req.User), "default")
	sessionID := firstNonEmpty(strings.TrimSpace(req.Session), userID)

	policy := map[string]any{}
	if req.MaxTokens != nil {
		policy["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		policy["temperature"] = *req.Temperature
	}

	env := h.builder.Build(req.Messages, policy, userID, sessionID)
	ack := conversation.BuildAckFallback(req.Messages)

	completionID := "convturn-" + uuid.NewString()[:12]
	created := time.Now().Unix()

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

	writeSSE(w, map[string]any{
		"id":      completionID,
		"object":  "conversation.turn",
		"created": created,
		"phase":   "start",
	})
	flusher.Flush()

	ackText := ""
	answerParts := []string{}
	for ev := range h.runtime.Run(r.Context(), env, conversation.RunOptions{AckFallback: ack}) {
		switch ev.EventType {
		case conversation.EventAck:
			ackText, _ = ev.Payload["text"].(string)
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "conversation.turn",
				"created": created,
				"phase":   "ack",
				"event":   ev.EventType,
				"text":    ackText,
			})
			flusher.Flush()
			time.Sleep(25 * time.Millisecond)
		case conversation.EventAnswer:
			text, _ := ev.Payload["text"].(string)
			answerParts = append(answerParts, text)
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "conversation.turn",
				"created": created,
				"phase":   "answer",
				"event":   ev.EventType,
				"text":    text,
			})
			flusher.Flush()
		case conversation.EventToolCall, conversation.EventToolResult, conversation.EventStatus:
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "conversation.turn",
				"created": created,
				"phase":   "act",
				"event":   ev.EventType,
				"payload": ev.Payload,
			})
			flusher.Flush()
		case conversation.EventError:
			msg, _ := ev.Payload["message"].(string)
			if msg == "" {
				msg = "unknown error"
			}
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "conversation.turn",
				"created": created,
				"phase":   "error",
				"event":   ev.EventType,
				"error":   msg,
			})
			flusher.Flush()
		case conversation.EventDone:
			writeSSE(w, map[string]any{
				"id":      completionID,
				"object":  "conversation.turn",
				"created": created,
				"phase":   "done",
				"event":   ev.EventType,
				"payload": ev.Payload,
			})
			flusher.Flush()
		}
	}

	if h.memory != nil {
		answer := strings.TrimSpace(strings.Join(answerParts, ""))
		if answer != "" {
			h.memory.RecordConversation(context.Background(), env.UserID, env.SessionID, llm.LatestUserTurnSummary(req.Messages), llm.LatestUserMessageContent(req.Messages), answer)
		}
	}

	log.Printf("conversation turn: user=%s session=%s", env.UserID, env.SessionID)
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
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
