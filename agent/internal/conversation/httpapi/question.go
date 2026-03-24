package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/mail"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// questionRequest represents a pending question from the agent to the user.
// The agent's ask_user tool creates one of these, blocks on answerCh,
// and the HTTP reply/reject endpoints resolve it.
type questionRequest struct {
	ID          string              `json:"id"`
	SessionID   string              `json:"session_id"`
	ToolCallID  string              `json:"tool_call_id,omitempty"`
	Question    string              `json:"question"`
	Header      string              `json:"header"`
	Kind        string              `json:"kind,omitempty"`
	Options     []questionOption    `json:"options"`
	AllowCustom bool                `json:"allow_custom"`
	Fields      []questionField     `json:"fields,omitempty"`
	SubmitLabel string              `json:"submit_label,omitempty"`
	CreatedAt   string              `json:"created_at"`
	answerCh    chan questionAnswer // internal channel, not serialized
}

// questionOption is a selectable option presented to the user.
type questionOption struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

type questionField struct {
	Name        string   `json:"name"`
	Label       string   `json:"label"`
	Type        string   `json:"type"`
	Required    bool     `json:"required,omitempty"`
	Placeholder string   `json:"placeholder,omitempty"`
	Options     []string `json:"options,omitempty"`
}

// questionAnswer carries the user's response back to the blocking tool.
type questionAnswer struct {
	Answers  []string       `json:"answers,omitempty"`
	Data     map[string]any `json:"data,omitempty"`
	Rejected bool           `json:"rejected"`
}

// questionManager tracks pending questions keyed by request ID.
// It is safe for concurrent use.
type questionManager struct {
	mu      sync.RWMutex
	pending map[string]*questionRequest
}

// newQuestionManager creates an empty question manager.
func newQuestionManager() *questionManager {
	return &questionManager{
		pending: make(map[string]*questionRequest),
	}
}

// ask stores a pending question and blocks until someone calls reply() or reject().
// The caller must populate all fields except ID and answerCh — those are set here.
// Returns the user's answer (or a rejected sentinel).
func (m *questionManager) ask(req *questionRequest) questionAnswer {
	// Assign a unique ID and create a buffered channel (size 1 so the
	// sender never blocks even if the tool goroutine hasn't selected yet).
	req.ID = uuid.New().String()
	req.answerCh = make(chan questionAnswer, 1)
	req.CreatedAt = time.Now().UTC().Format(time.RFC3339Nano)

	m.mu.Lock()
	m.pending[req.ID] = req
	m.mu.Unlock()

	// Block until the user replies or rejects.
	answer := <-req.answerCh

	// Clean up after resolution.
	m.mu.Lock()
	delete(m.pending, req.ID)
	m.mu.Unlock()

	return answer
}

// reply resolves a pending question with the given answers.
// Returns true if the question existed and was resolved.
func (m *questionManager) reply(id string, answers []string, data map[string]any) bool {
	m.mu.RLock()
	req, ok := m.pending[id]
	m.mu.RUnlock()
	if !ok {
		return false
	}
	req.answerCh <- questionAnswer{Answers: answers, Data: data, Rejected: false}
	return true
}

// reject resolves a pending question as dismissed/rejected.
// Returns true if the question existed and was resolved.
func (m *questionManager) reject(id string) bool {
	m.mu.RLock()
	req, ok := m.pending[id]
	m.mu.RUnlock()
	if !ok {
		return false
	}
	req.answerCh <- questionAnswer{Rejected: true}
	return true
}

// get returns a pending question by ID.
func (m *questionManager) get(id string) (*questionRequest, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	req, ok := m.pending[id]
	return req, ok
}

// listForSession returns all pending questions for a given session.
func (m *questionManager) listForSession(sessionID string) []*questionRequest {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []*questionRequest
	for _, req := range m.pending {
		if req.SessionID == sessionID {
			out = append(out, req)
		}
	}
	return out
}

func validateQuestionReply(req *questionRequest, answers []string, data map[string]any) ([]string, map[string]any, error) {
	if req == nil {
		return nil, nil, fmt.Errorf("question request is required")
	}

	switch strings.TrimSpace(strings.ToLower(req.Kind)) {
	case "form":
		if len(data) == 0 {
			return nil, nil, fmt.Errorf("form responses require data")
		}
		fields := map[string]questionField{}
		for _, field := range req.Fields {
			fields[field.Name] = field
		}
		if len(fields) == 0 {
			return nil, nil, fmt.Errorf("form request has no fields")
		}

		normalized := map[string]any{}
		for key, raw := range data {
			field, ok := fields[key]
			if !ok {
				return nil, nil, fmt.Errorf("unknown form field: %s", key)
			}
			value, include, err := normalizeQuestionFieldReply(field, raw)
			if err != nil {
				return nil, nil, err
			}
			if include {
				normalized[key] = value
			}
		}
		for _, field := range req.Fields {
			if !field.Required {
				continue
			}
			if _, ok := normalized[field.Name]; !ok {
				return nil, nil, fmt.Errorf("missing required field: %s", field.Name)
			}
		}
		if len(normalized) == 0 {
			return nil, nil, fmt.Errorf("form responses require at least one value")
		}
		return nil, normalized, nil
	default:
		if len(data) > 0 {
			return nil, nil, fmt.Errorf("only form prompts accept data responses")
		}
		normalized := make([]string, 0, len(answers))
		seen := map[string]struct{}{}
		for _, answer := range answers {
			answer = strings.TrimSpace(answer)
			if answer == "" {
				continue
			}
			if _, ok := seen[answer]; ok {
				continue
			}
			seen[answer] = struct{}{}
			normalized = append(normalized, answer)
		}
		if len(normalized) == 0 {
			return nil, nil, fmt.Errorf("answers array is required and must not be empty")
		}
		if strings.EqualFold(req.Kind, "confirm") && len(normalized) != 1 {
			return nil, nil, fmt.Errorf("confirm prompts require exactly one answer")
		}
		if !req.AllowCustom && len(req.Options) > 0 {
			allowed := map[string]struct{}{}
			for _, option := range req.Options {
				allowed[option.Label] = struct{}{}
			}
			for _, answer := range normalized {
				if _, ok := allowed[answer]; !ok {
					return nil, nil, fmt.Errorf("invalid answer: %s", answer)
				}
			}
		}
		return normalized, nil, nil
	}
}

func normalizeQuestionFieldReply(field questionField, raw any) (any, bool, error) {
	text := strings.TrimSpace(fmt.Sprintf("%v", raw))
	switch strings.TrimSpace(strings.ToLower(field.Type)) {
	case "number":
		if text == "" {
			if field.Required {
				return nil, false, fmt.Errorf("missing required field: %s", field.Name)
			}
			return nil, false, nil
		}
		if _, err := strconv.ParseFloat(text, 64); err != nil {
			return nil, false, fmt.Errorf("field %s must be a number", field.Name)
		}
		return text, true, nil
	case "email":
		if text == "" {
			if field.Required {
				return nil, false, fmt.Errorf("missing required field: %s", field.Name)
			}
			return nil, false, nil
		}
		if _, err := mail.ParseAddress(text); err != nil {
			return nil, false, fmt.Errorf("field %s must be a valid email", field.Name)
		}
		return text, true, nil
	case "select":
		if text == "" {
			if field.Required {
				return nil, false, fmt.Errorf("missing required field: %s", field.Name)
			}
			return nil, false, nil
		}
		for _, option := range field.Options {
			if option == text {
				return text, true, nil
			}
		}
		return nil, false, fmt.Errorf("field %s must be one of the allowed options", field.Name)
	default:
		if text == "" {
			if field.Required {
				return nil, false, fmt.Errorf("missing required field: %s", field.Name)
			}
			return nil, false, nil
		}
		return text, true, nil
	}
}

// ── QuestionAskerBridge ────────────────────────────────────────────────
// QuestionAskerBridge implements tools.QuestionAsker by delegating to a
// questionManager and emitting WS events via a notify callback.
// It is created in main.go and passed into the tools.ExecContext so the
// ask_user tool can block and emit events without importing the handler.
type QuestionAskerBridge struct {
	manager *questionManager
	notify  func(userID, eventType string, payload map[string]any)
}

// NewQuestionAskerBridge creates a bridge that connects the question manager
// to the WS notification system. Pass the handler's QuestionManager() and
// the same notify callback used by the handler.
func NewQuestionAskerBridge(mgr *questionManager, notify func(userID, eventType string, payload map[string]any)) *QuestionAskerBridge {
	return &QuestionAskerBridge{manager: mgr, notify: notify}
}

// AskQuestion implements tools.QuestionAsker. It stores the question,
// emits a "question.asked" WS event, and blocks until the user replies.
func (b *QuestionAskerBridge) AskQuestion(ctx context.Context, userID string, sessionID string, prompt tools.QuestionPrompt) (tools.QuestionResponse, error) {
	// Convert options to typed structs.
	var opts []questionOption
	for _, o := range prompt.Options {
		label := strings.TrimSpace(o.Label)
		desc := strings.TrimSpace(o.Description)
		if strings.TrimSpace(label) != "" {
			opts = append(opts, questionOption{Label: label, Description: desc})
		}
	}
	var fields []questionField
	for _, f := range prompt.Fields {
		name := strings.TrimSpace(f.Name)
		if name == "" {
			continue
		}
		label := strings.TrimSpace(f.Label)
		if label == "" {
			label = name
		}
		fieldType := strings.TrimSpace(f.Type)
		if fieldType == "" {
			fieldType = "text"
		}
		fields = append(fields, questionField{
			Name:        name,
			Label:       label,
			Type:        fieldType,
			Required:    f.Required,
			Placeholder: strings.TrimSpace(f.Placeholder),
			Options:     append([]string(nil), f.Options...),
		})
	}
	kind := strings.TrimSpace(strings.ToLower(prompt.Kind))
	if kind == "" {
		kind = "choice"
	}

	req := &questionRequest{
		SessionID:   sessionID,
		ToolCallID:  strings.TrimSpace(prompt.ToolCallID),
		Question:    prompt.Question,
		Header:      prompt.Header,
		Kind:        kind,
		Options:     opts,
		AllowCustom: prompt.AllowCustom,
		Fields:      fields,
		SubmitLabel: strings.TrimSpace(prompt.SubmitLabel),
	}

	// ask() assigns the ID and creates the channel. We need the ID
	// before blocking so we can emit the WS event. Use a goroutine
	// to emit the event after the request is stored but before blocking.
	//
	// Since ask() blocks, we run it in a goroutine and use a separate
	// channel to get the ID back before the block.
	idCh := make(chan string, 1)
	resultCh := make(chan questionAnswer, 1)

	go func() {
		// Assign ID and channel before storing.
		req.ID = uuid.New().String()
		req.answerCh = make(chan questionAnswer, 1)
		req.CreatedAt = time.Now().UTC().Format(time.RFC3339Nano)

		b.manager.mu.Lock()
		b.manager.pending[req.ID] = req
		b.manager.mu.Unlock()

		// Signal the ID so the WS event can be emitted.
		idCh <- req.ID

		// Block until user replies or context is cancelled.
		select {
		case answer := <-req.answerCh:
			// Clean up.
			b.manager.mu.Lock()
			delete(b.manager.pending, req.ID)
			b.manager.mu.Unlock()
			resultCh <- answer
		case <-ctx.Done():
			// Context cancelled — clean up and send rejection.
			b.manager.mu.Lock()
			delete(b.manager.pending, req.ID)
			b.manager.mu.Unlock()
			resultCh <- questionAnswer{Rejected: true}
		}
	}()

	// Wait for the ID to be assigned.
	questionID := <-idCh

	// Emit the "question.asked" WS event so the frontend can show the QuestionDock.
	if b.notify != nil {
		// Build serializable options for the payload.
		payloadOpts := make([]map[string]any, 0, len(opts))
		for _, o := range opts {
			entry := map[string]any{"label": o.Label}
			if o.Description != "" {
				entry["description"] = o.Description
			}
			payloadOpts = append(payloadOpts, entry)
		}
		b.notify(userID, "question.asked", map[string]any{
			"id":           questionID,
			"session_id":   sessionID,
			"tool_call_id": req.ToolCallID,
			"question":     req.Question,
			"header":       req.Header,
			"kind":         req.Kind,
			"options":      payloadOpts,
			"allow_custom": req.AllowCustom,
			"fields":       req.Fields,
			"submit_label": req.SubmitLabel,
			"created_at":   req.CreatedAt,
		})
	}

	// Block until the answer arrives.
	answer := <-resultCh

	if answer.Rejected {
		return tools.QuestionResponse{}, fmt.Errorf("question rejected")
	}
	return tools.QuestionResponse{Answers: answer.Answers, Data: answer.Data}, nil
}

// ── HTTP Handlers ──────────────────────────────────────────────────────

// handleQuestions routes question sub-paths:
//   - POST /agent/v1/questions/{id}/reply  — reply to a pending question
//   - POST /agent/v1/questions/{id}/reject — reject/dismiss a pending question
//   - GET  /agent/v1/questions/{id}        — get a specific pending question
func (h *Handler) handleQuestions(w http.ResponseWriter, r *http.Request) {
	// Parse path: /agent/v1/questions/{id}[/action]
	trimmed := trimConversationPathPrefix(r.URL.Path, agentPublicPrefix+"/questions/", legacyV1Prefix+"/questions/")
	trimmed = strings.Trim(trimmed, "/")
	parts := strings.SplitN(trimmed, "/", 2)

	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		httputil.WriteError(w, http.StatusBadRequest, "question id required")
		return
	}

	questionID := strings.TrimSpace(parts[0])
	action := ""
	if len(parts) > 1 {
		action = strings.TrimSpace(strings.ToLower(parts[1]))
	}

	switch {
	case action == "reply" && r.Method == http.MethodPost:
		h.handleQuestionReply(w, r, questionID)
	case action == "reject" && r.Method == http.MethodPost:
		h.handleQuestionReject(w, r, questionID)
	case action == "" && r.Method == http.MethodGet:
		h.handleQuestionGet(w, r, questionID)
	default:
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// handleQuestionsList handles GET /agent/v1/questions?session_id=xxx
func (h *Handler) handleQuestionsList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	sessionID := strings.TrimSpace(r.URL.Query().Get("session_id"))
	if sessionID == "" {
		httputil.WriteError(w, http.StatusBadRequest, "session_id query parameter is required")
		return
	}
	pending := h.questions.listForSession(sessionID)
	if pending == nil {
		pending = []*questionRequest{}
	}
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"questions": pending})
}

// handleQuestionReply resolves a pending question with user-provided answers.
func (h *Handler) handleQuestionReply(w http.ResponseWriter, r *http.Request, questionID string) {
	var body struct {
		Answers []string       `json:"answers"`
		Data    map[string]any `json:"data"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}
	if len(body.Answers) == 0 && len(body.Data) == 0 {
		httputil.WriteError(w, http.StatusBadRequest, "answers or data is required")
		return
	}
	req, ok := h.questions.get(questionID)
	if !ok {
		httputil.WriteError(w, http.StatusNotFound, "question not found or already answered")
		return
	}
	normalizedAnswers, normalizedData, err := validateQuestionReply(req, body.Answers, body.Data)
	if err != nil {
		httputil.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	if !h.questions.reply(questionID, normalizedAnswers, normalizedData) {
		httputil.WriteError(w, http.StatusNotFound, "question not found or already answered")
		return
	}
	h.emit(sessUserIDFromStore(h, r.Context(), req.SessionID), "question.replied", map[string]any{
		"session_id":   req.SessionID,
		"request_id":   questionID,
		"tool_call_id": req.ToolCallID,
		"updated_at":   time.Now().UTC().Format(time.RFC3339Nano),
	})
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"ok": true, "question_id": questionID})
}

// handleQuestionReject dismisses a pending question.
func (h *Handler) handleQuestionReject(w http.ResponseWriter, r *http.Request, questionID string) {
	req, ok := h.questions.get(questionID)
	if !ok {
		httputil.WriteError(w, http.StatusNotFound, "question not found or already answered")
		return
	}
	if !h.questions.reject(questionID) {
		httputil.WriteError(w, http.StatusNotFound, "question not found or already answered")
		return
	}
	h.emit(sessUserIDFromStore(h, r.Context(), req.SessionID), "question.rejected", map[string]any{
		"session_id":   req.SessionID,
		"request_id":   questionID,
		"tool_call_id": req.ToolCallID,
		"updated_at":   time.Now().UTC().Format(time.RFC3339Nano),
	})
	httputil.WriteJSON(w, http.StatusOK, map[string]any{"ok": true, "question_id": questionID, "rejected": true})
}

// handleQuestionGet returns a specific pending question.
func (h *Handler) handleQuestionGet(w http.ResponseWriter, r *http.Request, questionID string) {
	req, ok := h.questions.get(questionID)
	if !ok {
		httputil.WriteError(w, http.StatusNotFound, "question not found")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, req)
}
