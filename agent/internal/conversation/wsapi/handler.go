package wsapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
	agentcfg "github.com/suryaumapathy2812/core-ai/agent/internal/config"
	"github.com/suryaumapathy2812/core-ai/agent/internal/conversation"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

const (
	protocolVersion      = 1
	eventWindowLimit     = 1024
	maxAudioChunkBytes   = 512 * 1024
	defaultMaxAudioBytes = 12 * 1024 * 1024
	sessionControlTurnID = "session"
)

type Handler struct {
	runtime   *conversation.Runtime
	builder   *llm.ContextBuilder
	memory    *memory.Service
	store     *db.Store
	limits    agentcfg.MediaLimitsConfig
	notify    func(userID, eventType string, payload map[string]any)
	validator *agentauth.Validator

	upgrader websocket.Upgrader
}

type Options struct {
	Runtime   *conversation.Runtime
	Builder   *llm.ContextBuilder
	Memory    *memory.Service
	Store     *db.Store
	Limits    agentcfg.MediaLimitsConfig
	Notify    func(userID, eventType string, payload map[string]any)
	Validator *agentauth.Validator
}

type envelope struct {
	Version   int            `json:"v"`
	Type      string         `json:"type"`
	EventID   string         `json:"event_id"`
	SessionID string         `json:"session_id"`
	TurnID    string         `json:"turn_id"`
	Seq       int            `json:"seq"`
	TS        int64          `json:"ts"`
	Payload   map[string]any `json:"payload,omitempty"`
}

type turnMode string

const (
	turnModeText  turnMode = "text"
	turnModeVoice turnMode = "voice"
)

type audioChunk struct {
	Data []byte
	MIME string
}

type activeTurn struct {
	id         string
	mode       turnMode
	textParts  []string
	audio      []audioChunk
	audioBytes int
	running    bool
	cancel     context.CancelFunc
	cancelled  atomic.Bool
}

type connectionState struct {
	mu sync.Mutex

	userID string

	seenEventIDs  map[string]struct{}
	seenEventList []string

	lastSeqByTurn map[string]int
	outSeqByTurn  map[string]int
	outstanding   map[string]time.Time

	activeBySession map[string]*activeTurn
}

func New(opts Options) *Handler {
	return &Handler{
		runtime:   opts.Runtime,
		builder:   opts.Builder,
		memory:    opts.Memory,
		store:     opts.Store,
		limits:    opts.Limits,
		notify:    opts.Notify,
		validator: opts.Validator,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin:     func(r *http.Request) bool { return true },
		},
	}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	for _, path := range []string{"/agent/v1/ws/conversation", "/ws/conversation"} {
		mux.HandleFunc(path, h.handleWS)
	}
}

func (h *Handler) handleWS(w http.ResponseWriter, r *http.Request) {
	token := strings.TrimSpace(r.URL.Query().Get("token"))
	if token == "" {
		http.Error(w, "missing token", http.StatusUnauthorized)
		return
	}
	userID, ok := h.authorizeRequest(r, token)
	if !ok {
		http.Error(w, "invalid token", http.StatusUnauthorized)
		return
	}

	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("conversation ws: upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	state := &connectionState{
		userID:          userID,
		seenEventIDs:    map[string]struct{}{},
		lastSeqByTurn:   map[string]int{},
		outSeqByTurn:    map[string]int{},
		outstanding:     map[string]time.Time{},
		activeBySession: map[string]*activeTurn{},
		seenEventList:   make([]string, 0, eventWindowLimit),
	}

	writeMu := &sync.Mutex{}
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			h.cancelAll(state)
			return
		}

		var in envelope
		if err := json.Unmarshal(raw, &in); err != nil {
			h.writeError(conn, writeMu, state, "", "", "validation", "invalid envelope json")
			continue
		}
		in.EventID = strings.TrimSpace(in.EventID)
		in.SessionID = strings.TrimSpace(in.SessionID)
		in.TurnID = strings.TrimSpace(in.TurnID)
		in.Type = strings.TrimSpace(in.Type)

		if in.EventID == "" || in.SessionID == "" || in.TurnID == "" || in.Type == "" {
			h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "missing required envelope fields")
			continue
		}

		if duplicate := h.markSeen(state, in.EventID); duplicate {
			if in.Type != "ack" {
				h.writeAck(conn, writeMu, state, in.SessionID, in.TurnID, in.EventID)
			}
			continue
		}

		if in.Type != "ack" {
			h.writeAck(conn, writeMu, state, in.SessionID, in.TurnID, in.EventID)
		}

		if err := h.validateEnvelope(state, in); err != nil {
			h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", err.Error())
			continue
		}

		h.applyClientSeq(state, in)

		switch in.Type {
		case "ack":
			h.handleAck(state, in)
		case "session.start":
			h.handleSessionStart(conn, writeMu, state, in)
		case "session.stop":
			h.handleSessionStop(conn, writeMu, state, in)
		case "turn.start":
			h.handleTurnStart(conn, writeMu, state, in)
		case "turn.input.text":
			h.handleTurnInputText(conn, writeMu, state, in)
		case "turn.input.audio.chunk":
			h.handleTurnInputAudio(conn, writeMu, state, in)
		case "turn.commit":
			h.handleTurnCommit(conn, writeMu, state, in)
		case "turn.cancel":
			h.handleTurnCancel(conn, writeMu, state, in)
		default:
			h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "unsupported event type")
		}
	}
}

func (h *Handler) authorizeRequest(r *http.Request, token string) (string, bool) {
	if h.validator != nil && (agentauth.RequiresDirectToken(r) || agentauth.IsDirectToken(token)) {
		claims, err := h.validator.ValidateRequest(r)
		if err != nil {
			return "", false
		}
		return claims.UserID, true
	}
	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
	if userID == "" {
		userID = "default"
	}
	return userID, true
}

func (h *Handler) validateEnvelope(state *connectionState, in envelope) error {
	if in.Version != 0 && in.Version != protocolVersion {
		return fmt.Errorf("unsupported protocol version")
	}
	if in.Seq <= 0 {
		return fmt.Errorf("seq must be > 0")
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if last := state.lastSeqByTurn[in.TurnID]; in.Seq <= last {
		return fmt.Errorf("seq must be monotonic per turn")
	}
	return nil
}

func (h *Handler) applyClientSeq(state *connectionState, in envelope) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.lastSeqByTurn[in.TurnID] = in.Seq
}

func (h *Handler) markSeen(state *connectionState, eventID string) bool {
	state.mu.Lock()
	defer state.mu.Unlock()
	if _, ok := state.seenEventIDs[eventID]; ok {
		return true
	}
	state.seenEventIDs[eventID] = struct{}{}
	state.seenEventList = append(state.seenEventList, eventID)
	if len(state.seenEventList) > eventWindowLimit {
		evict := state.seenEventList[0]
		state.seenEventList = state.seenEventList[1:]
		delete(state.seenEventIDs, evict)
	}
	return false
}

func (h *Handler) handleAck(state *connectionState, in envelope) {
	acked, _ := in.Payload["event_id"].(string)
	acked = strings.TrimSpace(acked)
	if acked == "" {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	delete(state.outstanding, acked)
}

func (h *Handler) handleSessionStart(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	h.emit(conn, writeMu, state, in.SessionID, in.TurnID, "session.ready", map[string]any{
		"user_id": state.userID,
	})
}

func (h *Handler) handleSessionStop(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	h.cancelSessionTurn(state, in.SessionID)
	h.emit(conn, writeMu, state, in.SessionID, in.TurnID, "session.ready", map[string]any{"stopped": true})
}

func (h *Handler) handleTurnStart(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	mode := turnModeText
	if v, _ := in.Payload["mode"].(string); strings.EqualFold(strings.TrimSpace(v), string(turnModeVoice)) {
		mode = turnModeVoice
	}
	if v, _ := in.Payload["mode"].(string); v != "" {
		norm := strings.ToLower(strings.TrimSpace(v))
		if norm != string(turnModeText) && norm != string(turnModeVoice) {
			h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "turn.start mode must be text or voice")
			return
		}
	}

	state.mu.Lock()
	if current, ok := state.activeBySession[in.SessionID]; ok && current != nil {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "turn_conflict", "a turn is already active for this session")
		return
	}
	state.activeBySession[in.SessionID] = &activeTurn{id: in.TurnID, mode: mode}
	state.mu.Unlock()
	h.emit(conn, writeMu, state, in.SessionID, in.TurnID, "turn.accepted", map[string]any{"mode": string(mode)})
}

func (h *Handler) handleTurnInputText(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	text, _ := in.Payload["text"].(string)
	text = strings.TrimSpace(text)
	if text == "" {
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "turn.input.text payload.text is required")
		return
	}

	state.mu.Lock()
	turn, ok := state.activeBySession[in.SessionID]
	if !ok || turn == nil || turn.id != in.TurnID {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "turn_conflict", "no matching active turn")
		return
	}
	if turn.mode != turnModeText {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "turn.input.text requires text mode")
		return
	}
	turn.textParts = append(turn.textParts, text)
	state.mu.Unlock()
}

func (h *Handler) handleTurnInputAudio(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	audioB64, _ := in.Payload["audio"].(string)
	audioB64 = strings.TrimSpace(audioB64)
	if audioB64 == "" {
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "turn.input.audio.chunk payload.audio is required")
		return
	}
	data, err := base64.StdEncoding.DecodeString(audioB64)
	if err != nil {
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "turn.input.audio.chunk payload.audio must be base64")
		return
	}
	if len(data) == 0 {
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "audio chunk cannot be empty")
		return
	}
	if len(data) > maxAudioChunkBytes {
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "unsupported_audio", "audio chunk exceeds max size")
		return
	}

	mime, _ := in.Payload["mime_type"].(string)
	mime = strings.TrimSpace(strings.ToLower(mime))
	if mime == "" {
		mime = "audio/webm"
	}
	format, ok := audioFormatFromMime(mime)
	if !ok {
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "unsupported_audio", "unsupported audio mime_type")
		return
	}

	state.mu.Lock()
	turn, okTurn := state.activeBySession[in.SessionID]
	if !okTurn || turn == nil || turn.id != in.TurnID {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "turn_conflict", "no matching active turn")
		return
	}
	if turn.mode != turnModeVoice {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "turn.input.audio.chunk requires voice mode")
		return
	}
	maxAudioBytes := h.limits.MaxAudioBytes
	if maxAudioBytes <= 0 {
		maxAudioBytes = defaultMaxAudioBytes
	}
	if turn.audioBytes+len(data) > maxAudioBytes {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "unsupported_audio", "audio payload exceeds max bytes")
		return
	}
	turn.audio = append(turn.audio, audioChunk{Data: data, MIME: format})
	turn.audioBytes += len(data)
	state.mu.Unlock()
}

func (h *Handler) handleTurnCommit(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	state.mu.Lock()
	turn, ok := state.activeBySession[in.SessionID]
	if !ok || turn == nil || turn.id != in.TurnID {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "turn_conflict", "no matching active turn")
		return
	}
	if turn.running {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "turn_conflict", "turn already committed")
		return
	}

	if turn.mode == turnModeText && len(turn.textParts) == 0 {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "text turn has no input")
		return
	}
	if turn.mode == turnModeVoice && len(turn.audio) == 0 {
		state.mu.Unlock()
		h.writeError(conn, writeMu, state, in.SessionID, in.TurnID, "validation", "voice turn has no audio")
		return
	}

	runCtx, cancel := context.WithCancel(context.Background())
	turn.running = true
	turn.cancel = cancel
	state.mu.Unlock()

	go h.runTurn(runCtx, conn, writeMu, state, in.SessionID, in.TurnID, turn)
}

func (h *Handler) handleTurnCancel(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, in envelope) {
	state.mu.Lock()
	turn, ok := state.activeBySession[in.SessionID]
	if !ok || turn == nil || turn.id != in.TurnID {
		state.mu.Unlock()
		return
	}
	if turn.cancel != nil {
		turn.cancelled.Store(true)
		turn.cancel()
	}
	delete(state.activeBySession, in.SessionID)
	state.mu.Unlock()

	h.emit(conn, writeMu, state, in.SessionID, in.TurnID, "turn.cancelled", map[string]any{"reason": "client_cancel"})
	h.emitSessionIdle(state.userID, in.SessionID)
}

func (h *Handler) runTurn(ctx context.Context, conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, sessionID, turnID string, turn *activeTurn) {
	defer func() {
		state.mu.Lock()
		current := state.activeBySession[sessionID]
		if current == turn {
			delete(state.activeBySession, sessionID)
		}
		state.mu.Unlock()
		if !turn.cancelled.Load() {
			h.emitSessionIdle(state.userID, sessionID)
		}
	}()

	if h.runtime == nil || h.builder == nil {
		h.writeError(conn, writeMu, state, sessionID, turnID, "internal", "conversation runtime unavailable")
		return
	}

	latestUserModel, latestUserStore, userSummary, err := h.buildUserMessage(turn)
	if err != nil {
		h.writeError(conn, writeMu, state, sessionID, turnID, "validation", err.Error())
		return
	}

	messages := []map[string]any{}
	if h.store != nil {
		history := h.loadConversationHistory(ctx, state.userID, sessionID)
		messages = append(messages, history...)
		if err := h.store.AppendChatMessage(ctx, state.userID, sessionID, latestUserStore); err == nil {
			h.emitNotify(state.userID, "message.updated", map[string]any{"sessionID": sessionID, "role": "user", "updatedAt": nowRFC3339Nano()})
		}
		_ = h.store.TouchChatSession(ctx, sessionID)
	}
	messages = append(messages, latestUserModel)

	h.emitSessionBusy(state.userID, sessionID)

	policy := map[string]any{}
	if h.store != nil {
		if modelPref, err := h.store.GetUserPreference(ctx, state.userID, "model"); err == nil && strings.TrimSpace(modelPref) != "" {
			policy["model"] = strings.TrimSpace(modelPref)
		}
	}
	env := h.builder.Build(messages, policy, state.userID, sessionID)
	rCtx := tools.WithTaskRuntimeContext(ctx, tools.TaskRuntimeContext{UserID: state.userID})
	answerParts := []string{}
	pendingToolCalls := []map[string]any{}
	pendingToolCallIDs := map[string]struct{}{}
	assistantFlushedForToolBatch := false

	for ev := range h.runtime.Run(rCtx, env, conversation.RunOptions{}) {
		if turn.cancelled.Load() {
			return
		}
		select {
		case <-ctx.Done():
			return
		default:
		}

		switch ev.EventType {
		case conversation.EventTextDelta:
			delta, _ := ev.Payload["delta"].(string)
			if strings.TrimSpace(delta) == "" {
				continue
			}
			answerParts = append(answerParts, delta)
			h.emit(conn, writeMu, state, sessionID, turnID, "assistant.text.delta", map[string]any{"delta": delta})
			h.emitNotify(state.userID, "message.part.delta", map[string]any{"sessionID": sessionID, "messageID": "assistant-current", "partID": "text", "delta": delta, "updatedAt": nowRFC3339Nano()})

		case conversation.EventToolInputAvailable:
			name, _ := ev.Payload["toolName"].(string)
			callID, _ := ev.Payload["toolCallId"].(string)
			input := ev.Payload["input"]
			h.emit(conn, writeMu, state, sessionID, turnID, "assistant.tool-input-available", map[string]any{
				"toolName":   name,
				"toolCallId": callID,
				"input":      input,
			})
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
			callID, _ := ev.Payload["toolCallId"].(string)
			output, _ := ev.Payload["output"].(string)
			errorText, _ := ev.Payload["errorText"].(string)
			if ev.EventType == conversation.EventType("tool-output-error") {
				h.emit(conn, writeMu, state, sessionID, turnID, "assistant.tool-output-error", map[string]any{
					"toolCallId": callID,
					"errorText":  errorText,
				})
			} else {
				h.emit(conn, writeMu, state, sessionID, turnID, "assistant.tool-output-available", map[string]any{
					"toolCallId": callID,
					"output":     output,
				})
			}
			if h.store != nil && !assistantFlushedForToolBatch && len(pendingToolCalls) > 0 {
				_ = h.store.AppendChatMessage(ctx, state.userID, sessionID, map[string]any{
					"role":       "assistant",
					"tool_calls": pendingToolCalls,
				})
				assistantFlushedForToolBatch = true
			}
			if h.store != nil {
				content := output
				if ev.EventType == conversation.EventType("tool-output-error") {
					content = "[tool_error] " + errorText
				}
				if strings.TrimSpace(content) != "" {
					_ = h.store.AppendChatMessage(ctx, state.userID, sessionID, map[string]any{
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

		case conversation.EventError:
			errText, _ := ev.Payload["errorText"].(string)
			if strings.TrimSpace(errText) == "" {
				errText = "conversation error"
			}
			h.writeError(conn, writeMu, state, sessionID, turnID, "internal", errText)
			return
		}
	}

	if turn.cancelled.Load() || ctx.Err() != nil {
		return
	}

	answer := strings.TrimSpace(strings.Join(answerParts, ""))
	if h.memory != nil && answer != "" {
		h.memory.RecordConversation(context.Background(), state.userID, sessionID, userSummary, latestUserModel["content"], answer)
	}
	if answer != "" && h.store != nil {
		_ = h.store.AppendChatMessage(ctx, state.userID, sessionID, map[string]any{"role": "assistant", "content": answer})
		h.emitNotify(state.userID, "message.updated", map[string]any{"sessionID": sessionID, "role": "assistant", "updatedAt": nowRFC3339Nano()})
	}

	h.emit(conn, writeMu, state, sessionID, turnID, "assistant.done", map[string]any{"status": "completed"})
}

func (h *Handler) buildUserMessage(turn *activeTurn) (model map[string]any, stored map[string]any, summary string, err error) {
	if turn == nil {
		return nil, nil, "", fmt.Errorf("turn is required")
	}
	if turn.mode == turnModeText {
		text := strings.TrimSpace(strings.Join(turn.textParts, "\n"))
		if text == "" {
			return nil, nil, "", fmt.Errorf("text turn has no input")
		}
		msg := map[string]any{
			"role":    "user",
			"content": text,
			"metadata": map[string]any{
				"input_mode": "text",
			},
		}
		return cloneMap(msg), msg, text, nil
	}

	if len(turn.audio) == 0 {
		return nil, nil, "", fmt.Errorf("voice turn has no audio")
	}
	buf := make([]byte, 0, turn.audioBytes)
	format := "wav"
	for _, chunk := range turn.audio {
		buf = append(buf, chunk.Data...)
		if chunk.MIME != "" {
			format = chunk.MIME
		}
	}
	encoded := base64.StdEncoding.EncodeToString(buf)
	content := []any{
		map[string]any{"type": "text", "text": "[voice instruction]"},
		map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": encoded, "format": format}},
	}
	msg := map[string]any{
		"role":    "user",
		"content": content,
		"metadata": map[string]any{
			"input_mode": "voice",
		},
	}
	return cloneMap(msg), msg, "[voice instruction]", nil
}

func (h *Handler) cancelAll(state *connectionState) {
	state.mu.Lock()
	defer state.mu.Unlock()
	for _, turn := range state.activeBySession {
		if turn == nil || turn.cancel == nil {
			continue
		}
		turn.cancelled.Store(true)
		turn.cancel()
	}
	state.activeBySession = map[string]*activeTurn{}
}

func (h *Handler) cancelSessionTurn(state *connectionState, sessionID string) {
	state.mu.Lock()
	defer state.mu.Unlock()
	turn := state.activeBySession[sessionID]
	if turn == nil {
		return
	}
	if turn.cancel != nil {
		turn.cancelled.Store(true)
		turn.cancel()
	}
	delete(state.activeBySession, sessionID)
}

func (h *Handler) emit(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, sessionID, turnID, typ string, payload map[string]any) {
	if strings.TrimSpace(sessionID) == "" {
		return
	}
	if strings.TrimSpace(turnID) == "" {
		turnID = sessionControlTurnID
	}
	if payload == nil {
		payload = map[string]any{}
	}

	state.mu.Lock()
	seq := state.outSeqByTurn[turnID] + 1
	state.outSeqByTurn[turnID] = seq
	state.mu.Unlock()

	out := envelope{
		Version:   protocolVersion,
		Type:      typ,
		EventID:   newEventID("srv"),
		SessionID: sessionID,
		TurnID:    turnID,
		Seq:       seq,
		TS:        time.Now().UnixMilli(),
		Payload:   payload,
	}

	writeMu.Lock()
	err := conn.WriteJSON(out)
	writeMu.Unlock()
	if err != nil {
		return
	}

	if typ != "ack" {
		state.mu.Lock()
		state.outstanding[out.EventID] = time.Now().UTC()
		state.mu.Unlock()
	}
}

func (h *Handler) writeAck(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, sessionID, turnID, eventID string) {
	h.emit(conn, writeMu, state, sessionID, turnID, "ack", map[string]any{"event_id": eventID})
}

func (h *Handler) writeError(conn *websocket.Conn, writeMu *sync.Mutex, state *connectionState, sessionID, turnID, code, message string) {
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "unknown"
	}
	if strings.TrimSpace(turnID) == "" {
		turnID = sessionControlTurnID
	}
	if strings.TrimSpace(code) == "" {
		code = "internal"
	}
	if strings.TrimSpace(message) == "" {
		message = "unexpected error"
	}
	h.emit(conn, writeMu, state, sessionID, turnID, "error", map[string]any{
		"code":    code,
		"message": message,
	})
}

func (h *Handler) emitNotify(userID, eventType string, payload map[string]any) {
	if h == nil || h.notify == nil || strings.TrimSpace(userID) == "" || strings.TrimSpace(eventType) == "" {
		return
	}
	h.notify(userID, eventType, payload)
}

func (h *Handler) emitSessionBusy(userID, sessionID string) {
	h.emitNotify(userID, "session.status", map[string]any{
		"sessionID": sessionID,
		"status":    map[string]any{"type": "busy"},
		"updatedAt": nowRFC3339Nano(),
	})
}

func (h *Handler) emitSessionIdle(userID, sessionID string) {
	h.emitNotify(userID, "session.status", map[string]any{
		"sessionID": sessionID,
		"status":    map[string]any{"type": "idle"},
		"updatedAt": nowRFC3339Nano(),
	})
}

func (h *Handler) loadConversationHistory(ctx context.Context, userID, sessionID string) []map[string]any {
	if h == nil || h.store == nil {
		return []map[string]any{}
	}
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

func audioFormatFromMime(mime string) (string, bool) {
	mime = strings.ToLower(strings.TrimSpace(mime))
	switch {
	case strings.Contains(mime, "wav"):
		return "wav", true
	case strings.Contains(mime, "mpeg") || strings.Contains(mime, "mp3"):
		return "mp3", true
	case strings.Contains(mime, "ogg"):
		return "ogg", true
	case strings.Contains(mime, "flac"):
		return "flac", true
	case strings.Contains(mime, "aac"):
		return "aac", true
	case strings.Contains(mime, "aiff"):
		return "aiff", true
	case strings.Contains(mime, "mp4") || strings.Contains(mime, "m4a"):
		return "m4a", true
	case strings.Contains(mime, "webm"):
		return "webm", true
	default:
		return "", false
	}
}

func cloneMap(in map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range in {
		out[k] = v
	}
	return out
}

func nowRFC3339Nano() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

var eventCounter uint64

func newEventID(prefix string) string {
	if strings.TrimSpace(prefix) == "" {
		prefix = "evt"
	}
	n := atomic.AddUint64(&eventCounter, 1)
	return fmt.Sprintf("%s-%d-%d", prefix, time.Now().UnixNano(), n)
}
