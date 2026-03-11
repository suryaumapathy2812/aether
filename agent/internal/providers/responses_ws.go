package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/suryaumapathy2812/core-ai/agent/internal/config"
)

const (
	wsMaxMessageSize = 1024 * 1024 * 10
	wsMaxReadBuf     = 1024 * 512
	wsMaxWriteBuf    = 1024 * 512
)

type ResponsesWSConfig struct {
	APIKey     string
	BaseURL    string
	Model      string
	Store      bool
	MaxRetries int
}

type ResponsesWSSession struct {
	mu                 sync.RWMutex
	conn               *websocket.Conn
	config             ResponsesWSConfig
	responseID         string
	previousResponseID string
	inputItems         []map[string]any
	tools              []map[string]any
	closed             bool
}

func NewResponsesWSSession(cfg ResponsesWSConfig) *ResponsesWSSession {
	return &ResponsesWSSession{
		config: cfg,
	}
}

func (s *ResponsesWSSession) Connect(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn != nil {
		return nil
	}

	baseURL := s.config.BaseURL
	if baseURL == "" {
		baseURL = "wss://openrouter.ai/api/v1"
	}

	url := baseURL + "/responses"
	if s.config.Model != "" {
		url += "?model=" + s.config.Model
	}

	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.config.APIKey)
	header.Set("Content-Type", "application/json")

	conn, _, err := websocket.DefaultDialer.DialContext(ctx, url, header)
	if err != nil {
		return fmt.Errorf("websocket dial: %w", err)
	}

	conn.SetReadLimit(wsMaxMessageSize)
	s.conn = conn

	go s.readLoop()

	return nil
}

func (s *ResponsesWSSession) readLoop() {
	for {
		_, message, err := s.conn.ReadMessage()
		if err != nil {
			if !s.isClosed() {
				log.Printf("responses ws read error: %v", err)
			}
			s.mu.Lock()
			s.closed = true
			s.mu.Unlock()
			return
		}

		var event map[string]any
		if err := json.Unmarshal(message, &event); err != nil {
			continue
		}

		typ, _ := event["type"].(string)
		if typ == "response.created" {
			if resp, ok := event["response"].(map[string]any); ok {
				if id, ok := resp["id"].(string); ok {
					s.mu.Lock()
					s.responseID = id
					s.mu.Unlock()
				}
			}
		}
	}
}

func (s *ResponsesWSSession) isClosed() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.closed
}

func (s *ResponsesWSSession) ResponseID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.responseID
}

func (s *ResponsesWSSession) CreateResponse(ctx context.Context, input []map[string]any, tools []map[string]any, opts ResponsesOptions) (<-chan LLMStreamEvent, error) {
	s.mu.Lock()
	s.inputItems = input
	s.tools = tools
	s.mu.Unlock()

	if err := s.Connect(ctx); err != nil {
		return nil, err
	}

	req := map[string]any{
		"type":       "response.create",
		"model":      s.config.Model,
		"input":      input,
		"store":      s.config.Store,
		"max_tokens": opts.MaxTokens,
	}

	if s.previousResponseID != "" {
		req["previous_response_id"] = s.previousResponseID
	}

	if len(tools) > 0 {
		req["tools"] = convertToolsToResponses(tools)
	}

	if opts.Instructions != "" {
		req["instructions"] = opts.Instructions
	}

	if opts.Temperature > 0 {
		req["temperature"] = opts.Temperature
	}

	out := make(chan LLMStreamEvent, 64)

	go func() {
		defer close(out)

		err := s.conn.WriteJSON(req)
		if err != nil {
			out <- LLMStreamEvent{Type: EventError, Err: fmt.Errorf("write request: %w", err)}
			return
		}

		readCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		pendingCalls := map[string]*responsesFunctionCall{}

		go func() {
			for {
				_, message, err := s.conn.ReadMessage()
				if err != nil {
					cancel()
					return
				}

				var event map[string]any
				if err := json.Unmarshal(message, &event); err != nil {
					continue
				}

				typ, _ := event["type"].(string)
				switch typ {
				case "response.output_text.delta":
					content, _ := event["delta"].(string)
					if content != "" {
						out <- LLMStreamEvent{Type: EventToken, Content: content}
					}
				case "response.function_call.begin":
					callID, _ := event["id"].(string)
					name, _ := event["name"].(string)
					pendingCalls[callID] = &responsesFunctionCall{ID: callID, Name: name}
				case "response.function_call.argument.delta":
					callID, _ := event["id"].(string)
					delta, _ := event["delta"].(string)
					if pc, ok := pendingCalls[callID]; ok {
						pc.Arguments += delta
					}
				case "response.function_call.done":
					callID, _ := event["id"].(string)
					if pc, ok := pendingCalls[callID]; ok {
						args := map[string]any{}
						if strings.TrimSpace(pc.Arguments) != "" {
							_ = json.Unmarshal([]byte(pc.Arguments), &args)
						}
						out <- LLMStreamEvent{
							Type:      EventToolCalls,
							ToolCalls: []LLMToolCall{{ID: pc.ID, Name: pc.Name, Arguments: args}},
						}
						delete(pendingCalls, callID)
					}
				case "response.function_call_output":
					callID, _ := event["call_id"].(string)
					output, _ := event["output"].(string)
					out <- LLMStreamEvent{Type: EventToolResult, CallID: callID, Content: output}
				case "response.completed":
					s.mu.Lock()
					s.previousResponseID = s.responseID
					s.mu.Unlock()
					out <- LLMStreamEvent{Type: EventDone, FinishReason: "stop"}
				case "response.incomplete":
					out <- LLMStreamEvent{Type: EventDone, FinishReason: "length"}
				case "error":
					errMsg, _ := event["message"].(string)
					if errMsg == "" {
						errMsg = "unknown error"
					}
					out <- LLMStreamEvent{Type: EventError, Err: fmt.Errorf(errMsg)}
				}
			}
		}()

		<-readCtx.Done()
	}()

	return out, nil
}

func (s *ResponsesWSSession) AppendToolResult(callID, output string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		return fmt.Errorf("not connected")
	}

	req := map[string]any{
		"type": "response.create",
		"input": []map[string]any{
			{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  output,
			},
		},
		"previous_response_id": s.responseID,
	}

	return s.conn.WriteJSON(req)
}

func (s *ResponsesWSSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		return nil
	}

	err := s.conn.Close()
	s.conn = nil
	s.closed = true
	return err
}

type ResponsesWSManager struct {
	mu         sync.RWMutex
	sessions   map[string]*ResponsesWSSession
	config     ResponsesWSConfig
	httpClient *http.Client
}

func NewResponsesWSManager(cfg config.LLMConfig) *ResponsesWSManager {
	apiKey := cfg.APIKey
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL != "" && !strings.HasPrefix(baseURL, "wss://") {
		baseURL = "wss://" + strings.TrimPrefix(baseURL, "https://")
	}

	return &ResponsesWSManager{
		sessions: make(map[string]*ResponsesWSSession),
		config: ResponsesWSConfig{
			APIKey:  apiKey,
			BaseURL: baseURL,
			Model:   cfg.Model,
			Store:   false,
		},
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

func (m *ResponsesWSManager) GetOrCreateSession(sessionID string) *ResponsesWSSession {
	m.mu.Lock()
	defer m.mu.Unlock()

	if sess, ok := m.sessions[sessionID]; ok {
		return sess
	}

	sess := NewResponsesWSSession(m.config)
	m.sessions[sessionID] = sess
	return sess
}

func (m *ResponsesWSManager) RemoveSession(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if sess, ok := m.sessions[sessionID]; ok {
		sess.Close()
		delete(m.sessions, sessionID)
	}
}

func (m *ResponsesWSManager) CloseAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, sess := range m.sessions {
		sess.Close()
	}
	m.sessions = make(map[string]*ResponsesWSSession)
}
