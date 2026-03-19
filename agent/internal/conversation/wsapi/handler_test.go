package wsapi

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/suryaumapathy2812/core-ai/agent/internal/conversation"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type scriptedProvider struct {
	events []providers.LLMStreamEvent
}

func (p *scriptedProvider) Name() string { return "scripted" }

func (p *scriptedProvider) Capabilities() providers.ProviderCapabilities {
	return providers.DefaultCapabilities
}

func (p *scriptedProvider) StreamWithTools(ctx context.Context, opts providers.GenerateOptions) (<-chan providers.LLMStreamEvent, error) {
	_ = opts
	out := make(chan providers.LLMStreamEvent, len(p.events)+1)
	go func() {
		defer close(out)
		for _, ev := range p.events {
			select {
			case <-ctx.Done():
				return
			case out <- ev:
			}
		}
	}()
	return out, nil
}

func TestWSConversationTextTurnAndPersistence(t *testing.T) {
	t.Parallel()

	assetsDir := t.TempDir()
	store, err := db.OpenInAssets(assetsDir, "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	sess, err := store.CreateChatSession(context.Background(), "user-1", "chat")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	registry := tools.NewRegistry()
	provider := &scriptedProvider{events: []providers.LLMStreamEvent{{Type: providers.EventToken, Content: "hello"}, {Type: providers.EventToken, Content: " world"}, {Type: providers.EventDone, FinishReason: "stop"}}}
	core := llm.NewCore(provider, tools.NewOrchestrator(registry, tools.ExecContext{}))
	runtime := conversation.NewRuntime(conversation.RuntimeOptions{Core: core})
	builder := llm.NewContextBuilder(registry, nil, nil, store, llm.ContextBuilderConfig{})

	mux := http.NewServeMux()
	h := New(Options{Runtime: runtime, Builder: builder, Store: store})
	h.RegisterRoutes(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws/conversation?token=test&user_id=user-1"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial websocket: %v", err)
	}
	defer conn.Close()

	seq := map[string]int{}
	send := func(typ, sessionID, turnID string, payload map[string]any) {
		seq[turnID]++
		err := conn.WriteJSON(envelope{Version: protocolVersion, Type: typ, EventID: newEventID("cli"), SessionID: sessionID, TurnID: turnID, Seq: seq[turnID], TS: time.Now().UnixMilli(), Payload: payload})
		if err != nil {
			t.Fatalf("write %s: %v", typ, err)
		}
	}

	readUntil := func(wantType string) envelope {
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			_ = conn.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
			var ev envelope
			if err := conn.ReadJSON(&ev); err != nil {
				if websocket.IsCloseError(err, websocket.CloseNormalClosure) {
					t.Fatalf("unexpected close")
				}
				continue
			}
			if ev.Type == wantType {
				return ev
			}
		}
		t.Fatalf("timeout waiting for %s", wantType)
		return envelope{}
	}

	send("session.start", sess.ID, sessionControlTurnID, map[string]any{"user_id": "user-1"})
	_ = readUntil("session.ready")

	turnID := "turn-1"
	send("turn.start", sess.ID, turnID, map[string]any{"mode": "text"})
	_ = readUntil("turn.accepted")
	send("turn.input.text", sess.ID, turnID, map[string]any{"text": "hi there"})
	send("turn.commit", sess.ID, turnID, nil)

	firstDelta := readUntil("assistant.text.delta")
	if firstDelta.Payload["delta"] == nil {
		t.Fatalf("assistant.text.delta missing delta payload")
	}
	_ = readUntil("assistant.done")

	rows, err := store.ListChatMessages(context.Background(), "user-1", sess.ID, 20)
	if err != nil {
		t.Fatalf("list chat messages: %v", err)
	}
	if len(rows) < 2 {
		t.Fatalf("expected at least 2 chat messages, got %d", len(rows))
	}

	user := rows[len(rows)-2].Content
	assistant := rows[len(rows)-1].Content
	if role, _ := user["role"].(string); role != "user" {
		t.Fatalf("expected persisted user role, got %q", role)
	}
	if meta, _ := user["metadata"].(map[string]any); meta["input_mode"] != "text" {
		t.Fatalf("expected user input_mode text, got %#v", meta)
	}
	if role, _ := assistant["role"].(string); role != "assistant" {
		t.Fatalf("expected persisted assistant role, got %q", role)
	}
}

func TestBuildUserMessageVoiceIncludesAudioAndMetadata(t *testing.T) {
	t.Parallel()
	h := New(Options{})
	turn := &activeTurn{mode: turnModeVoice, audio: []audioChunk{{Data: []byte("abc"), MIME: "webm"}}, audioBytes: 3}

	model, stored, _, err := h.buildUserMessage(turn)
	if err != nil {
		t.Fatalf("buildUserMessage: %v", err)
	}

	meta, _ := stored["metadata"].(map[string]any)
	if meta["input_mode"] != "voice" {
		t.Fatalf("expected input_mode voice, got %#v", meta["input_mode"])
	}

	content, ok := model["content"].([]any)
	if !ok || len(content) != 2 {
		t.Fatalf("expected multimodal content with 2 parts")
	}
	second, _ := content[1].(map[string]any)
	inputAudio, _ := second["input_audio"].(map[string]any)
	data, _ := inputAudio["data"].(string)
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		t.Fatalf("decode audio: %v", err)
	}
	if string(decoded) != "abc" {
		t.Fatalf("unexpected decoded audio data: %q", string(decoded))
	}
}
