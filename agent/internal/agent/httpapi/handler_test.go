package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

func TestTaskEndpoints(t *testing.T) {
	ctx := context.Background()
	store, err := db.Open(filepath.Join(t.TempDir(), "state.db"), "")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()
	task, err := store.CreateAgentTask(ctx, db.AgentTaskCreate{UserID: "u1", SessionID: "s1", Title: "hello", Goal: "world"})
	if err != nil {
		t.Fatalf("create task: %v", err)
	}

	h := New(store, nil)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/agent/tasks?user_id=u1", nil)
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("list tasks expected 200 got %d body=%s", w.Code, w.Body.String())
	}

	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/v1/agent/tasks/"+task.ID, nil)
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("get task expected 200 got %d body=%s", w.Code, w.Body.String())
	}

	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/agent/tasks/"+task.ID+"/cancel", nil)
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("cancel task expected 200 got %d body=%s", w.Code, w.Body.String())
	}

	task2, err := store.CreateAgentTask(ctx, db.AgentTaskCreate{UserID: "u1", SessionID: "s1", Title: "approve", Goal: "need approval"})
	if err != nil {
		t.Fatalf("create task2: %v", err)
	}
	claimed, err := store.ClaimNextAgentTask(ctx, time.Now().UTC(), 20*time.Second)
	if err != nil {
		t.Fatalf("claim task2: %v", err)
	}
	if claimed.ID != task2.ID {
		t.Fatalf("claimed wrong task: %s", claimed.ID)
	}
	if err := store.SetAgentTaskWaitingInput(ctx, task2.ID, claimed.LockToken, "Approve?", map[string]any{"question": "Approve?"}); err != nil {
		t.Fatalf("set waiting input: %v", err)
	}
	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/v1/agent/tasks?user_id=u1&status=waiting_input", nil)
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("list waiting tasks expected 200 got %d body=%s", w.Code, w.Body.String())
	}
	if !bytes.Contains(w.Body.Bytes(), []byte(task2.ID)) {
		t.Fatalf("expected waiting task in filtered list body=%s", w.Body.String())
	}
	body, _ := json.Marshal(map[string]any{"user_id": "u1", "message": "approved"})
	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/agent/tasks/"+task2.ID+"/resume", bytes.NewReader(body))
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("resume task expected 200 got %d body=%s", w.Code, w.Body.String())
	}

	_, err = store.CreateAgentTask(ctx, db.AgentTaskCreate{UserID: "u1", SessionID: "s1", Title: "approve path", Goal: "need approval again", Priority: 100})
	if err != nil {
		t.Fatalf("create task3: %v", err)
	}
	claimed, err = store.ClaimNextAgentTask(ctx, time.Now().UTC(), 20*time.Second)
	if err != nil {
		t.Fatalf("claim task3: %v", err)
	}
	task3ID := claimed.ID
	if err := store.SetAgentTaskWaitingInput(ctx, task3ID, claimed.LockToken, "Approve 3?", map[string]any{"question": "Approve 3?"}); err != nil {
		t.Fatalf("set waiting input task3: %v", err)
	}
	body, _ = json.Marshal(map[string]any{"user_id": "u1", "decision": "approved", "reason": "looks good", "instructions": "proceed"})
	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/agent/tasks/"+task3ID+"/approve", bytes.NewReader(body))
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("approve task expected 200 got %d body=%s", w.Code, w.Body.String())
	}

	_, err = store.CreateAgentTask(ctx, db.AgentTaskCreate{UserID: "u1", SessionID: "s1", Title: "reject path", Goal: "need rejection", Priority: 90})
	if err != nil {
		t.Fatalf("create task4: %v", err)
	}
	claimed, err = store.ClaimNextAgentTask(ctx, time.Now().UTC(), 20*time.Second)
	if err != nil {
		t.Fatalf("claim task4: %v", err)
	}
	task4ID := claimed.ID
	if err := store.SetAgentTaskWaitingInput(ctx, task4ID, claimed.LockToken, "Approve 4?", map[string]any{"question": "Approve 4?"}); err != nil {
		t.Fatalf("set waiting input task4: %v", err)
	}
	body, _ = json.Marshal(map[string]any{"user_id": "u1", "reason": "not safe", "next_action": "ask again"})
	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/agent/tasks/"+task4ID+"/reject", bytes.NewReader(body))
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("reject task expected 200 got %d body=%s", w.Code, w.Body.String())
	}
}
