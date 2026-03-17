package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── questionManager unit tests ─────────────────────────────────────────

func TestQuestionManagerAskReply(t *testing.T) {
	mgr := newQuestionManager()

	var answer questionAnswer
	var wg sync.WaitGroup
	wg.Add(1)

	// ask() blocks, so run it in a goroutine.
	go func() {
		defer wg.Done()
		answer = mgr.ask(&questionRequest{
			SessionID:   "sess-1",
			Question:    "Which framework?",
			Header:      "Question",
			Options:     []questionOption{{Label: "React"}, {Label: "Vue"}},
			AllowCustom: true,
		})
	}()

	// Wait for the question to appear in pending.
	var questionID string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := mgr.listForSession("sess-1")
		if len(pending) == 1 {
			questionID = pending[0].ID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if questionID == "" {
		t.Fatal("question did not appear in pending within timeout")
	}

	// Verify get() works.
	req, ok := mgr.get(questionID)
	if !ok {
		t.Fatal("get() returned false for pending question")
	}
	if req.Question != "Which framework?" {
		t.Fatalf("unexpected question text: %s", req.Question)
	}

	// Reply and unblock the ask() goroutine.
	if !mgr.reply(questionID, []string{"React"}) {
		t.Fatal("reply() returned false")
	}

	wg.Wait()

	if answer.Rejected {
		t.Fatal("expected non-rejected answer")
	}
	if len(answer.Answers) != 1 || answer.Answers[0] != "React" {
		t.Fatalf("unexpected answers: %v", answer.Answers)
	}

	// After resolution, the question should be cleaned up.
	if _, ok := mgr.get(questionID); ok {
		t.Fatal("question should be removed after reply")
	}
}

func TestQuestionManagerAskReject(t *testing.T) {
	mgr := newQuestionManager()

	var answer questionAnswer
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		answer = mgr.ask(&questionRequest{
			SessionID: "sess-2",
			Question:  "Continue?",
		})
	}()

	// Wait for the question to appear.
	var questionID string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := mgr.listForSession("sess-2")
		if len(pending) == 1 {
			questionID = pending[0].ID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if questionID == "" {
		t.Fatal("question did not appear in pending within timeout")
	}

	// Reject.
	if !mgr.reject(questionID) {
		t.Fatal("reject() returned false")
	}

	wg.Wait()

	if !answer.Rejected {
		t.Fatal("expected rejected answer")
	}
}

func TestQuestionManagerReplyNonExistent(t *testing.T) {
	mgr := newQuestionManager()
	if mgr.reply("nonexistent-id", []string{"x"}) {
		t.Fatal("reply() should return false for nonexistent question")
	}
	if mgr.reject("nonexistent-id") {
		t.Fatal("reject() should return false for nonexistent question")
	}
}

func TestQuestionManagerListForSession(t *testing.T) {
	mgr := newQuestionManager()

	// Start two questions on different sessions.
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		mgr.ask(&questionRequest{SessionID: "sess-a", Question: "Q1"})
	}()
	go func() {
		defer wg.Done()
		mgr.ask(&questionRequest{SessionID: "sess-b", Question: "Q2"})
	}()

	// Wait for both to appear.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		a := mgr.listForSession("sess-a")
		b := mgr.listForSession("sess-b")
		if len(a) == 1 && len(b) == 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	sessA := mgr.listForSession("sess-a")
	sessB := mgr.listForSession("sess-b")
	if len(sessA) != 1 {
		t.Fatalf("expected 1 question for sess-a, got %d", len(sessA))
	}
	if len(sessB) != 1 {
		t.Fatalf("expected 1 question for sess-b, got %d", len(sessB))
	}
	if sessA[0].Question != "Q1" {
		t.Fatalf("unexpected question for sess-a: %s", sessA[0].Question)
	}
	if sessB[0].Question != "Q2" {
		t.Fatalf("unexpected question for sess-b: %s", sessB[0].Question)
	}

	// Clean up.
	mgr.reject(sessA[0].ID)
	mgr.reject(sessB[0].ID)
	wg.Wait()
}

// ── QuestionAskerBridge tests ──────────────────────────────────────────

func TestQuestionAskerBridgeAskAndReply(t *testing.T) {
	mgr := newQuestionManager()
	var notifiedPayload map[string]any
	var notifiedEvent string
	var mu sync.Mutex

	notify := func(userID, eventType string, payload map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		notifiedEvent = eventType
		notifiedPayload = payload
	}

	bridge := NewQuestionAskerBridge(mgr, notify)

	var answers []string
	var askErr error
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		answers, askErr = bridge.AskQuestion(
			context.Background(),
			"sess-bridge",
			"Pick a color",
			"Color",
			[]map[string]any{
				{"label": "Red", "description": "Warm"},
				{"label": "Blue"},
			},
			true,
		)
	}()

	// Wait for the question to appear.
	var questionID string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := mgr.listForSession("sess-bridge")
		if len(pending) == 1 {
			questionID = pending[0].ID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if questionID == "" {
		t.Fatal("question did not appear in pending within timeout")
	}

	// Verify the WS event was emitted.
	mu.Lock()
	if notifiedEvent != "question.asked" {
		t.Fatalf("expected question.asked event, got: %s", notifiedEvent)
	}
	if notifiedPayload["question"] != "Pick a color" {
		t.Fatalf("unexpected question in payload: %v", notifiedPayload["question"])
	}
	if notifiedPayload["header"] != "Color" {
		t.Fatalf("unexpected header in payload: %v", notifiedPayload["header"])
	}
	mu.Unlock()

	// Reply.
	if !mgr.reply(questionID, []string{"Blue"}) {
		t.Fatal("reply() returned false")
	}

	wg.Wait()

	if askErr != nil {
		t.Fatalf("unexpected error: %v", askErr)
	}
	if len(answers) != 1 || answers[0] != "Blue" {
		t.Fatalf("unexpected answers: %v", answers)
	}
}

func TestQuestionAskerBridgeContextCancellation(t *testing.T) {
	mgr := newQuestionManager()
	bridge := NewQuestionAskerBridge(mgr, nil)

	ctx, cancel := context.WithCancel(context.Background())

	var answers []string
	var askErr error
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		answers, askErr = bridge.AskQuestion(ctx, "sess-cancel", "Will this cancel?", "Test", nil, true)
	}()

	// Wait for the question to appear.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := mgr.listForSession("sess-cancel")
		if len(pending) == 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Cancel the context.
	cancel()

	wg.Wait()

	if askErr == nil {
		t.Fatal("expected error from context cancellation")
	}
	if answers != nil {
		t.Fatalf("expected nil answers, got: %v", answers)
	}
}

// ── HTTP endpoint tests ────────────────────────────────────────────────

func newTestHandler() *Handler {
	return &Handler{
		questions: newQuestionManager(),
		notify:    func(userID, eventType string, payload map[string]any) {},
	}
}

func TestHandleQuestionReplyEndpoint(t *testing.T) {
	h := newTestHandler()

	// Start a question in the background.
	var answer questionAnswer
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		answer = h.questions.ask(&questionRequest{
			SessionID: "sess-http",
			Question:  "Which DB?",
		})
	}()

	// Wait for the question to appear.
	var questionID string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := h.questions.listForSession("sess-http")
		if len(pending) == 1 {
			questionID = pending[0].ID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if questionID == "" {
		t.Fatal("question did not appear")
	}

	// POST /v1/questions/{id}/reply
	body := `{"answers": ["PostgreSQL"]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/questions/"+questionID+"/reply", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.handleQuestions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	wg.Wait()

	if answer.Rejected {
		t.Fatal("expected non-rejected answer")
	}
	if len(answer.Answers) != 1 || answer.Answers[0] != "PostgreSQL" {
		t.Fatalf("unexpected answers: %v", answer.Answers)
	}
}

func TestHandleQuestionRejectEndpoint(t *testing.T) {
	h := newTestHandler()

	var answer questionAnswer
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		answer = h.questions.ask(&questionRequest{
			SessionID: "sess-reject",
			Question:  "Proceed?",
		})
	}()

	var questionID string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := h.questions.listForSession("sess-reject")
		if len(pending) == 1 {
			questionID = pending[0].ID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if questionID == "" {
		t.Fatal("question did not appear")
	}

	// POST /v1/questions/{id}/reject
	req := httptest.NewRequest(http.MethodPost, "/v1/questions/"+questionID+"/reject", nil)
	w := httptest.NewRecorder()
	h.handleQuestions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	wg.Wait()

	if !answer.Rejected {
		t.Fatal("expected rejected answer")
	}
}

func TestHandleQuestionGetEndpoint(t *testing.T) {
	h := newTestHandler()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		h.questions.ask(&questionRequest{
			SessionID:   "sess-get",
			Question:    "What color?",
			Header:      "Preference",
			Options:     []questionOption{{Label: "Red"}, {Label: "Blue"}},
			AllowCustom: false,
		})
	}()

	var questionID string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := h.questions.listForSession("sess-get")
		if len(pending) == 1 {
			questionID = pending[0].ID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if questionID == "" {
		t.Fatal("question did not appear")
	}

	// GET /v1/questions/{id}
	req := httptest.NewRequest(http.MethodGet, "/v1/questions/"+questionID, nil)
	w := httptest.NewRecorder()
	h.handleQuestions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var result map[string]any
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if result["question"] != "What color?" {
		t.Fatalf("unexpected question: %v", result["question"])
	}
	if result["header"] != "Preference" {
		t.Fatalf("unexpected header: %v", result["header"])
	}

	// Clean up.
	h.questions.reject(questionID)
	wg.Wait()
}

func TestHandleQuestionsListEndpoint(t *testing.T) {
	h := newTestHandler()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		h.questions.ask(&questionRequest{
			SessionID: "sess-list",
			Question:  "Q1",
		})
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		pending := h.questions.listForSession("sess-list")
		if len(pending) == 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// GET /v1/questions?session_id=sess-list
	req := httptest.NewRequest(http.MethodGet, "/v1/questions?session_id=sess-list", nil)
	w := httptest.NewRecorder()
	h.handleQuestionsList(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var result map[string]any
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	questions, ok := result["questions"].([]any)
	if !ok || len(questions) != 1 {
		t.Fatalf("expected 1 question, got: %v", result["questions"])
	}

	// Clean up.
	pending := h.questions.listForSession("sess-list")
	if len(pending) > 0 {
		h.questions.reject(pending[0].ID)
	}
	wg.Wait()
}

func TestHandleQuestionReplyNotFound(t *testing.T) {
	h := newTestHandler()

	body := `{"answers": ["x"]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/questions/nonexistent/reply", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.handleQuestions(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleQuestionRejectNotFound(t *testing.T) {
	h := newTestHandler()

	req := httptest.NewRequest(http.MethodPost, "/v1/questions/nonexistent/reject", nil)
	w := httptest.NewRecorder()
	h.handleQuestions(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleQuestionsListMissingSessionID(t *testing.T) {
	h := newTestHandler()

	req := httptest.NewRequest(http.MethodGet, "/v1/questions", nil)
	w := httptest.NewRecorder()
	h.handleQuestionsList(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleQuestionReplyEmptyAnswers(t *testing.T) {
	h := newTestHandler()

	body := `{"answers": []}`
	req := httptest.NewRequest(http.MethodPost, "/v1/questions/some-id/reply", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.handleQuestions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}
