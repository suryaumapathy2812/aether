package builtin

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// mockQuestionAsker implements tools.QuestionAsker for testing.
type mockQuestionAsker struct {
	answers  []string
	data     map[string]any
	rejected bool
}

func (m *mockQuestionAsker) AskQuestion(ctx context.Context, userID string, sessionID string, prompt tools.QuestionPrompt) (tools.QuestionResponse, error) {
	if m.rejected {
		return tools.QuestionResponse{}, fmt.Errorf("question rejected")
	}
	return tools.QuestionResponse{Answers: m.answers, Data: m.data}, nil
}

func TestQuestionToolDefinition(t *testing.T) {
	tool := &QuestionTool{}
	def := tool.Definition()

	if def.Name != "ask_user" {
		t.Fatalf("expected tool name 'ask_user', got '%s'", def.Name)
	}
	if def.StatusText == "" {
		t.Fatal("expected non-empty status text")
	}

	// Verify required parameters.
	hasQuestion := false
	for _, p := range def.Parameters {
		if p.Name == "question" && p.Required {
			hasQuestion = true
		}
	}
	if !hasQuestion {
		t.Fatal("expected required 'question' parameter")
	}
}

func TestQuestionToolExecuteSuccess(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{answers: []string{"React", "Vue"}}

	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question":     "Which frameworks?",
			"header":       "Frameworks",
			"allow_custom": true,
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if result.Error {
		t.Fatalf("expected success, got error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "React") || !strings.Contains(result.Output, "Vue") {
		t.Fatalf("expected answers in output, got: %s", result.Output)
	}
}

func TestQuestionToolExecuteRejected(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{rejected: true}

	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question": "Continue?",
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if !result.Error {
		t.Fatal("expected error for rejected question")
	}
	if !strings.Contains(result.Output, "dismissed") {
		t.Fatalf("expected 'dismissed' in error output, got: %s", result.Output)
	}
}

func TestQuestionToolExecuteNoAsker(t *testing.T) {
	tool := &QuestionTool{}

	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question": "Test?",
		},
		Ctx: tools.ExecContext{}, // No QuestionAsker
	})

	if !result.Error {
		t.Fatal("expected error when QuestionAsker is nil")
	}
	if !strings.Contains(result.Output, "not available") {
		t.Fatalf("expected 'not available' in error, got: %s", result.Output)
	}
}

func TestQuestionToolExecuteMissingQuestion(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{answers: []string{"x"}}

	// The ValidateArgs in safeExecute should catch this, but test the tool directly too.
	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question": "", // empty
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if !result.Error {
		t.Fatal("expected error for empty question")
	}
}

func TestQuestionToolHeaderTruncation(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{answers: []string{"ok"}}

	// Header longer than 30 chars should be truncated.
	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question": "Test?",
			"header":   "This is a very long header that exceeds thirty characters",
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if result.Error {
		t.Fatalf("expected success, got error: %s", result.Output)
	}
}

func TestQuestionToolWithOptions(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{answers: []string{"Option A"}}

	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question": "Pick one",
			"options": []any{
				map[string]any{"label": "Option A", "description": "First option"},
				map[string]any{"label": "Option B"},
			},
			"allow_custom": false,
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if result.Error {
		t.Fatalf("expected success, got error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Option A") {
		t.Fatalf("expected 'Option A' in output, got: %s", result.Output)
	}
}

func TestQuestionToolExecuteFormSuccess(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{data: map[string]any{"email": "surya@example.com", "name": "Surya"}}

	result := tool.Execute(context.Background(), tools.Call{
		ID: "call-form",
		Args: map[string]any{
			"question":     "Fill this out",
			"kind":         "form",
			"submit_label": "Save",
			"fields": []any{
				map[string]any{"name": "email", "label": "Email", "type": "email", "required": true},
				map[string]any{"name": "name", "label": "Name", "type": "text"},
			},
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if result.Error {
		t.Fatalf("expected success, got error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "surya@example.com") {
		t.Fatalf("expected form data json in output, got: %s", result.Output)
	}
	if kind, _ := result.Metadata["kind"].(string); kind != "form" {
		t.Fatalf("expected kind=form metadata, got: %#v", result.Metadata)
	}
	if _, ok := result.Metadata["data"].(map[string]any); !ok {
		t.Fatalf("expected form data metadata, got: %#v", result.Metadata)
	}
}

func TestQuestionToolExecuteFormRequiresFields(t *testing.T) {
	tool := &QuestionTool{}
	asker := &mockQuestionAsker{}

	result := tool.Execute(context.Background(), tools.Call{
		Args: map[string]any{
			"question": "Fill this out",
			"kind":     "form",
		},
		Ctx: tools.ExecContext{QuestionAsker: asker},
	})

	if !result.Error {
		t.Fatal("expected error for form prompt without fields")
	}
	if !strings.Contains(result.Output, "at least one field") {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

var _ tools.Tool = (*QuestionTool)(nil)
