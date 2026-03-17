package builtin

import (
	"context"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// QuestionTool implements the ask_user tool, which allows the agent to pose
// a blocking question to the user when it cannot proceed without input.
// The tool blocks until the user replies or dismisses the question.
type QuestionTool struct{}

func (t *QuestionTool) Definition() tools.Definition {
	return tools.Definition{
		Name: "ask_user",
		Description: "Ask the user a question when you are truly blocked and cannot proceed without their input. " +
			"Provide clear options when possible. Only use this when you cannot resolve the ambiguity using available tools.",
		StatusText: "Waiting for your response...",
		Parameters: []tools.Param{
			{
				Name:        "question",
				Type:        "string",
				Required:    true,
				Description: "The question to ask the user",
			},
			{
				Name:        "header",
				Type:        "string",
				Required:    false,
				Default:     "Question",
				Description: "Short header label (max 30 chars)",
			},
			{
				Name:        "options",
				Type:        "array",
				Required:    false,
				Description: "Selectable options for the user to choose from",
				Items: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"label":       map[string]any{"type": "string"},
						"description": map[string]any{"type": "string"},
					},
				},
			},
			{
				Name:        "allow_custom",
				Type:        "boolean",
				Required:    false,
				Default:     true,
				Description: "Whether the user can type a custom answer instead of selecting an option",
			},
		},
	}
}

func (t *QuestionTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	// Extract arguments.
	question, _ := call.Args["question"].(string)
	question = strings.TrimSpace(question)
	if question == "" {
		return tools.Fail("question is required", nil)
	}

	header, _ := call.Args["header"].(string)
	header = strings.TrimSpace(header)
	if header == "" {
		header = "Question"
	}
	// Enforce max 30 chars for header.
	if len(header) > 30 {
		header = header[:30]
	}

	allowCustom := true
	if v, ok := call.Args["allow_custom"].(bool); ok {
		allowCustom = v
	}

	// Parse options array.
	var options []map[string]any
	if rawOpts, ok := call.Args["options"]; ok && rawOpts != nil {
		switch opts := rawOpts.(type) {
		case []any:
			for _, item := range opts {
				if m, ok := item.(map[string]any); ok {
					options = append(options, m)
				}
			}
		case []map[string]any:
			options = opts
		}
	}

	// Resolve session ID from the task runtime context.
	sessionID := ""
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok {
		sessionID = strings.TrimSpace(taskCtx.UserID) // userID doubles as session context key
	}

	// The QuestionAsker is required — it's wired by the handler.
	if call.Ctx.QuestionAsker == nil {
		return tools.Fail("Question system is not available in this context", nil)
	}

	// This call blocks until the user replies or rejects.
	answers, err := call.Ctx.QuestionAsker.AskQuestion(ctx, sessionID, question, header, options, allowCustom)
	if err != nil {
		return tools.Fail("User dismissed the question", nil)
	}

	if len(answers) == 0 {
		return tools.Fail("User dismissed the question", nil)
	}

	return tools.Success(strings.Join(answers, ", "), map[string]any{
		"answers": answers,
	})
}

var _ tools.Tool = (*QuestionTool)(nil)
