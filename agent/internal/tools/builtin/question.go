package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

var allowedQuestionFieldTypes = map[string]struct{}{
	"text":     {},
	"email":    {},
	"number":   {},
	"textarea": {},
	"select":   {},
}

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
				Name:        "kind",
				Type:        "string",
				Required:    false,
				Default:     "choice",
				Enum:        []string{"choice", "confirm", "form"},
				Description: "Interactive request type: choice, confirm, or form",
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
			{
				Name:        "fields",
				Type:        "array",
				Required:    false,
				Description: "Form fields to render when kind=form",
				Items: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name":        map[string]any{"type": "string"},
						"label":       map[string]any{"type": "string"},
						"type":        map[string]any{"type": "string", "enum": []string{"text", "email", "number", "textarea", "select"}},
						"required":    map[string]any{"type": "boolean"},
						"placeholder": map[string]any{"type": "string"},
						"options":     map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					},
				},
			},
			{
				Name:        "submit_label",
				Type:        "string",
				Required:    false,
				Default:     "Submit",
				Description: "Custom submit button label for interactive prompts",
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

	kind, _ := call.Args["kind"].(string)
	kind = strings.TrimSpace(strings.ToLower(kind))
	if kind == "" {
		kind = "choice"
	}

	allowCustom := true
	if v, ok := call.Args["allow_custom"].(bool); ok {
		allowCustom = v
	}

	// Parse options array.
	var options []tools.QuestionOption
	if rawOpts, ok := call.Args["options"]; ok && rawOpts != nil {
		switch opts := rawOpts.(type) {
		case []any:
			for _, item := range opts {
				if m, ok := item.(map[string]any); ok {
					label, _ := m["label"].(string)
					label = strings.TrimSpace(label)
					if label == "" {
						continue
					}
					description, _ := m["description"].(string)
					options = append(options, tools.QuestionOption{Label: label, Description: strings.TrimSpace(description)})
				}
			}
		}
	}

	var fields []tools.QuestionField
	if rawFields, ok := call.Args["fields"]; ok && rawFields != nil {
		switch items := rawFields.(type) {
		case []any:
			for _, item := range items {
				field, ok := item.(map[string]any)
				if !ok {
					continue
				}
				name, _ := field["name"].(string)
				name = strings.TrimSpace(name)
				if name == "" {
					continue
				}
				label, _ := field["label"].(string)
				label = strings.TrimSpace(label)
				if label == "" {
					label = name
				}
				fieldType, _ := field["type"].(string)
				fieldType = strings.TrimSpace(strings.ToLower(fieldType))
				if fieldType == "" {
					fieldType = "text"
				}
				required, _ := field["required"].(bool)
				placeholder, _ := field["placeholder"].(string)
				parsed := tools.QuestionField{
					Name:        name,
					Label:       label,
					Type:        fieldType,
					Required:    required,
					Placeholder: strings.TrimSpace(placeholder),
				}
				if rawOptions, ok := field["options"].([]any); ok {
					for _, opt := range rawOptions {
						if s, ok := opt.(string); ok && strings.TrimSpace(s) != "" {
							parsed.Options = append(parsed.Options, strings.TrimSpace(s))
						}
					}
				}
				fields = append(fields, parsed)
			}
		}
	}

	submitLabel, _ := call.Args["submit_label"].(string)
	submitLabel = strings.TrimSpace(submitLabel)
	if submitLabel == "" {
		submitLabel = "Submit"
	}

	if err := validateQuestionPrompt(kind, options, allowCustom, fields); err != nil {
		return tools.Fail(err.Error(), nil)
	}

	// Resolve session ID from the task runtime context.
	userID := ""
	sessionID := ""
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok {
		userID = strings.TrimSpace(taskCtx.UserID)
		sessionID = strings.TrimSpace(taskCtx.SessionID)
	}
	if sessionID == "" {
		sessionID = userID
	}

	// The QuestionAsker is required — it's wired by the handler.
	if call.Ctx.QuestionAsker == nil {
		return tools.Fail("Question system is not available in this context", nil)
	}

	// This call blocks until the user replies or rejects.
	reply, err := call.Ctx.QuestionAsker.AskQuestion(ctx, userID, sessionID, tools.QuestionPrompt{
		ToolCallID:  call.ID,
		Question:    question,
		Header:      header,
		Kind:        kind,
		Options:     options,
		AllowCustom: allowCustom,
		Fields:      fields,
		SubmitLabel: submitLabel,
	})
	if err != nil {
		return tools.Fail("User dismissed the question", nil)
	}

	if kind == "form" {
		data := reply.Data
		if data == nil {
			data = map[string]any{}
		}
		encoded, marshalErr := json.Marshal(data)
		if marshalErr != nil {
			return tools.Fail("Failed to encode form response", nil)
		}
		return tools.Success(string(encoded), map[string]any{
			"kind": kind,
			"data": data,
		})
	}

	if len(reply.Answers) == 0 {
		return tools.Fail("User dismissed the question", nil)
	}

	return tools.Success(strings.Join(reply.Answers, ", "), map[string]any{
		"kind":    kind,
		"answers": reply.Answers,
	})
}

func validateQuestionPrompt(kind string, options []tools.QuestionOption, allowCustom bool, fields []tools.QuestionField) error {
	switch kind {
	case "choice", "confirm":
		if len(fields) > 0 {
			return fmt.Errorf("fields are only supported when kind=form")
		}
		if kind == "choice" && len(options) == 0 && !allowCustom {
			return fmt.Errorf("choice prompts require options or allow_custom=true")
		}
	case "form":
		if len(fields) == 0 {
			return fmt.Errorf("form prompts require at least one field")
		}
		seen := map[string]struct{}{}
		for _, field := range fields {
			name := strings.TrimSpace(field.Name)
			if name == "" {
				return fmt.Errorf("form fields require a name")
			}
			if _, ok := seen[name]; ok {
				return fmt.Errorf("duplicate form field: %s", name)
			}
			seen[name] = struct{}{}
			fieldType := strings.TrimSpace(strings.ToLower(field.Type))
			if _, ok := allowedQuestionFieldTypes[fieldType]; !ok {
				return fmt.Errorf("unsupported form field type: %s", field.Type)
			}
			if fieldType == "select" && len(field.Options) == 0 {
				return fmt.Errorf("select field %s requires options", name)
			}
		}
	default:
		return fmt.Errorf("unsupported question kind: %s", kind)
	}

	return nil
}

var _ tools.Tool = (*QuestionTool)(nil)
