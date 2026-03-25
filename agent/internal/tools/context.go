package tools

import (
	"context"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/integrations"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
)

// PushDeliverer abstracts Web Push delivery so the tools package
// does not import the ws package directly (avoids circular deps).
type PushDeliverer interface {
	DeliverPush(ctx context.Context, userID string, title, body, tag string) (sent int, failed int, err error)
}

type QuestionOption struct {
	Label       string
	Description string
}

type QuestionField struct {
	Name        string
	Label       string
	Type        string
	Required    bool
	Placeholder string
	Options     []string
}

type QuestionPrompt struct {
	ToolCallID  string
	Question    string
	Header      string
	Kind        string
	Options     []QuestionOption
	AllowCustom bool
	Fields      []QuestionField
	SubmitLabel string
}

type QuestionResponse struct {
	Answers []string
	Data    map[string]any
}

// QuestionAsker abstracts the ask_user tool's ability to pose a blocking
// question to the user and wait for a reply. The handler implements this
// interface and wires it into ExecContext so the tool can block during
// execution while the HTTP/WS layer delivers the question to the client.
type QuestionAsker interface {
	// AskQuestion blocks until the user replies or dismisses the question.
	// Returns the submitted answers or form data, or an error if the request
	// was rejected/cancelled.
	AskQuestion(ctx context.Context, userID string, sessionID string, prompt QuestionPrompt) (QuestionResponse, error)
}

type ExecContext struct {
	WorkingDir       string
	Store            *db.Store
	Skills           *skills.Manager
	Integrations     *integrations.Manager
	IntegrationName  string
	IntegrationState integrations.PluginState
	RuntimeHints     map[string]any
	PushDeliverer    PushDeliverer
	// QuestionAsker enables the ask_user tool to pose blocking questions (optional).
	QuestionAsker QuestionAsker
	// EmbeddingProvider generates embeddings for semantic search (optional)
	EmbeddingProvider interface {
		Embed(ctx context.Context, texts []string) ([][]float32, error)
		EmbedSingle(ctx context.Context, text string) ([]float32, error)
	}
}
