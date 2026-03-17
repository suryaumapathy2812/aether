package tools

import (
	"context"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
)

// PushDeliverer abstracts Web Push delivery so the tools package
// does not import the ws package directly (avoids circular deps).
type PushDeliverer interface {
	DeliverPush(ctx context.Context, userID string, title, body, tag string) (sent int, failed int, err error)
}

// QuestionAsker abstracts the ask_user tool's ability to pose a blocking
// question to the user and wait for a reply. The handler implements this
// interface and wires it into ExecContext so the tool can block during
// execution while the HTTP/WS layer delivers the question to the client.
type QuestionAsker interface {
	// AskQuestion blocks until the user replies or dismisses the question.
	// options is a slice of {label, description} maps. Returns the selected
	// answers or an error if the question was rejected/cancelled.
	AskQuestion(ctx context.Context, sessionID string, question string, header string, options []map[string]any, allowCustom bool) ([]string, error)
}

type ExecContext struct {
	WorkingDir    string
	Store         *db.Store
	Skills        *skills.Manager
	Plugins       *plugins.Manager
	PluginName    string
	PluginState   plugins.PluginState
	RuntimeHints  map[string]any
	PushDeliverer PushDeliverer
	// QuestionAsker enables the ask_user tool to pose blocking questions (optional).
	QuestionAsker QuestionAsker
	// VectorDB provides semantic memory search using CortexDB (optional)
	VectorDB *cortexdb.DB
	// EmbeddingProvider generates embeddings for semantic search (optional)
	EmbeddingProvider interface {
		EmbedSingle(ctx context.Context, text string) ([]float32, error)
	}
}
