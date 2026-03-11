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

type ExecContext struct {
	WorkingDir    string
	Store         *db.Store
	Skills        *skills.Manager
	Plugins       *plugins.Manager
	PluginName    string
	PluginState   plugins.PluginState
	RuntimeHints  map[string]any
	PushDeliverer PushDeliverer
	// VectorDB provides semantic memory search using CortexDB (optional)
	VectorDB *cortexdb.DB
	// EmbeddingProvider generates embeddings for semantic search (optional)
	EmbeddingProvider interface {
		EmbedSingle(ctx context.Context, text string) ([]float32, error)
	}
}
