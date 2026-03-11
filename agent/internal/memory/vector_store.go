package memory

import (
	"context"
	"sync"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// VectorStore provides semantic memory search using CortexDB
// This wraps the CortexDB connection for use in memory tools
type VectorStore struct {
	db *cortexdb.DB
	mu sync.RWMutex
}

// VectorMemoryResult represents a memory search result with score
type VectorMemoryResult struct {
	Content    string
	Score      float64
	MemoryType string
	EntityName string
	EntityType string
	Metadata   map[string]string
}

// NewVectorStore creates a new vector memory store from an existing CortexDB instance
func NewVectorStore(db *cortexdb.DB) *VectorStore {
	return &VectorStore{
		db: db,
	}
}

// AddMemory stores a memory with its embedding for semantic search
func (vs *VectorStore) AddMemory(ctx context.Context, userID, memoryType, content string, embedding []float32) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	// Store using Quick API - requires embedding to be provided
	// The content is stored as metadata
	vs.db.Quick().Add(ctx, embedding, content)

	return nil
}

// SearchMemories searches for similar memories using semantic vector search
func (vs *VectorStore) SearchMemories(ctx context.Context, userID string, queryEmbedding []float32, limit int) ([]VectorMemoryResult, error) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	// Use Quick API for simple vector search
	results, err := vs.db.Quick().Search(ctx, queryEmbedding, limit)
	if err != nil {
		return nil, err
	}

	memoryResults := make([]VectorMemoryResult, 0, len(results))
	for _, r := range results {
		memoryResults = append(memoryResults, VectorMemoryResult{
			Content:    r.Content,
			Score:      r.Score,
			MemoryType: "semantic",
		})
	}

	return memoryResults, nil
}

// AddText stores text directly for hybrid search
// Note: Requires an embedder to be configured for automatic embedding
func (vs *VectorStore) AddText(ctx context.Context, userID, content string) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	// Insert text - requires embedder for automatic embedding
	// Without embedder, use AddMemory with pre-computed embeddings
	vs.db.InsertText(ctx, content, content, nil)

	return nil
}

// SearchText performs keyword + semantic hybrid search
// Note: Requires an embedder to be configured
func (vs *VectorStore) SearchText(ctx context.Context, userID, query string, limit int) ([]VectorMemoryResult, error) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	// Hybrid search combines semantic + keyword
	results, err := vs.db.HybridSearchText(ctx, query, limit)
	if err != nil {
		return nil, err
	}

	memoryResults := make([]VectorMemoryResult, 0, len(results))
	for _, r := range results {
		memoryResults = append(memoryResults, VectorMemoryResult{
			Content: r.Content,
			Score:   r.Score,
		})
	}

	return memoryResults, nil
}

// Close closes the vector store
func (vs *VectorStore) Close() error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	if vs.db != nil {
		return vs.db.Close()
	}
	return nil
}
