package memory

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

const (
	// CronModuleMemory is the cron module name for memory maintenance jobs.
	CronModuleMemory = "memory"
	// CronJobTypeDedup is the job type for the recurring dedup cycle.
	CronJobTypeDedup = "dedup"
	// Default dedup interval: 12 hours.
	DefaultDedupIntervalSecs = 43200
)

// DedupEngine runs periodic deduplication across memory tables.
type DedupEngine struct {
	store        *db.Store
	intervalSecs int
}

// DedupOptions configures a new DedupEngine.
type DedupOptions struct {
	Store        *db.Store
	IntervalSecs int
}

// DedupStats tracks what was cleaned up in a single dedup run.
type DedupStats struct {
	FactsRemoved     int
	MemoriesRemoved  int
	DecisionsRemoved int
	EntitiesMerged   int
	ItemsArchived    int
}

// NewDedupEngine creates a new dedup engine.
func NewDedupEngine(opts DedupOptions) *DedupEngine {
	if opts.IntervalSecs <= 0 {
		opts.IntervalSecs = DefaultDedupIntervalSecs
	}
	return &DedupEngine{
		store:        opts.Store,
		intervalSecs: opts.IntervalSecs,
	}
}

// RegisterCronHandlers registers the dedup cron handler with the scheduler.
func RegisterDedupCronHandlers(scheduler *cron.Scheduler, engine *DedupEngine) {
	if scheduler == nil || engine == nil {
		return
	}
	scheduler.RegisterHandler(CronModuleMemory, CronJobTypeDedup, func(ctx context.Context, job cron.Job) error {
		if engine == nil {
			return fmt.Errorf("dedup engine not initialized")
		}
		return engine.RunDedup(ctx)
	})
}

// EnsureCronJobs schedules the recurring dedup job if it doesn't already exist.
func (e *DedupEngine) EnsureCronJobs(ctx context.Context) error {
	if e == nil || e.store == nil {
		return nil
	}
	scope := e.store.ScopeCronModule(CronModuleMemory)
	jobs, err := scope.List(ctx)
	if err != nil {
		return err
	}
	for _, j := range jobs {
		if j.JobType == CronJobTypeDedup && (j.Status == db.CronStatusScheduled || j.Status == db.CronStatusRunning) {
			return nil // already scheduled
		}
	}
	interval := int64(e.intervalSecs)
	_, err = scope.ScheduleRecurring(ctx, CronJobTypeDedup, map[string]any{}, time.Now().Add(30*time.Second), interval, 5)
	return err
}

// RunDedup performs a full dedup pass across facts, memories, decisions, and entities.
func (e *DedupEngine) RunDedup(ctx context.Context) error {
	if e == nil || e.store == nil {
		return fmt.Errorf("dedup engine not initialized")
	}
	start := time.Now()
	stats := DedupStats{}

	users, err := e.store.ListMemoryUsers(ctx)
	if err != nil {
		return err
	}
	for _, userID := range users {
		removed, err := e.dedupFacts(ctx, userID)
		if err != nil {
			log.Printf("memory dedup: facts error user=%s: %v", userID, err)
		}
		stats.FactsRemoved += removed

		removed, err = e.dedupMemories(ctx, userID)
		if err != nil {
			log.Printf("memory dedup: memories error user=%s: %v", userID, err)
		}
		stats.MemoriesRemoved += removed

		removed, err = e.dedupDecisions(ctx, userID)
		if err != nil {
			log.Printf("memory dedup: decisions error user=%s: %v", userID, err)
		}
		stats.DecisionsRemoved += removed

		merged, err := e.dedupEntities(ctx, userID)
		if err != nil {
			log.Printf("memory dedup: entities error user=%s: %v", userID, err)
		}
		stats.EntitiesMerged += merged

		archived, err := e.store.ArchiveStaleMemoryItems(ctx, userID, time.Now().UTC().Add(-45*24*time.Hour), true)
		if err != nil {
			log.Printf("memory dedup: archive error user=%s: %v", userID, err)
		} else {
			stats.ItemsArchived += int(archived)
		}
	}

	total := stats.FactsRemoved + stats.MemoriesRemoved + stats.DecisionsRemoved + stats.EntitiesMerged + stats.ItemsArchived
	if total > 0 {
		log.Printf("memory dedup: completed in %v — facts_removed=%d memories_removed=%d decisions_removed=%d entities_merged=%d items_archived=%d",
			time.Since(start).Round(time.Millisecond), stats.FactsRemoved, stats.MemoriesRemoved, stats.DecisionsRemoved, stats.EntitiesMerged, stats.ItemsArchived)
	} else {
		log.Printf("memory dedup: completed in %v — nothing to clean", time.Since(start).Round(time.Millisecond))
	}
	return nil
}

// dedupFacts finds facts with high token overlap and removes the older/shorter one.
// The fact_key UNIQUE constraint catches exact dupes at insert time, but this
// catches semantically similar facts like:
//   - "The user works at DET" vs "The user works at Deshpande Educational Trust"
func (e *DedupEngine) dedupFacts(ctx context.Context, userID string) (int, error) {
	facts, err := e.store.ListFacts(ctx, userID)
	if err != nil {
		return 0, err
	}
	return dedupTextRecords(ctx, facts, func(r db.FactRecord) (int64, string) {
		return r.ID, r.Fact
	}, func(ctx context.Context, id int64) error {
		return e.store.DeleteFact(ctx, id)
	})
}

// dedupMemories finds memories with high token overlap and removes duplicates.
func (e *DedupEngine) dedupMemories(ctx context.Context, userID string) (int, error) {
	memories, err := e.store.ListMemories(ctx, userID, "", 2000)
	if err != nil {
		return 0, err
	}
	return dedupTextRecords(ctx, memories, func(r db.MemoryRecord) (int64, string) {
		return r.ID, r.Memory
	}, func(ctx context.Context, id int64) error {
		return e.store.DeleteMemory(ctx, id)
	})
}

// dedupDecisions finds decisions with high token overlap and removes duplicates.
func (e *DedupEngine) dedupDecisions(ctx context.Context, userID string) (int, error) {
	decisions, err := e.store.ListDecisions(ctx, userID, "", false)
	if err != nil {
		return 0, err
	}
	return dedupTextRecords(ctx, decisions, func(r db.DecisionRecord) (int64, string) {
		return r.ID, r.Decision
	}, func(ctx context.Context, id int64) error {
		return e.store.DeleteDecision(ctx, id)
	})
}

// dedupTextRecords is a generic dedup function for any table with ID + text content.
// It compares all pairs using token overlap. When two records have >0.8 overlap,
// the shorter one is deleted (the longer one is assumed to be more informative).
func dedupTextRecords[T any](
	ctx context.Context,
	records []T,
	extract func(T) (int64, string),
	deleteFn func(context.Context, int64) error,
) (int, error) {
	if len(records) < 2 {
		return 0, nil
	}

	type entry struct {
		id     int64
		text   string
		tokens map[string]struct{}
	}

	entries := make([]entry, 0, len(records))
	for _, r := range records {
		id, text := extract(r)
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		entries = append(entries, entry{id: id, text: text, tokens: dedupTokenize(text)})
	}

	deleted := map[int64]bool{}
	removed := 0

	for i := 0; i < len(entries); i++ {
		if deleted[entries[i].id] {
			continue
		}
		for j := i + 1; j < len(entries); j++ {
			if deleted[entries[j].id] {
				continue
			}
			score := dedupTokenOverlap(entries[i].tokens, entries[j].tokens)
			if score >= 0.8 {
				// Delete the shorter one (less informative). If same length, delete the later one.
				var victim int
				if len(entries[i].text) < len(entries[j].text) {
					victim = i
				} else {
					victim = j
				}
				if err := deleteFn(ctx, entries[victim].id); err == nil {
					deleted[entries[victim].id] = true
					removed++
				}
			}
		}
	}
	return removed, nil
}

// dedupEntities finds entity duplicates within the same type using fuzzy matching
// and merges them. Uses the same logic as the extraction dedup but across ALL entities.
func (e *DedupEngine) dedupEntities(ctx context.Context, userID string) (int, error) {
	entities, err := e.store.ListEntities(ctx, userID, "", 2000)
	if err != nil {
		return 0, err
	}
	if len(entities) < 2 {
		return 0, nil
	}

	// Group by entity type — only compare within the same type.
	byType := map[string][]db.EntityRecord{}
	for _, ent := range entities {
		byType[ent.EntityType] = append(byType[ent.EntityType], ent)
	}

	merged := 0
	mergedIDs := map[string]bool{}

	for _, group := range byType {
		if len(group) < 2 {
			continue
		}
		for i := 0; i < len(group); i++ {
			if mergedIDs[group[i].ID] {
				continue
			}
			for j := i + 1; j < len(group); j++ {
				if mergedIDs[group[j].ID] {
					continue
				}
				if shouldMergeEntities(group[i], group[j]) {
					// Keep the one with more interactions, or the older one if tied.
					target, source := group[i], group[j]
					if source.InteractionCount > target.InteractionCount ||
						(source.InteractionCount == target.InteractionCount && source.CreatedAt.Before(target.CreatedAt)) {
						target, source = source, target
					}
					if err := e.store.MergeEntities(ctx, target.ID, source.ID); err != nil {
						log.Printf("memory dedup: entity merge failed (%s into %s): %v", source.Name, target.Name, err)
						continue
					}
					log.Printf("memory dedup: merged entity %q into %q", source.Name, target.Name)
					mergedIDs[source.ID] = true
					merged++
				}
			}
		}
	}
	return merged, nil
}

// shouldMergeEntities determines if two entities of the same type are duplicates.
// Criteria: exact substring match OR high token overlap (>0.7).
func shouldMergeEntities(a, b db.EntityRecord) bool {
	aLower := strings.ToLower(strings.TrimSpace(a.Name))
	bLower := strings.ToLower(strings.TrimSpace(b.Name))

	// Exact match (shouldn't happen due to upsert, but just in case).
	if aLower == bLower {
		return true
	}

	// Substring: "Bhuvan" in "Bhuvan T" or vice versa.
	if strings.Contains(aLower, bLower) || strings.Contains(bLower, aLower) {
		return true
	}

	// Check aliases for substring matches.
	allANames := append([]string{aLower}, lowercaseAll(a.Aliases)...)
	allBNames := append([]string{bLower}, lowercaseAll(b.Aliases)...)
	for _, an := range allANames {
		for _, bn := range allBNames {
			if an == bn || strings.Contains(an, bn) || strings.Contains(bn, an) {
				return true
			}
		}
	}

	// Token overlap on names.
	aTokens := dedupTokenize(aLower)
	bTokens := dedupTokenize(bLower)
	if dedupTokenOverlap(aTokens, bTokens) >= 0.7 {
		return true
	}

	return false
}

// lowercaseAll returns a new slice with all strings lowercased.
func lowercaseAll(ss []string) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = strings.ToLower(strings.TrimSpace(s))
	}
	return out
}

// dedupTokenize splits text into lowercase alphanumeric tokens.
func dedupTokenize(v string) map[string]struct{} {
	re := regexp.MustCompile(`[a-z0-9]+`)
	parts := re.FindAllString(strings.ToLower(v), -1)
	out := map[string]struct{}{}
	for _, p := range parts {
		out[p] = struct{}{}
	}
	return out
}

// dedupTokenOverlap computes overlap between two token sets.
// Uses the smaller set as denominator for a stricter match.
func dedupTokenOverlap(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	common := 0
	for k := range a {
		if _, ok := b[k]; ok {
			common++
		}
	}
	if common == 0 {
		return 0
	}
	smaller := len(a)
	if len(b) < smaller {
		smaller = len(b)
	}
	return float64(common) / float64(smaller)
}
