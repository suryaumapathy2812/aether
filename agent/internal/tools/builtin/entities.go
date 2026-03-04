package builtin

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// SaveEntityTool saves or updates an entity in memory.
type SaveEntityTool struct{}

func (t *SaveEntityTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "save_entity",
		Description: "Save or update an entity (person, project, organization, topic, etc.) in memory.",
		StatusText:  "Saving entity...",
		Parameters: []tools.Param{
			{Name: "name", Type: "string", Description: "The entity name.", Required: true},
			{Name: "entity_type", Type: "string", Description: "The entity type.", Required: true, Enum: []string{"person", "project", "organization", "topic", "place", "tool"}},
			{Name: "observations", Type: "array", Description: "Observations or traits about the entity.", Required: false, Items: map[string]any{"type": "string"}},
			{Name: "properties", Type: "object", Description: "Additional properties for the entity.", Required: false},
		},
	}
}

func (t *SaveEntityTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	name = strings.TrimSpace(name)
	if name == "" {
		return tools.Fail("Entity name is required", nil)
	}
	entityType, _ := call.Args["entity_type"].(string)
	entityType = strings.TrimSpace(entityType)
	if entityType == "" {
		return tools.Fail("Entity type is required", nil)
	}

	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	// Parse optional observations.
	var observations []string
	if rawObs, ok := call.Args["observations"]; ok {
		switch v := rawObs.(type) {
		case []any:
			for _, item := range v {
				if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
					observations = append(observations, strings.TrimSpace(s))
				}
			}
		case []string:
			for _, s := range v {
				if strings.TrimSpace(s) != "" {
					observations = append(observations, strings.TrimSpace(s))
				}
			}
		}
	}

	// Parse optional properties.
	var properties map[string]any
	if rawProps, ok := call.Args["properties"]; ok {
		if p, ok := rawProps.(map[string]any); ok {
			properties = p
		}
	}

	entityID, err := call.Ctx.Store.UpsertEntity(ctx, userID, entityType, name, nil, properties)
	if err != nil {
		return tools.Fail("Failed to save entity: "+err.Error(), nil)
	}

	// Store observations.
	savedObs := 0
	for _, obs := range observations {
		if err := call.Ctx.Store.AddEntityObservation(ctx, entityID, userID, obs, "trait", "explicit"); err == nil {
			savedObs++
		}
	}

	return tools.Success(
		fmt.Sprintf("Saved entity '%s' (%s) with %d observations.", name, entityType, savedObs),
		map[string]any{"entity_id": entityID, "observations_saved": savedObs},
	)
}

// AddEntityObservationTool adds an observation or trait about a known entity.
type AddEntityObservationTool struct{}

func (t *AddEntityObservationTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "add_entity_observation",
		Description: "Add an observation or trait about a known entity.",
		StatusText:  "Adding observation...",
		Parameters: []tools.Param{
			{Name: "entity_name", Type: "string", Description: "The entity name to add the observation to.", Required: true},
			{Name: "observation", Type: "string", Description: "The observation or trait.", Required: true},
			{Name: "category", Type: "string", Description: "Observation category.", Required: false, Default: "trait", Enum: []string{"trait", "fact", "note", "preference"}},
		},
	}
}

func (t *AddEntityObservationTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	entityName, _ := call.Args["entity_name"].(string)
	entityName = strings.TrimSpace(entityName)
	if entityName == "" {
		return tools.Fail("Entity name is required", nil)
	}
	observation, _ := call.Args["observation"].(string)
	observation = strings.TrimSpace(observation)
	if observation == "" {
		return tools.Fail("Observation is required", nil)
	}
	category, _ := call.Args["category"].(string)
	if strings.TrimSpace(category) == "" {
		category = "trait"
	}

	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	// Find entity by searching.
	entities, err := call.Ctx.Store.SearchEntities(ctx, userID, entityName, 1)
	if err != nil || len(entities) == 0 {
		return tools.Fail(fmt.Sprintf("Entity '%s' not found. Save it first with save_entity.", entityName), nil)
	}
	entity := entities[0]

	if err := call.Ctx.Store.AddEntityObservation(ctx, entity.ID, userID, observation, category, "explicit"); err != nil {
		return tools.Fail("Failed to add observation: "+err.Error(), nil)
	}
	return tools.Success(
		fmt.Sprintf("Added %s observation to '%s': %s", category, entity.Name, observation),
		map[string]any{"entity_id": entity.ID, "category": category},
	)
}

// SearchEntitiesTool searches for known entities in memory.
type SearchEntitiesTool struct{}

func (t *SearchEntitiesTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "search_entities",
		Description: "Search for known entities (people, projects, organizations, etc.) in memory.",
		StatusText:  "Searching entities...",
		Parameters: []tools.Param{
			{Name: "query", Type: "string", Description: "What to search for.", Required: true},
			{Name: "entity_type", Type: "string", Description: "Filter by entity type.", Required: false},
			{Name: "limit", Type: "integer", Description: "Maximum results.", Required: false, Default: 5},
		},
	}
}

func (t *SearchEntitiesTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	query, _ := call.Args["query"].(string)
	query = strings.TrimSpace(query)
	if query == "" {
		return tools.Fail("Cannot search with empty query", nil)
	}
	limit, _ := call.Args["limit"].(int)
	if limit <= 0 {
		limit = 5
	}

	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	results, err := call.Ctx.Store.SearchEntities(ctx, userID, query, limit)
	if err != nil {
		return tools.Fail("Failed to search entities: "+err.Error(), nil)
	}

	// Optionally filter by entity_type.
	entityType, _ := call.Args["entity_type"].(string)
	entityType = strings.TrimSpace(entityType)
	if entityType != "" {
		var filtered []db.EntityRecord
		for _, r := range results {
			if strings.EqualFold(r.EntityType, entityType) {
				filtered = append(filtered, r)
			}
		}
		results = filtered
	}

	if len(results) == 0 {
		return tools.Success("No matching entities found.", map[string]any{"count": 0})
	}
	lines := make([]string, 0, len(results))
	for i, r := range results {
		line := fmt.Sprintf("%d. [%s] %s", i+1, r.EntityType, r.Name)
		if r.Summary != "" {
			line += ": " + truncateOutput(r.Summary, 120)
		}
		lines = append(lines, line)
	}
	return tools.Success("Found entities:\n"+strings.Join(lines, "\n"), map[string]any{"count": len(results)})
}

// RelateEntitiesTool creates a relationship between two entities.
type RelateEntitiesTool struct{}

func (t *RelateEntitiesTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "relate_entities",
		Description: "Create a relationship between two entities.",
		StatusText:  "Linking entities...",
		Parameters: []tools.Param{
			{Name: "source_name", Type: "string", Description: "The source entity name.", Required: true},
			{Name: "relation", Type: "string", Description: "The relationship type (e.g. 'works_with', 'manages', 'part_of').", Required: true},
			{Name: "target_name", Type: "string", Description: "The target entity name.", Required: true},
			{Name: "context", Type: "string", Description: "Additional context about the relationship.", Required: false},
		},
	}
}

func (t *RelateEntitiesTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	sourceName, _ := call.Args["source_name"].(string)
	sourceName = strings.TrimSpace(sourceName)
	if sourceName == "" {
		return tools.Fail("Source entity name is required", nil)
	}
	relation, _ := call.Args["relation"].(string)
	relation = strings.TrimSpace(relation)
	if relation == "" {
		return tools.Fail("Relation is required", nil)
	}
	targetName, _ := call.Args["target_name"].(string)
	targetName = strings.TrimSpace(targetName)
	if targetName == "" {
		return tools.Fail("Target entity name is required", nil)
	}
	relContext, _ := call.Args["context"].(string)

	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	// Find source entity.
	sourceEntities, err := call.Ctx.Store.SearchEntities(ctx, userID, sourceName, 1)
	if err != nil || len(sourceEntities) == 0 {
		return tools.Fail(fmt.Sprintf("Source entity '%s' not found.", sourceName), nil)
	}
	// Find target entity.
	targetEntities, err := call.Ctx.Store.SearchEntities(ctx, userID, targetName, 1)
	if err != nil || len(targetEntities) == 0 {
		return tools.Fail(fmt.Sprintf("Target entity '%s' not found.", targetName), nil)
	}

	source := sourceEntities[0]
	target := targetEntities[0]

	if err := call.Ctx.Store.AddEntityRelation(ctx, userID, source.ID, relation, target.ID, relContext); err != nil {
		return tools.Fail("Failed to create relation: "+err.Error(), nil)
	}
	return tools.Success(
		fmt.Sprintf("Linked '%s' -[%s]-> '%s'.", source.Name, relation, target.Name),
		map[string]any{"source_id": source.ID, "target_id": target.ID, "relation": relation},
	)
}

// GetEntityDetailsTool gets full details about a known entity.
type GetEntityDetailsTool struct{}

func (t *GetEntityDetailsTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "get_entity_details",
		Description: "Get full details about a known entity including observations, relations, and recent interactions.",
		StatusText:  "Loading entity details...",
		Parameters: []tools.Param{
			{Name: "name", Type: "string", Description: "The entity name.", Required: true},
		},
	}
}

func (t *GetEntityDetailsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.Store == nil {
		return tools.Fail("State store is unavailable", nil)
	}
	name, _ := call.Args["name"].(string)
	name = strings.TrimSpace(name)
	if name == "" {
		return tools.Fail("Entity name is required", nil)
	}

	userID := "default"
	if taskCtx, ok := tools.TaskRuntimeContextFromContext(ctx); ok && strings.TrimSpace(taskCtx.UserID) != "" {
		userID = strings.TrimSpace(taskCtx.UserID)
	}

	// Find entity.
	entities, err := call.Ctx.Store.SearchEntities(ctx, userID, name, 1)
	if err != nil || len(entities) == 0 {
		return tools.Fail(fmt.Sprintf("Entity '%s' not found.", name), nil)
	}
	entity := entities[0]

	// Build detail output.
	lines := []string{
		fmt.Sprintf("Entity: %s", entity.Name),
		fmt.Sprintf("Type: %s", entity.EntityType),
	}
	if entity.Summary != "" {
		lines = append(lines, fmt.Sprintf("Summary: %s", entity.Summary))
	}
	if len(entity.Aliases) > 0 {
		lines = append(lines, fmt.Sprintf("Aliases: %s", strings.Join(entity.Aliases, ", ")))
	}
	lines = append(lines, fmt.Sprintf("Interactions: %d", entity.InteractionCount))
	lines = append(lines, fmt.Sprintf("First seen: %s", entity.FirstSeenAt.Format(time.RFC3339)))
	lines = append(lines, fmt.Sprintf("Last seen: %s", entity.LastSeenAt.Format(time.RFC3339)))

	// Observations.
	observations, _ := call.Ctx.Store.ListEntityObservations(ctx, entity.ID, 20)
	if len(observations) > 0 {
		lines = append(lines, "", "Observations:")
		for _, obs := range observations {
			lines = append(lines, fmt.Sprintf("  - [%s] %s", obs.Category, obs.Observation))
		}
	}

	// Relations.
	relations, _ := call.Ctx.Store.ListEntityRelations(ctx, entity.ID)
	if len(relations) > 0 {
		lines = append(lines, "", "Relations:")
		for _, rel := range relations {
			// Determine direction.
			if rel.SourceEntityID == entity.ID {
				// This entity is the source.
				targetEnt, err := call.Ctx.Store.GetEntity(ctx, rel.TargetEntityID)
				targetName := rel.TargetEntityID
				if err == nil && targetEnt != nil {
					targetName = targetEnt.Name
				}
				line := fmt.Sprintf("  - %s -[%s]-> %s", entity.Name, rel.Relation, targetName)
				if rel.Context != "" {
					line += " (" + truncateOutput(rel.Context, 80) + ")"
				}
				lines = append(lines, line)
			} else {
				// This entity is the target.
				sourceEnt, err := call.Ctx.Store.GetEntity(ctx, rel.SourceEntityID)
				sourceName := rel.SourceEntityID
				if err == nil && sourceEnt != nil {
					sourceName = sourceEnt.Name
				}
				line := fmt.Sprintf("  - %s -[%s]-> %s", sourceName, rel.Relation, entity.Name)
				if rel.Context != "" {
					line += " (" + truncateOutput(rel.Context, 80) + ")"
				}
				lines = append(lines, line)
			}
		}
	}

	// Recent interactions.
	interactions, _ := call.Ctx.Store.ListEntityInteractions(ctx, entity.ID, 10)
	if len(interactions) > 0 {
		lines = append(lines, "", "Recent interactions:")
		for _, inter := range interactions {
			lines = append(lines, fmt.Sprintf("  - [%s] %s (%s)", inter.InteractionAt.Format(time.RFC3339), truncateOutput(inter.Summary, 120), inter.Source))
		}
	}

	return tools.Success(strings.Join(lines, "\n"), map[string]any{"entity_id": entity.ID})
}
