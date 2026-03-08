import Foundation

final class MemoryService {
    private let api: APIClient

    init(api: APIClient) {
        self.api = api
    }

    func fetchFacts() async throws -> [String] {
        let raw = try await api.get(path: "/api/memory/facts")
        return (dictValue(raw)["facts"] as? [String]) ?? []
    }

    func fetchConversations(limit: Int = 30) async throws -> [MemoryConversationItem] {
        let raw = try await api.get(path: "/api/memory/conversations?limit=\(limit)")
        return arrayValue(raw, key: "conversations").enumerated().map { idx, row in
            MemoryConversationItem(
                id: intValue(row, "id", "ID") != 0 ? intValue(row, "id", "ID") : idx,
                userMessage: stringValue(row, "user_message", "UserMessage"),
                assistantMessage: stringValue(row, "assistant_message", "AssistantMessage"),
                timestamp: doubleValue(row, "timestamp", "Timestamp")
            )
        }
    }

    func fetchMemories(limit: Int = 100) async throws -> [MemoryItem] {
        let raw = try await api.get(path: "/api/memory/memories?limit=\(limit)")
        return arrayValue(raw, key: "memories").map { row in
            MemoryItem(
                id: intValue(row, "id", "ID"),
                memory: stringValue(row, "memory", "Memory"),
                category: stringValue(row, "category", "Category"),
                confidence: doubleValue(row, "confidence", "Confidence"),
                createdAt: stringValue(row, "created_at", "CreatedAt")
            )
        }
    }

    func fetchDecisions() async throws -> [DecisionItem] {
        let raw = try await api.get(path: "/api/memory/decisions?active_only=true")
        return arrayValue(raw, key: "decisions").map { row in
            DecisionItem(
                id: intValue(row, "id", "ID"),
                decision: stringValue(row, "decision", "Decision"),
                category: stringValue(row, "category", "Category"),
                source: stringValue(row, "source", "Source"),
                active: boolValue(row, "active", "Active"),
                confidence: doubleValue(row, "confidence", "Confidence"),
                updatedAt: stringValue(row, "updated_at", "UpdatedAt")
            )
        }
    }

    func fetchEntities(limit: Int = 50) async throws -> [EntityItem] {
        let raw = try await api.get(path: "/api/memory/entities?limit=\(limit)")
        return arrayValue(raw, key: "entities").map { row in
            EntityItem(
                id: stringValue(row, "id", "ID"),
                entityType: stringValue(row, "entity_type", "EntityType"),
                name: stringValue(row, "name", "Name"),
                aliases: (row["aliases"] as? [String]) ?? (row["Aliases"] as? [String]) ?? [],
                summary: stringValue(row, "summary", "Summary"),
                interactionCount: intValue(row, "interaction_count", "InteractionCount"),
                firstSeenAt: stringValue(row, "first_seen_at", "FirstSeenAt"),
                lastSeenAt: stringValue(row, "last_seen_at", "LastSeenAt")
            )
        }
    }

    func fetchEntityDetails(entityID: String) async throws -> EntityDetails {
        let encodedID = entityID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? entityID
        let raw = try await api.get(path: "/api/memory/entities/\(encodedID)")
        let root = dictValue(raw)
        let entityRaw = dictValue(root["entity"] as Any)
        let entity = EntityItem(
            id: stringValue(entityRaw, "id", "ID"),
            entityType: stringValue(entityRaw, "entity_type", "EntityType"),
            name: stringValue(entityRaw, "name", "Name"),
            aliases: (entityRaw["aliases"] as? [String]) ?? (entityRaw["Aliases"] as? [String]) ?? [],
            summary: stringValue(entityRaw, "summary", "Summary"),
            interactionCount: intValue(entityRaw, "interaction_count", "InteractionCount"),
            firstSeenAt: stringValue(entityRaw, "first_seen_at", "FirstSeenAt"),
            lastSeenAt: stringValue(entityRaw, "last_seen_at", "LastSeenAt")
        )
        let observations = ((root["observations"] as? [[String: Any]]) ?? []).map { row in
            EntityObservation(
                id: intValue(row, "id", "ID"),
                observation: stringValue(row, "observation", "Observation"),
                category: stringValue(row, "category", "Category"),
                source: stringValue(row, "source", "Source"),
                confidence: doubleValue(row, "confidence", "Confidence")
            )
        }
        let interactions = ((root["interactions"] as? [[String: Any]]) ?? []).map { row in
            EntityInteraction(
                id: intValue(row, "id", "ID"),
                summary: stringValue(row, "summary", "Summary"),
                source: stringValue(row, "source", "Source"),
                interactionAt: stringValue(row, "interaction_at", "InteractionAt")
            )
        }
        let relations = ((root["relations"] as? [[String: Any]]) ?? []).map { row in
            EntityRelation(
                id: intValue(row, "id", "ID"),
                sourceEntityID: stringValue(row, "source_entity_id", "SourceEntityID"),
                relation: stringValue(row, "relation", "Relation"),
                targetEntityID: stringValue(row, "target_entity_id", "TargetEntityID"),
                context: stringValue(row, "context", "Context"),
                confidence: doubleValue(row, "confidence", "Confidence")
            )
        }
        return EntityDetails(entity: entity, observations: observations, interactions: interactions, relations: relations)
    }
}
