import Foundation

struct MemoryConversationItem: Identifiable, Hashable {
    let id: Int
    let userMessage: String
    let assistantMessage: String
    let timestamp: TimeInterval
}

struct MemoryItem: Identifiable, Hashable {
    let id: Int
    let memory: String
    let category: String
    let confidence: Double
    let createdAt: String
}

struct DecisionItem: Identifiable, Hashable {
    let id: Int
    let decision: String
    let category: String
    let source: String
    let active: Bool
    let confidence: Double
    let updatedAt: String
}

struct EntityItem: Identifiable, Hashable {
    let id: String
    let entityType: String
    let name: String
    let aliases: [String]
    let summary: String
    let interactionCount: Int
    let firstSeenAt: String
    let lastSeenAt: String
}

struct EntityObservation: Identifiable, Hashable {
    let id: Int
    let observation: String
    let category: String
    let source: String
    let confidence: Double
}

struct EntityInteraction: Identifiable, Hashable {
    let id: Int
    let summary: String
    let source: String
    let interactionAt: String
}

struct EntityRelation: Identifiable, Hashable {
    let id: Int
    let sourceEntityID: String
    let relation: String
    let targetEntityID: String
    let context: String
    let confidence: Double
}

struct EntityDetails {
    let entity: EntityItem
    let observations: [EntityObservation]
    let interactions: [EntityInteraction]
    let relations: [EntityRelation]
}

struct PluginConfigField: Hashable {
    let key: String
    let label: String
    let type: String
    let required: Bool
    let description: String
}

struct PluginItem: Identifiable, Hashable {
    var id: String { name }
    let name: String
    let displayName: String
    let description: String
    let authType: String
    let installed: Bool
    let enabled: Bool
    let connected: Bool
    let needsReconnect: Bool
    let configFields: [PluginConfigField]
}
