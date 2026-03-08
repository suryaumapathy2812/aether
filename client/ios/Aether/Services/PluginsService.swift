import Foundation

final class PluginsService {
    private let api: APIClient

    init(api: APIClient) {
        self.api = api
    }

    func listPlugins() async throws -> [PluginItem] {
        let raw = try await api.get(path: "/api/plugins")
        let rows = rootArrayValue(raw).isEmpty ? arrayValue(raw, key: "plugins") : rootArrayValue(raw)
        return rows.map { row in
            let fields: [PluginConfigField] = ((row["config_fields"] as? [[String: Any]]) ?? []).map { field in
                PluginConfigField(
                    key: stringValue(field, "key"),
                    label: stringValue(field, "label"),
                    type: stringValue(field, "type"),
                    required: boolValue(field, "required"),
                    description: stringValue(field, "description")
                )
            }
            return PluginItem(
                name: stringValue(row, "name"),
                displayName: stringValue(row, "display_name"),
                description: stringValue(row, "description"),
                authType: stringValue(row, "auth_type"),
                installed: boolValue(row, "installed"),
                enabled: boolValue(row, "enabled"),
                connected: boolValue(row, "connected"),
                needsReconnect: boolValue(row, "needs_reconnect"),
                configFields: fields
            )
        }
    }

    func install(name: String) async throws {
        _ = try await api.post(path: "/api/plugins/\(name)/install")
    }

    func enable(name: String) async throws {
        _ = try await api.post(path: "/api/plugins/\(name)/enable")
    }

    func disable(name: String) async throws {
        _ = try await api.post(path: "/api/plugins/\(name)/disable")
    }

    func getConfig(name: String) async throws -> [String: String] {
        let raw = try await api.get(path: "/api/plugins/\(name)/config")
        let dict = dictValue(raw)
        var out: [String: String] = [:]
        for (k, v) in dict {
            if let s = v as? String { out[k] = s }
        }
        return out
    }

    func saveConfig(name: String, config: [String: String]) async throws {
        _ = try await api.post(path: "/api/plugins/\(name)/config", body: ["config": config])
    }
}
