import Foundation

enum APIClientError: Error {
    case invalidURL
    case invalidResponse
    case requestFailed(String)
}

final class APIClient {
    private let baseURL: String
    private let token: String
    private let proxyPrefix = "/api/go"

    init(baseURL: String, token: String) {
        self.baseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.token = token
    }

    func get(path: String) async throws -> Any {
        try await request(path: path, method: "GET", body: nil)
    }

    func post(path: String, body: [String: Any]?) async throws -> Any {
        let payload = try body.map { try JSONSerialization.data(withJSONObject: $0) }
        return try await request(path: path, method: "POST", body: payload)
    }

    func post(path: String) async throws -> Any {
        try await request(path: path, method: "POST", body: nil)
    }

    private func request(path: String, method: String, body: Data?) async throws -> Any {
        let normalizedPath = path.hasPrefix("/") ? path : "/\(path)"
        guard let url = URL(string: "\(baseURL)\(proxyPrefix)\(normalizedPath)") else {
            throw APIClientError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIClientError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw APIClientError.requestFailed(text)
        }
        if data.isEmpty {
            return [:]
        }
        return (try? JSONSerialization.jsonObject(with: data)) ?? [:]
    }
}

func dictValue(_ raw: Any) -> [String: Any] {
    raw as? [String: Any] ?? [:]
}

func arrayValue(_ raw: Any, key: String) -> [[String: Any]] {
    (dictValue(raw)[key] as? [[String: Any]]) ?? []
}

func rootArrayValue(_ raw: Any) -> [[String: Any]] {
    raw as? [[String: Any]] ?? []
}

func stringValue(_ dict: [String: Any], _ keys: String...) -> String {
    for key in keys {
        if let value = dict[key] as? String, !value.isEmpty { return value }
    }
    return ""
}

func intValue(_ dict: [String: Any], _ keys: String...) -> Int {
    for key in keys {
        if let value = dict[key] as? Int { return value }
        if let value = dict[key] as? Double { return Int(value) }
        if let value = dict[key] as? String, let parsed = Int(value) { return parsed }
    }
    return 0
}

func boolValue(_ dict: [String: Any], _ keys: String...) -> Bool {
    for key in keys {
        if let value = dict[key] as? Bool { return value }
        if let value = dict[key] as? String {
            let v = value.lowercased()
            if v == "true" || v == "1" { return true }
            if v == "false" || v == "0" { return false }
        }
    }
    return false
}

func doubleValue(_ dict: [String: Any], _ keys: String...) -> Double {
    for key in keys {
        if let value = dict[key] as? Double { return value }
        if let value = dict[key] as? Int { return Double(value) }
        if let value = dict[key] as? String, let parsed = Double(value) { return parsed }
    }
    return 0
}
