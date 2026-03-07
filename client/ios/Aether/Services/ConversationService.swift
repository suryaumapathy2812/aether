import Foundation

struct ConversationTurnEvent {
    let phase: String
    let event: String?
    let text: String?
    let error: String?
    let payload: [String: Any]?
}

final class ConversationService {
    private let baseURL: String
    private let token: String
    private let proxyPrefix = "/api/go"

    init(baseURL: String, token: String) {
        self.baseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.token = token
    }

    func streamTurn(messages: [[String: Any]], sessionID: String, onEvent: @escaping (ConversationTurnEvent) -> Void) async throws {
        guard let url = URL(string: "\(baseURL)\(proxyPrefix)/v1/conversations/turn") else {
            throw NSError(domain: "ConversationService", code: 1, userInfo: [NSLocalizedDescriptionKey: "invalid URL"])
        }

        let body: [String: Any] = [
            "messages": messages,
            "user": "",
            "user_id": "",
            "session": sessionID,
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "ConversationService", code: 2, userInfo: [NSLocalizedDescriptionKey: "invalid response"])
        }
        guard (200...299).contains(http.statusCode) else {
            var errorBody = ""
            for try await line in bytes.lines {
                errorBody += line
                if errorBody.count > 1000 { break }
            }
            throw NSError(domain: "ConversationService", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: errorBody.isEmpty ? "conversation request failed" : errorBody])
        }

        for try await line in bytes.lines {
            if line.isEmpty || !line.hasPrefix("data:") {
                continue
            }

            let raw = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if raw == "[DONE]" {
                break
            }
            guard let data = raw.data(using: .utf8), let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            let event = ConversationTurnEvent(
                phase: obj["phase"] as? String ?? "",
                event: obj["event"] as? String,
                text: obj["text"] as? String,
                error: obj["error"] as? String,
                payload: obj["payload"] as? [String: Any]
            )
            onEvent(event)
        }
    }
}
