import Foundation

struct MediaRef {
    let bucket: String?
    let key: String
    let url: String?
    let mime: String?
    let size: Int?
    let fileName: String?
    let format: String?

    func asDictionary() -> [String: Any] {
        var out: [String: Any] = ["key": key]
        if let bucket { out["bucket"] = bucket }
        if let url { out["url"] = url }
        if let mime { out["mime"] = mime }
        if let size { out["size"] = size }
        if let fileName { out["file_name"] = fileName }
        if let format { out["format"] = format }
        return out
    }
}

final class MediaUploadService {
    private let baseURL: String
    private let token: String

    init(baseURL: String, token: String) {
        self.baseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.token = token
    }

    func uploadAudio(fileURL: URL, sessionID: String) async throws -> MediaRef {
        let fileName = fileURL.lastPathComponent
        let mimeType = mimeTypeForAudio(pathExtension: fileURL.pathExtension)
        let size = try fileSize(for: fileURL)

        let initBody: [String: Any] = [
            "user_id": "",
            "session_id": sessionID,
            "file_name": fileName,
            "content_type": mimeType,
            "size": size,
            "kind": "audio",
        ]
        let initResponse = try await postJSON(path: "/agent/v1/media/upload/init", body: initBody)

        guard
            let uploadURL = initResponse["upload_url"] as? String,
            let uploadURLValue = URL(string: uploadURL),
            let objectKey = initResponse["object_key"] as? String
        else {
            throw NSError(domain: "MediaUploadService", code: 1, userInfo: [NSLocalizedDescriptionKey: "invalid upload init response"])
        }

        var putRequest = URLRequest(url: uploadURLValue)
        putRequest.httpMethod = "PUT"
        if let headers = initResponse["headers"] as? [String: Any] {
            for (key, value) in headers {
                putRequest.setValue("\(value)", forHTTPHeaderField: key)
            }
        }

        let (_, putResponse) = try await URLSession.shared.upload(for: putRequest, fromFile: fileURL)
        guard let httpPut = putResponse as? HTTPURLResponse, (200...299).contains(httpPut.statusCode) else {
            throw NSError(domain: "MediaUploadService", code: 2, userInfo: [NSLocalizedDescriptionKey: "upload failed"])
        }

        let completeBody: [String: Any] = [
            "user_id": "",
            "bucket": initResponse["bucket"] as? String ?? "",
            "object_key": objectKey,
            "file_name": fileName,
            "content_type": mimeType,
            "size": size,
            "kind": "audio",
        ]
        let completeResponse = try await postJSON(path: "/agent/v1/media/upload/complete", body: completeBody)
        guard let media = completeResponse["media"] as? [String: Any], let key = media["key"] as? String else {
            throw NSError(domain: "MediaUploadService", code: 3, userInfo: [NSLocalizedDescriptionKey: "invalid upload complete response"])
        }

        return MediaRef(
            bucket: media["bucket"] as? String,
            key: key,
            url: media["url"] as? String,
            mime: media["mime"] as? String,
            size: media["size"] as? Int,
            fileName: media["file_name"] as? String,
            format: media["format"] as? String
        )
    }

    private func postJSON(path: String, body: [String: Any]) async throws -> [String: Any] {
        guard let url = URL(string: "\(baseURL)\(path)") else {
            throw NSError(domain: "MediaUploadService", code: 100, userInfo: [NSLocalizedDescriptionKey: "invalid URL"])
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "MediaUploadService", code: 101, userInfo: [NSLocalizedDescriptionKey: "invalid response"])
        }
        guard (200...299).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "request failed"
            throw NSError(domain: "MediaUploadService", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: msg])
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "MediaUploadService", code: 102, userInfo: [NSLocalizedDescriptionKey: "invalid json response"])
        }
        return json
    }

    private func fileSize(for fileURL: URL) throws -> Int {
        let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = attrs[.size] as? NSNumber
        return fileSize?.intValue ?? 0
    }

    private func mimeTypeForAudio(pathExtension: String) -> String {
        switch pathExtension.lowercased() {
        case "wav":
            return "audio/wav"
        case "mp3":
            return "audio/mpeg"
        case "flac":
            return "audio/flac"
        case "aac":
            return "audio/aac"
        case "ogg":
            return "audio/ogg"
        case "m4a":
            return "audio/mp4"
        default:
            return "audio/mp4"
        }
    }
}
