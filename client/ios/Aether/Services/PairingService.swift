import Foundation
import Combine
import UIKit

/// Handles device pairing with the Aether orchestrator.
/// Generates a code, registers it, polls for confirmation.
class PairingService: ObservableObject {
    @Published var pairingCode: String = ""
    @Published var isPaired: Bool = false
    @Published var status: String = "ready"
    @Published var orchestratorURL: String

    /// In-memory token (primary) — Keychain is fallback for app restarts.
    private var cachedToken: String?
    private var pollTimer: Timer?
    private static let defaultOrchestratorURL = "https://aether.suryaumapathy.in"

    init() {
        // Load from UserDefaults or use default
        self.orchestratorURL = UserDefaults.standard.string(forKey: "orchestrator_url")
            ?? Self.defaultOrchestratorURL

        // Check if already paired (try Keychain, also try UserDefaults as fallback)
        if let token = KeychainHelper.load(key: "device_token") {
            self.cachedToken = token
            self.isPaired = true
            print("[PairingService] Restored token from Keychain")
        } else if let token = UserDefaults.standard.string(forKey: "device_token") {
            self.cachedToken = token
            self.isPaired = true
            print("[PairingService] Restored token from UserDefaults")
        }
    }

    /// Generate a new pairing code and register with orchestrator.
    func startPairing() {
        let normalizedBaseURL = normalizedOrchestratorURL(orchestratorURL)
        guard let pairURL = URL(string: "\(normalizedBaseURL)/go/v1/pair/request") else {
            DispatchQueue.main.async {
                self.status = "invalid orchestrator URL"
            }
            return
        }

        // Persist normalized URL so VoiceOrb uses the same host.
        saveOrchestratorURL(normalizedBaseURL)

        let code = generateCode()
        self.pairingCode = code
        self.status = "waiting for pairing..."

        // Register the code with orchestrator
        var request = URLRequest(url: pairURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let deviceName = UIDevice.current.name
        let body: [String: String] = [
            "code": code,
            "device_type": "ios",
            "device_name": deviceName,
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    self?.status = "failed: \(error.localizedDescription)"
                }
                return
            }

            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard statusCode == 200 || statusCode == 201 else {
                DispatchQueue.main.async {
                    self?.status = "failed: HTTP \(statusCode)"
                }
                return
            }

            // Start polling on main thread (Timer needs main RunLoop)
            DispatchQueue.main.async {
                self?.startPolling(code: code, baseURL: normalizedBaseURL)
            }
        }.resume()
    }

    /// Poll orchestrator to check if code was confirmed.
    private func startPolling(code: String, baseURL: String) {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            guard let url = URL(string: "\(baseURL)/go/v1/pair/status/\(code)") else {
                DispatchQueue.main.async {
                    self.status = "invalid orchestrator URL"
                }
                self.pollTimer?.invalidate()
                return
            }
            URLSession.shared.dataTask(with: url) { data, _, _ in
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let status = json["status"] as? String
                else { return }

                if status == "paired",
                   let token = json["device_token"] as? String {
                    DispatchQueue.main.async {
                        print("[PairingService] Paired! Token: \(token.prefix(20))...")
                        // Save in memory (primary), UserDefaults (reliable), Keychain (secure)
                        self.cachedToken = token
                        UserDefaults.standard.set(token, forKey: "device_token")
                        KeychainHelper.save(key: "device_token", value: token)
                        self.isPaired = true
                        self.status = "paired"
                        self.pollTimer?.invalidate()
                    }
                }
            }.resume()
        }
    }

    func saveOrchestratorURL(_ value: String) {
        let normalized = normalizedOrchestratorURL(value)
        orchestratorURL = normalized
        UserDefaults.standard.set(normalized, forKey: "orchestrator_url")
    }

    private func normalizedOrchestratorURL(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return Self.defaultOrchestratorURL
        }
        let normalized: String
        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") {
            normalized = trimmed
        } else {
            normalized = "https://\(trimmed)"
        }

        return normalized
    }

    func getDeviceToken() -> String? {
        if let token = cachedToken {
            return token
        }
        // Fallback: try Keychain then UserDefaults
        if let token = KeychainHelper.load(key: "device_token") {
            cachedToken = token
            return token
        }
        if let token = UserDefaults.standard.string(forKey: "device_token") {
            cachedToken = token
            return token
        }
        return nil
    }

    func unpair() {
        cachedToken = nil
        KeychainHelper.delete(key: "device_token")
        UserDefaults.standard.removeObject(forKey: "device_token")
        isPaired = false
        pairingCode = ""
        status = "ready"
    }

    private func generateCode() -> String {
        let chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        let seg1 = String((0..<4).map { _ in chars.randomElement()! })
        let seg2 = String((0..<4).map { _ in chars.randomElement()! })
        return "AETHER-\(seg1)-\(seg2)"
    }
}

// MARK: - Simple Keychain Helper

enum KeychainHelper {
    static func save(key: String, value: String) {
        let data = value.data(using: .utf8)!
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
        ]
        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }

    static func load(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
        ]
        var result: AnyObject?
        SecItemCopyMatching(query as CFDictionary, &result)
        guard let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func delete(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
