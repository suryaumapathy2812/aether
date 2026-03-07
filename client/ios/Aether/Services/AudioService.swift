import Foundation
import CoreGraphics

enum OrbState: Equatable {
    case idle
    case recording
    case uploading
    case thinking
    case speaking
    case error
}

@MainActor
final class AudioService: ObservableObject {
    @Published var state: OrbState = .idle
    @Published var statusText: String = "hold to record"
    @Published var lastResponse: String = ""
    @Published var userSpeechLevel: CGFloat = 0
    @Published var agentSpeechLevel: CGFloat = 0
    @Published var recordingDuration: TimeInterval = 0
    @Published var micPermissionDenied: Bool = false

    private let recorder = AudioRecorderService()
    private var uploader: MediaUploadService?
    private var conversation: ConversationService?

    private var token = ""
    private var baseURL = ""
    private var sessionID = "ios-voice"

    private var history: [[String: Any]] = []
    private let maxHistoryMessages = 20
    private var speechDecayTimer: Timer?

    deinit {
        speechDecayTimer?.invalidate()
        recorder.stopRecording(discard: true)
    }

    func configure(token: String, orchestratorURL: String) {
        self.token = token
        self.baseURL = orchestratorURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.uploader = MediaUploadService(baseURL: self.baseURL, token: token)
        self.conversation = ConversationService(baseURL: self.baseURL, token: token)

        recorder.onPowerLevel = { [weak self] level in
            self?.userSpeechLevel = max(self?.userSpeechLevel ?? 0, level)
        }
        recorder.onDuration = { [weak self] duration in
            self?.recordingDuration = duration
        }
        startSpeechDecayTimer()
        statusText = "hold to record"
    }

    func beginRecording() {
        guard state == .idle || state == .error else { return }
        guard !token.isEmpty else {
            state = .error
            statusText = "missing device token"
            return
        }

        Task { @MainActor in
            let granted = await AudioRecorderService.requestMicrophonePermission()
            if !granted {
                micPermissionDenied = true
                state = .error
                statusText = "microphone access required"
                return
            }

            do {
                try recorder.startRecording()
                state = .recording
                statusText = "recording... release to send"
                lastResponse = ""
                recordingDuration = 0
            } catch {
                state = .error
                statusText = "failed to start recording"
            }
        }
    }

    func endRecordingAndSend() {
        guard state == .recording else { return }
        guard let fileURL = recorder.finishRecording() else {
            state = .error
            statusText = "no audio captured"
            return
        }

        Task {
            await MainActor.run {
                self.state = .uploading
                self.statusText = "uploading voice..."
                self.userSpeechLevel = 0
                self.recordingDuration = 0
            }

            do {
                guard let uploader else { throw NSError(domain: "AudioService", code: 1, userInfo: [NSLocalizedDescriptionKey: "uploader unavailable"]) }
                let media = try await uploader.uploadAudio(fileURL: fileURL, sessionID: sessionID)
                try? FileManager.default.removeItem(at: fileURL)
                try await sendTurn(userContent: [
                    ["type": "audio_ref", "media": media.asDictionary()],
                ], userSummary: "[voice]")
            } catch {
                await MainActor.run {
                    self.state = .error
                    self.statusText = "voice send failed"
                }
            }
        }
    }

    func cancelRecording() {
        guard state == .recording else { return }
        recorder.stopRecording(discard: true)
        state = .idle
        statusText = "hold to record"
        recordingDuration = 0
    }

    func sendText(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        Task {
            do {
                try await sendTurn(userContent: [["type": "text", "text": trimmed]], userSummary: trimmed)
            } catch {
                await MainActor.run {
                    self.state = .error
                    self.statusText = "text send failed"
                }
            }
        }
    }

    private func sendTurn(userContent: [[String: Any]], userSummary: String) async throws {
        guard let conversation else { throw NSError(domain: "AudioService", code: 2, userInfo: [NSLocalizedDescriptionKey: "conversation unavailable"]) }

        var messages = history
        let userMessage: [String: Any] = ["role": "user", "content": userContent]
        messages.append(userMessage)

        await MainActor.run {
            self.state = .thinking
            self.statusText = "thinking..."
            self.lastResponse = ""
            self.agentSpeechLevel = 0
        }

        var assistantAnswer = ""
        var lastRenderedCount = 0
        var lastRenderedAt = Date.distantPast
        try await conversation.streamTurn(messages: messages, sessionID: sessionID) { [weak self] (event: ConversationTurnEvent) in
            guard let self else { return }
            DispatchQueue.main.async {
                switch event.phase {
                case "ack":
                    if assistantAnswer.isEmpty {
                        self.statusText = event.text ?? "thinking..."
                    }
                case "answer":
                    let chunk = event.text ?? ""
                    assistantAnswer += chunk
                    let shouldRender =
                        (assistantAnswer.count - lastRenderedCount) >= 24 ||
                        Date().timeIntervalSince(lastRenderedAt) >= 0.1 ||
                        chunk.contains("\n")
                    if shouldRender {
                        self.lastResponse = assistantAnswer
                        self.bumpAgentSpeechLevel(for: chunk)
                        lastRenderedCount = assistantAnswer.count
                        lastRenderedAt = Date()
                    }
                    self.state = .speaking
                    self.statusText = "responding..."
                case "error":
                    self.state = .error
                    self.statusText = event.error ?? "conversation failed"
                case "done":
                    self.lastResponse = assistantAnswer
                    self.state = .idle
                    self.statusText = "hold to record"
                default:
                    break
                }
            }
        }

        if !assistantAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            history.append(["role": "user", "content": userContent, "summary": userSummary])
            history.append(["role": "assistant", "content": assistantAnswer])
            if history.count > maxHistoryMessages {
                history = Array(history.suffix(maxHistoryMessages))
            }
        }
    }

    private func startSpeechDecayTimer() {
        speechDecayTimer?.invalidate()
        speechDecayTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.userSpeechLevel = max(0, self.userSpeechLevel * 0.86)
                self.agentSpeechLevel = max(0, self.agentSpeechLevel * 0.82)
            }
        }
    }

    private func bumpAgentSpeechLevel(for chunk: String) {
        let normalized = min(1.0, CGFloat(chunk.count) / 24.0)
        agentSpeechLevel = max(agentSpeechLevel, max(0.18, normalized))
    }
}
