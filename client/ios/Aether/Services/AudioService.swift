import Foundation
import AVFoundation
import WebRTC

// MARK: - Orb State

enum OrbState: Equatable {
    case idle, connecting, listening, thinking, speaking, muted
}

// MARK: - Audio Service (WebRTC)
//
// Voice-to-voice via WebRTC:
//   - Audio in:  local mic → WebRTC audio track → server (STT)
//   - Audio out: server (TTS) → WebRTC audio track → speaker (automatic)
//   - Events:    data channel carries status, transcript, text_chunk, stream_end, etc.
//
// The server's SmallWebRTCTransport + CoreHandler handle STT/LLM/TTS.
// This client manages the WebRTC connection, mic permission, and UI state.

class AudioService: NSObject, ObservableObject {
    @Published var state: OrbState = .connecting
    @Published var statusText: String = "connecting..."
    @Published var lastResponse: String = ""
    @Published var liveTranscript: String = ""
    @Published var isStreaming: Bool = false
    @Published var isMuted: Bool = false
    @Published var userSpeechLevel: CGFloat = 0
    @Published var agentSpeechLevel: CGFloat = 0
    @Published var micPermissionDenied: Bool = false

    private let webrtc = WebRTCService()
    private var token = ""
    private var baseURL = "http://localhost:3080"

    /// Reconnection state
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 10
    private let baseReconnectDelay: TimeInterval = 3.0
    private let maxReconnectDelay: TimeInterval = 30.0
    private var shouldAutoStart = true
    private var speechDecayTimer: Timer?

    /// Tracks streaming intent synchronously (before the async @Published update lands).
    /// Use this in callbacks that fire before the main-thread dispatch completes.
    private var _streamingIntent = false

    /// Set to true when WE initiated the disconnect (stream_stop) so we don't
    /// trigger a reconnect loop for an expected server-side close.
    private var _intentionalDisconnect = false

    deinit {
        speechDecayTimer?.invalidate()
        webrtc.disconnect()
    }

    func configure(token: String, orchestratorURL: String) {
        self.token = token
        self.baseURL = orchestratorURL

        webrtc.configure(token: token, orchestratorURL: orchestratorURL)
        webrtc.onMessage = { [weak self] json in
            self?.handleServerMessage(json)
        }
        webrtc.onConnectionStateChange = { [weak self] state in
            self?.handleConnectionStateChange(state)
        }
        webrtc.onDataChannelReady = { [weak self] isReady in
            self?.handleDataChannelReady(isReady)
        }

        startSpeechDecayTimer()

        // Request mic permission first, then connect
        WebRTCService.requestMicrophonePermission { [weak self] granted in
            guard let self else { return }
            if granted {
                self.connectWebRTC()
            } else {
                self.micPermissionDenied = true
                self.state = .idle
                self.statusText = "microphone access required"
                print("[AudioService] Microphone permission denied")
            }
        }
    }

    // MARK: - WebRTC Connection

    private func connectWebRTC() {
        DispatchQueue.main.async {
            self.state = .connecting
            self.statusText = "connecting..."
        }

        Task {
            do {
                try await webrtc.connect()
                print("[AudioService] WebRTC connected")
                reconnectAttempts = 0
            } catch {
                print("[AudioService] WebRTC connect failed: \(error)")
                scheduleReconnect(wasStreaming: isStreaming)
            }
        }
    }

    private func handleConnectionStateChange(_ iceState: RTCIceConnectionState) {
        switch iceState {
        case .connected, .completed:
            DispatchQueue.main.async {
                if !self.isStreaming {
                    self.state = .connecting
                    self.statusText = "connected"
                }
            }
            maybeAutoStartStreaming()

        case .disconnected, .failed:
            // If we sent stream_stop, the server closes the WebRTC session — that's
            // expected. Don't reconnect; just reset state and wait for user to tap again.
            if _intentionalDisconnect {
                print("[AudioService] Expected disconnect after stream_stop — not reconnecting")
                _intentionalDisconnect = false
                DispatchQueue.main.async {
                    self.isStreaming = false
                    self.isMuted = false
                    self.userSpeechLevel = 0
                    self.agentSpeechLevel = 0
                    self.state = .idle
                    self.statusText = "tap to speak"
                }
                return
            }
            let wasStreaming = _streamingIntent
            DispatchQueue.main.async {
                self.isStreaming = false
                self.isMuted = false
                self.userSpeechLevel = 0
                self.agentSpeechLevel = 0
            }
            scheduleReconnect(wasStreaming: wasStreaming)

        case .closed:
            DispatchQueue.main.async {
                self.state = .idle
                self.statusText = "disconnected"
            }

        default:
            break
        }
    }

    private func handleDataChannelReady(_ isReady: Bool) {
        if isReady {
            maybeAutoStartStreaming()
        }
    }

    private func maybeAutoStartStreaming() {
        guard shouldAutoStart else { return }
        guard webrtc.isConnected && webrtc.isDataChannelOpen else { return }
        shouldAutoStart = false
        _intentionalDisconnect = false
        startStreaming()
    }

    private func scheduleReconnect(wasStreaming: Bool) {
        guard reconnectAttempts < maxReconnectAttempts else {
            print("[AudioService] Max reconnect attempts reached")
            DispatchQueue.main.async {
                self.state = .idle
                self.statusText = "connection lost — restart app"
            }
            return
        }

        reconnectAttempts += 1
        let delay = min(baseReconnectDelay * pow(2.0, Double(reconnectAttempts - 1)), maxReconnectDelay)
        print("[AudioService] Reconnecting in \(delay)s (attempt \(reconnectAttempts)/\(maxReconnectAttempts))...")

        DispatchQueue.main.async {
            self.state = .connecting
            self.statusText = "reconnecting (\(self.reconnectAttempts)/\(self.maxReconnectAttempts))..."
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.webrtc.disconnect()
            self?.shouldAutoStart = wasStreaming
            self?.connectWebRTC()
        }
    }

    // MARK: - Text Input (data channel)

    func sendText(_ text: String) {
        guard webrtc.isConnected else {
            print("[AudioService] Not connected, can't send text")
            return
        }
        print("[AudioService] Sending text: \(text.prefix(60))")

        webrtc.send(["type": "text", "data": text])

        DispatchQueue.main.async {
            self.state = .thinking
            self.statusText = "thinking..."
            self.lastResponse = ""
            self.liveTranscript = ""
        }
    }

    // MARK: - Toggle Streaming (tap orb)

    func toggleStreaming() {
        shouldAutoStart = false
        if isMuted {
            toggleMute()
            return
        }
        if isStreaming {
            stopStreaming()
        } else {
            startStreaming()
        }
    }

    private func startStreaming() {
        guard webrtc.isConnected && webrtc.isDataChannelOpen else {
            print("[AudioService] Not ready, can't start streaming")
            return
        }

        print("[AudioService] startStreaming")
        _streamingIntent = true
        _intentionalDisconnect = false

        webrtc.setMicEnabled(true)
        webrtc.send(["type": "stream_start"])

        DispatchQueue.main.async {
            self.isStreaming = true
            self.isMuted = false
            self.state = .listening
            self.statusText = "listening..."
            self.lastResponse = ""
            self.liveTranscript = ""
        }
    }

    private func stopStreaming() {
        print("[AudioService] stopStreaming")
        _streamingIntent = false
        _intentionalDisconnect = true

        webrtc.setMicEnabled(false)
        webrtc.send(["type": "stream_stop"])

        DispatchQueue.main.async {
            self.isStreaming = false
            self.isMuted = false
            self.state = .idle
            self.statusText = "tap to speak"
            self.liveTranscript = ""
            self.userSpeechLevel = 0
            self.agentSpeechLevel = 0
        }
    }

    // MARK: - Mute

    func toggleMute() {
        guard isStreaming else { return }
        isMuted.toggle()

        if isMuted {
            webrtc.setMicEnabled(false)
            webrtc.send(["type": "mute"])
            DispatchQueue.main.async {
                self.state = .muted
                self.statusText = "muted"
                self.liveTranscript = ""
            }
            print("[AudioService] Muted")
        } else {
            webrtc.setMicEnabled(true)
            webrtc.send(["type": "unmute"])
            DispatchQueue.main.async {
                self.state = .listening
                self.statusText = "listening..."
            }
            print("[AudioService] Unmuted")
        }
    }

    // MARK: - Speech Level Decay

    private func startSpeechDecayTimer() {
        speechDecayTimer?.invalidate()
        speechDecayTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.userSpeechLevel = max(0, self.userSpeechLevel * 0.85)
            self.agentSpeechLevel = max(0, self.agentSpeechLevel * 0.82)
        }
    }

    private func bumpUserSpeechLevel(for transcript: String) {
        let normalized = min(1.0, CGFloat(transcript.count) / 18.0)
        userSpeechLevel = max(userSpeechLevel, max(0.15, normalized))
    }

    private func bumpAgentSpeechLevel(for chunk: String) {
        let normalized = min(1.0, CGFloat(chunk.count) / 24.0)
        agentSpeechLevel = max(agentSpeechLevel, max(0.18, normalized))
    }

    // MARK: - Handle Server Messages (via data channel)

    private func handleServerMessage(_ json: [String: Any]) {
        guard let type = json["type"] as? String else { return }
        let payload = json["data"] as? String ?? ""

        switch type {
        case "status":
            print("[AudioService] Status: \(payload)")
            DispatchQueue.main.async {
                if self.isStreaming {
                    switch payload {
                    case "listening...":
                        self.state = .listening
                        self.statusText = "listening..."
                    case "thinking...":
                        self.state = .thinking
                        self.statusText = "thinking..."
                    case "muted":
                        self.state = .muted
                        self.statusText = "muted"
                    default:
                        self.statusText = payload
                    }
                }
            }

        case "transcript":
            let interim = json["interim"] as? Bool ?? false
            print("[AudioService] Transcript (interim=\(interim)): \(payload.prefix(60))")
            DispatchQueue.main.async {
                if interim {
                    self.liveTranscript = payload
                    self.bumpUserSpeechLevel(for: payload)
                } else {
                    self.liveTranscript = ""
                    self.state = .thinking
                    self.statusText = "thinking..."
                }
            }

        case "text_chunk":
            print("[AudioService] Text chunk: \(payload.prefix(60))")
            DispatchQueue.main.async {
                self.state = .speaking
                self.statusText = "speaking..."
                self.bumpAgentSpeechLevel(for: payload)
                // Accumulate chunks into lastResponse (displayed in VoiceOrbView)
                if self.lastResponse.isEmpty {
                    self.lastResponse = payload
                } else {
                    self.lastResponse += payload
                }
            }

        case "audio_chunk":
            // With WebRTC, TTS audio arrives via the audio track automatically.
            print("[AudioService] Audio chunk via data channel (unexpected in WebRTC mode)")
            DispatchQueue.main.async {
                self.state = .speaking
                self.statusText = "speaking..."
                self.bumpAgentSpeechLevel(for: payload)
            }

        case "stream_end":
            print("[AudioService] Stream end — streamingIntent=\(_streamingIntent)")
            // Use _streamingIntent (sync) not isStreaming (@Published, may lag behind)
            let shouldListen = _streamingIntent && !isMuted
            DispatchQueue.main.async {
                self.agentSpeechLevel = 0
                if shouldListen {
                    self.state = .listening
                    self.statusText = "listening..."
                } else {
                    self.state = .idle
                    self.statusText = "tap to speak"
                    self.userSpeechLevel = 0
                }
            }

        case "error":
            print("[AudioService] Error: \(payload)")
            DispatchQueue.main.async {
                self.statusText = payload
            }

        case "notification":
            if let notificationData = json["data"] as? [String: Any],
               let level = notificationData["level"] as? String,
               let text = notificationData["text"] as? String {
                print("[AudioService] Notification (\(level)): \(text)")

                switch level {
                case "speak":
                    DispatchQueue.main.async {
                        self.state = .speaking
                        self.statusText = "notification"
                        self.lastResponse = text
                    }
                case "nudge":
                    DispatchQueue.main.async {
                        self.state = .idle
                        self.statusText = "tap orb for update"
                    }
                case "batch":
                    let itemCount = (notificationData["items"] as? [[String: Any]])?.count ?? 0
                    DispatchQueue.main.async {
                        self.state = .idle
                        self.statusText = "\(itemCount) updates — tap orb"
                    }
                default:
                    break
                }

                webrtc.send([
                    "type": "notification_feedback",
                    "data": [
                        "event_id": notificationData["event_id"] as? String ?? "",
                        "action": "received",
                        "plugin": "unknown",
                        "sender": ""
                    ] as [String: Any]
                ])
            }

        default:
            print("[AudioService] Unknown type: \(type)")
        }
    }
}
