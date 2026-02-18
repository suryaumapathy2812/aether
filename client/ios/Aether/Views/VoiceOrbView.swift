import SwiftUI
import AVFoundation
import WebRTC

/// Voice orb screen — continuous conversation mode via WebRTC.
/// Tap orb to start listening, tap again to stop. Mute button to pause mic.
///
/// Audio flows natively via WebRTC audio tracks (no more base64 over WebSocket).
/// Text/events flow via the WebRTC data channel (same JSON protocol as before).
struct VoiceOrbView: View {
    @EnvironmentObject var pairing: PairingService
    @StateObject private var audio = AudioService()

    // Breathing animation
    @State private var breathe = false

    var orbBaseSize: CGFloat {
        switch audio.state {
        case .idle: return 120
        case .connecting: return 100
        case .listening: return 130
        case .thinking: return 110
        case .speaking: return 140
        case .muted: return 100
        }
    }

    /// Pulsing scale on top of base size
    var orbScale: CGFloat {
        switch audio.state {
        case .listening: return breathe ? 1.08 : 0.95
        case .thinking: return breathe ? 1.04 : 0.96
        case .speaking: return breathe ? 1.1 : 0.92
        case .connecting: return breathe ? 1.02 : 0.98
        default: return 1.0
        }
    }

    var orbOpacity: Double {
        switch audio.state {
        case .idle: return 0.06
        case .connecting: return 0.04
        case .listening: return 0.15
        case .thinking: return 0.1
        case .speaking: return 0.22
        case .muted: return 0.03
        }
    }

    var glowRadius: CGFloat {
        switch audio.state {
        case .idle: return 15
        case .connecting: return 8
        case .listening: return breathe ? 55 : 35
        case .thinking: return breathe ? 35 : 20
        case .speaking: return breathe ? 70 : 40
        case .muted: return 3
        }
    }

    var pulseSpeed: Double {
        switch audio.state {
        case .listening: return 2.0
        case .thinking: return 1.2
        case .speaking: return 0.8
        default: return 3.0
        }
    }

    var body: some View {
        ZStack {
            Color(hex: "111111").ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                // Orb — tap to toggle streaming
                ZStack {
                    // Outer glow ring
                    Circle()
                        .fill(Color.white.opacity(orbOpacity * 0.3))
                        .frame(width: orbBaseSize + 40, height: orbBaseSize + 40)
                        .blur(radius: 20)
                        .scaleEffect(orbScale * 1.1)

                    // Main orb
                    Circle()
                        .fill(Color.white.opacity(orbOpacity))
                        .frame(width: orbBaseSize, height: orbBaseSize)
                        .shadow(color: .white.opacity(orbOpacity * 0.6), radius: glowRadius)
                        .scaleEffect(orbScale)
                }
                .animation(.easeInOut(duration: pulseSpeed), value: breathe)
                .animation(.easeInOut(duration: 0.6), value: audio.state)
                .onTapGesture {
                    audio.toggleStreaming()
                }

                // Minimal status — only when needed
                Text(audio.statusText)
                    .font(.system(size: 10, weight: .light))
                    .tracking(3)
                    .foregroundColor(.white.opacity(0.2))
                    .padding(.top, 32)

                Spacer()
                Spacer()

                // Mute — minimal, only when streaming
                if audio.isStreaming {
                    Button(action: { audio.toggleMute() }) {
                        Circle()
                            .stroke(audio.isMuted ? Color.red.opacity(0.3) : Color.white.opacity(0.1), lineWidth: 1)
                            .frame(width: 44, height: 44)
                            .overlay(
                                Image(systemName: audio.isMuted ? "mic.slash" : "mic")
                                    .font(.system(size: 14, weight: .light))
                                    .foregroundColor(audio.isMuted ? .red.opacity(0.5) : .white.opacity(0.25))
                            )
                    }
                    .padding(.bottom, 20)
                    .transition(.opacity)
                }

                // Brand
                Text("aether")
                    .font(.system(size: 11, weight: .ultraLight))
                    .tracking(8)
                    .foregroundColor(.white.opacity(0.1))
                    .padding(.bottom, 40)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            let token = pairing.getDeviceToken() ?? ""
            print("[VoiceOrbView] Configuring with token: \(token.isEmpty ? "(empty!)" : String(token.prefix(20)) + "...")")
            audio.configure(token: token, orchestratorURL: pairing.orchestratorURL)
            startBreathing()
        }
        .onChange(of: audio.state) {
            startBreathing()
        }
    }

    private func startBreathing() {
        // Continuous breathing animation
        withAnimation(.easeInOut(duration: pulseSpeed).repeatForever(autoreverses: true)) {
            breathe = true
        }
    }
}

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
// This client just manages the WebRTC connection and UI state.

class AudioService: NSObject, ObservableObject {
    @Published var state: OrbState = .connecting
    @Published var statusText: String = "connecting..."
    @Published var lastResponse: String = ""
    @Published var liveTranscript: String = ""
    @Published var isStreaming: Bool = false
    @Published var isMuted: Bool = false

    private let webrtc = WebRTCService()
    private var token = ""
    private var baseURL = "http://localhost:9000"

    /// Reconnection state
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 10
    private let baseReconnectDelay: TimeInterval = 3.0
    private let maxReconnectDelay: TimeInterval = 30.0

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

        connectWebRTC()
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
                // Connection state callback will update UI when ICE connects
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
                    self.state = .idle
                    self.statusText = "tap the orb to speak"
                }
            }

        case .disconnected, .failed:
            let wasStreaming = isStreaming
            DispatchQueue.main.async {
                self.isStreaming = false
                self.isMuted = false
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
            self?.connectWebRTC()
            if wasStreaming {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    self?.startStreaming()
                }
            }
        }
    }

    // MARK: - Toggle Streaming (tap orb)

    func toggleStreaming() {
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
        guard webrtc.isConnected else {
            print("[AudioService] Not connected, can't start streaming")
            return
        }

        print("[AudioService] startStreaming")

        // Setup audio session for playback + recording
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker])
            try audioSession.setActive(true)
        } catch {
            print("[AudioService] Audio session error: \(error)")
        }

        // Enable mic track
        webrtc.setMicEnabled(true)

        // Tell server to start voice session (opens STT, etc.)
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

        // Disable mic track
        webrtc.setMicEnabled(false)

        // Tell server to stop
        webrtc.send(["type": "stream_stop"])

        DispatchQueue.main.async {
            self.isStreaming = false
            self.isMuted = false
            self.state = .idle
            self.statusText = "tap the orb to speak"
            self.liveTranscript = ""
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
                } else {
                    self.liveTranscript = ""
                    self.state = .thinking
                    self.statusText = "thinking..."
                }
            }

        case "text_chunk":
            let index = json["index"] as? Int ?? 0
            print("[AudioService] Text chunk [\(index)]: \(payload.prefix(60))")
            DispatchQueue.main.async {
                self.state = .speaking
                self.statusText = "speaking..."
                if index == 0 {
                    self.lastResponse = payload
                } else {
                    self.lastResponse += " " + payload
                }
            }

        case "audio_chunk":
            // With WebRTC, TTS audio arrives via the audio track automatically.
            // This message type is only for data-channel fallback (shouldn't happen
            // in normal WebRTC mode, but handle gracefully).
            print("[AudioService] Audio chunk via data channel (unexpected in WebRTC mode)")
            DispatchQueue.main.async {
                self.state = .speaking
                self.statusText = "speaking..."
            }

        case "stream_end":
            print("[AudioService] Stream end")
            DispatchQueue.main.async {
                // Resume listening after agent finishes speaking
                // With WebRTC, audio playback is handled by the audio track,
                // so we just need to update the UI state.
                if self.isStreaming && !self.isMuted {
                    self.state = .listening
                    self.statusText = "listening..."
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

                // Send feedback
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
