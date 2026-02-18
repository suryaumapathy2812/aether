import SwiftUI
import AVFoundation

/// Voice orb screen — continuous conversation mode.
/// Tap orb to start listening, tap again to stop. Mute button to pause mic.
/// Matches the web client protocol exactly.
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

// MARK: - Audio Service
// Continuous conversation mode matching web client protocol:
//   Tap orb → stream_start + mic open → audio_chunk (base64 JSON) flows continuously
//   Tap orb again → stream_stop + mic close
//   Mute → mute/unmute messages, mic stays open but audio not sent
//   Agent handles silence detection via Deepgram utterance_end + debounce

class AudioService: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var state: OrbState = .connecting
    @Published var statusText: String = "connecting..."
    @Published var lastResponse: String = ""
    @Published var liveTranscript: String = ""
    @Published var isStreaming: Bool = false
    @Published var isMuted: Bool = false

    private var ws: URLSessionWebSocketTask?
    private let engine = AVAudioEngine()
    private var token = ""
    private var baseURL = "http://localhost:9000"
    private var isConnected = false

    /// Echo cancellation: suppress mic while agent is speaking
    private var isSpeaking = false
    /// Audio playback queue
    private var audioQueue: [Data] = []
    private var audioPlayer: AVAudioPlayer?
    private var isPlaying = false
    /// Whether we received stream_end (response complete)
    private var streamEnded = false

    /// Reconnection state
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 10
    private let baseReconnectDelay: TimeInterval = 3.0
    private let maxReconnectDelay: TimeInterval = 30.0

    func configure(token: String, orchestratorURL: String) {
        self.token = token
        self.baseURL = orchestratorURL
        connectWebSocket()
    }

    // MARK: - WebSocket

    private func connectWebSocket() {
        ws?.cancel(with: .goingAway, reason: nil)

        let wsURL = baseURL
            .replacingOccurrences(of: "http://", with: "ws://")
            .replacingOccurrences(of: "https://", with: "wss://")
            + "/api/ws?token=\(token)"
        print("[AudioService] Connecting to: \(wsURL.prefix(80))...")
        guard let url = URL(string: wsURL) else {
            print("[AudioService] Invalid URL!")
            return
        }

        let session = URLSession(configuration: .default)
        ws = session.webSocketTask(with: url)
        ws?.resume()
        isConnected = true
        reconnectAttempts = 0
        print("[AudioService] WebSocket resumed")

        DispatchQueue.main.async {
            self.state = .idle
            self.statusText = "tap the orb to speak"
        }

        listenForMessages()
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
            self?.connectWebSocket()
            if wasStreaming {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self?.startStreaming()
                }
            }
        }
    }

    private func sendJSON(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        ws?.send(.string(text)) { error in
            if let error = error {
                print("[AudioService] Send error: \(error.localizedDescription)")
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
        print("[AudioService] startStreaming")

        // Clean up any existing tap before installing a new one
        // (prevents crash if called twice, e.g. after reconnect)
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }

        // Send stream_start to agent (opens Deepgram STT connection)
        sendJSON(["type": "stream_start"])

        // Setup audio session
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try audioSession.setActive(true)
        } catch {
            print("[AudioService] Audio session error: \(error)")
        }

        // Install mic tap — continuously send audio chunks
        // Deepgram expects 16kHz mono linear16. The Simulator mic is 48kHz.
        // We use AVAudioConverter to resample properly.
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        print("[AudioService] Input: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount)ch")

        let targetSampleRate: Double = 16000
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetSampleRate, channels: 1, interleaved: false) else {
            print("[AudioService] Failed to create target format")
            return
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            print("[AudioService] Failed to create converter")
            return
        }

        var chunkCount = 0
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self, !self.isMuted, !self.isSpeaking else { return }

            // Resample to 16kHz
            let ratio = targetSampleRate / inputFormat.sampleRate
            let outputFrames = AVAudioFrameCount(Double(buffer.frameLength) * ratio)
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrames) else { return }

            var error: NSError?
            let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            guard status != .error, error == nil else {
                print("[AudioService] Convert error: \(error?.localizedDescription ?? "unknown")")
                return
            }

            // Convert float32 → PCM16 little-endian
            guard let channelData = outputBuffer.floatChannelData?[0] else { return }
            let frames = Int(outputBuffer.frameLength)
            var pcm16 = Data(capacity: frames * 2)
            for i in 0..<frames {
                let sample = Int16(max(-1, min(1, channelData[i])) * 32767)
                var le = sample.littleEndian
                pcm16.append(Data(bytes: &le, count: 2))
            }

            // Send as base64 JSON text message
            let b64 = pcm16.base64EncodedString()
            self.sendJSON(["type": "audio_chunk", "data": b64])

            chunkCount += 1
            if chunkCount <= 3 || chunkCount % 100 == 0 {
                print("[AudioService] Audio chunk #\(chunkCount) (\(pcm16.count) bytes, 16kHz)")
            }
        }

        do {
            try engine.start()
            print("[AudioService] Engine started")
        } catch {
            print("[AudioService] Engine start error: \(error)")
            return
        }

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

        // Stop mic safely
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }

        // Tell agent to close STT stream
        sendJSON(["type": "stream_stop"])

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
            sendJSON(["type": "mute"])
            DispatchQueue.main.async {
                self.state = .muted
                self.statusText = "muted"
                self.liveTranscript = ""
            }
            print("[AudioService] Muted")
        } else {
            sendJSON(["type": "unmute"])
            DispatchQueue.main.async {
                self.state = .listening
                self.statusText = "listening..."
            }
            print("[AudioService] Unmuted")
        }
    }

    // MARK: - Receive

    private func listenForMessages() {
        ws?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    self?.handleServerMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        self?.handleServerMessage(text)
                    }
                @unknown default:
                    break
                }
                self?.listenForMessages()

            case .failure(let error):
                print("[AudioService] WS error: \(error.localizedDescription)")
                let wasStreaming = self?.isStreaming ?? false

                // Check for auth errors (don't reconnect on 4001/4002)
                let errorDesc = error.localizedDescription.lowercased()
                let isAuthError = errorDesc.contains("4001") || errorDesc.contains("invalid token")
                if isAuthError {
                    print("[AudioService] Auth error — not reconnecting")
                    DispatchQueue.main.async {
                        self?.isConnected = false
                        self?.state = .idle
                        self?.statusText = "auth failed — re-pair device"
                    }
                    return
                }

                DispatchQueue.main.async {
                    self?.isConnected = false
                    // Stop engine safely during reconnect
                    if self?.engine.isRunning == true {
                        self?.engine.inputNode.removeTap(onBus: 0)
                        self?.engine.stop()
                    }
                    self?.isStreaming = false
                    self?.isSpeaking = false
                    self?.audioQueue.removeAll()
                    self?.isPlaying = false
                }
                self?.scheduleReconnect(wasStreaming: wasStreaming)
            }
        }
    }

    private func handleServerMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String
        else {
            print("[AudioService] Unparseable: \(text.prefix(80))")
            return
        }

        let payload = json["data"] as? String ?? ""

        switch type {
        case "status":
            print("[AudioService] Status: \(payload)")
            DispatchQueue.main.async {
                // Only update state if we're streaming — don't override idle
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
                    // Final transcript — user's utterance confirmed
                    self.liveTranscript = ""
                    self.state = .thinking
                    self.statusText = "thinking..."
                }
            }

        case "text_chunk":
            let index = json["index"] as? Int ?? 0
            print("[AudioService] Text chunk [\(index)]: \(payload.prefix(60))")
            isSpeaking = true  // Suppress mic to prevent echo
            if index == 0 {
                streamEnded = false  // New response starting
            }
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
            if let audioData = Data(base64Encoded: payload) {
                print("[AudioService] Audio chunk: \(audioData.count) bytes")
                DispatchQueue.main.async {
                    self.state = .speaking
                    self.statusText = "speaking..."
                }
                queueAudio(data: audioData)
            }

        case "stream_end":
            print("[AudioService] Stream end")
            DispatchQueue.main.async {
                self.streamEnded = true
                // If nothing is playing, resume mic now
                if !self.isPlaying {
                    self.resumeListening()
                }
                // Otherwise, resumeListening will be called when playback finishes
            }

        case "error":
            print("[AudioService] Error: \(payload)")
            DispatchQueue.main.async {
                self.statusText = payload
            }

        default:
            print("[AudioService] Unknown type: \(type)")
        }
    }

    // MARK: - Playback Queue

    private func queueAudio(data: Data) {
        audioQueue.append(data)
        if !isPlaying {
            playNext()
        }
    }

    private func playNext() {
        guard !audioQueue.isEmpty else {
            isPlaying = false
            // All audio finished playing — if stream also ended, resume mic
            if streamEnded {
                resumeListening()
            }
            return
        }

        isPlaying = true
        let data = audioQueue.removeFirst()
        do {
            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayer?.delegate = self
            audioPlayer?.play()
        } catch {
            print("[AudioService] Playback error: \(error)")
            playNext()  // Skip failed chunk
        }
    }

    // AVAudioPlayerDelegate — called when a chunk finishes playing
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        playNext()
    }

    /// Resume mic after agent finishes speaking
    private func resumeListening() {
        print("[AudioService] Resuming listening (playback done)")
        isSpeaking = false
        streamEnded = false
        DispatchQueue.main.async {
            if self.isStreaming && !self.isMuted {
                self.state = .listening
                self.statusText = "listening..."
            }
        }
    }
}
