import SwiftUI

/// Voice orb screen — continuous conversation mode via WebRTC.
/// Tap orb to start listening, tap again to stop. Mute button to pause mic.
struct VoiceOrbView: View {
    @EnvironmentObject var pairing: PairingService
    @StateObject private var audio = AudioService()

    // Single breathing phase — toggled once, drives a repeatForever animation
    @State private var breathe = false
    @State private var textInput = ""
    @FocusState private var isTextFieldFocused: Bool

    // MARK: - Computed orb properties

    private var orbBaseSize: CGFloat {
        switch audio.state {
        case .idle:       return 120
        case .connecting: return 100
        case .listening:  return 130
        case .thinking:   return 110
        case .speaking:   return 140
        case .muted:      return 100
        }
    }

    private var orbScale: CGFloat {
        let userBoost  = audio.userSpeechLevel  * 0.12
        let agentBoost = audio.agentSpeechLevel * 0.16
        switch audio.state {
        case .listening:  return (breathe ? 1.08 : 0.95) + userBoost
        case .thinking:   return breathe ? 1.04 : 0.96
        case .speaking:   return (breathe ? 1.10 : 0.92) + agentBoost
        case .connecting: return breathe ? 1.02 : 0.98
        default:          return 1.0
        }
    }

    private var orbOpacity: Double {
        switch audio.state {
        case .idle:       return 0.06
        case .connecting: return 0.04
        case .listening:  return 0.18
        case .thinking:   return 0.12
        case .speaking:   return 0.26
        case .muted:      return 0.03
        }
    }

    private var glowRadius: CGFloat {
        switch audio.state {
        case .idle:       return 15
        case .connecting: return 8
        case .listening:  return breathe ? 55 : 35
        case .thinking:   return breathe ? 35 : 20
        case .speaking:   return breathe ? 70 : 40
        case .muted:      return 3
        }
    }

    private var pulseSpeed: Double {
        switch audio.state {
        case .listening:  return 2.0
        case .thinking:   return 1.2
        case .speaking:   return 0.8
        default:          return 3.0
        }
    }

    /// Accent colour tint — shifts subtly per state
    private var orbTint: Color {
        switch audio.state {
        case .listening: return Color(red: 0.85, green: 0.95, blue: 1.0)   // cool blue-white
        case .thinking:  return Color(red: 0.95, green: 0.90, blue: 1.0)   // soft lavender
        case .speaking:  return Color(red: 1.0,  green: 0.97, blue: 0.88)  // warm cream
        case .muted:     return Color(red: 1.0,  green: 0.5,  blue: 0.5)   // muted red
        default:         return .white
        }
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            Color(hex: "111111").ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                orbSection

                statusSection

                Spacer()
                Spacer()

                bottomControls
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            let token = pairing.getDeviceToken() ?? ""
            audio.configure(token: token, orchestratorURL: pairing.orchestratorURL)
            startBreathing(speed: pulseSpeed)
        }
        .onChange(of: audio.state) {
            // Restart breathing with new speed — cancel old animation first
            breathe = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                startBreathing(speed: pulseSpeed)
            }
        }
        .alert("Microphone Access Required", isPresented: $audio.micPermissionDenied) {
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Aether needs microphone access to hear you. Please enable it in Settings.")
        }
    }

    // MARK: - Orb Section

    private var orbSection: some View {
        Button(action: { audio.toggleStreaming() }) {
            ZStack {
                // Outermost ambient halo — only when active
                if audio.state == .listening || audio.state == .speaking {
                    Circle()
                        .fill(orbTint.opacity(orbOpacity * 0.12))
                        .frame(width: orbBaseSize + 80, height: orbBaseSize + 80)
                        .blur(radius: 30)
                        .scaleEffect(orbScale * 1.15)
                        .animation(.easeInOut(duration: pulseSpeed), value: breathe)
                }

                // Outer glow ring
                Circle()
                    .fill(orbTint.opacity(orbOpacity * 0.3))
                    .frame(width: orbBaseSize + 40, height: orbBaseSize + 40)
                    .blur(radius: 20)
                    .scaleEffect(orbScale * 1.1)
                    .animation(.easeInOut(duration: pulseSpeed), value: breathe)

                // Rotating ring — visible when listening or speaking
                if audio.state == .listening || audio.state == .speaking || audio.state == .thinking {
                    RotatingRingView(
                        diameter: orbBaseSize + 28,
                        tint: orbTint,
                        opacity: orbOpacity * 0.6,
                        state: audio.state
                    )
                    .scaleEffect(orbScale)
                    .animation(.easeInOut(duration: 0.5), value: audio.state)
                }

                // Main orb
                Circle()
                    .fill(orbTint.opacity(orbOpacity))
                    .frame(width: orbBaseSize, height: orbBaseSize)
                    .shadow(color: orbTint.opacity(orbOpacity * 0.8), radius: glowRadius)
                    .scaleEffect(orbScale)
                    .animation(.easeInOut(duration: pulseSpeed), value: breathe)

                // Inner highlight — small bright core
                Circle()
                    .fill(orbTint.opacity(orbOpacity * 0.5))
                    .frame(width: orbBaseSize * 0.35, height: orbBaseSize * 0.35)
                    .blur(radius: 6)
                    .offset(x: -orbBaseSize * 0.12, y: -orbBaseSize * 0.12)
                    .scaleEffect(orbScale)
                    .animation(.easeInOut(duration: pulseSpeed), value: breathe)
            }
            .animation(.easeInOut(duration: 0.6), value: audio.state)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(audio.isStreaming ? "Stop listening" : "Start listening")
    }

    // MARK: - Status Section

    private var statusSection: some View {
        VStack(spacing: 10) {
            // Primary status label
            Text(audio.statusText)
                .font(.system(size: 10, weight: .light))
                .tracking(3)
                .foregroundStyle(.white.opacity(0.25))
                .padding(.top, 28)
                .animation(.easeInOut(duration: 0.4), value: audio.statusText)

            // Live transcript — user's words appearing in real time
            if !audio.liveTranscript.isEmpty {
                Text(audio.liveTranscript)
                    .font(.system(size: 13, weight: .light))
                    .foregroundStyle(.white.opacity(0.45))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .padding(.horizontal, 40)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            // Agent response text
            if !audio.lastResponse.isEmpty {
                Text(audio.lastResponse)
                    .font(.system(size: 15, weight: .light))
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                    .lineLimit(5)
                    .padding(.horizontal, 32)
                    .padding(.top, audio.liveTranscript.isEmpty ? 4 : 0)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.35), value: audio.liveTranscript)
        .animation(.easeInOut(duration: 0.35), value: audio.lastResponse)
    }

    // MARK: - Bottom Controls

    private var bottomControls: some View {
        VStack(spacing: 0) {
            // Mute button — only when streaming
            if audio.isStreaming {
                Button(action: { audio.toggleMute() }) {
                    ZStack {
                        Circle()
                            .strokeBorder(
                                audio.isMuted ? Color.red.opacity(0.4) : Color.white.opacity(0.12),
                                lineWidth: 1
                            )
                            .frame(width: 48, height: 48)

                        Image(systemName: audio.isMuted ? "mic.slash" : "mic")
                            .font(.system(size: 15, weight: .light))
                            .foregroundStyle(audio.isMuted ? Color.red.opacity(0.6) : Color.white.opacity(0.3))
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(audio.isMuted ? "Unmute" : "Mute")
                .padding(.bottom, 20)
                .transition(.opacity.combined(with: .scale(scale: 0.85)))
            }

            // Text input
            textInputRow
                .padding(.bottom, 12)

            // Brand
            Text("aether")
                .font(.system(size: 11, weight: .ultraLight))
                .tracking(8)
                .foregroundStyle(.white.opacity(0.1))
                .padding(.bottom, 40)
        }
        .animation(.easeInOut(duration: 0.3), value: audio.isStreaming)
    }

    private var textInputRow: some View {
        HStack(spacing: 8) {
            TextField("type a message...", text: $textInput)
                .textFieldStyle(.plain)
                .font(.system(size: 14, weight: .light))
                .foregroundStyle(.white.opacity(0.7))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.06))
                .clipShape(.rect(cornerRadius: 20))
                .focused($isTextFieldFocused)
                .onSubmit { sendTextMessage() }

            if !textInput.isEmpty {
                Button(action: { sendTextMessage() }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(.white.opacity(0.3))
                }
                .buttonStyle(.plain)
                .transition(.opacity.combined(with: .scale(scale: 0.8)))
            }
        }
        .padding(.horizontal, 20)
        .animation(.easeInOut(duration: 0.2), value: textInput.isEmpty)
    }

    // MARK: - Helpers

    private func sendTextMessage() {
        let text = textInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        textInput = ""
        isTextFieldFocused = false
        audio.sendText(text)
    }

    private func startBreathing(speed: Double) {
        withAnimation(.easeInOut(duration: speed).repeatForever(autoreverses: true)) {
            breathe = true
        }
    }
}

// MARK: - Rotating Ring

/// A dashed ring that slowly rotates — adds life to the orb during active states.
private struct RotatingRingView: View {
    let diameter: CGFloat
    let tint: Color
    let opacity: Double
    let state: OrbState

    @State private var rotation: Double = 0

    private var dashPattern: [CGFloat] {
        switch state {
        case .listening: return [4, 12]
        case .speaking:  return [8, 6]
        case .thinking:  return [2, 16]
        default:         return [4, 12]
        }
    }

    private var rotationSpeed: Double {
        switch state {
        case .listening: return 8.0
        case .speaking:  return 4.0
        case .thinking:  return 12.0
        default:         return 10.0
        }
    }

    var body: some View {
        Circle()
            .strokeBorder(
                style: StrokeStyle(lineWidth: 1, dash: dashPattern)
            )
            .foregroundStyle(tint.opacity(opacity))
            .frame(width: diameter, height: diameter)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: rotationSpeed).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
            .onChange(of: state) {
                rotation = 0
                withAnimation(.linear(duration: rotationSpeed).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
    }
}
