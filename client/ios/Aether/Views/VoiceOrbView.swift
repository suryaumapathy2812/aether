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
        case .idle:       return 0.07
        case .connecting: return 0.05
        case .listening:  return 0.2
        case .thinking:   return 0.13
        case .speaking:   return 0.3
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
        case .listening: return Color(red: 0.86, green: 0.93, blue: 1.0)
        case .thinking:  return Color(red: 0.91, green: 0.93, blue: 0.99)
        case .speaking:  return Color(red: 1.0,  green: 0.96, blue: 0.88)
        case .muted:     return Color(red: 1.0,  green: 0.55, blue: 0.55)
        default:         return .white
        }
    }

    private var sceneBackground: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.03, green: 0.04, blue: 0.05),
                    Color(red: 0.05, green: 0.07, blue: 0.09),
                    Color(red: 0.04, green: 0.05, blue: 0.07)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color.white.opacity(0.1), .clear],
                        center: .center,
                        startRadius: 10,
                        endRadius: 320
                    )
                )
                .frame(width: 360, height: 360)
                .offset(x: -160, y: -300)
                .blur(radius: 12)

            Circle()
                .fill(
                    RadialGradient(
                        colors: [orbTint.opacity(0.14), .clear],
                        center: .center,
                        startRadius: 20,
                        endRadius: 310
                    )
                )
                .frame(width: 380, height: 380)
                .offset(x: 170, y: 260)
                .blur(radius: 16)

            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.black.opacity(0.52), .clear, Color.black.opacity(0.64)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .ignoresSafeArea()
        }
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            sceneBackground

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
                if audio.state == .listening || audio.state == .speaking {
                    Circle()
                        .fill(orbTint.opacity(orbOpacity * 0.14))
                        .frame(width: orbBaseSize + 96, height: orbBaseSize + 96)
                        .blur(radius: 34)
                        .scaleEffect(orbScale * 1.15)
                        .animation(.easeInOut(duration: pulseSpeed), value: breathe)
                }

                Circle()
                    .fill(orbTint.opacity(orbOpacity * 0.36))
                    .frame(width: orbBaseSize + 46, height: orbBaseSize + 46)
                    .blur(radius: 22)
                    .scaleEffect(orbScale * 1.1)
                    .animation(.easeInOut(duration: pulseSpeed), value: breathe)

                if audio.state == .listening || audio.state == .speaking || audio.state == .thinking {
                    RotatingRingView(
                        diameter: orbBaseSize + 32,
                        tint: orbTint,
                        opacity: orbOpacity * 0.55,
                        state: audio.state
                    )
                    .scaleEffect(orbScale)
                    .animation(.easeInOut(duration: 0.5), value: audio.state)
                }

                Circle()
                    .fill(
                        LinearGradient(
                            colors: [
                                orbTint.opacity(orbOpacity * 1.06),
                                orbTint.opacity(orbOpacity * 0.64)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: orbBaseSize, height: orbBaseSize)
                    .overlay(
                        Circle()
                            .strokeBorder(Color.white.opacity(0.22), lineWidth: 0.8)
                    )
                    .shadow(color: orbTint.opacity(orbOpacity * 0.9), radius: glowRadius)
                    .scaleEffect(orbScale)
                    .animation(.easeInOut(duration: pulseSpeed), value: breathe)

                Circle()
                    .fill(Color.white.opacity(orbOpacity * 0.56))
                    .frame(width: orbBaseSize * 0.31, height: orbBaseSize * 0.31)
                    .blur(radius: 7)
                    .offset(x: -orbBaseSize * 0.17, y: -orbBaseSize * 0.17)
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
            Text(audio.statusText)
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .tracking(3.2)
                .foregroundStyle(.white.opacity(0.34))
                .padding(.top, 28)
                .animation(.easeInOut(duration: 0.4), value: audio.statusText)

            if !audio.liveTranscript.isEmpty {
                Text(audio.liveTranscript)
                    .font(.system(size: 13, weight: .light, design: .rounded))
                    .foregroundStyle(.white.opacity(0.52))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .padding(.horizontal, 40)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            if !audio.lastResponse.isEmpty {
                Text(audio.lastResponse)
                    .font(.system(size: 16, weight: .light, design: .serif))
                    .foregroundStyle(.white.opacity(0.72))
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
            if audio.isStreaming {
                Button(action: { audio.toggleMute() }) {
                    ZStack {
                        Circle()
                            .fill(Color.white.opacity(0.04))
                            .frame(width: 48, height: 48)
                            .overlay(
                                Circle()
                                    .strokeBorder(
                                        audio.isMuted ? Color.red.opacity(0.45) : Color.white.opacity(0.2),
                                        lineWidth: 1
                                    )
                            )

                        Image(systemName: audio.isMuted ? "mic.slash" : "mic")
                            .font(.system(size: 15, weight: .light))
                            .foregroundStyle(audio.isMuted ? Color.red.opacity(0.68) : Color.white.opacity(0.44))
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(audio.isMuted ? "Unmute" : "Mute")
                .padding(.bottom, 20)
                .transition(.opacity.combined(with: .scale(scale: 0.85)))
            }

            textInputRow
                .padding(.bottom, 12)

            Text("aether")
                .font(.system(size: 11, weight: .ultraLight, design: .serif))
                .tracking(9)
                .foregroundStyle(.white.opacity(0.17))
                .padding(.bottom, 40)
        }
        .animation(.easeInOut(duration: 0.3), value: audio.isStreaming)
    }

    private var textInputRow: some View {
        HStack(spacing: 8) {
            TextField("type a message...", text: $textInput)
                .textFieldStyle(.plain)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundStyle(.white.opacity(0.74))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.07))
                .clipShape(.rect(cornerRadius: 20))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Color.white.opacity(0.18), lineWidth: 0.8)
                )
                .focused($isTextFieldFocused)
                .onSubmit { sendTextMessage() }

            if !textInput.isEmpty {
                Button(action: { sendTextMessage() }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(.white.opacity(0.46))
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
