import SwiftUI

struct VoiceOrbView: View {
    @EnvironmentObject var pairing: PairingService
    @Environment(\.openURL) private var openURL
    @StateObject private var audio = AudioService()

    @State private var breathe = false
    @State private var isPressingOrb = false
    @State private var dragOffsetX: CGFloat = 0
    @State private var cancelArmed = false
    @State private var didRunIntro = false
    @State private var textInput = ""
    @FocusState private var isTextFieldFocused: Bool

    private let cancelThreshold: CGFloat = -90

    private var orbBaseSize: CGFloat {
        switch audio.state {
        case .idle: return 116
        case .recording: return 126
        case .uploading: return 112
        case .thinking: return 114
        case .speaking: return 120
        case .error: return 112
        }
    }

    private var orbScale: CGFloat {
        let userBoost = audio.userSpeechLevel * 0.08
        let agentBoost = audio.agentSpeechLevel * 0.07
        switch audio.state {
        case .recording: return (breathe ? 1.05 : 0.98) + userBoost
        case .speaking: return (breathe ? 1.04 : 0.98) + agentBoost
        case .thinking: return breathe ? 1.02 : 0.99
        case .uploading: return breathe ? 1.01 : 0.99
        default: return breathe ? 1.01 : 1.0
        }
    }

    private var orbOpacity: Double {
        switch audio.state {
        case .idle: return 0.12
        case .recording: return 0.22
        case .uploading: return 0.14
        case .thinking: return 0.15
        case .speaking: return 0.18
        case .error: return 0.2
        }
    }

    private var glowRadius: CGFloat {
        switch audio.state {
        case .recording: return breathe ? 24 : 16
        case .speaking: return breathe ? 20 : 14
        case .thinking: return breathe ? 18 : 12
        case .uploading: return breathe ? 14 : 10
        case .error: return 18
        case .idle: return 10
        }
    }

    private var pulseSpeed: Double {
        switch audio.state {
        case .recording: return 0.95
        case .uploading: return 1.3
        case .thinking: return 1.2
        case .speaking: return 1.0
        default: return 1.6
        }
    }

    private var orbTint: Color {
        switch audio.state {
        case .recording: return Color(red: 0.90, green: 0.52, blue: 0.47)
        case .error: return Color(red: 0.93, green: 0.42, blue: 0.42)
        default: return Color.white
        }
    }

    var body: some View {
        ZStack {
            Color(hex: "111111").ignoresSafeArea()

            GeometryReader { geo in
                VStack(spacing: 0) {
                    Spacer(minLength: 34)

                    orbSection
                        .padding(.top, 6)

                    statusSection
                        .padding(.top, 16)

                    Spacer(minLength: 24)

                    bottomControls
                        .padding(.bottom, 10)
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            let token = pairing.getDeviceToken() ?? ""
            audio.configure(token: token, orchestratorURL: pairing.orchestratorURL)
            startBreathing(speed: pulseSpeed)
            if !didRunIntro {
                didRunIntro = true
            }
        }
        .onChange(of: audio.state) {
            breathe = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.04) {
                startBreathing(speed: pulseSpeed)
            }
        }
        .alert("Microphone Access Required", isPresented: $audio.micPermissionDenied) {
            Button("Open Settings") {
                if let url = URL(string: "app-settings:") {
                    openURL(url)
                }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Aether needs microphone access to capture your voice.")
        }
    }

    private var orbSection: some View {
        ZStack {
            Circle()
                .fill(orbTint.opacity(orbOpacity * 0.34))
                .frame(width: orbBaseSize + 34, height: orbBaseSize + 34)
                .blur(radius: 10)
                .scaleEffect(orbScale * 1.04)

            Circle()
                .fill(
                    LinearGradient(
                        colors: [orbTint.opacity(orbOpacity * 1.04), orbTint.opacity(orbOpacity * 0.82)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: orbBaseSize, height: orbBaseSize)
                .overlay(Circle().strokeBorder(Color.white.opacity(0.2), lineWidth: 0.7))
                .shadow(color: orbTint.opacity(orbOpacity * 0.64), radius: glowRadius)
                .scaleEffect(orbScale)
        }
        .animation(.easeInOut(duration: pulseSpeed), value: breathe)
        .animation(.easeInOut(duration: 0.45), value: audio.state)
        .offset(x: audio.state == .recording ? max(-60, min(0, dragOffsetX * 0.35)) : 0)
        .scaleEffect(didRunIntro ? 1.0 : 0.72, anchor: .center)
        .animation(.spring(response: 0.45, dampingFraction: 0.82), value: didRunIntro)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    guard !isPressingOrb else { return }
                    isPressingOrb = true
                    dragOffsetX = 0
                    cancelArmed = false
                    audio.beginRecording()
                }
                .onChanged { value in
                    guard isPressingOrb else { return }
                    dragOffsetX = value.translation.width
                    cancelArmed = value.translation.width <= cancelThreshold
                }
                .onEnded { _ in
                    guard isPressingOrb else { return }
                    isPressingOrb = false
                    let shouldCancel = cancelArmed
                    dragOffsetX = 0
                    cancelArmed = false
                    if shouldCancel {
                        audio.cancelRecording()
                    } else {
                        audio.endRecordingAndSend()
                    }
                }
        )
        .accessibilityLabel("Hold to record voice")
    }

    private var statusSection: some View {
        VStack(spacing: 10) {
            Text(audio.statusText)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .tracking(1.4)
                .foregroundStyle(.white.opacity(0.58))
                .padding(.top, 24)

            if !audio.lastResponse.isEmpty {
                Text(audio.lastResponse)
                    .font(.system(size: 24, weight: .regular, design: .rounded))
                    .foregroundStyle(.white.opacity(0.9))
                    .multilineTextAlignment(.leading)
                    .lineLimit(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(minHeight: 58, alignment: .topLeading)
                    .padding(.horizontal, 14)
                    .padding(.horizontal, 22)
                    .padding(.top, 2)
                    .transition(.opacity)
            }

            if audio.state == .recording {
                HStack(spacing: 8) {
                    Circle()
                        .fill(cancelArmed ? Color.red.opacity(0.9) : Color.white.opacity(0.8))
                        .frame(width: 6, height: 6)
                    Text(formatDuration(audio.recordingDuration))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.84))
                    Text(cancelArmed ? "release to cancel" : "slide left to cancel")
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(cancelArmed ? Color.red.opacity(0.9) : Color.white.opacity(0.6))
                }
                .padding(.top, 8)
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.3), value: audio.lastResponse)
        .animation(.easeInOut(duration: 0.2), value: cancelArmed)
    }

    private var bottomControls: some View {
        VStack(spacing: 0) {
            textInputRow
                .padding(.bottom, 28)
        }
    }

    private var textInputRow: some View {
        HStack(spacing: 8) {
            TextField("type a message...", text: $textInput)
                .textFieldStyle(.plain)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundStyle(.white.opacity(0.86))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.05))
                .clipShape(.rect(cornerRadius: 20))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Color.white.opacity(0.11), lineWidth: 0.8)
                )
                .focused($isTextFieldFocused)
                .onSubmit { sendTextMessage() }

            if !textInput.isEmpty {
                Button(action: { sendTextMessage() }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 25))
                        .foregroundStyle(.white.opacity(0.62))
                }
                .buttonStyle(.plain)
                .transition(.opacity.combined(with: .scale(scale: 0.8)))
            }
        }
        .padding(.horizontal, 20)
        .animation(.easeInOut(duration: 0.2), value: textInput.isEmpty)
    }

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

    private func formatDuration(_ duration: TimeInterval) -> String {
        let total = max(0, Int(duration.rounded(.down)))
        let minutes = total / 60
        let seconds = total % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}
