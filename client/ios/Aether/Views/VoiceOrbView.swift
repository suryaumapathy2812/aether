import SwiftUI

struct VoiceOrbView: View {
    @EnvironmentObject var pairing: PairingService
    @Environment(\.openURL) private var openURL
    @StateObject private var audio = AudioService()

    @State private var breathe = false
    @State private var isPressingOrb = false
    @State private var dragOffsetX: CGFloat = 0
    @State private var cancelArmed = false
    @State private var textInput = ""
    @FocusState private var isTextFieldFocused: Bool

    private let cancelThreshold: CGFloat = -90

    private var orbBaseSize: CGFloat {
        switch audio.state {
        case .idle: return 122
        case .recording: return 144
        case .uploading: return 110
        case .thinking: return 118
        case .speaking: return 136
        case .error: return 102
        }
    }

    private var orbScale: CGFloat {
        let userBoost = audio.userSpeechLevel * 0.18
        let agentBoost = audio.agentSpeechLevel * 0.16
        switch audio.state {
        case .recording: return (breathe ? 1.10 : 0.94) + userBoost
        case .speaking: return (breathe ? 1.08 : 0.95) + agentBoost
        case .thinking: return breathe ? 1.04 : 0.97
        case .uploading: return breathe ? 1.03 : 0.98
        default: return breathe ? 1.02 : 0.99
        }
    }

    private var orbOpacity: Double {
        switch audio.state {
        case .idle: return 0.07
        case .recording: return 0.28
        case .uploading: return 0.11
        case .thinking: return 0.14
        case .speaking: return 0.26
        case .error: return 0.06
        }
    }

    private var glowRadius: CGFloat {
        switch audio.state {
        case .recording: return breathe ? 66 : 42
        case .speaking: return breathe ? 58 : 34
        case .thinking: return breathe ? 40 : 22
        case .uploading: return breathe ? 26 : 14
        case .error: return 8
        case .idle: return 16
        }
    }

    private var pulseSpeed: Double {
        switch audio.state {
        case .recording: return 0.55
        case .uploading: return 1.8
        case .thinking: return 1.2
        case .speaking: return 0.9
        default: return 2.8
        }
    }

    private var orbTint: Color {
        switch audio.state {
        case .recording: return Color(red: 1.0, green: 0.53, blue: 0.44)
        case .speaking: return Color(red: 0.99, green: 0.95, blue: 0.82)
        case .thinking: return Color(red: 0.88, green: 0.92, blue: 1.0)
        case .uploading: return Color(red: 0.78, green: 0.85, blue: 0.98)
        case .error: return Color(red: 1.0, green: 0.60, blue: 0.60)
        case .idle: return .white
        }
    }

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
                .fill(orbTint.opacity(orbOpacity * 0.16))
                .frame(width: orbBaseSize + 92, height: orbBaseSize + 92)
                .blur(radius: 18)
                .scaleEffect(orbScale * 1.12)

            Circle()
                .fill(orbTint.opacity(orbOpacity * 0.34))
                .frame(width: orbBaseSize + 46, height: orbBaseSize + 46)
                .blur(radius: 12)
                .scaleEffect(orbScale * 1.08)

            Circle()
                .fill(
                    LinearGradient(
                        colors: [orbTint.opacity(orbOpacity * 1.06), orbTint.opacity(orbOpacity * 0.62)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: orbBaseSize, height: orbBaseSize)
                .overlay(Circle().strokeBorder(Color.white.opacity(0.22), lineWidth: 0.8))
                .shadow(color: orbTint.opacity(orbOpacity * 0.88), radius: glowRadius)
                .scaleEffect(orbScale)

            Circle()
                .fill(Color.white.opacity(orbOpacity * 0.5))
                .frame(width: orbBaseSize * 0.30, height: orbBaseSize * 0.30)
                .blur(radius: 3)
                .offset(x: -orbBaseSize * 0.16, y: -orbBaseSize * 0.16)
                .scaleEffect(orbScale)
        }
        .animation(.easeInOut(duration: pulseSpeed), value: breathe)
        .animation(.easeInOut(duration: 0.45), value: audio.state)
        .offset(x: audio.state == .recording ? max(-60, min(0, dragOffsetX * 0.35)) : 0)
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
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .tracking(3.1)
                .foregroundStyle(.white.opacity(0.34))
                .padding(.top, 28)

            if !audio.lastResponse.isEmpty {
                Text(audio.lastResponse)
                    .font(.system(size: 16, weight: .light, design: .serif))
                    .foregroundStyle(.white.opacity(0.72))
                    .multilineTextAlignment(.center)
                    .lineLimit(6)
                    .padding(.horizontal, 32)
                    .padding(.top, 4)
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
                .padding(.bottom, 12)

            Text("aether")
                .font(.system(size: 11, weight: .ultraLight, design: .serif))
                .tracking(9)
                .foregroundStyle(.white.opacity(0.17))
                .padding(.bottom, 40)
        }
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

    private var sceneBackground: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.03, green: 0.04, blue: 0.05),
                    Color(red: 0.05, green: 0.07, blue: 0.09),
                    Color(red: 0.04, green: 0.05, blue: 0.07),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            Circle()
                .fill(RadialGradient(colors: [Color.white.opacity(0.1), .clear], center: .center, startRadius: 10, endRadius: 320))
                .frame(width: 360, height: 360)
                .offset(x: -160, y: -300)
                .blur(radius: 6)

            Circle()
                .fill(RadialGradient(colors: [orbTint.opacity(0.14), .clear], center: .center, startRadius: 20, endRadius: 310))
                .frame(width: 380, height: 380)
                .offset(x: 170, y: 260)
                .blur(radius: 8)

            Rectangle()
                .fill(LinearGradient(colors: [Color.black.opacity(0.52), .clear, Color.black.opacity(0.64)], startPoint: .top, endPoint: .bottom))
                .ignoresSafeArea()
        }
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
