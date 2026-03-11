import SwiftUI

struct VoiceOrbView: View {
    @EnvironmentObject var audio: AudioService
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var showHistorySheet: Bool
    @State private var breathe = false
    @State private var reveal = false

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
        GeometryReader { geo in
            let compact = geo.size.height < 760
            ZStack(alignment: .bottom) {
                LinearGradient(
                    colors: [
                        Color(red: 0.04, green: 0.06, blue: 0.09),
                        Color(red: 0.08, green: 0.13, blue: 0.19),
                        Color(red: 0.05, green: 0.07, blue: 0.10)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                Circle()
                    .fill(Color.blue.opacity(0.24))
                    .blur(radius: compact ? 70 : 90)
                    .frame(width: compact ? 240 : 300, height: compact ? 240 : 300)
                    .offset(y: compact ? -190 : -220)

                VStack(spacing: compact ? 10 : 14) {
                    Spacer(minLength: compact ? 8 : 20)

                    orbSection
                        .scaleEffect(compact ? 0.92 : 1.0)

                    statusSection(compact: compact)
                        .padding(.top, compact ? 10 : 18)

                    Spacer(minLength: compact ? 12 : 20)

                    Button {
                        AetherHaptics.tap()
                        showHistorySheet = true
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "chevron.up")
                            Text("Pull up for full history")
                        }
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(.white.opacity(0.62))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.07), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open full history")
                    .padding(.bottom, compact ? 4 : 6)
                }
                .padding(.horizontal, compact ? 14 : 22)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            startBreathing(speed: pulseSpeed)
            reveal = true
        }
        .onChange(of: audio.state) {
            breathe = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.04) {
                startBreathing(speed: pulseSpeed)
            }
        }
        .gesture(
            DragGesture(minimumDistance: 20)
                .onEnded { value in
                    if value.translation.height < -70 {
                        AetherHaptics.tap()
                        showHistorySheet = true
                    }
                }
        )
        .alert("Microphone Access Required", isPresented: $audio.micPermissionDenied) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Aether needs microphone access to capture your voice from the global button.")
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
        .scaleEffect(reveal ? 1.0 : 0.72, anchor: .center)
        .animation(.spring(response: 0.45, dampingFraction: 0.82), value: reveal)
    }

    private func statusSection(compact: Bool) -> some View {
        VStack(spacing: 12) {
            Text(audio.statusText)
                .font(.system(size: compact ? 11 : 12, weight: .medium, design: .rounded))
                .tracking(1.4)
                .foregroundStyle(.white.opacity(0.58))
                .padding(.top, compact ? 6 : 12)

            if !audio.lastResponse.isEmpty {
                AetherGlassCard(cornerRadius: compact ? 22 : 26) {
                    Text(audio.lastResponse)
                        .font(.system(size: compact ? 22 : 26, weight: .regular, design: .rounded))
                        .foregroundStyle(.white.opacity(0.92))
                        .multilineTextAlignment(.leading)
                        .lineLimit(compact ? 5 : 6)
                        .minimumScaleFactor(0.84)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .frame(minHeight: compact ? 52 : 68, alignment: .topLeading)
                        .padding(.horizontal, compact ? 16 : 22)
                        .padding(.vertical, compact ? 16 : 22)
                        .transition(.opacity)
                }
            }
        }
        .animation(.easeInOut(duration: 0.3), value: audio.lastResponse)
    }

    private func startBreathing(speed: Double) {
        guard !reduceMotion else {
            breathe = true
            return
        }
        withAnimation(.easeInOut(duration: speed).repeatForever(autoreverses: true)) {
            breathe = true
        }
    }
}
