import SwiftUI

struct VoiceOrbView: View {
    @EnvironmentObject var audio: AudioService
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var reveal = false

    private var headline: String {
        if !audio.lastResponse.isEmpty {
            return audio.lastResponse
        }
        switch audio.state {
        case .recording:  return "Listening..."
        case .uploading:  return "Sending your instruction"
        case .thinking:   return "Thinking through your request"
        case .speaking:   return "Preparing your next action"
        case .error:      return "I hit a temporary issue"
        case .idle:       return "Hello, ready when you are."
        }
    }

    @ViewBuilder
    private var responseText: some View {
        if !audio.lastResponse.isEmpty {
            if let attributed = try? AttributedString(markdown: audio.lastResponse, options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnly)) {
                Text(attributed)
                    .font(.system(size: 34))
                    .foregroundStyle(.white.opacity(0.88))
                    .multilineTextAlignment(.leading)
                    .lineSpacing(5)
            } else {
                Text(audio.lastResponse)
                    .font(.system(size: 34, weight: .regular, design: .default))
                    .foregroundStyle(.white.opacity(0.88))
                    .multilineTextAlignment(.leading)
                    .lineSpacing(5)
            }
        } else {
            Text(headline)
                .font(.system(size: 34, weight: .light, design: .default))
                .foregroundStyle(.white.opacity(0.88))
                .multilineTextAlignment(.center)
                .lineSpacing(5)
        }
    }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height

            ZStack {
                // Full-bleed atmospheric gradient
                AetherTheme.atmosphericGradient
                    .ignoresSafeArea()

                VStack(spacing: 0) {
                    Spacer()

                    // Main headline — centered and scrollable for long responses
                    ScrollView {
                        responseText
                            .padding(.horizontal, 36)
                    }
                    .frame(maxHeight: h * 0.6)
                    .scaleEffect(reveal ? 1.0 : 0.97)
                    .opacity(reveal ? 1.0 : 0.0)
                    .animation(
                        reduceMotion ? .linear(duration: 0.01) :
                            .easeOut(duration: 0.5).delay(0.1),
                        value: reveal
                    )

                    Spacer()
                    // Extra spacer for voice button clearance
                    Spacer()
                }
                .frame(width: w, height: h)
            }
        }
        .onAppear {
            reveal = true
        }
        .onDisappear {
            reveal = false
        }
        .onChange(of: audio.state) { _, _ in }
        .alert("Microphone Access Required", isPresented: $audio.micPermissionDenied) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Aether needs microphone access to capture your voice.")
        }
    }
}
