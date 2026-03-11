import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

private enum RootTab: Hashable {
    case home
    case history
    case memory
    case plugins
}

private enum AetherTheme {
    static let shellGradient = LinearGradient(
        colors: [
            Color(red: 0.04, green: 0.05, blue: 0.07),
            Color(red: 0.07, green: 0.09, blue: 0.12),
            Color(red: 0.05, green: 0.06, blue: 0.09)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let cardFill = Color.white.opacity(0.08)
    static let cardStroke = Color.white.opacity(0.16)
    static let softText = Color.white.opacity(0.58)
    static let pressedSpring = Animation.spring(response: 0.35, dampingFraction: 0.86)
}

struct AetherGlassCard<Content: View>: View {
    let cornerRadius: CGFloat
    @ViewBuilder var content: Content

    var body: some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(AetherTheme.cardFill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(AetherTheme.cardStroke, lineWidth: 0.8)
            )
    }
}

enum AetherHaptics {
    static func tap() {
#if canImport(UIKit)
        UISelectionFeedbackGenerator().selectionChanged()
#endif
    }

    static func success() {
#if canImport(UIKit)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
#endif
    }

    static func warning() {
#if canImport(UIKit)
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
#endif
    }
}

struct AppRootView: View {
    @EnvironmentObject var pairing: PairingService
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @StateObject private var audio = AudioService()
    @State private var selectedTab: RootTab = .home
    @State private var showHistorySheet = false
    @State private var showComposer = false
    @State private var composerText = ""
    @FocusState private var composerFocused: Bool

    private var showGlobalButton: Bool {
        !showComposer
    }

    private var tabTransitionAnimation: Animation {
        reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.22)
    }

    var body: some View {
        ZStack {
            AetherTheme.shellGradient.ignoresSafeArea()

            VStack(spacing: 0) {
                topSwitcher
                    .padding(.horizontal, 14)
                    .padding(.top, 6)

                if selectedTab != .home && audio.state != .idle {
                    globalStatusBanner
                        .padding(.horizontal, 16)
                        .padding(.top, 2)
                }

                currentTabContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.bottom, showGlobalButton ? 96 : 0)

                if showGlobalButton {
                    GlobalActionButton(audio: audio) {
                        AetherHaptics.tap()
                        withAnimation(reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.2)) {
                            showComposer = true
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                            composerFocused = true
                        }
                    }
                    .padding(.bottom, 8)
                    .transition(AnyTransition.opacity.combined(with: AnyTransition.move(edge: .bottom)))
                }
            }

            if showComposer {
                Color.black.opacity(0.34)
                    .ignoresSafeArea()
                    .onTapGesture {
                        composerFocused = false
                        withAnimation(reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.2)) {
                            showComposer = false
                        }
                    }
                composerOverlay
                    .transition(AnyTransition.move(edge: .bottom).combined(with: AnyTransition.opacity))
            }
        }
        .sheet(isPresented: $showHistorySheet) {
            NavigationStack {
                HistoryView(embedded: true)
                    .navigationTitle("Conversation")
                    .navigationBarTitleDisplayMode(.inline)
            }
            .presentationDetents([.fraction(0.38), .large])
            .presentationDragIndicator(.visible)
            .presentationBackground(.black.opacity(0.92))
        }
        .task(id: pairing.getDeviceToken() ?? "") {
            let token = pairing.getDeviceToken() ?? ""
            audio.configure(token: token, orchestratorURL: pairing.orchestratorURL)
        }
        .onChange(of: audio.state) {
            if audio.state == .error {
                AetherHaptics.warning()
            }
            if audio.state == .recording || audio.state == .uploading {
                withAnimation(reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.2)) {
                    showComposer = false
                }
                composerFocused = false
            }
        }
        .environmentObject(audio)
        .preferredColorScheme(.dark)
        .tint(.white)
    }

    private var currentTabContent: some View {
        ZStack {
            switch selectedTab {
            case .home:
                VoiceOrbView(showHistorySheet: $showHistorySheet)
                    .transition(.opacity)
            case .history:
                HistoryView(embedded: true)
                    .transition(.opacity)
            case .memory:
                MemoryView(embedded: true)
                    .transition(.opacity)
            case .plugins:
                PluginsView(embedded: true)
                    .transition(.opacity)
            }
        }
        .animation(tabTransitionAnimation, value: selectedTab)
    }

    private var globalStatusBanner: some View {
        AetherGlassCard(cornerRadius: 16) {
            HStack(spacing: 8) {
                Circle()
                    .fill(audio.state == .recording ? Color.red.opacity(0.84) : Color.white.opacity(0.78))
                    .frame(width: 6, height: 6)
                Text(audio.statusText)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.72))
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }

    private var topSwitcher: some View {
        HStack(spacing: 8) {
            switchButton(title: "Home", icon: "circle.grid.2x1", tab: .home)
            switchButton(title: "History", icon: "text.bubble", tab: .history)
            switchButton(title: "Memory", icon: "brain.head.profile", tab: .memory)
            switchButton(title: "Plugins", icon: "puzzlepiece.extension", tab: .plugins)
        }
        .padding(.vertical, 8)
    }

    private var composerOverlay: some View {
        VStack {
            Spacer()
            AetherGlassCard(cornerRadius: 22) {
                HStack(spacing: 10) {
                    TextField("type instruction...", text: $composerText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 15, weight: .regular, design: .rounded))
                        .foregroundStyle(.white.opacity(0.9))
                        .accessibilityLabel("Instruction input")
                        .focused($composerFocused)
                        .submitLabel(.send)
                        .onSubmit {
                            sendComposerText()
                        }

                    Button("Cancel") {
                        composerFocused = false
                        withAnimation(reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.2)) {
                            showComposer = false
                        }
                    }
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.62))

                    Button {
                        sendComposerText()
                    } label: {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 14, weight: .semibold))
                            .frame(width: 34, height: 34)
                            .background(Color.white.opacity(composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.08 : 0.24), in: Circle())
                    }
                    .disabled(composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityLabel("Send instruction")
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
        }
    }

    private func sendComposerText() {
        let trimmed = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard audio.state != .recording && audio.state != .uploading else {
            AetherHaptics.warning()
            return
        }
        audio.sendText(trimmed)
        AetherHaptics.success()
        composerText = ""
        composerFocused = false
        withAnimation(reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.2)) {
            showComposer = false
        }
    }

    private func switchButton(title: String, icon: String, tab: RootTab) -> some View {
        let selected = selectedTab == tab
        return Button {
            AetherHaptics.tap()
            withAnimation(AetherTheme.pressedSpring) {
                selectedTab = tab
            }
        } label: {
            VStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                    .minimumScaleFactor(0.8)
                Text(title)
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(.white.opacity(selected ? 0.95 : 0.45))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(selected ? Color.white.opacity(0.14) : .clear)
            )
        }
        .accessibilityIdentifier("top-switch-\(title.lowercased())")
        .buttonStyle(.plain)
    }
}

private struct GlobalActionButton: View {
    @ObservedObject var audio: AudioService
    var onTapCompose: () -> Void

    @State private var isPressingOrb = false
    @State private var dragOffsetX: CGFloat = 0
    @State private var cancelArmed = false
    @State private var hasStartedRecording = false
    @State private var delayedStart: DispatchWorkItem?
    private let cancelThreshold: CGFloat = -90

    var body: some View {
        VStack(spacing: 8) {
            Text(audio.state == .recording
                 ? (cancelArmed ? "release to cancel" : "slide left to cancel")
                 : audio.statusText)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(AetherTheme.softText)

            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.12))
                    .frame(width: 66, height: 66)

                Circle()
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    .frame(width: 66, height: 66)

                Image(systemName: hasStartedRecording ? "waveform.badge.mic" : "waveform")
                    .font(.system(size: 19, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.95))
            }
            .offset(x: audio.state == .recording ? max(-66, min(0, dragOffsetX * 0.4)) : 0)
            .animation(.easeInOut(duration: 0.2), value: dragOffsetX)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if !isPressingOrb {
                            isPressingOrb = true
                            dragOffsetX = 0
                            cancelArmed = false
                            hasStartedRecording = false
                            let work = DispatchWorkItem {
                                audio.beginRecording()
                                hasStartedRecording = true
                                AetherHaptics.tap()
                            }
                            delayedStart = work
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.18, execute: work)
                        }

                        guard isPressingOrb, hasStartedRecording else { return }
                        dragOffsetX = value.translation.width
                        cancelArmed = value.translation.width <= cancelThreshold
                    }
                    .onEnded { value in
                        guard isPressingOrb else { return }
                        isPressingOrb = false

                        delayedStart?.cancel()
                        delayedStart = nil

                        if hasStartedRecording {
                            let shouldCancel = cancelArmed
                            dragOffsetX = 0
                            cancelArmed = false
                            hasStartedRecording = false
                            if shouldCancel {
                                audio.cancelRecording()
                                AetherHaptics.warning()
                            } else {
                                audio.endRecordingAndSend()
                                AetherHaptics.success()
                            }
                        } else if abs(value.translation.width) < 10 && abs(value.translation.height) < 10 {
                            onTapCompose()
                        }
                    }
            )
            .accessibilityLabel("Hold to record instruction")
            .accessibilityHint("Tap to type instead")
            .accessibilityIdentifier("global-voice-action")
        }
        .padding(.bottom, 8)
        .onDisappear {
            delayedStart?.cancel()
            delayedStart = nil
        }
    }
}
