import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Theme

enum AetherTheme {
    static let listBg = Color.black

    /// Top color of the atmospheric gradient — used for status bar sync
    static let gradientTop = Color(red: 0.10, green: 0.12, blue: 0.16)

    static let softText = Color.white.opacity(0.50)
    static let mutedText = Color.white.opacity(0.35)
    static let bodyText = Color.white.opacity(0.88)

    static let rowDivider = Color.white.opacity(0.10)

    /// Standard corner radius for cards, fields, buttons
    static let cardRadius: CGFloat = 16
    /// Smaller radius for inline elements (tags, pills)
    static let smallRadius: CGFloat = 10
    /// Larger radius for overlays, sheets
    static let sheetRadius: CGFloat = 24

    /// Home / page-selection atmospheric gradient
    static let atmosphericGradient = LinearGradient(
        stops: [
            .init(color: Color(red: 0.10, green: 0.12, blue: 0.16), location: 0.0),
            .init(color: Color(red: 0.16, green: 0.20, blue: 0.26), location: 0.25),
            .init(color: Color(red: 0.28, green: 0.32, blue: 0.38), location: 0.50),
            .init(color: Color(red: 0.42, green: 0.42, blue: 0.42), location: 0.72),
            .init(color: Color(red: 0.56, green: 0.54, blue: 0.52), location: 0.88),
            .init(color: Color(red: 0.62, green: 0.60, blue: 0.56), location: 1.0),
        ],
        startPoint: .top,
        endPoint: .bottom
    )
}

// MARK: - Glass Card (composer overlay)

struct AetherGlassCard<Content: View>: View {
    var cornerRadius: CGFloat = AetherTheme.sheetRadius
    @ViewBuilder var content: Content

    var body: some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Color.white.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(Color.white.opacity(0.12), lineWidth: 0.6)
            )
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}

// MARK: - Haptics

enum AetherHaptics {
    static func tap() {
#if canImport(UIKit)
        UISelectionFeedbackGenerator().selectionChanged()
#endif
    }

    static func light() {
#if canImport(UIKit)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
#endif
    }

    static func medium() {
#if canImport(UIKit)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
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

// MARK: - Navigation State

enum AppPage: Equatable {
    case voice
    case pageSelection
    case history
    case memory
    case plugins
}

// MARK: - App Root

struct AppRootView: View {
    @EnvironmentObject var pairing: PairingService
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @StateObject private var audio = AudioService()
    @State private var currentPage: AppPage = .voice
    @State private var showComposer = false
    @State private var composerText = ""
    @FocusState private var composerFocused: Bool

    private var anim: Animation {
        reduceMotion ? .linear(duration: 0.01) : .spring(response: 0.45, dampingFraction: 0.88)
    }

    private var quickAnim: Animation {
        reduceMotion ? .linear(duration: 0.01) : .easeInOut(duration: 0.25)
    }

    var body: some View {
        ZStack {
            // Consistent gradient background behind everything — prevents black flash
            AetherTheme.atmosphericGradient
                .ignoresSafeArea()

            // Current page content — fills entire screen
            Group {
                switch currentPage {
                case .voice:
                    VoiceOrbView()
                        .transition(.opacity.combined(with: .scale(scale: 0.97)))

                case .pageSelection:
                    PageSelectionView(onSelect: { page in
                        AetherHaptics.light()
                        withAnimation(anim) {
                            currentPage = page
                        }
                    })
                    .transition(.opacity.combined(with: .scale(scale: 0.97)))

                case .history:
                    HistoryView()
                        .transition(.asymmetric(
                            insertion: .opacity.combined(with: .scale(scale: 0.95)),
                            removal: .opacity.combined(with: .scale(scale: 0.95))
                        ))

                case .memory:
                    MemoryView()
                        .transition(.asymmetric(
                            insertion: .opacity.combined(with: .scale(scale: 0.95)),
                            removal: .opacity.combined(with: .scale(scale: 0.95))
                        ))

                case .plugins:
                    PluginsView()
                        .transition(.asymmetric(
                            insertion: .opacity.combined(with: .scale(scale: 0.95)),
                            removal: .opacity.combined(with: .scale(scale: 0.95))
                        ))
                }
            }
            .animation(anim, value: currentPage)

            // Voice button — fixed floating at bottom center, always visible
            // Placed outside the page content stack so it never participates in page transitions
            if !showComposer {
                GlobalActionButton(
                    audio: audio,
                    currentPage: currentPage,
                    onSingleTap: {
                        AetherHaptics.tap()
                        withAnimation(anim) {
                            currentPage = .voice
                        }
                    },
                    onDoubleTap: {
                        AetherHaptics.medium()
                        withAnimation(anim) {
                            currentPage = .pageSelection
                        }
                    },
                    onTapCompose: {
                        AetherHaptics.tap()
                        withAnimation(quickAnim) {
                            showComposer = true
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                            composerFocused = true
                        }
                    }
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                .padding(.bottom, 28)
                .ignoresSafeArea(.keyboard)
                .transition(.opacity)
                .animation(quickAnim, value: showComposer)
            }

            // Text composer overlay
            if showComposer {
                Color.black.opacity(0.5)
                    .ignoresSafeArea()
                    .onTapGesture {
                        composerFocused = false
                        withAnimation(quickAnim) {
                            showComposer = false
                        }
                    }
                composerOverlay
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
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
                withAnimation(quickAnim) {
                    showComposer = false
                }
                composerFocused = false
            }
        }
        .environmentObject(audio)
        .preferredColorScheme(.dark)
        .tint(.white)
    }

    // MARK: - Composer Overlay

    private var composerOverlay: some View {
        VStack {
            Spacer()
            AetherGlassCard(cornerRadius: AetherTheme.sheetRadius) {
                HStack(spacing: 10) {
                    TextField("type instruction...", text: $composerText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 15, weight: .regular, design: .default))
                        .foregroundStyle(.white.opacity(0.9))
                        .accessibilityLabel("Instruction input")
                        .focused($composerFocused)
                        .submitLabel(.send)
                        .onSubmit { sendComposerText() }

                    Button("Cancel") {
                        composerFocused = false
                        withAnimation(quickAnim) {
                            showComposer = false
                        }
                    }
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.white.opacity(0.45))

                    Button {
                        sendComposerText()
                    } label: {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.black.opacity(0.7))
                            .frame(width: 32, height: 32)
                            .background(
                                Color.white.opacity(
                                    composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.15 : 0.85
                                ),
                                in: Circle()
                            )
                    }
                    .disabled(composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityLabel("Send instruction")
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
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
        withAnimation(quickAnim) {
            showComposer = false
        }
        withAnimation(anim) {
            currentPage = .voice
        }
    }
}

// MARK: - Page Selection View

struct PageSelectionView: View {
    var onSelect: (AppPage) -> Void
    @State private var appeared = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let pages: [(page: AppPage, title: String, subtitle: String)] = [
        (.history, "History", "Your conversations"),
        (.memory, "Memory", "What I remember about you"),
        (.plugins, "Plugins", "Connected services"),
    ]

    var body: some View {
        ZStack {
            AetherTheme.atmosphericGradient
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                // Page cards
                VStack(spacing: 12) {
                    ForEach(Array(pages.enumerated()), id: \.element.page) { index, item in
                        Button {
                            onSelect(item.page)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(item.title)
                                        .font(.system(size: 26, weight: .light))
                                        .foregroundStyle(.white.opacity(0.88))

                                    Text(item.subtitle)
                                        .font(.system(size: 13, weight: .regular))
                                        .foregroundStyle(.white.opacity(0.40))
                                }
                                Spacer()

                                Image(systemName: "chevron.right")
                                    .font(.system(size: 14, weight: .light))
                                    .foregroundStyle(.white.opacity(0.25))
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 22)
                            .background(
                                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                                    .fill(Color.white.opacity(0.06))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                                    .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
                            )
                        }
                        .buttonStyle(PageRowButtonStyle())
                        .opacity(appeared ? 1 : 0)
                        .offset(y: appeared ? 0 : 20)
                        .animation(
                            (reduceMotion ? Animation.linear(duration: 0.01) : Animation.easeOut(duration: 0.45)).delay(Double(index) * 0.08),
                            value: appeared
                        )
                    }
                }
                .padding(.horizontal, 20)

                Spacer()
                // Extra spacer for voice button clearance
                Spacer()
            }
        }
        .onAppear {
            withAnimation(reduceMotion ? .linear(duration: 0.01) : .easeOut(duration: 0.5)) {
                appeared = true
            }
        }
        .onDisappear {
            appeared = false
        }
    }
}

// Subtle press effect for page rows
private struct PageRowButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

// MARK: - Global Voice Action Button

struct GlobalActionButton: View {
    @ObservedObject var audio: AudioService
    var currentPage: AppPage
    var onSingleTap: () -> Void
    var onDoubleTap: () -> Void
    var onTapCompose: () -> Void

    @State private var isPressingOrb = false
    @State private var dragOffset: CGSize = .zero
    @State private var cancelArmed = false
    @State private var hasStartedRecording = false
    @State private var delayedStart: DispatchWorkItem?
    @State private var buttonScale: CGFloat = 1.0
    @State private var ringPulse = false

    // Double-tap detection
    @State private var lastTapTime: Date = .distantPast
    @State private var pendingSingleTap: DispatchWorkItem?

    private let cancelThreshold: CGFloat = -80
    private let doubleTapInterval: TimeInterval = 0.28

    var body: some View {
        ZStack {
            // Outer pulse ring (only when recording)
            if hasStartedRecording {
                Circle()
                    .stroke(Color.white.opacity(ringPulse ? 0.05 : 0.20), lineWidth: 1.2)
                    .frame(width: ringPulse ? 72 : 58, height: ringPulse ? 72 : 58)
                    .animation(
                        .easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                        value: ringPulse
                    )
            }

            // White button circle — fixed size, scale applied to the ZStack
            Circle()
                .fill(Color.white)
                .frame(width: 50, height: 50)

            // Dark waveform icon
            Image(systemName: hasStartedRecording ? "waveform.badge.mic" : "waveform")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Color.black.opacity(0.75))
        }
        .scaleEffect(buttonScale)
        .shadow(color: .white.opacity(0.06), radius: 16, y: 0)
        // When recording, allow dragging in any direction. Slide left to cancel.
        .offset(hasStartedRecording ? CGSize(width: dragOffset.width * 0.4, height: min(0, dragOffset.height * 0.2)) : .zero)
        .animation(.interactiveSpring(response: 0.2, dampingFraction: 0.8), value: dragOffset)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    if !isPressingOrb {
                        isPressingOrb = true
                        dragOffset = .zero
                        cancelArmed = false
                        hasStartedRecording = false

                        // Gentle scale down — no jump
                        withAnimation(.easeOut(duration: 0.12)) {
                            buttonScale = 0.92
                        }

                        let work = DispatchWorkItem {
                            guard isPressingOrb else { return }
                            pendingSingleTap?.cancel()
                            pendingSingleTap = nil

                            audio.beginRecording()
                            hasStartedRecording = true
                            ringPulse = true
                            AetherHaptics.tap()
                        }
                        delayedStart = work
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.22, execute: work)
                    }

                    guard isPressingOrb, hasStartedRecording else { return }
                    dragOffset = value.translation
                    cancelArmed = value.translation.width <= cancelThreshold
                }
                .onEnded { value in
                    guard isPressingOrb else { return }
                    isPressingOrb = false

                    delayedStart?.cancel()
                    delayedStart = nil

                    if hasStartedRecording {
                        let shouldCancel = cancelArmed
                        dragOffset = .zero
                        cancelArmed = false
                        hasStartedRecording = false
                        ringPulse = false

                        if shouldCancel {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                buttonScale = 1.0
                            }
                            audio.cancelRecording()
                            AetherHaptics.warning()
                        } else {
                            // Send — bounce up
                            withAnimation(.spring(response: 0.2, dampingFraction: 0.5)) {
                                buttonScale = 1.08
                            }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                    buttonScale = 1.0
                                }
                            }
                            audio.endRecordingAndSend()
                            AetherHaptics.success()
                        }
                    } else if abs(value.translation.width) < 10 && abs(value.translation.height) < 10 {
                        // Short tap — detect single vs double
                        // Quick bounce
                        withAnimation(.spring(response: 0.2, dampingFraction: 0.6)) {
                            buttonScale = 1.05
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            withAnimation(.spring(response: 0.25, dampingFraction: 0.7)) {
                                buttonScale = 1.0
                            }
                        }

                        let now = Date()
                        let elapsed = now.timeIntervalSince(lastTapTime)

                        if elapsed < doubleTapInterval {
                            pendingSingleTap?.cancel()
                            pendingSingleTap = nil
                            lastTapTime = .distantPast
                            onDoubleTap()
                        } else {
                            lastTapTime = now
                            let work = DispatchWorkItem {
                                if currentPage == .voice {
                                    onTapCompose()
                                } else {
                                    onSingleTap()
                                }
                            }
                            pendingSingleTap = work
                            DispatchQueue.main.asyncAfter(deadline: .now() + doubleTapInterval, execute: work)
                        }
                    } else {
                        // Drag ended without recording — just reset
                        withAnimation(.spring(response: 0.25, dampingFraction: 0.7)) {
                            buttonScale = 1.0
                        }
                    }
                }
        )
        .accessibilityLabel("Voice button")
        .accessibilityHint("Tap to go home or type. Double-tap for pages. Hold to record.")
        .accessibilityIdentifier("global-voice-action")
        .onDisappear {
            delayedStart?.cancel()
            delayedStart = nil
            pendingSingleTap?.cancel()
            pendingSingleTap = nil
        }
    }
}
