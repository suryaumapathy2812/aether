import SwiftUI

/// Pairing screen — shows a code, user enters it on the dashboard.
/// Light Phone aesthetic: dark bg, centered content, minimal.
struct PairingView: View {
    @EnvironmentObject var pairing: PairingService

    @State private var codePulse = false

    var body: some View {
        ZStack {
            Color(hex: "111111").ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                // Brand
                Text("aether")
                    .font(.system(size: 12, weight: .light))
                    .tracking(6)
                    .foregroundStyle(.white.opacity(0.25))
                    .padding(.bottom, 56)

                if pairing.pairingCode.isEmpty {
                    initialState
                } else {
                    codeState
                }

                Spacer()
                Spacer()
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Initial State

    private var initialState: some View {
        VStack(spacing: 20) {
            Text("pair your device")
                .font(.system(size: 13, weight: .regular))
                .tracking(2)
                .foregroundStyle(.white.opacity(0.5))

            Button(action: { pairing.startPairing() }) {
                Text("start")
                    .font(.system(size: 13, weight: .regular))
                    .tracking(3)
                    .foregroundStyle(.white.opacity(0.75))
                    .padding(.horizontal, 36)
                    .padding(.vertical, 13)
                    .overlay(
                        Rectangle()
                            .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Code State

    private var codeState: some View {
        VStack(spacing: 28) {
            Text("enter this code on your dashboard")
                .font(.system(size: 11, weight: .regular))
                .tracking(1)
                .foregroundStyle(.white.opacity(0.35))

            // Pairing code with subtle pulse
            Text(pairing.pairingCode)
                .font(.system(size: 26, weight: .light, design: .monospaced))
                .tracking(3)
                .foregroundStyle(.white.opacity(codePulse ? 0.95 : 0.7))
                .onAppear {
                    withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) {
                        codePulse = true
                    }
                }

            // Status
            HStack(spacing: 6) {
                // Animated dot
                Circle()
                    .fill(.white.opacity(codePulse ? 0.5 : 0.15))
                    .frame(width: 4, height: 4)

                Text(pairing.status)
                    .font(.system(size: 10, weight: .regular))
                    .tracking(1)
                    .foregroundStyle(.white.opacity(0.3))
            }
            .animation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true), value: codePulse)
        }
    }
}

// MARK: - Color hex extension

extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        var rgb: UInt64 = 0
        scanner.scanHexInt64(&rgb)
        self.init(
            red: Double((rgb >> 16) & 0xFF) / 255.0,
            green: Double((rgb >> 8) & 0xFF) / 255.0,
            blue: Double(rgb & 0xFF) / 255.0
        )
    }
}
