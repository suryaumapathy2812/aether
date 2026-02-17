import SwiftUI

/// Pairing screen — shows a code, user enters it on the dashboard.
/// Light Phone aesthetic: dark bg, centered content, minimal.
struct PairingView: View {
    @EnvironmentObject var pairing: PairingService

    var body: some View {
        ZStack {
            Color(hex: "1c1c1c").ignoresSafeArea()

            VStack(spacing: 40) {
                Spacer()

                // Brand
                Text("aether")
                    .font(.system(size: 12, weight: .light))
                    .tracking(6)
                    .foregroundColor(.white.opacity(0.3))

                if pairing.pairingCode.isEmpty {
                    // Initial state
                    VStack(spacing: 16) {
                        Text("pair your device")
                            .font(.system(size: 14, weight: .regular))
                            .tracking(2)
                            .foregroundColor(.white.opacity(0.6))

                        Button(action: { pairing.startPairing() }) {
                            Text("start")
                                .font(.system(size: 14, weight: .regular))
                                .tracking(2)
                                .foregroundColor(.white.opacity(0.8))
                                .padding(.horizontal, 32)
                                .padding(.vertical, 12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 0)
                                        .stroke(Color.white.opacity(0.3), lineWidth: 1)
                                )
                        }
                    }
                } else {
                    // Show code
                    VStack(spacing: 24) {
                        Text("enter this code on your dashboard")
                            .font(.system(size: 12, weight: .regular))
                            .tracking(1)
                            .foregroundColor(.white.opacity(0.4))

                        Text(pairing.pairingCode)
                            .font(.system(size: 28, weight: .light, design: .monospaced))
                            .tracking(4)
                            .foregroundColor(.white.opacity(0.9))

                        // Status
                        Text(pairing.status)
                            .font(.system(size: 11, weight: .regular))
                            .foregroundColor(.white.opacity(0.3))
                    }
                }

                Spacer()
                Spacer()
            }
        }
        .preferredColorScheme(.dark)
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
