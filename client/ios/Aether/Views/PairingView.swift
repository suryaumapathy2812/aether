import SwiftUI

struct PairingView: View {
    @EnvironmentObject var pairing: PairingService

    @State private var codePulse = false

    var body: some View {
        ZStack {
            pairingBackground

            VStack(spacing: 0) {
                Spacer()

                Text("aether")
                    .font(.system(size: 12, weight: .light, design: .serif))
                    .tracking(8)
                    .foregroundStyle(.white.opacity(0.34))
                    .padding(.bottom, 52)

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

    private var pairingBackground: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(hex: "0A0B0D"),
                    Color(hex: "101216"),
                    Color(hex: "0B0D10")
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color.white.opacity(0.11), .clear],
                        center: .center,
                        startRadius: 10,
                        endRadius: 270
                    )
                )
                .frame(width: 340, height: 340)
                .offset(x: -120, y: -280)
                .blur(radius: 8)

            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color(red: 0.72, green: 0.78, blue: 0.88).opacity(0.1), .clear],
                        center: .center,
                        startRadius: 20,
                        endRadius: 260
                    )
                )
                .frame(width: 320, height: 320)
                .offset(x: 130, y: 260)
                .blur(radius: 10)

            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.black.opacity(0.46), .clear, Color.black.opacity(0.62)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .ignoresSafeArea()
        }
    }

    private var initialState: some View {
        VStack(spacing: 24) {
            Text("pair your device")
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .tracking(2.4)
                .foregroundStyle(.white.opacity(0.58))

            Button(action: { pairing.startPairing() }) {
                Text("start")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .tracking(3.8)
                    .foregroundStyle(.white.opacity(0.9))
                    .padding(.horizontal, 40)
                    .padding(.vertical, 14)
                    .background(
                        Capsule()
                            .fill(Color.white.opacity(0.04))
                    )
                    .overlay(
                        Capsule()
                            .strokeBorder(.white.opacity(0.22), lineWidth: 1)
                    )
                    .shadow(color: .white.opacity(0.08), radius: 20, y: 6)
            }
            .buttonStyle(.plain)
        }
    }

    private var codeState: some View {
        VStack(spacing: 22) {
            Text("enter this code on your dashboard")
                .font(.system(size: 11, weight: .regular, design: .rounded))
                .tracking(1.4)
                .foregroundStyle(.white.opacity(0.44))

            VStack(spacing: 14) {
                Text(pairing.pairingCode)
                    .font(.system(size: 25, weight: .light, design: .monospaced))
                    .tracking(3.6)
                    .foregroundStyle(.white.opacity(codePulse ? 0.97 : 0.76))

                HStack(spacing: 7) {
                    Circle()
                        .fill(.white.opacity(codePulse ? 0.55 : 0.2))
                        .frame(width: 5, height: 5)

                    Text(pairing.status)
                        .font(.system(size: 10, weight: .regular, design: .rounded))
                        .tracking(1.2)
                        .foregroundStyle(.white.opacity(0.46))
                }
            }
            .padding(.horizontal, 26)
            .padding(.vertical, 20)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(Color.white.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.white.opacity(0.17), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.45), radius: 28, y: 12)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true)) {
                    codePulse = true
                }
            }
            .animation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true), value: codePulse)
        }
        .padding(.horizontal, 24)
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
