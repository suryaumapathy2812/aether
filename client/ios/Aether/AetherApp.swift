import SwiftUI

@main
struct AetherApp: App {
    @StateObject private var pairingService = PairingService()
    @State private var showOrb = false

    var body: some Scene {
        WindowGroup {
            ZStack {
                Color(hex: "111111").ignoresSafeArea()

                if pairingService.isPaired {
                    VoiceOrbView()
                        .environmentObject(pairingService)
                        .opacity(showOrb ? 1 : 0)
                        .scaleEffect(showOrb ? 1 : 0.8)
                } else {
                    PairingView()
                        .environmentObject(pairingService)
                        .opacity(showOrb ? 0 : 1)
                        .scaleEffect(showOrb ? 1.05 : 1)
                }
            }
            .animation(.easeInOut(duration: 0.8), value: showOrb)
            .onChange(of: pairingService.isPaired) {
                if pairingService.isPaired {
                    // Brief pause, then crossfade to orb
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        showOrb = true
                    }
                } else {
                    showOrb = false
                }
            }
            .onAppear {
                // If already paired on launch, skip animation
                if pairingService.isPaired {
                    showOrb = true
                }
            }
        }
    }
}
