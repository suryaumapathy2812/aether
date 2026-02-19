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
                        .scaleEffect(showOrb ? 1 : 0.92)
                } else {
                    PairingView()
                        .environmentObject(pairingService)
                        .opacity(showOrb ? 0 : 1)
                        .scaleEffect(showOrb ? 1.04 : 1)
                }
            }
            .animation(.easeInOut(duration: 0.7), value: showOrb)
            .onChange(of: pairingService.isPaired) {
                if pairingService.isPaired {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        showOrb = true
                    }
                } else {
                    showOrb = false
                }
            }
            .onAppear {
                if pairingService.isPaired {
                    showOrb = true
                }
            }
        }
    }
}
