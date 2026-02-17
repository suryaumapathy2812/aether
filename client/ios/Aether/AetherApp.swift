import SwiftUI

@main
struct AetherApp: App {
    @StateObject private var pairingService = PairingService()

    var body: some Scene {
        WindowGroup {
            if pairingService.isPaired {
                VoiceOrbView()
                    .environmentObject(pairingService)
            } else {
                PairingView()
                    .environmentObject(pairingService)
            }
        }
    }
}
