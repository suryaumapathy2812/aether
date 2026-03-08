import SwiftUI

@main
struct AetherApp: App {
    @StateObject private var pairingService = PairingService()
    @State private var showMainUI = false

    var body: some Scene {
        WindowGroup {
            ZStack {
                Color(hex: "111111").ignoresSafeArea()

                if pairingService.isPaired {
                    AppRootView()
                        .environmentObject(pairingService)
                        .opacity(showMainUI ? 1 : 0)
                        .scaleEffect(showMainUI ? 1 : 0.98)
                } else {
                    PairingView()
                        .environmentObject(pairingService)
                        .opacity(showMainUI ? 0 : 1)
                        .scaleEffect(showMainUI ? 1.04 : 1)
                }
            }
            .animation(.easeInOut(duration: 0.7), value: showMainUI)
            .onChange(of: pairingService.isPaired) {
                if pairingService.isPaired {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        showMainUI = true
                    }
                } else {
                    showMainUI = false
                }
            }
            .onAppear {
                if pairingService.isPaired {
                    showMainUI = true
                }
            }
        }
    }
}
