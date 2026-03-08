import SwiftUI

private enum RootTab: Hashable {
    case history
    case memory
    case plugins
    case speak
}

struct AppRootView: View {
    @State private var selectedTab: RootTab = .speak

    var body: some View {
        TabView(selection: $selectedTab) {
            HistoryView()
                .tabItem {
                    Label("History", systemImage: "text.bubble")
                }
                .tag(RootTab.history)

            MemoryView()
                .tabItem {
                    Label("Memory", systemImage: "brain.head.profile")
                }
                .tag(RootTab.memory)

            PluginsView()
                .tabItem {
                    Label("Plugins", systemImage: "puzzlepiece.extension")
                }
                .tag(RootTab.plugins)

            NavigationStack {
                VoiceOrbView()
                    .navigationTitle("Speak")
            }
                .tabItem {
                    Label("Speak", systemImage: "waveform.circle")
                }
                .tag(RootTab.speak)
        }
        .preferredColorScheme(.dark)
        .tint(.white)
    }
}
