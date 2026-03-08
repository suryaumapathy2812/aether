import SwiftUI

@MainActor
final class HistoryViewModel: ObservableObject {
    @Published var loading = false
    @Published var error = ""
    @Published var conversations: [MemoryConversationItem] = []

    private var service: MemoryService?

    func configure(baseURL: String, token: String) {
        guard !token.isEmpty else { return }
        service = MemoryService(api: APIClient(baseURL: baseURL, token: token))
    }

    func load() async {
        guard let service else {
            error = "device not paired"
            return
        }
        loading = true
        error = ""
        do {
            conversations = try await service.fetchConversations(limit: 60)
        } catch {
            self.error = error.localizedDescription
        }
        loading = false
    }
}

struct HistoryView: View {
    @EnvironmentObject var pairing: PairingService
    @StateObject private var model = HistoryViewModel()

    var body: some View {
        NavigationStack {
            Group {
                if model.loading {
                    ProgressView("Loading history...")
                        .tint(.white)
                } else if !model.error.isEmpty {
                    ContentUnavailableView("Could not load history", systemImage: "exclamationmark.triangle", description: Text(model.error))
                } else if model.conversations.isEmpty {
                    ContentUnavailableView("No messages yet", systemImage: "text.bubble")
                } else {
                    ScrollView {
                        LazyVStack(spacing: 14) {
                            ForEach(model.conversations.reversed()) { item in
                                VStack(spacing: 8) {
                                    HStack {
                                        Spacer(minLength: 48)
                                        Text(item.userMessage)
                                            .font(.system(size: 14, weight: .medium, design: .rounded))
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 10)
                                            .background(Color.white.opacity(0.11), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                                    }
                                    HStack {
                                        Text(item.assistantMessage)
                                            .font(.system(size: 14, weight: .regular, design: .rounded))
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 10)
                                            .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                                        Spacer(minLength: 48)
                                    }
                                    if item.timestamp > 0 {
                                        Text(relativeTime(fromEpoch: item.timestamp))
                                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                                            .foregroundStyle(.white.opacity(0.4))
                                    }
                                }
                                .padding(.horizontal, 12)
                            }
                        }
                        .padding(.vertical, 10)
                    }
                    .refreshable {
                        await model.load()
                    }
                    .background(Color(hex: "111111"))
                }
            }
            .navigationTitle("History")
        }
        .preferredColorScheme(.dark)
        .task {
            let token = pairing.getDeviceToken() ?? ""
            model.configure(baseURL: pairing.orchestratorURL, token: token)
            await model.load()
        }
    }
}

private func relativeTime(fromEpoch value: TimeInterval) -> String {
    if value <= 0 { return "-" }
    let now = Date().timeIntervalSince1970
    let diff = max(0, now - value)
    if diff < 60 { return "just now" }
    if diff < 3600 { return "\(Int(diff / 60))m ago" }
    if diff < 86400 { return "\(Int(diff / 3600))h ago" }
    return "\(Int(diff / 86400))d ago"
}
