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
    @State private var lastLoadedToken = ""

    var body: some View {
        ZStack {
            // Same atmospheric gradient as home/menu for visual consistency
            AetherTheme.atmosphericGradient
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("HISTORY")
                            .font(.system(size: 11, weight: .semibold))
                            .tracking(3.0)
                            .foregroundStyle(AetherTheme.softText)

                        Text("\(model.conversations.count) conversations")
                            .font(.system(size: 13, weight: .regular))
                            .foregroundStyle(AetherTheme.mutedText)
                    }
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 16)

                Rectangle()
                    .fill(Color.white.opacity(0.12))
                    .frame(height: 0.5)

                // Content
                if model.loading {
                    Spacer()
                    ProgressView()
                        .tint(.white.opacity(0.3))
                    Spacer()
                } else if !model.error.isEmpty {
                    Spacer()
                    Text(model.error)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(AetherTheme.mutedText)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                    Spacer()
                } else if model.conversations.isEmpty {
                    Spacer()
                    Text("No messages yet")
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(AetherTheme.mutedText)
                    Spacer()
                } else {
                    ScrollView {
                        LazyVStack(spacing: 10) {
                            ForEach(model.conversations.reversed()) { item in
                                conversationCard(item)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 12)
                        .padding(.bottom, 100)
                    }
                    .scrollIndicators(.hidden)
                }
            }
        }
        .preferredColorScheme(.dark)
        .task(id: pairing.getDeviceToken() ?? "") {
            let token = pairing.getDeviceToken() ?? ""
            model.configure(baseURL: pairing.orchestratorURL, token: token)
            if token != lastLoadedToken {
                lastLoadedToken = token
                await model.load()
            }
        }
    }

    private func conversationCard(_ item: MemoryConversationItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(item.userMessage)
                .font(.system(size: 13, weight: .medium))
                .tracking(0.4)
                .foregroundStyle(.white.opacity(0.80))
                .lineLimit(2)

            Text(item.assistantMessage)
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(.white.opacity(0.40))
                .lineLimit(3)

            if item.timestamp > 0 {
                Text(relativeTime(fromEpoch: item.timestamp))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(AetherTheme.mutedText)
                    .padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
        )
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
