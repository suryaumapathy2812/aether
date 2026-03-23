import SwiftUI

enum MemorySection: String, CaseIterable, Identifiable {
    case about = "About"
    case entities = "Entities"
    case memories = "Memories"
    case decisions = "Decisions"

    var id: String { rawValue }

    var subtitle: String {
        switch self {
        case .about:     return "FACTS ABOUT YOU"
        case .entities:  return "PEOPLE & THINGS"
        case .memories:  return "EPISODIC RECALL"
        case .decisions: return "RULES & PREFERENCES"
        }
    }
}

@MainActor
final class MemoryViewModel: ObservableObject {
    @Published var loading = false
    @Published var error = ""
    @Published var facts: [String] = []
    @Published var memories: [MemoryItem] = []
    @Published var decisions: [DecisionItem] = []
    @Published var entities: [EntityItem] = []
    @Published var selectedEntity: EntityDetails?

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
            async let itemsTask = service.fetchMemoryItems(limit: 200)
            async let e = service.fetchEntities(limit: 50)
            let items = try await itemsTask
            facts = items.filter { $0.kind == "fact" }.map(\.content)
            memories = items.filter { $0.kind == "memory" || $0.kind == "summary" }.map {
                MemoryItem(
                    id: $0.id,
                    memory: $0.content,
                    category: $0.kind == "summary" ? "summary:\($0.category.isEmpty ? "general" : $0.category)" : $0.category,
                    confidence: $0.confidence,
                    createdAt: $0.createdAt
                )
            }
            decisions = items.filter { $0.kind == "decision" }.map {
                DecisionItem(
                    id: $0.id,
                    decision: $0.content,
                    category: $0.category,
                    source: $0.sourceType,
                    active: $0.status == "active",
                    confidence: $0.confidence,
                    updatedAt: $0.updatedAt
                )
            }
            entities = try await e
        } catch {
            self.error = error.localizedDescription
        }
        loading = false
    }

    func loadEntityDetails(entityID: String) async {
        guard let service else { return }
        do {
            selectedEntity = try await service.fetchEntityDetails(entityID: entityID)
        } catch {
            self.error = error.localizedDescription
        }
    }

    func count(for section: MemorySection) -> Int {
        switch section {
        case .about:     return facts.count
        case .entities:  return entities.count
        case .memories:  return memories.count
        case .decisions: return decisions.count
        }
    }
}

struct MemoryView: View {
    @EnvironmentObject var pairing: PairingService
    @StateObject private var model = MemoryViewModel()
    @State private var lastLoadedToken = ""
    @State private var activeSection: MemorySection?
    @State private var showEntitySheet = false

    var body: some View {
        ZStack {
            AetherTheme.atmosphericGradient
                .ignoresSafeArea()

            if let section = activeSection {
                sectionDetailView(section)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            } else {
                mainContent
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: activeSection)
        .preferredColorScheme(.dark)
        .gesture(
            DragGesture()
                .onEnded { value in
                    if value.translation.width > 100 && activeSection != nil {
                        AetherHaptics.tap()
                        activeSection = nil
                    }
                }
        )
        .task(id: pairing.getDeviceToken() ?? "") {
            let token = pairing.getDeviceToken() ?? ""
            model.configure(baseURL: pairing.orchestratorURL, token: token)
            if token != lastLoadedToken {
                lastLoadedToken = token
                await model.load()
            }
        }
    }

    // MARK: - Main

    private var mainContent: some View {
        VStack(spacing: 0) {
            HStack {
                Text("MEMORY")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(3.0)
                    .foregroundStyle(AetherTheme.softText)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 16)

            Rectangle()
                .fill(Color.white.opacity(0.12))
                .frame(height: 0.5)

            if model.loading {
                Spacer()
                ProgressView().tint(.white.opacity(0.3))
                Spacer()
            } else if !model.error.isEmpty {
                Spacer()
                Text(model.error)
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(AetherTheme.mutedText)
                    .padding(.horizontal, 40)
                Spacer()
            } else {
                ScrollView {
                    VStack(spacing: 10) {
                        ForEach(MemorySection.allCases) { section in
                            sectionCard(section)
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

    private func sectionCard(_ section: MemorySection) -> some View {
        Button {
            AetherHaptics.tap()
            activeSection = section
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    HStack(spacing: 0) {
                        Text(section.rawValue.uppercased())
                            .font(.system(size: 13, weight: .medium))
                            .tracking(1.8)
                            .foregroundStyle(.white.opacity(0.78))
                        Text("  ·  \(model.count(for: section))")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.white.opacity(0.40))
                    }
                    Text(section.subtitle)
                        .font(.system(size: 11, weight: .regular))
                        .tracking(0.8)
                        .foregroundStyle(.white.opacity(0.30))
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.25))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 18)
            .background(
                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                    .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
            )
        }
        .buttonStyle(MemoryCardButtonStyle())
    }

    // MARK: - Section Detail Views

    @ViewBuilder
    private func sectionDetailView(_ section: MemorySection) -> some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Button {
                    AetherHaptics.tap()
                    activeSection = nil
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 13, weight: .medium))
                        Text("BACK")
                            .font(.system(size: 11, weight: .medium))
                            .tracking(1.5)
                    }
                    .foregroundStyle(AetherTheme.softText)
                }
                .buttonStyle(.plain)

                Spacer()

                Text(section.rawValue.uppercased())
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(2.0)
                    .foregroundStyle(AetherTheme.softText)

                Spacer()
                Color.clear.frame(width: 60, height: 1)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            Rectangle()
                .fill(Color.white.opacity(0.12))
                .frame(height: 0.5)

            switch section {
            case .about:     aboutDetail
            case .entities:  entitiesDetail
            case .memories:  memoriesDetail
            case .decisions: decisionsDetail
            }
        }
    }

    private func detailCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
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

    private var aboutDetail: some View {
        Group {
            if model.facts.isEmpty {
                Spacer()
                Text("No facts yet")
                    .font(.system(size: 13)).foregroundStyle(AetherTheme.mutedText)
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(Array(model.facts.enumerated()), id: \.offset) { _, fact in
                            detailCard {
                                Text(fact)
                                    .font(.system(size: 13, weight: .regular))
                                    .foregroundStyle(.white.opacity(0.72))
                            }
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

    private var entitiesDetail: some View {
        Group {
            if model.entities.isEmpty {
                Spacer()
                Text("No entities yet")
                    .font(.system(size: 13)).foregroundStyle(AetherTheme.mutedText)
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 16) {
                        ForEach(Array(groupedEntities.keys).sorted(), id: \.self) { entityType in
                            VStack(spacing: 8) {
                                // Entity type header
                                Text(entityType.uppercased())
                                    .font(.system(size: 10, weight: .semibold))
                                    .tracking(2.0)
                                    .foregroundStyle(.white.opacity(0.35))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 16)
                                    .padding(.top, 8)
                                
                                // Entities of this type
                                ForEach(groupedEntities[entityType]!) { entity in
                                    Button {
                                        Task { await model.loadEntityDetails(entityID: entity.id) }
                                        showEntitySheet = true
                                    } label: {
                                        detailCard {
                                            VStack(alignment: .leading, spacing: 5) {
                                                Text(entity.name.uppercased())
                                                    .font(.system(size: 13, weight: .medium))
                                                    .tracking(1.2)
                                                    .foregroundStyle(.white.opacity(0.78))
                                                if !entity.summary.isEmpty {
                                                    Text(entity.summary)
                                                        .font(.system(size: 11, weight: .regular))
                                                        .foregroundStyle(.white.opacity(0.35))
                                                        .lineLimit(2)
                                                }
                                            }
                                        }
                                    }
                                    .buttonStyle(MemoryCardButtonStyle())
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .padding(.bottom, 100)
                }
                .scrollIndicators(.hidden)
                .sheet(isPresented: $showEntitySheet) {
                    if let details = model.selectedEntity {
                        EntityDetailOverlay(details: details)
                    }
                }
            }
        }
    }

    private var groupedEntities: [String: [EntityItem]] {
        Dictionary(grouping: model.entities, by: \.entityType)
    }

    private var memoriesDetail: some View {
        Group {
            if model.memories.isEmpty {
                Spacer()
                Text("No episodic memories yet")
                    .font(.system(size: 13)).foregroundStyle(AetherTheme.mutedText)
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(model.memories) { item in
                            detailCard {
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(item.memory)
                                        .font(.system(size: 13, weight: .regular))
                                        .foregroundStyle(.white.opacity(0.72))
                                    Text("\(item.category.uppercased())  ·  \(Int((item.confidence * 100).rounded()))%")
                                        .font(.system(size: 10, weight: .medium))
                                        .tracking(1.0)
                                        .foregroundStyle(.white.opacity(0.30))
                                }
                            }
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

    private var decisionsDetail: some View {
        Group {
            if model.decisions.isEmpty {
                Spacer()
                Text("No decisions yet")
                    .font(.system(size: 13)).foregroundStyle(AetherTheme.mutedText)
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(model.decisions) { item in
                            detailCard {
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(item.decision)
                                        .font(.system(size: 13, weight: .regular))
                                        .foregroundStyle(.white.opacity(0.72))
                                    Text("\(item.category.uppercased())  ·  \(item.active ? "ACTIVE" : "INACTIVE")")
                                        .font(.system(size: 10, weight: .medium))
                                        .tracking(1.0)
                                        .foregroundStyle(.white.opacity(0.30))
                                }
                            }
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
}

// MARK: - Entity Detail Overlay

struct EntityDetailOverlay: View {
    let details: EntityDetails
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            AetherTheme.atmosphericGradient
                .ignoresSafeArea()

            VStack(spacing: 0) {
                HStack {
                    Button { dismiss() } label: {
                        Text("CLOSE")
                            .font(.system(size: 11, weight: .medium))
                            .tracking(1.5)
                            .foregroundStyle(AetherTheme.softText)
                    }
                    .buttonStyle(.plain)

                    Spacer()

                    Text(details.entity.name.uppercased())
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(2.0)
                        .foregroundStyle(AetherTheme.softText)

                    Spacer()
                    Color.clear.frame(width: 50, height: 1)
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 12)

                Rectangle().fill(Color.white.opacity(0.12)).frame(height: 0.5)

                ScrollView {
                    VStack(spacing: 12) {
                        if !details.entity.summary.isEmpty {
                            entityDetailCard {
                                Text(details.entity.summary)
                                    .font(.system(size: 13, weight: .regular))
                                    .foregroundStyle(.white.opacity(0.65))
                            }
                        }

                        if !details.observations.isEmpty {
                            detailSectionHeader("OBSERVATIONS")
                            ForEach(details.observations) { row in
                                entityDetailCard {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(row.observation)
                                            .font(.system(size: 13, weight: .regular))
                                            .foregroundStyle(.white.opacity(0.65))
                                        Text("\(row.category.uppercased())  ·  \(row.source.uppercased())")
                                            .font(.system(size: 10, weight: .medium))
                                            .tracking(0.8)
                                            .foregroundStyle(.white.opacity(0.28))
                                    }
                                }
                            }
                        }

                        if !details.relations.isEmpty {
                            detailSectionHeader("RELATIONS")
                            ForEach(details.relations) { row in
                                entityDetailCard {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("\(row.sourceEntityID) → \(row.relation) → \(row.targetEntityID)")
                                            .font(.system(size: 13, weight: .regular))
                                            .foregroundStyle(.white.opacity(0.65))
                                        if !row.context.isEmpty {
                                            Text(row.context)
                                                .font(.system(size: 11, weight: .regular))
                                                .foregroundStyle(.white.opacity(0.28))
                                        }
                                    }
                                }
                            }
                        }

                        if !details.interactions.isEmpty {
                            detailSectionHeader("INTERACTIONS")
                            ForEach(details.interactions) { row in
                                entityDetailCard {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(row.summary)
                                            .font(.system(size: 13, weight: .regular))
                                            .foregroundStyle(.white.opacity(0.65))
                                        Text(row.interactionAt)
                                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                                            .foregroundStyle(.white.opacity(0.28))
                                    }
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .padding(.bottom, 40)
                }
                .scrollIndicators(.hidden)
            }
        }
        .preferredColorScheme(.dark)
    }

    private func entityDetailCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                    .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
            )
    }

    private func detailSectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 10, weight: .semibold))
            .tracking(2.0)
            .foregroundStyle(.white.opacity(0.28))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 12)
            .padding(.bottom, 2)
    }
}

private struct MemoryCardButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}
