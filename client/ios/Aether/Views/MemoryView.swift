import SwiftUI

enum MemorySection: String, CaseIterable, Identifiable {
    case about = "About"
    case entities = "Entities"
    case memories = "Memories"
    case decisions = "Decisions"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .about: return "person.text.rectangle"
        case .entities: return "person.2"
        case .memories: return "archivebox"
        case .decisions: return "checkmark.seal"
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
            async let f = service.fetchFacts()
            async let m = service.fetchMemories(limit: 100)
            async let d = service.fetchDecisions()
            async let e = service.fetchEntities(limit: 50)
            facts = try await f
            memories = try await m
            decisions = try await d
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
        case .about: return facts.count
        case .entities: return entities.count
        case .memories: return memories.count
        case .decisions: return decisions.count
        }
    }
}

struct MemoryView: View {
    @EnvironmentObject var pairing: PairingService
    @StateObject private var model = MemoryViewModel()
    var embedded = false
    @State private var lastLoadedToken = ""

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Memory")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar(embedded ? .hidden : .visible, for: .navigationBar)
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
        .sheet(isPresented: Binding(
            get: { model.selectedEntity != nil },
            set: { if !$0 { model.selectedEntity = nil } }
        )) {
            if let details = model.selectedEntity {
                EntityDetailSheet(details: details)
            }
        }
    }

    private var content: some View {
        Group {
            if model.loading {
                ProgressView("Loading memory...")
                    .tint(.white)
            } else if !model.error.isEmpty {
                ContentUnavailableView("Could not load memory", systemImage: "exclamationmark.triangle", description: Text(model.error))
            } else {
                List {
                    ForEach(MemorySection.allCases) { section in
                        NavigationLink {
                            destinationView(for: section)
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: section.icon)
                                    .foregroundStyle(.white.opacity(0.78))
                                Text(section.rawValue)
                                    .font(.system(size: 15, weight: .medium, design: .rounded))
                                    .foregroundStyle(.white.opacity(0.9))
                                Spacer()
                                Text("\(model.count(for: section))")
                                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.46))
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
                .background(Color.black.opacity(0.15))
                .refreshable {
                    await model.load()
                }
            }
        }
    }

    @ViewBuilder
    private func destinationView(for section: MemorySection) -> some View {
        switch section {
        case .about:
            List {
                if model.facts.isEmpty {
                    Text("No facts yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(model.facts.enumerated()), id: \.offset) { _, fact in
                        Text(fact)
                    }
                }
            }
            .navigationTitle("About")
        case .entities:
            List {
                if model.entities.isEmpty {
                    Text("No entities yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(model.entities) { entity in
                        Button {
                            Task { await model.loadEntityDetails(entityID: entity.id) }
                        } label: {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(entity.name)
                                    .foregroundStyle(.white.opacity(0.9))
                                if !entity.summary.isEmpty {
                                    Text(entity.summary)
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .navigationTitle("Entities")
        case .memories:
            List {
                if model.memories.isEmpty {
                    Text("No episodic memories yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(model.memories) { item in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(item.memory)
                            Text("\(item.category) · \(Int((item.confidence * 100).rounded()))%")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Memories")
        case .decisions:
            List {
                if model.decisions.isEmpty {
                    Text("No decisions yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(model.decisions) { item in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(item.decision)
                            Text("\(item.category) · \(item.active ? "active" : "inactive")")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Decisions")
        }
    }
}

private struct EntityDetailSheet: View {
    let details: EntityDetails

    var body: some View {
        NavigationStack {
            List {
                Section("Entity") {
                    Text(details.entity.name)
                    Text(details.entity.summary)
                        .foregroundStyle(.secondary)
                }

                if !details.observations.isEmpty {
                    Section("Observations") {
                        ForEach(details.observations) { row in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(row.observation)
                                Text("\(row.category) · \(row.source)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                if !details.relations.isEmpty {
                    Section("Relations") {
                        ForEach(details.relations) { row in
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(row.sourceEntityID) \(row.relation) \(row.targetEntityID)")
                                if !row.context.isEmpty {
                                    Text(row.context)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                if !details.interactions.isEmpty {
                    Section("Interactions") {
                        ForEach(details.interactions) { row in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(row.summary)
                                Text(row.interactionAt)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle(details.entity.name)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
