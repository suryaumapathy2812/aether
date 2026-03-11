import SwiftUI

@MainActor
final class PluginsViewModel: ObservableObject {
    @Published var loading = false
    @Published var error = ""
    @Published var plugins: [PluginItem] = []

    private var service: PluginsService?

    func configure(baseURL: String, token: String) {
        guard !token.isEmpty else { return }
        service = PluginsService(api: APIClient(baseURL: baseURL, token: token))
    }

    func reload() async {
        guard let service else {
            error = "device not paired"
            return
        }
        loading = true
        error = ""
        do {
            plugins = try await service.listPlugins()
        } catch {
            self.error = error.localizedDescription
        }
        loading = false
    }

    func installIfNeeded(_ plugin: PluginItem) async throws {
        guard !plugin.installed, let service else { return }
        try await service.install(name: plugin.name)
    }

    func setEnabled(_ plugin: PluginItem, enabled: Bool) async throws {
        guard let service else { return }
        if enabled {
            try await service.enable(name: plugin.name)
        } else {
            try await service.disable(name: plugin.name)
        }
    }

    func currentConfig(for plugin: PluginItem) async throws -> [String: String] {
        guard let service else { return [:] }
        return try await service.getConfig(name: plugin.name)
    }

    func saveConfig(for plugin: PluginItem, config: [String: String]) async throws {
        guard let service else { return }
        try await service.saveConfig(name: plugin.name, config: config)
    }
}

struct PluginsView: View {
    @EnvironmentObject var pairing: PairingService
    @StateObject private var model = PluginsViewModel()
    var embedded = false
    @State private var lastLoadedToken = ""

    var body: some View {
        NavigationStack {
            Group {
                if model.loading {
                    ProgressView("Loading plugins...")
                        .tint(.white)
                } else if !model.error.isEmpty {
                    ContentUnavailableView("Could not load plugins", systemImage: "exclamationmark.triangle", description: Text(model.error))
                } else if model.plugins.isEmpty {
                    ContentUnavailableView("No plugins", systemImage: "puzzlepiece.extension")
                } else {
                    List(model.plugins) { plugin in
                        NavigationLink {
                            PluginDetailView(plugin: plugin, model: model, baseURL: pairing.orchestratorURL, token: pairing.getDeviceToken() ?? "")
                        } label: {
                            HStack(alignment: .top, spacing: 12) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(plugin.displayName.isEmpty ? plugin.name : plugin.displayName)
                                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                                        .foregroundStyle(.white.opacity(0.9))
                                    Text(plugin.description.isEmpty ? "Use this plugin with Aether." : plugin.description)
                                        .font(.system(size: 12, weight: .regular, design: .rounded))
                                        .foregroundStyle(.white.opacity(0.62))
                                        .lineLimit(2)
                                }
                                Spacer()
                                statusBadge(plugin)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .listStyle(.insetGrouped)
                    .scrollContentBackground(.hidden)
                    .background(Color.black.opacity(0.15))
                    .refreshable {
                        await model.reload()
                    }
                }
            }
            .navigationTitle("Plugins")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar(embedded ? .hidden : .visible, for: .navigationBar)
        }
        .preferredColorScheme(.dark)
        .task(id: pairing.getDeviceToken() ?? "") {
            let token = pairing.getDeviceToken() ?? ""
            model.configure(baseURL: pairing.orchestratorURL, token: token)
            if token != lastLoadedToken {
                lastLoadedToken = token
                await model.reload()
            }
        }
    }

    private func statusBadge(_ plugin: PluginItem) -> some View {
        let text: String
        let color: Color
        if !plugin.installed {
            text = "set up"
            color = .blue
        } else if plugin.needsReconnect || (plugin.authType == "oauth2" && !plugin.connected) {
            text = "needs attention"
            color = .orange
        } else if !plugin.enabled {
            text = "off"
            color = .gray
        } else {
            text = "connected"
            color = .green
        }
        return Text(text)
            .font(.system(size: 10, weight: .medium, design: .rounded))
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(color.opacity(0.2), in: Capsule())
            .foregroundStyle(color.opacity(0.95))
    }
}

private struct PluginDetailView: View {
    let plugin: PluginItem
    @ObservedObject var model: PluginsViewModel
    let baseURL: String
    let token: String

    @State private var config: [String: String] = [:]
    @State private var busy = false
    @State private var info = ""
    @State private var error = ""
    @Environment(\.openURL) private var openURL

    var body: some View {
        List {
            Section {
                Text(plugin.description)
                    .font(.system(size: 14, weight: .regular, design: .rounded))
            }

            if !plugin.configFields.isEmpty {
                Section("Configuration") {
                    ForEach(plugin.configFields, id: \.key) { field in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(field.label)
                                .font(.system(size: 12, weight: .medium, design: .rounded))
                            if field.type == "password" {
                                SecureField(field.description.isEmpty ? field.label : field.description, text: binding(for: field.key))
                                    .textInputAutocapitalization(.never)
                            } else {
                                TextField(field.description.isEmpty ? field.label : field.description, text: binding(for: field.key))
                                    .textInputAutocapitalization(.never)
                            }
                        }
                        .padding(.vertical, 4)
                    }

                    Button(busy ? "Saving..." : "Save") {
                        Task { await saveConfig() }
                    }
                    .disabled(busy)
                }
            }

            Section("Connection") {
                if plugin.authType == "oauth2" {
                    Button(plugin.connected ? "Reconnect" : "Connect") {
                        startOAuth()
                    }
                    .disabled(token.isEmpty)
                }

                Button(plugin.enabled ? "Turn Off" : "Turn On") {
                    Task { await toggleEnabled() }
                }
                .disabled(busy)
            }

            if !info.isEmpty {
                Section {
                    Text(info)
                        .foregroundStyle(.green)
                }
            }
            if !error.isEmpty {
                Section {
                    Text(error)
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle(plugin.displayName.isEmpty ? plugin.name : plugin.displayName)
        .navigationBarTitleDisplayMode(.inline)
        .refreshable {
            await loadConfig()
            await model.reload()
        }
        .task {
            await loadConfig()
        }
    }

    private func binding(for key: String) -> Binding<String> {
        Binding(
            get: { config[key] ?? "" },
            set: { config[key] = $0 }
        )
    }

    private func loadConfig() async {
        do {
            config = try await model.currentConfig(for: plugin)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func saveConfig() async {
        busy = true
        defer { busy = false }
        error = ""
        info = ""
        do {
            try await model.installIfNeeded(plugin)
            try await model.saveConfig(for: plugin, config: config)
            info = "Saved"
            await model.reload()
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func toggleEnabled() async {
        busy = true
        defer { busy = false }
        error = ""
        info = ""
        do {
            try await model.installIfNeeded(plugin)
            try await model.setEnabled(plugin, enabled: !plugin.enabled)
            info = plugin.enabled ? "Turned off" : "Turned on"
            await model.reload()
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func startOAuth() {
        guard !token.isEmpty else { return }
        let encodedName = plugin.name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? plugin.name
        let queryToken = token.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? token
        guard let url = URL(string: "\(baseURL)/plugins/\(encodedName)/oauth/start?token=\(queryToken)") else {
            error = "invalid oauth url"
            return
        }
        openURL(url)
        info = "Finish connection in browser, then return and refresh."
    }
}
