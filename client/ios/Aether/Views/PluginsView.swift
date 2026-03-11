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
    @State private var lastLoadedToken = ""
    @State private var selectedPlugin: PluginItem?

    var body: some View {
        ZStack {
            AetherTheme.atmosphericGradient
                .ignoresSafeArea()

            if let plugin = selectedPlugin {
                PluginDetailView(
                    plugin: plugin,
                    model: model,
                    baseURL: pairing.orchestratorURL,
                    token: pairing.getDeviceToken() ?? "",
                    onBack: {
                        AetherHaptics.tap()
                        selectedPlugin = nil
                    }
                )
                .transition(.move(edge: .trailing).combined(with: .opacity))
            } else {
                mainContent
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: selectedPlugin?.id)
        .gesture(
            DragGesture()
                .onEnded { value in
                    if value.translation.width > 100 && selectedPlugin != nil {
                        AetherHaptics.tap()
                        selectedPlugin = nil
                    }
                }
        )
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

    private var mainContent: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 6) {
                    Text("PLUGINS")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(3.0)
                        .foregroundStyle(AetherTheme.softText)

                    Text("\(model.plugins.count) available")
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
            } else if model.plugins.isEmpty {
                Spacer()
                Text("No plugins available")
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(AetherTheme.mutedText)
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(model.plugins) { plugin in
                            pluginCard(plugin)
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

    private func pluginCard(_ plugin: PluginItem) -> some View {
        Button {
            AetherHaptics.tap()
            selectedPlugin = plugin
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text((plugin.displayName.isEmpty ? plugin.name : plugin.displayName).uppercased())
                        .font(.system(size: 13, weight: .medium))
                        .tracking(1.2)
                        .foregroundStyle(.white.opacity(0.78))
                    Spacer()
                    statusText(plugin)
                }
                Text(plugin.description.isEmpty ? "Use this plugin with Aether." : plugin.description)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundStyle(.white.opacity(0.30))
                    .lineLimit(2)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                    .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
            )
        }
        .buttonStyle(PluginCardButtonStyle())
    }

    private func statusText(_ plugin: PluginItem) -> some View {
        let text: String
        let color: Color
        if !plugin.installed {
            text = "SET UP"
            color = .blue
        } else if plugin.needsReconnect || (plugin.authType == "oauth2" && !plugin.connected) {
            text = "ATTENTION"
            color = .orange
        } else if !plugin.enabled {
            text = "OFF"
            color = .gray
        } else {
            text = "CONNECTED"
            color = .green
        }
        return Text(text)
            .font(.system(size: 10, weight: .medium))
            .tracking(1.0)
            .foregroundStyle(color.opacity(0.70))
    }
}

private struct PluginCardButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

// MARK: - Plugin Detail

private struct PluginDetailView: View {
    let plugin: PluginItem
    @ObservedObject var model: PluginsViewModel
    let baseURL: String
    let token: String
    var onBack: () -> Void

    @State private var config: [String: String] = [:]
    @State private var busy = false
    @State private var info = ""
    @State private var error = ""
    @Environment(\.openURL) private var openURL

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Button {
                    onBack()
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

                Text((plugin.displayName.isEmpty ? plugin.name : plugin.displayName).uppercased())
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(2.0)
                    .foregroundStyle(AetherTheme.softText)

                Spacer()
                Color.clear.frame(width: 60, height: 1)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            Rectangle().fill(Color.white.opacity(0.12)).frame(height: 0.5)

            ScrollView {
                VStack(spacing: 12) {
                    if !plugin.description.isEmpty {
                        pluginDetailCard {
                            Text(plugin.description)
                                .font(.system(size: 13, weight: .regular))
                                .foregroundStyle(.white.opacity(0.60))
                        }
                    }

                    if !plugin.configFields.isEmpty {
                        sectionHeader("CONFIGURATION")

                        VStack(spacing: 12) {
                            ForEach(plugin.configFields, id: \.key) { field in
                                VStack(alignment: .leading, spacing: 8) {
                                    Text(field.label.uppercased())
                                        .font(.system(size: 10, weight: .medium))
                                        .tracking(1.2)
                                        .foregroundStyle(.white.opacity(0.40))

                                    if field.type == "password" {
                                        SecureField(
                                            field.description.isEmpty ? field.label : field.description,
                                            text: binding(for: field.key)
                                        )
                                        .textInputAutocapitalization(.never)
                                        .font(.system(size: 14, weight: .regular))
                                        .foregroundStyle(.white.opacity(0.85))
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: AetherTheme.smallRadius, style: .continuous)
                                                .fill(Color.white.opacity(0.05))
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: AetherTheme.smallRadius, style: .continuous)
                                                .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
                                        )
                                    } else {
                                        TextField(
                                            field.description.isEmpty ? field.label : field.description,
                                            text: binding(for: field.key)
                                        )
                                        .textInputAutocapitalization(.never)
                                        .font(.system(size: 14, weight: .regular))
                                        .foregroundStyle(.white.opacity(0.85))
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: AetherTheme.smallRadius, style: .continuous)
                                                .fill(Color.white.opacity(0.05))
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: AetherTheme.smallRadius, style: .continuous)
                                                .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
                                        )
                                    }
                                }
                            }
                        }
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

                        Button {
                            Task { await saveConfig() }
                        } label: {
                            Text(busy ? "SAVING..." : "SAVE")
                                .font(.system(size: 12, weight: .medium))
                                .tracking(1.5)
                                .foregroundStyle(.white.opacity(busy ? 0.3 : 0.75))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(
                                    RoundedRectangle(cornerRadius: AetherTheme.smallRadius, style: .continuous)
                                        .fill(Color.white.opacity(0.08))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: AetherTheme.smallRadius, style: .continuous)
                                        .stroke(Color.white.opacity(0.10), lineWidth: 0.5)
                                )
                        }
                        .buttonStyle(.plain)
                        .disabled(busy)
                    }

                    sectionHeader("CONNECTION")

                    VStack(spacing: 0) {
                        if plugin.authType == "oauth2" {
                            actionRow(
                                title: plugin.connected ? "RECONNECT" : "CONNECT",
                                disabled: token.isEmpty
                            ) {
                                startOAuth()
                            }

                            Rectangle().fill(Color.white.opacity(0.08)).frame(height: 0.5)
                                .padding(.horizontal, 16)
                        }

                        actionRow(
                            title: plugin.enabled ? "TURN OFF" : "TURN ON",
                            disabled: busy
                        ) {
                            Task { await toggleEnabled() }
                        }
                    }
                    .background(
                        RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                            .fill(Color.white.opacity(0.06))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous)
                            .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: AetherTheme.cardRadius, style: .continuous))

                    if !info.isEmpty {
                        Text(info)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.green.opacity(0.70))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    if !error.isEmpty {
                        Text(error)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.red.opacity(0.70))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 100)
            }
            .scrollIndicators(.hidden)
        }
        .task {
            await loadConfig()
        }
    }

    private func pluginDetailCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
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

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 10, weight: .semibold))
            .tracking(2.0)
            .foregroundStyle(.white.opacity(0.28))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 12)
            .padding(.bottom, 2)
    }

    private func actionRow(title: String, disabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .tracking(1.2)
                    .foregroundStyle(.white.opacity(disabled ? 0.20 : 0.65))
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 16)
        }
        .buttonStyle(.plain)
        .disabled(disabled)
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
