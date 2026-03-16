"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams, useSearchParams } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { useSession } from "@/lib/auth-client";
import {
  listPlugins,
  installPlugin,
  uninstallPlugin,
  getPluginConfig,
  savePluginConfig,
  enablePlugin,
  disablePlugin,
  getOAuthStartUrl,
  PluginInfo,
} from "@/lib/api";
import {
  ExternalLink,
  Check,
  AlertCircle,
  Mail,
  Calendar,
  Contact2,
  HardDrive,
  CloudSun,
  Compass,
  MapPin,
  Rss,
  Music,
  BookOpen,
  Calculator,
  Puzzle,
  Trash2,
} from "lucide-react";

// ── Icon mapping (shared with list page) ──

const PLUGIN_ICONS: Record<string, React.ElementType> = {
  gmail: Mail,
  google_calendar: Calendar,
  google_contacts: Contact2,
  google_drive: HardDrive,
  weather: CloudSun,
  brave_search: Compass,
  local_search: MapPin,
  rss: Rss,
  spotify: Music,
  wikipedia: BookOpen,
  wolfram: Calculator,
};

function getPluginIcon(name: string): React.ElementType {
  return PLUGIN_ICONS[name] || Puzzle;
}

// ── Detail page ──

export default function PluginDetailPage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const pluginName = params.name as string;

  const [plugin, setPlugin] = useState<PluginInfo | null>(null);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [saving, setSaving] = useState(false);
  const [toggling, setToggling] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [uninstalling, setUninstalling] = useState(false);

  const justConnected = searchParams.get("connected") === "true";
  const oauthError = searchParams.get("error") || "";

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    loadPluginData();
  }, [session, isPending, router]);

  async function loadPluginData() {
    try {
      setLoading(true);
      setError("");
      const plugins = await listPlugins();
      const found = plugins.find((p) => p.name === pluginName);
      if (!found) {
        setError("Plugin not found");
        return;
      }
      setPlugin(found);

      if (found.installed) {
        const cfg = await getPluginConfig(pluginName);
        setConfig(cfg);
        setFormValues(cfg);
      }

      // Auto-install zero-config plugins on first visit
      if (!found.installed && found.auth_type === "none") {
        await installPlugin(pluginName);
        const refreshed = await listPlugins();
        const updated = refreshed.find((p) => p.name === pluginName);
        if (updated) setPlugin(updated);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not load plugin");
    } finally {
      setLoading(false);
    }
  }

  function getEditableFields() {
    if (!plugin?.config_fields) return [];
    return plugin.config_fields.filter((f) => f.key !== "account_email");
  }

  function hasRequiredConfig(): boolean {
    const required = getEditableFields().filter((f) => f.required);
    return required.every((f) => formValues[f.key]?.trim());
  }

  async function handleGetPlugin() {
    if (!plugin) return;
    setInstalling(true);
    setError("");
    try {
      await installPlugin(pluginName);
      const refreshed = await listPlugins();
      const updated = refreshed.find((p) => p.name === pluginName);
      if (updated) {
        setPlugin(updated);
        if (updated.installed) {
          const cfg = await getPluginConfig(pluginName);
          setConfig(cfg);
          setFormValues(cfg);
        }
      }
      setSuccess("Installed");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not install plugin");
    } finally {
      setInstalling(false);
    }
  }

  async function handleOAuthConnect() {
    if (!plugin) return;
    setError("");
    try {
      if (!plugin.installed) await installPlugin(pluginName);
      window.location.href = getOAuthStartUrl(pluginName);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not start connection");
    }
  }

  async function handleSave() {
    if (!plugin) return;
    setSaving(true);
    setError("");
    setSuccess("");
    try {
      if (!plugin.installed) {
        await installPlugin(pluginName);
        setPlugin({ ...plugin, installed: true });
      }
      const result = await savePluginConfig(pluginName, formValues);
      setConfig(formValues);
      if (result.auto_enabled) {
        setPlugin((p) =>
          p
            ? { ...p, installed: true, enabled: true, connected: true }
            : p
        );
        setSuccess("Saved and enabled");
      } else {
        setSuccess("Saved");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not save");
    } finally {
      setSaving(false);
    }
  }

  async function handleToggle(enabled: boolean) {
    if (!plugin) return;
    setToggling(true);
    setError("");
    setSuccess("");
    try {
      if (enabled) {
        await enablePlugin(pluginName);
        setPlugin({ ...plugin, enabled: true });
      } else {
        await disablePlugin(pluginName);
        setPlugin({ ...plugin, enabled: false });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setToggling(false);
    }
  }

  async function handleUninstall() {
    if (!plugin) return;
    setUninstalling(true);
    setError("");
    try {
      await uninstallPlugin(pluginName);
      router.push("/plugins");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not uninstall");
      setUninstalling(false);
    }
  }

  if (isPending || !session) return null;

  const Icon = getPluginIcon(pluginName);
  const editableFields = getEditableFields();
  const canEnable =
    plugin?.auth_type === "none" ||
    (plugin?.auth_type === "oauth2" && plugin.connected) ||
    (plugin?.auth_type === "api_key" && hasRequiredConfig());

  const status = plugin
    ? plugin.needs_reconnect ||
      (plugin.auth_type === "oauth2" && plugin.installed && !plugin.connected)
      ? "attention"
      : plugin.installed && plugin.enabled && plugin.connected
        ? "active"
        : plugin.installed
          ? "disabled"
          : "not_installed"
    : "not_installed";

  return (
    <ContentShell title={plugin?.display_name || "Plugin"} back="/plugins">
      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : !plugin ? (
        <p className="text-muted-foreground text-xs">
          {error || "Plugin not found"}
        </p>
      ) : (
        <div className="space-y-6">
          {/* ── Hero header ── */}
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 rounded-2xl bg-white/[0.06] flex items-center justify-center shrink-0">
              <Icon className="size-6 text-muted-foreground" strokeWidth={1.5} />
            </div>
            <div className="flex-1 min-w-0 pt-0.5">
              <div className="flex items-center gap-2.5">
                <h2 className="text-[15px] font-medium text-foreground truncate">
                  {plugin.display_name}
                </h2>
                {status === "active" && (
                  <Badge
                    variant="outline"
                    className="text-[9px] px-1.5 py-0 h-4 border-emerald-500/20 text-emerald-400/80 bg-emerald-500/[0.06]"
                  >
                    Active
                  </Badge>
                )}
                {status === "attention" && (
                  <Badge
                    variant="outline"
                    className="text-[9px] px-1.5 py-0 h-4 border-amber-500/20 text-amber-400/80 bg-amber-500/[0.06]"
                  >
                    Needs Setup
                  </Badge>
                )}
                {status === "disabled" && (
                  <Badge
                    variant="outline"
                    className="text-[9px] px-1.5 py-0 h-4 border-white/[0.06] text-muted-foreground/50 bg-white/[0.02]"
                  >
                    Disabled
                  </Badge>
                )}
              </div>
              <p className="text-[12px] text-muted-foreground/70 mt-1 leading-relaxed">
                {plugin.description}
              </p>
            </div>
          </div>

          {/* ── Primary CTA ── */}
          <div>
            {!plugin.installed ? (
              <Button
                variant="aether"
                size="aether"
                onClick={handleGetPlugin}
                disabled={installing}
                className="w-full"
              >
                {installing ? "Installing..." : "Get"}
              </Button>
            ) : plugin.auth_type === "oauth2" && !plugin.connected ? (
              <Button
                variant="aether"
                size="aether"
                onClick={handleOAuthConnect}
                className="w-full"
              >
                <ExternalLink className="size-3.5 mr-2" />
                Connect Account
              </Button>
            ) : plugin.auth_type === "api_key" && !hasRequiredConfig() ? (
              <div className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-amber-500/[0.06] border border-amber-500/10">
                <AlertCircle className="size-3.5 text-amber-400/70 shrink-0" />
                <p className="text-[12px] text-amber-400/70">
                  Add your API key below to activate
                </p>
              </div>
            ) : status === "active" ? (
              <div className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-emerald-500/[0.06] border border-emerald-500/10">
                <Check className="size-3.5 text-emerald-400/70 shrink-0" />
                <p className="text-[12px] text-emerald-400/70">
                  Connected and active
                </p>
              </div>
            ) : null}
          </div>

          {/* ── Feedback messages ── */}
          {(success || justConnected) && (
            <p className="text-[12px] text-emerald-400/80 animate-[fade-in_0.3s_ease]">
              {success || "Connected successfully"}
            </p>
          )}
          {(error || oauthError) && (
            <p className="text-[12px] text-red-400/80">
              {error || "Could not finish connection. Please try again."}
            </p>
          )}

          {/* ── OAuth reconnect (when already connected) ── */}
          {plugin.auth_type === "oauth2" && plugin.connected && (
            <div className="flex items-center justify-between py-3 border-t border-white/[0.06]">
              <div>
                <p className="text-[13px] text-foreground">Account</p>
                <p className="text-[11px] text-muted-foreground/60 mt-0.5">
                  Connected via {plugin.auth_provider || "OAuth"}
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleOAuthConnect}
                className="text-[12px] h-8 px-3 text-muted-foreground"
              >
                <ExternalLink className="size-3 mr-1.5" />
                Reconnect
              </Button>
            </div>
          )}

          {/* ── Configuration fields ── */}
          {editableFields.length > 0 &&
            plugin.installed &&
            (plugin.auth_type === "api_key" ||
              (plugin.auth_type === "oauth2" &&
                editableFields.length > 0)) && (
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 space-y-4">
                <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/60 font-medium">
                  Configuration
                </p>
                {editableFields.map((field) => (
                  <div key={field.key}>
                    <label className="text-[12px] text-muted-foreground mb-1.5 block">
                      {field.label}
                      {field.required && (
                        <span className="text-red-400/60 ml-0.5">*</span>
                      )}
                    </label>
                    <input
                      type={field.type === "password" ? "password" : "text"}
                      value={formValues[field.key] || ""}
                      onChange={(e) =>
                        setFormValues({
                          ...formValues,
                          [field.key]: e.target.value,
                        })
                      }
                      placeholder={field.description || field.label}
                      className="w-full h-9 px-3 text-[13px] bg-white/[0.03] border border-white/[0.06] rounded-lg focus:outline-none focus:border-white/[0.12] text-foreground placeholder:text-muted-foreground/30 transition-colors"
                    />
                    {field.description && (
                      <p className="text-[10px] text-muted-foreground/40 mt-1">
                        {field.description}
                      </p>
                    )}
                  </div>
                ))}
                <Button
                  variant="aether"
                  size="aether"
                  onClick={handleSave}
                  disabled={saving || !hasRequiredConfig()}
                  className="w-full"
                >
                  {saving ? "Saving..." : "Save Configuration"}
                </Button>
              </div>
            )}

          {/* ── Enable / Disable toggle ── */}
          {plugin.installed && (
            <div className="flex items-center justify-between py-3 border-t border-white/[0.06]">
              <div>
                <p className="text-[13px] text-foreground">Enabled</p>
                {!canEnable && !plugin.enabled && (
                  <p className="text-[11px] text-muted-foreground/60 mt-0.5">
                    {plugin.auth_type === "oauth2"
                      ? "Connect your account first"
                      : "Complete configuration first"}
                  </p>
                )}
              </div>
              <Switch
                checked={plugin.enabled}
                onCheckedChange={handleToggle}
                disabled={toggling || (!canEnable && !plugin.enabled)}
                size="sm"
              />
            </div>
          )}

          {/* ── Uninstall ── */}
          {plugin.installed && (
            <div className="pt-2 border-t border-white/[0.06]">
              <button
                onClick={handleUninstall}
                disabled={uninstalling}
                className="flex items-center gap-2 text-[12px] text-muted-foreground/50 hover:text-red-400/70 transition-colors disabled:opacity-50"
              >
                <Trash2 className="size-3" />
                {uninstalling ? "Removing..." : "Uninstall plugin"}
              </button>
            </div>
          )}
        </div>
      )}
    </ContentShell>
  );
}
