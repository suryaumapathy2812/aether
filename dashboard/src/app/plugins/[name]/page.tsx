"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useParams, useSearchParams } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  listPlugins,
  installPlugin,
  getPluginConfig,
  savePluginConfig,
  enablePlugin,
  disablePlugin,
  getOAuthStartUrl,
  PluginInfo,
} from "@/lib/api";

/**
 * Plugin detail / setup page.
 *
 * UX intent — no "installed but not configured" limbo:
 *
 *   auth_type = "oauth2"
 *     Not installed → show "Connect with <provider>" button.
 *     Clicking it silently installs then redirects to OAuth.
 *     After OAuth callback → plugin is installed + connected + enabled.
 *
 *   auth_type = "api_key"
 *     Not installed → show config form immediately.
 *     Saving silently installs then saves config in one shot.
 *     Auto-enable fires server-side when all required fields are present.
 *
 *   auth_type = "none"
 *     Not installed → silently install + enable on page load.
 *
 * The user only ever sees "set up" or "active" — never "installed, disabled".
 */
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
  const [connecting, setConnecting] = useState(false);

  // Show success message after OAuth redirect
  const justConnected = searchParams.get("connected") === "true";

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

      // Auto-install + enable no-auth plugins on first visit
      if (!found.installed && found.auth_type === "none") {
        await installPlugin(pluginName);
        // Reload to get updated state
        const refreshed = await listPlugins();
        const updated = refreshed.find((p) => p.name === pluginName);
        if (updated) setPlugin(updated);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load plugin");
    } finally {
      setLoading(false);
    }
  }

  function hasRequiredConfig(): boolean {
    if (!plugin?.config_fields) return true;
    return plugin.config_fields
      .filter((f) => f.required)
      .every((f) => formValues[f.key]?.trim());
  }

  function getMissingFields(): string[] {
    if (!plugin?.config_fields) return [];
    return plugin.config_fields
      .filter((f) => f.required && !formValues[f.key]?.trim())
      .map((f) => f.label);
  }

  /** OAuth2: install silently then redirect to provider */
  async function handleConnect() {
    if (!plugin) return;
    setConnecting(true);
    setError("");
    try {
      if (!plugin.installed) {
        await installPlugin(pluginName);
      }
      window.location.href = getOAuthStartUrl(pluginName);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to connect");
      setConnecting(false);
    }
  }

  /** api_key: install (if needed) + save config in one shot */
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
        setPlugin((p) => p ? { ...p, installed: true, enabled: true, connected: true } : p);
        setSuccess("Configuration saved — plugin enabled");
      } else {
        setSuccess("Configuration saved");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save config");
    } finally {
      setSaving(false);
    }
  }

  async function handleEnable() {
    if (!plugin) return;
    setToggling(true);
    setError("");
    setSuccess("");
    try {
      await enablePlugin(pluginName);
      setPlugin({ ...plugin, enabled: true });
      setSuccess("Plugin enabled");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to enable plugin");
    } finally {
      setToggling(false);
    }
  }

  async function handleDisable() {
    if (!plugin) return;
    setToggling(true);
    setError("");
    setSuccess("");
    try {
      await disablePlugin(pluginName);
      setPlugin({ ...plugin, enabled: false });
      setSuccess("Plugin disabled");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to disable plugin");
    } finally {
      setToggling(false);
    }
  }

  if (isPending || !session) return null;

  const missingFields = getMissingFields();
  const canEnable =
    plugin?.auth_type === "none" ||
    (plugin?.auth_type === "oauth2" && plugin.connected) ||
    (plugin?.auth_type === "api_key" && hasRequiredConfig());

  return (
    <PageShell
      title={plugin?.display_name || "Plugin"}
      back="/plugins"
      centered={loading || !!error}
    >
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider">
          loading...
        </p>
      ) : error && !plugin ? (
        <div>
          <p className="text-muted-foreground text-xs mb-4">{error}</p>
          <Button variant="aether" size="aether" onClick={loadPluginData}>
            retry
          </Button>
        </div>
      ) : !plugin ? (
        <p className="text-muted-foreground text-xs">plugin not found</p>
      ) : (
        <div className="space-y-6">
          {/* Description */}
          <div>
            <p className="text-sm text-secondary-foreground leading-relaxed font-normal max-w-[78ch]">
              {plugin.description}
            </p>
          </div>

          {/* Feedback messages */}
          {success && (
            <div className="py-3 text-[12px] text-green-400 tracking-wider animate-[fade-in_0.3s_ease]">
              {success}
            </div>
          )}
          {justConnected && !success && (
            <div className="py-3 text-[12px] text-green-400 tracking-wider animate-[fade-in_0.3s_ease]">
              connected successfully
            </div>
          )}
          {error && (
            <div className="py-3 text-[12px] text-red-400 tracking-wider">
              {error}
            </div>
          )}

          {/* ── OAuth2 ── */}
          {plugin.auth_type === "oauth2" && (
            <div className="py-4">
              {plugin.connected ? (
                <div className="space-y-3">
                  {/* Connected row */}
                  <div className="flex items-center justify-between">
                    <span className="text-[14px] text-secondary-foreground font-light">
                      {config.account_email
                        ? `Connected as ${config.account_email}`
                        : "Connected"}
                    </span>
                    <Button
                      variant="aether-link"
                      size="aether-link"
                      onClick={handleConnect}
                      disabled={connecting}
                      className="text-[11px] tracking-wider"
                    >
                      {connecting ? "..." : "reconnect"}
                    </Button>
                  </div>

                  {/* Needs reconnect warning */}
                  {plugin.needs_reconnect && (
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-xl bg-amber-500/10 border border-amber-500/20 px-4 py-3">
                      <span className="text-[12px] text-amber-400 tracking-wide">
                        New permissions required — reconnect to continue using this plugin
                      </span>
                      <Button
                        variant="aether"
                        size="aether"
                        onClick={handleConnect}
                        disabled={connecting}
                        className="shrink-0 self-start sm:self-auto"
                      >
                        {connecting ? "..." : "reconnect"}
                      </Button>
                    </div>
                  )}
                </div>
              ) : (
                /* Not yet connected — primary CTA */
                <Button
                  variant="aether"
                  size="aether"
                  onClick={handleConnect}
                  disabled={connecting}
                >
                  {connecting ? "connecting..." : `Connect with ${plugin.auth_provider}`}
                </Button>
              )}
              <Separator className="mt-4" />
            </div>
          )}

          {/* ── API key config ── */}
          {plugin.auth_type === "api_key" && (
            <div className="space-y-4">
              <h2 className="text-xs tracking-widest text-muted-foreground uppercase font-normal">
                Configuration
              </h2>
              <div className="space-y-4">
                {plugin.config_fields?.map((field) => (
                  <MinimalInput
                    key={field.key}
                    label={field.label}
                    type={field.type === "password" ? "password" : "text"}
                    value={formValues[field.key] || ""}
                    onChange={(v) =>
                      setFormValues({ ...formValues, [field.key]: v })
                    }
                    placeholder={field.description || field.label}
                  />
                ))}
              </div>
              <Button
                variant="aether"
                size="aether"
                onClick={handleSave}
                disabled={saving || !hasRequiredConfig()}
              >
                {saving ? "..." : plugin.installed ? "save configuration" : "save & activate"}
              </Button>
              {plugin.installed && <Separator className="mt-4" />}
            </div>
          )}

          {/* ── Enable / Disable (only shown once installed) ── */}
          {plugin.installed && plugin.auth_type !== "none" && (
            <div className="py-2">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-2xl bg-white/5 border border-border/60 px-4 py-3">
                <div className="flex flex-col">
                  <span className="text-[11px] tracking-[0.12em] uppercase text-muted-foreground font-medium">
                    Status
                  </span>
                  {!canEnable && !plugin.enabled && (
                    <span className="text-[10px] text-red-400/80 mt-1">
                      {missingFields.length > 0
                        ? `Missing: ${missingFields.join(", ")}`
                        : "Configuration required"}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 self-start sm:self-auto">
                  <span
                    className={`text-[12px] tracking-wider ${
                      plugin.enabled ? "text-green-400" : "text-muted-foreground"
                    }`}
                  >
                    {toggling ? "..." : plugin.enabled ? "enabled" : "disabled"}
                  </span>
                  {plugin.enabled ? (
                    <Button
                      variant="aether-link"
                      size="aether-link"
                      onClick={handleDisable}
                      disabled={toggling}
                      className="text-red-400/80 hover:text-red-400"
                    >
                      disable
                    </Button>
                  ) : (
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={handleEnable}
                      disabled={toggling || !canEnable}
                    >
                      {toggling ? "..." : "enable"}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Not available in catalogue */}
          {!plugin.installed && plugin.auth_type !== "oauth2" && plugin.auth_type !== "api_key" && plugin.auth_type !== "none" && (
            <div className="text-[12px] text-muted-foreground">
              Install this plugin from the{" "}
              <Link href="/plugins" className="text-secondary-foreground hover:underline">
                plugins page
              </Link>
              .
            </div>
          )}
        </div>
      )}
    </PageShell>
  );
}
