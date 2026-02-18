"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams, useSearchParams } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { useSession } from "@/lib/auth-client";
import {
  listPlugins,
  getPluginConfig,
  savePluginConfig,
  enablePlugin,
  disablePlugin,
  getOAuthStartUrl,
  PluginInfo,
} from "@/lib/api";

/**
 * Plugin detail/config page.
 * Shows plugin info, OAuth connect button, config form, and enable/disable toggle.
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
  const [saving, setSaving] = useState(false);
  const [toggling, setToggling] = useState(false);

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
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load plugin");
    } finally {
      setLoading(false);
    }
  }

  async function handleSave() {
    if (!plugin) return;
    setSaving(true);
    try {
      await savePluginConfig(pluginName, formValues);
      setConfig(formValues);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to save config";
      setError(msg);
    } finally {
      setSaving(false);
    }
  }

  async function handleToggleEnable() {
    if (!plugin) return;
    setToggling(true);
    try {
      if (plugin.enabled) {
        await disablePlugin(pluginName);
        setPlugin({ ...plugin, enabled: false });
      } else {
        await enablePlugin(pluginName);
        setPlugin({ ...plugin, enabled: true });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to toggle plugin";
      setError(msg);
    } finally {
      setToggling(false);
    }
  }

  function handleConnect() {
    window.location.href = getOAuthStartUrl(pluginName);
  }

  if (isPending || !session) return null;

  return (
    <PageShell
      title={plugin?.display_name || "Plugin"}
      back="/plugins"
      centered={loading || !!error}
    >
      {loading ? (
        <p className="text-[var(--color-text-muted)] text-xs tracking-wider">
          loading...
        </p>
      ) : error ? (
        <div>
          <p className="text-[var(--color-text-muted)] text-xs mb-4">{error}</p>
          <button onClick={loadPluginData} className="btn text-xs">
            retry
          </button>
        </div>
      ) : !plugin ? (
        <p className="text-[var(--color-text-muted)] text-xs">
          plugin not found
        </p>
      ) : (
        <div className="space-y-6">
          {/* Plugin description */}
          <div>
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed font-light">
              {plugin.description}
            </p>
          </div>

          {/* Just connected success message */}
          {justConnected && (
            <div className="py-3 text-[12px] text-[var(--color-text-secondary)] tracking-wider animate-[fade-in_0.3s_ease]">
              connected successfully
            </div>
          )}

          {/* OAuth2 connect / connected status */}
          {plugin.auth_type === "oauth2" && plugin.installed && (
            <div className="py-4 border-b border-[var(--color-border)]">
              {plugin.connected ? (
                <div className="flex items-center justify-between">
                  <span className="text-[14px] text-[var(--color-text-secondary)] font-light">
                    {config.account_email
                      ? `Connected as ${config.account_email}`
                      : "Connected"}
                  </span>
                  <button
                    onClick={handleConnect}
                    className="text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors duration-300 tracking-wider"
                  >
                    reconnect
                  </button>
                </div>
              ) : (
                <button onClick={handleConnect} className="btn text-xs">
                  connect with {plugin.auth_provider}
                </button>
              )}
            </div>
          )}

          {/* Enable/Disable toggle */}
          {plugin.installed && plugin.connected && (
            <div className="py-4 border-b border-[var(--color-border)]">
              <button
                onClick={handleToggleEnable}
                disabled={toggling}
                className="w-full flex items-center justify-between group"
              >
                <span className="text-[14px] text-[var(--color-text-secondary)] group-hover:text-[var(--color-text)] transition-colors duration-300 font-light">
                  Status
                </span>
                <span className="text-[12px] text-[var(--color-text-muted)] tracking-wider disabled:opacity-30">
                  {toggling ? "..." : plugin.enabled ? "enabled" : "disabled"}
                </span>
              </button>
            </div>
          )}

          {/* Config form (only for non-OAuth fields, if any) */}
          {plugin.installed &&
            plugin.config_fields &&
            plugin.config_fields.length > 0 && (
              <div>
                <h2 className="text-xs tracking-widest text-[var(--color-text-muted)] uppercase mb-4 font-normal">
                  Configuration
                </h2>
                <div>
                  {plugin.config_fields.map((field) => (
                    <MinimalInput
                      key={field.key}
                      label={field.label}
                      type={field.type === "password" ? "password" : "text"}
                      value={formValues[field.key] || ""}
                      onChange={(v) =>
                        setFormValues({ ...formValues, [field.key]: v })
                      }
                      placeholder={field.label}
                    />
                  ))}
                </div>

                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="btn text-xs disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  {saving ? "..." : "save"}
                </button>
              </div>
            )}
        </div>
      )}
    </PageShell>
  );
}
