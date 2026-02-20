"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams, useSearchParams } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
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
        <p className="text-muted-foreground text-xs tracking-wider">
          loading...
        </p>
      ) : error ? (
        <div>
          <p className="text-muted-foreground text-xs mb-4">{error}</p>
          <Button
            variant="aether"
            size="aether"
            onClick={loadPluginData}
          >
            retry
          </Button>
        </div>
      ) : !plugin ? (
        <p className="text-muted-foreground text-xs">
          plugin not found
        </p>
      ) : (
        <div className="space-y-6">
          {/* Plugin description */}
          <div>
            <p className="text-sm text-secondary-foreground leading-relaxed font-normal max-w-[78ch]">
              {plugin.description}
            </p>
          </div>

          {/* Just connected success message */}
          {justConnected && (
            <div className="py-3 text-[12px] text-secondary-foreground tracking-wider animate-[fade-in_0.3s_ease]">
              connected successfully
            </div>
          )}

          {/* OAuth2 connect / connected status */}
          {plugin.auth_type === "oauth2" && plugin.installed && (
            <div className="py-4">
              {plugin.connected ? (
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
                    className="text-[11px] tracking-wider"
                  >
                    reconnect
                  </Button>
                </div>
              ) : (
                <Button
                  variant="aether"
                  size="aether"
                  onClick={handleConnect}
                >
                  connect with {plugin.auth_provider}
                </Button>
              )}
              <Separator className="mt-4" />
            </div>
          )}

          {/* Enable/Disable toggle */}
          {plugin.installed && plugin.connected && (
            <div className="py-2">
              <div className="flex items-center justify-between rounded-2xl bg-white/5 border border-border/60 px-4 py-3">
                <Label className="text-[11px] tracking-[0.12em] uppercase text-muted-foreground font-medium">
                  Status
                </Label>
                <div className="flex items-center gap-3">
                  <span className="text-[12px] text-secondary-foreground tracking-wider">
                    {toggling ? "..." : plugin.enabled ? "enabled" : "disabled"}
                  </span>
                  <Switch
                    checked={plugin.enabled}
                    onCheckedChange={handleToggleEnable}
                    disabled={toggling}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Config form (only for non-OAuth fields, if any) */}
          {plugin.installed &&
            plugin.config_fields &&
            plugin.config_fields.length > 0 && (
              <div>
                <h2 className="text-xs tracking-widest text-muted-foreground uppercase mb-4 font-normal">
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

                <Button
                  variant="aether"
                  size="aether"
                  onClick={handleSave}
                  disabled={saving}
                >
                  {saving ? "..." : "save"}
                </Button>
              </div>
            )}
        </div>
      )}
    </PageShell>
  );
}
