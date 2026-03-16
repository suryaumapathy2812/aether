"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams, useSearchParams } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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
import { ExternalLink } from "lucide-react";

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

  const justConnected = searchParams.get("connected") === "true";
  const oauthError = searchParams.get("error") || "";

  useEffect(() => {
    if (isPending) return;
    if (!session) { router.push("/"); return; }
    loadPluginData();
  }, [session, isPending, router]);

  async function loadPluginData() {
    try {
      setLoading(true);
      setError("");
      const plugins = await listPlugins();
      const found = plugins.find((p) => p.name === pluginName);
      if (!found) { setError("Plugin not found"); return; }
      setPlugin(found);

      if (found.installed) {
        const cfg = await getPluginConfig(pluginName);
        setConfig(cfg);
        setFormValues(cfg);
      }

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
    setError(""); setSuccess("");
    try {
      if (!plugin.installed) {
        await installPlugin(pluginName);
        setPlugin({ ...plugin, installed: true });
      }
      const result = await savePluginConfig(pluginName, formValues);
      setConfig(formValues);
      if (result.auto_enabled) {
        setPlugin((p) => p ? { ...p, installed: true, enabled: true, connected: true } : p);
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
    setError(""); setSuccess("");
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

  if (isPending || !session) return null;

  const editableFields = getEditableFields();
  const canEnable =
    plugin?.auth_type === "none" ||
    (plugin?.auth_type === "oauth2" && plugin.connected) ||
    (plugin?.auth_type === "api_key" && hasRequiredConfig());

  return (
    <ContentShell title={plugin?.display_name || "Plugin"} back="/plugins">
      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : !plugin ? (
        <p className="text-muted-foreground text-xs">{error || "Plugin not found"}</p>
      ) : (
        <div className="space-y-8">
          {/* Description + status */}
          <div>
            <p className="text-[13px] text-muted-foreground leading-relaxed">
              {plugin.description}
            </p>
          </div>

          {/* Feedback */}
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

          {/* OAuth connect */}
          {plugin.auth_type === "oauth2" && (
            <div className="py-4 border-b border-border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[13px] text-foreground">Account</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    {plugin.connected ? "Connected" : "Connect your account to get started"}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleOAuthConnect}
                  className="text-[12px] h-8 px-3"
                >
                  <ExternalLink className="size-3 mr-1.5" />
                  {plugin.connected ? "Reconnect" : "Connect"}
                </Button>
              </div>
            </div>
          )}

          {/* API key / config fields */}
          {editableFields.length > 0 && (plugin.auth_type === "api_key" || (plugin.auth_type === "oauth2" && editableFields.length > 0)) && (
            <div className="space-y-4">
              <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/60 font-medium">
                Configuration
              </p>
              {editableFields.map((field) => (
                <div key={field.key}>
                  <label className="text-[12px] text-muted-foreground mb-1.5 block">
                    {field.label}
                    {field.required && <span className="text-red-400/60 ml-0.5">*</span>}
                  </label>
                  <input
                    type={field.type === "password" ? "password" : "text"}
                    value={formValues[field.key] || ""}
                    onChange={(e) => setFormValues({ ...formValues, [field.key]: e.target.value })}
                    placeholder={field.description || field.label}
                    className="w-full h-8 px-3 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring text-foreground placeholder:text-muted-foreground/40"
                  />
                </div>
              ))}
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSave}
                disabled={saving || !hasRequiredConfig()}
                className="text-[12px] h-8 px-3"
              >
                {saving ? "..." : "Save"}
              </Button>
            </div>
          )}

          {/* Enable/disable toggle */}
          {plugin.installed && (
            <div className="flex items-center justify-between py-4 border-t border-border">
              <div>
                <p className="text-[13px] text-foreground">Enabled</p>
                {!canEnable && !plugin.enabled && (
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    {plugin.auth_type === "oauth2" ? "Connect your account first" : "Complete configuration first"}
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
        </div>
      )}
    </ContentShell>
  );
}
