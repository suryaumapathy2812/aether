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

  // Show success message after OAuth redirect
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
        setError("Connection not found");
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
      setError(e instanceof Error ? e.message : "Could not load this connection right now");
    } finally {
      setLoading(false);
    }
  }

  function hasRequiredConfig(): boolean {
    return getRequiredFieldsForSetup().every((f) => formValues[f.key]?.trim());
  }

  function getMissingFields(): string[] {
    return getRequiredFieldsForSetup()
      .filter((f) => !formValues[f.key]?.trim())
      .map((f) => f.label);
  }

  function getRequiredFieldsForSetup() {
    if (!plugin?.config_fields) return [];
    return plugin.config_fields.filter(
      (f) =>
        f.required &&
        !(plugin.auth_type === "oauth2" && f.key === "account_email")
    );
  }

  function getEditableConfigFields() {
    if (!plugin?.config_fields) return [];
    return plugin.config_fields.filter((f) => f.key !== "account_email");
  }

  function getOAuthSetupFields() {
    if (!plugin || plugin.auth_type !== "oauth2") return [];
    // When the backend has env-backed credentials (e.g. GOOGLE_CLIENT_ID),
    // it filters client_id/client_secret out of config_fields.
    // In that case, no setup fields are needed — just show the Connect button.
    return getEditableConfigFields();
  }

  function hasOAuthSetupValues(): boolean {
    const fields = getOAuthSetupFields();
    if (fields.length === 0) return true;
    return fields.every((f) => formValues[f.key]?.trim());
  }

  async function handleOAuthConnect() {
    if (!plugin) return;
    setError("");
    try {
      if (!plugin.installed) {
        await installPlugin(pluginName);
      }
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
        setPlugin((p) => p ? { ...p, installed: true, enabled: true, connected: true } : p);
        setSuccess("Saved and turned on");
      } else {
        setSuccess("Saved");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not save right now");
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
      setSuccess("Turned on");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not turn this on right now");
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
      setSuccess("Turned off");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not turn this off right now");
    } finally {
      setToggling(false);
    }
  }

  if (isPending || !session) return null;

  const missingFields = getMissingFields();
  const editableConfigFields = getEditableConfigFields();
  const oauthSetupFields = getOAuthSetupFields();
  const canEnable =
    plugin?.auth_type === "none" ||
    (plugin?.auth_type === "oauth2" && plugin.connected) ||
    (plugin?.auth_type === "api_key" && hasRequiredConfig());
  const oauthSetupError =
    oauthError.includes("needs+setup+details") ||
    oauthError.includes("needs setup details");

  const blockedReason = !canEnable && !plugin?.enabled
    ? missingFields.length > 0
        ? `Please add: ${missingFields.join(", ")}`
        : plugin?.auth_type === "oauth2"
          ? "Connect first"
        : "Finish setup first"
    : "";

  return (
    <PageShell
      title={plugin?.display_name || "Connection"}
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
            try again
          </Button>
        </div>
      ) : !plugin ? (
        <p className="text-muted-foreground text-xs">Connection not found</p>
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
              Connected
            </div>
          )}
          {error && (
            <div className="py-3 text-[12px] text-red-400 tracking-wider">
              {error}
            </div>
          )}
          {!!oauthError && (
            <div className="py-3 text-[12px] text-red-400 tracking-wider">
              {oauthSetupError
                ? "Add your app details once, then connect."
                : "Could not finish connection. Please try again."}
            </div>
          )}

          {plugin.auth_type === "oauth2" && (
            <div className="py-4">
              <div className="space-y-4">
                <p className="text-xs text-muted-foreground leading-relaxed max-w-[70ch]">
                  Connect your account once to start using this app with Aether.
                </p>
                {oauthSetupFields.length > 0 && (
                  <>
                    <div className="space-y-4">
                      {oauthSetupFields.map((field) => (
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
                      disabled={saving || !hasOAuthSetupValues()}
                    >
                      {saving ? "..." : "Save details"}
                    </Button>
                  </>
                )}
                <Button
                  variant="aether"
                  size="aether"
                  onClick={handleOAuthConnect}
                  disabled={!hasRequiredConfig()}
                >
                  {plugin.connected ? "Reconnect" : "Connect"}
                </Button>
                {plugin.connected && (
                  <span className="block text-[12px] text-green-400 tracking-wider">
                    Connected
                  </span>
                )}
              </div>
              <Separator className="mt-4" />
            </div>
          )}

          {plugin.auth_type === "api_key" && (
            <div className="space-y-4">
              <h2 className="text-xs tracking-widest text-muted-foreground uppercase font-normal">
                Set up
              </h2>
              <div className="space-y-4">
                {editableConfigFields.map((field) => (
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
                  {saving ? "..." : plugin.installed ? "Save" : "Save and turn on"}
                </Button>
              {plugin.installed && <Separator className="mt-4" />}
            </div>
          )}

          {plugin.installed && (
            <div className="py-2">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-2xl bg-white/5 border border-border/60 px-4 py-3">
                <div className="flex flex-col">
                  <span className="text-[11px] tracking-[0.12em] uppercase text-muted-foreground font-medium">
                    App status
                  </span>
                  {!canEnable && !plugin.enabled && (
                    <span className="text-[10px] text-red-400/80 mt-1">
                      {blockedReason}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 self-start sm:self-auto">
                  <span
                    className={`text-[12px] tracking-wider ${
                      plugin.enabled ? "text-green-400" : "text-muted-foreground"
                    }`}
                  >
                    {toggling ? "..." : plugin.enabled ? "On" : "Off"}
                  </span>
                  {plugin.enabled ? (
                    <Button
                      variant="aether-link"
                      size="aether-link"
                      onClick={handleDisable}
                      disabled={toggling}
                      className="text-red-400/80 hover:text-red-400"
                    >
                      turn off
                    </Button>
                  ) : (
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={handleEnable}
                      disabled={toggling || !canEnable}
                    >
                      {toggling ? "..." : "turn on"}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}

          {!plugin.installed && plugin.auth_type !== "oauth2" && plugin.auth_type !== "api_key" && plugin.auth_type !== "none" && (
            <div className="text-[12px] text-muted-foreground">
              This app can be set up from the{" "}
              <Link href="/plugins" className="text-secondary-foreground hover:underline">
                connections page
              </Link>
              .
            </div>
          )}
        </div>
      )}
    </PageShell>
  );
}
