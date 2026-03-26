import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "@tanstack/react-router";
import ContentShell from "#/components/ContentShell";
import { Button } from "#/components/ui/button";
import { Switch } from "#/components/ui/switch";
import { useSession } from "#/lib/auth-client";
import {
  listIntegrations,
  installIntegration,
  getIntegrationConfig,
  saveIntegrationConfig,
  enableIntegration,
  disableIntegration,
  getOAuthStartUrl,
} from "#/lib/api";
import type { IntegrationInfo } from "#/lib/api";
import { IconExternalLink, IconAlertCircle } from "@tabler/icons-react";
import { z } from "zod";

const integrationDetailSearchSchema = z.object({
  connected: z.string().optional().catch(undefined),
  error: z.string().optional().catch(undefined),
});

export const Route = createFileRoute("/integrations/$name")({
  validateSearch: integrationDetailSearchSchema,
  component: IntegrationDetailPage,
});

function IntegrationDetailPage() {
  const navigate = useNavigate();
  const { name: integrationName } = useParams({ from: "/integrations/$name" });
  const { connected, error: oauthError } = Route.useSearch();
  const { data: session, isPending } = useSession();

  const [integration, setIntegration] = useState<IntegrationInfo | null>(null);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [saving, setSaving] = useState(false);
  const [toggling, setToggling] = useState(false);
  const [installing, setInstalling] = useState(false);

  const justConnected = connected === "true";

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      navigate({ to: "/" });
      return;
    }
    loadIntegrationData();
  }, [session, isPending, navigate]);

  async function loadIntegrationData() {
    try {
      setLoading(true);
      const integrations = await listIntegrations();
      const found = integrations.find((p) => p.name === integrationName);
      if (!found) {
        setError("Integration not found");
        return;
      }
      setIntegration(found);
      try {
        const cfg = await getIntegrationConfig(integrationName);
        setConfig(cfg);
        setFormValues(cfg);
      } catch {
        /* no config yet */
      }
      if (!found.installed && found.config_fields?.every((f) => !f.required)) {
        await installIntegration(integrationName);
        const refreshed = await listIntegrations();
        const updated = refreshed.find((p) => p.name === integrationName);
        if (updated) setIntegration(updated);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load integration");
    } finally {
      setLoading(false);
    }
  }

  function visibleFields() {
    if (!integration?.config_fields) return [];
    return integration.config_fields.filter((f) => f.key !== "account_email");
  }

  async function handleInstall() {
    try {
      setInstalling(true);
      setError("");
      await installIntegration(integrationName);
      await loadIntegrationData();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Install failed");
    } finally {
      setInstalling(false);
    }
  }

  async function handleSave() {
    try {
      setSaving(true);
      setError("");
      setSuccess("");
      const res = await saveIntegrationConfig(integrationName, formValues);
      setConfig({ ...formValues });
      setSuccess("Config saved.");
      if (res.auto_enabled) {
        setIntegration((prev) => (prev ? { ...prev, enabled: true, connected: true } : prev));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function handleToggle(checked: boolean) {
    try {
      setToggling(true);
      setError("");
      if (checked) {
        await enableIntegration(integrationName);
      } else {
        await disableIntegration(integrationName);
      }
      setIntegration((prev) => (prev ? { ...prev, enabled: checked, connected: checked } : prev));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Toggle failed");
    } finally {
      setToggling(false);
    }
  }

  if (loading) {
    return (
      <ContentShell title="Loading..." back="/integrations">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">Loading integration...</p>
      </ContentShell>
    );
  }

  if (error && !integration) {
    return (
      <ContentShell title="Error" back="/integrations">
        <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
          <IconAlertCircle className="size-4" />
          <span>{error}</span>
        </div>
      </ContentShell>
    );
  }

  if (!integration) return null;

  const fields = visibleFields();
  const isOAuth = integration.auth_type === "oauth2";
  const hasOAuthEnv = integration.oauth_env_configured;
  const hasConfig = Object.keys(config).length > 0;
  const missingRequired = fields
    .filter((f) => f.required)
    .some((f) => !formValues[f.key]?.trim());

  return (
    <ContentShell title={integration.display_name} back="/integrations">
      <div className="space-y-6">
        <p className="text-sm text-neutral-600 dark:text-neutral-300">
          {integration.description}
        </p>

        {(error || oauthError) && (
          <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
            <IconAlertCircle className="size-4" />
            <span>{oauthError || error}</span>
          </div>
        )}

        {justConnected && (
          <div className="rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-700 dark:border-green-800 dark:bg-green-950 dark:text-green-300">
            Connected successfully.
          </div>
        )}

        {success && (
          <div className="rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-700 dark:border-green-800 dark:bg-green-950 dark:text-green-300">
            {success}
          </div>
        )}

        {!integration.installed && (
          <Button onClick={handleInstall} disabled={installing} size="sm">
            {installing ? "Installing..." : "Install"}
          </Button>
        )}

        {integration.installed && (
          <div className="flex items-center gap-3">
            <Switch
              checked={integration.enabled}
              onCheckedChange={handleToggle}
              disabled={toggling || missingRequired}
            />
            <span className="text-sm text-neutral-600 dark:text-neutral-300">
              {integration.enabled ? "Enabled" : "Disabled"}
            </span>
          </div>
        )}

        {integration.installed && fields.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Configuration</h3>
            {fields.map((field) => (
              <div key={field.key} className="space-y-1">
                <label className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                  {field.label}
                  {field.required && <span className="ml-1 text-red-500">*</span>}
                </label>
                {field.description && (
                  <p className="text-xs text-neutral-400 dark:text-neutral-500">
                    {field.description}
                  </p>
                )}
                <input
                  type={field.type === "password" ? "password" : "text"}
                  value={formValues[field.key] || ""}
                  onChange={(e) =>
                    setFormValues((prev) => ({ ...prev, [field.key]: e.target.value }))
                  }
                  className="w-full rounded-md border bg-transparent px-3 py-1.5 text-sm dark:border-neutral-700"
                  placeholder={field.type === "password" ? "••••••••" : ""}
                />
              </div>
            ))}
            <Button onClick={handleSave} disabled={saving} size="sm">
              {saving ? "Saving..." : "Save Config"}
            </Button>
          </div>
        )}

        {integration.installed && (hasConfig || hasOAuthEnv) && isOAuth && !integration.connected && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Connect</h3>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              Authorize access to your account.
            </p>
            <Button
              size="sm"
              onClick={() => navigate({ to: getOAuthStartUrl(integrationName) })}
            >
              <IconExternalLink className="mr-1 size-4" />
              Connect Account
            </Button>
          </div>
        )}

        {integration.installed && integration.connected && (
          <div className="rounded-md border p-3 text-sm dark:border-neutral-700">
            <span className="font-medium text-green-600 dark:text-green-400">Connected</span>
            {config.account_email && (
              <span className="ml-2 text-neutral-500 dark:text-neutral-400">
                ({config.account_email})
              </span>
            )}
            {integration.needs_reconnect && (
              <div className="mt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate({ to: getOAuthStartUrl(integrationName) })}
                >
                  Reconnect
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </ContentShell>
  );
}
