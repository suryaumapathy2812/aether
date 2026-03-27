import { createFileRoute, useNavigate, useParams } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import ContentShell from "#/components/ContentShell";
import { Button } from "#/components/ui/button";
import { useSession } from "#/lib/auth-client";
import {
  listIntegrations,
  installIntegration,
  getIntegrationConfig,
  saveIntegrationConfig,
  disconnectIntegration,
  getOAuthStartUrl,
} from "#/lib/api";
import type { IntegrationInfo } from "#/lib/api";
import { IconAlertCircle, IconPlugConnected, IconPlugConnectedX } from "@tabler/icons-react";
import { z } from "zod";

const integrationDetailSearchSchema = z.object({
  connected: z.string().optional().catch(undefined),
  error: z.string().optional().catch(undefined),
});

export const Route = createFileRoute("/integrations/$name/")({
  validateSearch: integrationDetailSearchSchema,
  component: IntegrationDetailPage,
});

function IntegrationDetailPage() {
  const navigate = useNavigate();
  const { name: integrationName } = useParams({ from: "/integrations/$name/" });
  const { connected, error: oauthError } = Route.useSearch();
  const { data: session, isPending } = useSession();

  const [integration, setIntegration] = useState<IntegrationInfo | null>(null);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [saving, setSaving] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);
  const [installing, setInstalling] = useState(false);

  const justConnected = connected === "true";

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      navigate({ to: "/" });
      return;
    }
    void loadIntegrationData();
  }, [session, isPending, navigate, integrationName]);

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
        // no config yet
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

  async function handleDisconnect() {
    try {
      setDisconnecting(true);
      setError("");
      await disconnectIntegration(integrationName);
      setIntegration((prev) =>
        prev ? { ...prev, enabled: false, connected: false, needs_reconnect: false } : prev,
      );
      setConfig({});
      setFormValues({});
      setSuccess("Disconnected.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Disconnect failed");
    } finally {
      setDisconnecting(false);
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
        <p className="text-sm text-neutral-600 dark:text-neutral-300">{integration.description}</p>

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

        {integration.installed && !integration.connected && fields.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Configuration</h3>
            {fields.map((field) => (
              <div key={field.key} className="space-y-1">
                <label className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                  {field.label}
                  {field.required && <span className="ml-1 text-red-500">*</span>}
                </label>
                {field.description && (
                  <p className="text-xs text-neutral-400 dark:text-neutral-500">{field.description}</p>
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

        {integration.installed && !integration.connected && (
          <div className="space-y-2">
            {isOAuth && (hasConfig || hasOAuthEnv) ? (
              <>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  Authorize access to your account.
                </p>
                <Button
                  size="sm"
                  onClick={() => {
                    window.location.href = getOAuthStartUrl(integrationName);
                  }}
                >
                  <IconPlugConnected className="mr-1.5 size-4" />
                  Connect
                </Button>
              </>
            ) : !isOAuth && !missingRequired ? (
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Save your configuration above to connect.
              </p>
            ) : null}
          </div>
        )}

        {integration.installed && integration.connected && (
          <div className="rounded-md border p-3 dark:border-neutral-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm">
                <span className="font-medium text-green-600 dark:text-green-400">Connected</span>
                {config.account_email && (
                  <span className="text-neutral-500 dark:text-neutral-400">({config.account_email})</span>
                )}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleDisconnect}
                disabled={disconnecting}
                className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-950"
              >
                <IconPlugConnectedX className="mr-1.5 size-4" />
                {disconnecting ? "Disconnecting..." : "Disconnect"}
              </Button>
            </div>
            {integration.needs_reconnect && (
              <div className="mt-2 flex items-center gap-2">
                <p className="text-xs text-amber-600 dark:text-amber-400">Connection expired.</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    window.location.href = getOAuthStartUrl(integrationName);
                  }}
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
