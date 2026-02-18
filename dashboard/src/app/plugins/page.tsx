"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  listPlugins,
  installPlugin,
  enablePlugin,
  disablePlugin,
  getOAuthStartUrl,
  PluginInfo,
} from "@/lib/api";

/**
 * Plugins — manage available plugins.
 * Each plugin shown as a row; tap to edit config.
 * Install/connect/enable/disable actions inline.
 */
export default function PluginsPage() {
  return (
    <Suspense>
      <PluginsContent />
    </Suspense>
  );
}

function PluginsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);

  // Show OAuth error from redirect
  const oauthError = searchParams.get("error");

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    loadPlugins();
  }, [session, isPending, router]);

  async function loadPlugins() {
    try {
      setLoading(true);
      setError("");
      const data = await listPlugins();
      setPlugins(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load plugins");
    } finally {
      setLoading(false);
    }
  }

  async function handleInstall(name: string) {
    setActionInProgress(name);
    try {
      await installPlugin(name);
      await loadPlugins();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to install plugin");
    } finally {
      setActionInProgress(null);
    }
  }

  function handleConnect(name: string) {
    // Navigate to OAuth start — browser redirect (not fetch)
    window.location.href = getOAuthStartUrl(name);
  }

  async function handleEnable(name: string) {
    setActionInProgress(name);
    try {
      await enablePlugin(name);
      await loadPlugins();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to enable plugin");
    } finally {
      setActionInProgress(null);
    }
  }

  async function handleDisable(name: string) {
    setActionInProgress(name);
    try {
      await disablePlugin(name);
      await loadPlugins();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to disable plugin");
    } finally {
      setActionInProgress(null);
    }
  }

  if (isPending || !session) return null;

  return (
    <PageShell title="Plugins" back="/home" centered={loading || !!error}>
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider">
          loading...
        </p>
      ) : error || oauthError ? (
        <div>
          <p className="text-muted-foreground text-xs mb-4">
            {error || `OAuth error: ${oauthError}`}
          </p>
          <Button
            variant="aether"
            size="aether"
            onClick={loadPlugins}
          >
            retry
          </Button>
        </div>
      ) : plugins.length === 0 ? (
        <p className="text-muted-foreground text-xs">
          no plugins available
        </p>
      ) : (
        <div>
          {plugins.map((plugin, index) => (
            <div key={plugin.name}>
              <div className="py-4">
                <Link
                  href={`/plugins/${plugin.name}`}
                  className="block mb-3 group"
                >
                  <h3 className="text-[14px] text-foreground group-hover:text-secondary-foreground transition-colors duration-300 font-light mb-0.5">
                    {plugin.display_name}
                  </h3>
                  <p className="text-[12px] text-muted-foreground leading-relaxed font-light">
                    {plugin.description}
                  </p>
                </Link>

                <div className="flex items-center gap-3 text-xs">
                  {/* Not installed → install */}
                  {!plugin.installed ? (
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={() => handleInstall(plugin.name)}
                      disabled={actionInProgress === plugin.name}
                    >
                      {actionInProgress === plugin.name ? "..." : "install"}
                    </Button>
                  ) : /* Installed, OAuth needed, not connected, and no token_source shortcut */
                  plugin.auth_type === "oauth2" &&
                    !plugin.connected &&
                    !plugin.token_source ? (
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={() => handleConnect(plugin.name)}
                    >
                      connect with {plugin.auth_provider}
                    </Button>
                  ) : /* Has token_source but source not connected yet */
                  plugin.token_source && !plugin.connected ? (
                    <span className="text-muted-foreground tracking-wider">
                      connect {plugin.token_source} first
                    </span>
                  ) : /* Connected/ready but disabled */
                  !plugin.enabled ? (
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={() => handleEnable(plugin.name)}
                      disabled={actionInProgress === plugin.name}
                    >
                      {actionInProgress === plugin.name ? "..." : "enable"}
                    </Button>
                  ) : (
                    /* Enabled and connected */
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground tracking-wider">
                        {plugin.auth_type === "none" ? "active" : "connected"}
                      </span>
                      <Button
                        variant="aether-link"
                        size="aether-link"
                        onClick={() => handleDisable(plugin.name)}
                        disabled={actionInProgress === plugin.name}
                      >
                        {actionInProgress === plugin.name ? "..." : "disable"}
                      </Button>
                    </div>
                  )}
                </div>
              </div>
              {index < plugins.length - 1 && <Separator />}
            </div>
          ))}
        </div>
      )}
    </PageShell>
  );
}
