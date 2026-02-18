"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import PageShell from "@/components/PageShell";
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
        <p className="text-[var(--color-text-muted)] text-xs tracking-wider">
          loading...
        </p>
      ) : error || oauthError ? (
        <div>
          <p className="text-[var(--color-text-muted)] text-xs mb-4">
            {error || `OAuth error: ${oauthError}`}
          </p>
          <button onClick={loadPlugins} className="btn text-xs">
            retry
          </button>
        </div>
      ) : plugins.length === 0 ? (
        <p className="text-[var(--color-text-muted)] text-xs">
          no plugins available
        </p>
      ) : (
        <div>
          {plugins.map((plugin) => (
            <div
              key={plugin.name}
              className="py-4 border-b border-[var(--color-border)] last:border-b-0"
            >
              <Link
                href={`/plugins/${plugin.name}`}
                className="block mb-3 group"
              >
                <h3 className="text-[14px] text-[var(--color-text)] group-hover:text-[var(--color-text-secondary)] transition-colors duration-300 font-light mb-0.5">
                  {plugin.display_name}
                </h3>
                <p className="text-[12px] text-[var(--color-text-muted)] leading-relaxed font-light">
                  {plugin.description}
                </p>
              </Link>

              <div className="flex items-center gap-3 text-xs">
                {/* Not installed → install */}
                {!plugin.installed ? (
                  <button
                    onClick={() => handleInstall(plugin.name)}
                    disabled={actionInProgress === plugin.name}
                    className="btn disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    {actionInProgress === plugin.name ? "..." : "install"}
                  </button>
                ) : /* Installed but needs OAuth connection */
                plugin.auth_type === "oauth2" && !plugin.connected ? (
                  <button
                    onClick={() => handleConnect(plugin.name)}
                    className="btn"
                  >
                    connect with {plugin.auth_provider}
                  </button>
                ) : /* Connected but disabled */
                !plugin.enabled ? (
                  <button
                    onClick={() => handleEnable(plugin.name)}
                    disabled={actionInProgress === plugin.name}
                    className="btn disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    {actionInProgress === plugin.name ? "..." : "enable"}
                  </button>
                ) : (
                  /* Enabled and connected */
                  <div className="flex items-center gap-2">
                    <span className="text-[var(--color-text-muted)] tracking-wider">
                      connected
                    </span>
                    <button
                      onClick={() => handleDisable(plugin.name)}
                      disabled={actionInProgress === plugin.name}
                      className="text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors duration-300 disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      {actionInProgress === plugin.name ? "..." : "disable"}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </PageShell>
  );
}
