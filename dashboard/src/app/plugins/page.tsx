"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import { listPlugins, installPlugin, PluginInfo } from "@/lib/api";

/**
 * Plugins — list available plugins with install button and status.
 * Click plugin to go to detail page for enable/disable/config.
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
  const [installing, setInstalling] = useState<string | null>(null);

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
    setInstalling(name);
    try {
      await installPlugin(name);
      await loadPlugins();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to install plugin");
    } finally {
      setInstalling(null);
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
        <div className="space-y-1">
          {plugins.map((plugin) => (
            <div key={plugin.name}>
              <div className="py-5 flex items-start justify-between gap-5">
                <Link
                  href={`/plugins/${plugin.name}`}
                  className="block group min-w-0 flex-1"
                >
                  <h3 className="text-[14px] text-foreground group-hover:text-secondary-foreground transition-colors duration-300 font-normal mb-0.5">
                    {plugin.display_name}
                  </h3>
                  <p className="text-[12px] text-muted-foreground leading-relaxed font-normal">
                    {plugin.description}
                  </p>
                </Link>

                <div className="flex items-start justify-end text-xs shrink-0 pt-0.5">
                  {/* Not installed → install */}
                  {!plugin.installed ? (
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        handleInstall(plugin.name);
                      }}
                      disabled={installing === plugin.name}
                    >
                      {installing === plugin.name ? "..." : "install"}
                    </Button>
                  ) : (
                    /* Installed → show status badge */
                    <Link
                      href={`/plugins/${plugin.name}`}
                      className="block"
                    >
                      <span
                        className={`text-[11px] tracking-wider px-2 py-1 rounded-full ${
                          plugin.enabled
                            ? "bg-green-500/10 text-green-400"
                            : "bg-muted/50 text-muted-foreground"
                        }`}
                      >
                        {plugin.enabled ? "enabled" : "disabled"}
                      </span>
                    </Link>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </PageShell>
  );
}
