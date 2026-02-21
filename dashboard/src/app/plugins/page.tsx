"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import { listPlugins, PluginInfo } from "@/lib/api";

/**
 * Plugins — list available plugins.
 *
 * UX intent:
 *   - Not set up  → "set up →" link navigates to detail page where the user
 *                   connects / configures. Install happens silently there.
 *   - Set up      → status badge (enabled / disabled / reconnect) links to detail.
 *
 * There is no inline install button. The user never sees an "installed but
 * not configured" limbo state.
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
          <Button variant="aether" size="aether" onClick={loadPlugins}>
            retry
          </Button>
        </div>
      ) : plugins.length === 0 ? (
        <p className="text-muted-foreground text-xs">no plugins available</p>
      ) : (
        <div className="space-y-1">
          {plugins.map((plugin) => (
            <PluginRow key={plugin.name} plugin={plugin} />
          ))}
        </div>
      )}
    </PageShell>
  );
}

// ── Plugin row ──────────────────────────────────────────────

function PluginRow({ plugin }: { plugin: PluginInfo }) {
  return (
    <div className="py-5 flex items-start justify-between gap-5">
      {/* Name + description — always a link to detail */}
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

      {/* Right-side action / status */}
      <div className="flex items-start justify-end shrink-0 pt-0.5">
        <PluginStatus plugin={plugin} />
      </div>
    </div>
  );
}

function PluginStatus({ plugin }: { plugin: PluginInfo }) {
  // Not yet set up → prompt to set up
  if (!plugin.installed) {
    return (
      <Link
        href={`/plugins/${plugin.name}`}
        className="text-[11px] tracking-wider text-muted-foreground hover:text-secondary-foreground transition-colors duration-300"
      >
        set up →
      </Link>
    );
  }

  // Installed but needs reconnect
  if (plugin.needs_reconnect) {
    return (
      <Link href={`/plugins/${plugin.name}`} className="block">
        <span className="text-[11px] tracking-wider px-2 py-1 rounded-full bg-amber-500/10 text-amber-400">
          reconnect
        </span>
      </Link>
    );
  }

  // Installed + configured but not yet enabled (e.g. api_key saved, not enabled)
  if (plugin.installed && !plugin.enabled) {
    return (
      <Link href={`/plugins/${plugin.name}`} className="block">
        <span className="text-[11px] tracking-wider px-2 py-1 rounded-full bg-muted/50 text-muted-foreground">
          disabled
        </span>
      </Link>
    );
  }

  // Active
  return (
    <Link href={`/plugins/${plugin.name}`} className="block">
      <span className="text-[11px] tracking-wider px-2 py-1 rounded-full bg-green-500/10 text-green-400">
        enabled
      </span>
    </Link>
  );
}
