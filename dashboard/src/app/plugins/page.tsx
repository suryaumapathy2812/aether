"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import ListItem from "@/components/ListItem";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import { listPlugins, PluginInfo } from "@/lib/api";
import { IconSearch } from "@tabler/icons-react";

export default function PluginsPage() {
  return (
    <Suspense>
      <PluginsContent />
    </Suspense>
  );
}

// ── Main content ──

function PluginsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"installed" | "browse">("browse");
  const [search, setSearch] = useState("");

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

  // "Installed" = active (enabled + connected)
  // "Browse" = everything else (not installed, disabled, needs attention)
  const active = useMemo(
    () =>
      plugins.filter(
        (p) => p.installed && p.enabled && p.connected && !p.needs_reconnect
      ),
    [plugins]
  );
  const inactive = useMemo(
    () =>
      plugins.filter(
        (p) =>
          !p.installed ||
          !p.enabled ||
          !p.connected ||
          p.needs_reconnect
      ),
    [plugins]
  );

  const filtered = useMemo(() => {
    const list = tab === "installed" ? active : inactive;
    if (!search.trim()) return list;
    const q = search.toLowerCase();
    return list.filter(
      (p) =>
        p.display_name.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q)
    );
  }, [tab, active, inactive, search]);

  if (isPending || !session) return null;

  return (
    <ContentShell title="Plugins">
      {/* Tabs */}
      <div className="flex items-center gap-1 mb-6">
        <button
          onClick={() => setTab("browse")}
          className={`
            px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors
            ${
              tab === "browse"
                ? "bg-white/[0.08] text-foreground"
                : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]"
            }
          `}
        >
          Browse
        </button>
        <button
          onClick={() => setTab("installed")}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors
            ${
              tab === "installed"
                ? "bg-white/[0.08] text-foreground"
                : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]"
            }
          `}
        >
          Installed
          <span
            className={`
              text-[10px] tabular-nums min-w-[18px] text-center rounded-full px-1.5 py-0.5
              ${tab === "installed" ? "bg-white/[0.08] text-foreground/70" : "bg-white/[0.04] text-muted-foreground/60"}
            `}
          >
            {active.length}
          </span>
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <IconSearch className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/40" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={
            tab === "installed" ? "Search installed..." : "Search plugins..."
          }
          className="w-full h-9 pl-9 pr-3 text-[13px] bg-white/[0.03] border border-white/[0.06] rounded-lg focus:outline-none focus:border-white/[0.12] text-foreground placeholder:text-muted-foreground/40 transition-colors"
        />
      </div>

      {/* Error */}
      {(error || oauthError) && (
        <div className="mb-6">
          <p className="text-muted-foreground text-xs mb-3">
            {error || "Could not finish that connection. Please try again."}
          </p>
          <Button
            variant="aether"
            size="sm"
            onClick={loadPlugins}
            className="text-xs"
          >
            Try again
          </Button>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : filtered.length === 0 ? (
        <div className="py-12 text-center">
          <p className="text-muted-foreground/60 text-xs">
            {search.trim()
              ? "No plugins match your search."
              : tab === "installed"
                ? "No active plugins yet."
                : "All plugins are active."}
          </p>
          {tab === "installed" && !search.trim() && inactive.length > 0 && (
            <button
              onClick={() => setTab("browse")}
              className="mt-3 text-[12px] text-muted-foreground hover:text-foreground transition-colors"
            >
              Browse available plugins
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((plugin) => (
            <ListItem
              key={plugin.name}
              title={plugin.display_name}
              description={plugin.description}
              href={`/plugins/${plugin.name}`}
              action={
                !plugin.installed && tab === "browse" ? (
                  <span className="shrink-0 text-[11px] font-medium text-foreground/70 bg-white/[0.08] px-3 py-1 rounded-full">
                    Get
                  </span>
                ) : undefined
              }
            />
          ))}
        </div>
      )}
    </ContentShell>
  );
}
