"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useSession } from "@/lib/auth-client";
import { listPlugins, PluginInfo } from "@/lib/api";
import {
  Search,
  Mail,
  Calendar,
  Contact2,
  HardDrive,
  CloudSun,
  Compass,
  MapPin,
  Rss,
  Music,
  BookOpen,
  Calculator,
  Puzzle,
} from "lucide-react";

export default function PluginsPage() {
  return (
    <Suspense>
      <PluginsContent />
    </Suspense>
  );
}

// ── Icon mapping ──

const PLUGIN_ICONS: Record<string, React.ElementType> = {
  gmail: Mail,
  google_calendar: Calendar,
  google_contacts: Contact2,
  google_drive: HardDrive,
  weather: CloudSun,
  brave_search: Compass,
  local_search: MapPin,
  rss: Rss,
  spotify: Music,
  wikipedia: BookOpen,
  wolfram: Calculator,
};

function getPluginIcon(name: string): React.ElementType {
  return PLUGIN_ICONS[name] || Puzzle;
}

// ── Main content ──

function PluginsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"installed" | "browse">("installed");
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

  const installed = useMemo(
    () => plugins.filter((p) => p.installed),
    [plugins]
  );
  const available = useMemo(
    () => plugins.filter((p) => !p.installed),
    [plugins]
  );

  const filtered = useMemo(() => {
    const list = tab === "installed" ? installed : available;
    if (!search.trim()) return list;
    const q = search.toLowerCase();
    return list.filter(
      (p) =>
        p.display_name.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q)
    );
  }, [tab, installed, available, search]);

  if (isPending || !session) return null;

  return (
    <ContentShell title="Plugins">
      {/* Tabs */}
      <div className="flex items-center gap-1 mb-6">
        <TabButton
          active={tab === "installed"}
          onClick={() => setTab("installed")}
          count={installed.length}
        >
          Installed
        </TabButton>
        <TabButton
          active={tab === "browse"}
          onClick={() => setTab("browse")}
          count={available.length}
        >
          Browse
        </TabButton>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/40" />
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
            variant="ghost"
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
                ? "No plugins installed yet."
                : "No more plugins available."}
          </p>
          {tab === "installed" && !search.trim() && available.length > 0 && (
            <button
              onClick={() => setTab("browse")}
              className="mt-3 text-[12px] text-muted-foreground hover:text-foreground transition-colors"
            >
              Browse available plugins
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {filtered.map((plugin) => (
            <PluginCard key={plugin.name} plugin={plugin} tab={tab} />
          ))}
        </div>
      )}
    </ContentShell>
  );
}

// ── Tab button ──

function TabButton({
  active,
  onClick,
  count,
  children,
}: {
  active: boolean;
  onClick: () => void;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors
        ${
          active
            ? "bg-white/[0.08] text-foreground"
            : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]"
        }
      `}
    >
      {children}
      <span
        className={`
          text-[10px] tabular-nums min-w-[18px] text-center rounded-full px-1.5 py-0.5
          ${active ? "bg-white/[0.08] text-foreground/70" : "bg-white/[0.04] text-muted-foreground/60"}
        `}
      >
        {count}
      </span>
    </button>
  );
}

// ── Plugin card ──

function PluginCard({
  plugin,
  tab,
}: {
  plugin: PluginInfo;
  tab: "installed" | "browse";
}) {
  const Icon = getPluginIcon(plugin.name);
  const status = getPluginStatus(plugin);

  return (
    <Link
      href={`/plugins/${plugin.name}`}
      className="group flex items-start gap-3.5 p-4 rounded-xl border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/[0.1] transition-all"
    >
      {/* Icon */}
      <div className="w-10 h-10 rounded-xl bg-white/[0.06] flex items-center justify-center shrink-0 group-hover:bg-white/[0.08] transition-colors">
        <Icon className="size-4.5 text-muted-foreground" strokeWidth={1.5} />
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-medium text-foreground truncate">
            {plugin.display_name}
          </span>
          {tab === "installed" && (
            <StatusBadge status={status} />
          )}
        </div>
        <p className="text-[11px] text-muted-foreground/70 mt-0.5 line-clamp-2 leading-relaxed">
          {plugin.description}
        </p>
      </div>

      {/* Browse tab: Get button */}
      {tab === "browse" && (
        <span className="shrink-0 mt-0.5 text-[11px] font-medium text-foreground/70 bg-white/[0.08] hover:bg-white/[0.12] px-3 py-1 rounded-full transition-colors">
          Get
        </span>
      )}
    </Link>
  );
}

// ── Status helpers ──

type PluginStatus = "active" | "attention" | "disabled";

function getPluginStatus(plugin: PluginInfo): PluginStatus {
  if (
    plugin.needs_reconnect ||
    (plugin.auth_type === "oauth2" && plugin.installed && !plugin.connected)
  ) {
    return "attention";
  }
  if (plugin.installed && plugin.enabled && plugin.connected) {
    return "active";
  }
  return "disabled";
}

function StatusBadge({ status }: { status: PluginStatus }) {
  if (status === "active") {
    return (
      <Badge
        variant="outline"
        className="text-[9px] px-1.5 py-0 h-4 border-emerald-500/20 text-emerald-400/80 bg-emerald-500/[0.06]"
      >
        Active
      </Badge>
    );
  }
  if (status === "attention") {
    return (
      <Badge
        variant="outline"
        className="text-[9px] px-1.5 py-0 h-4 border-amber-500/20 text-amber-400/80 bg-amber-500/[0.06]"
      >
        Setup
      </Badge>
    );
  }
  return (
    <Badge
      variant="outline"
      className="text-[9px] px-1.5 py-0 h-4 border-white/[0.06] text-muted-foreground/50 bg-white/[0.02]"
    >
      Off
    </Badge>
  );
}
