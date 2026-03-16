"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import { listPlugins, PluginInfo } from "@/lib/api";
import { ChevronRight } from "lucide-react";

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

  // Group plugins by status
  const connected = plugins.filter((p) => p.installed && p.enabled && p.connected);
  const attention = plugins.filter((p) => p.needs_reconnect || (p.installed && !p.enabled) || (p.auth_type === "oauth2" && p.installed && !p.connected));
  const available = plugins.filter((p) => !p.installed);

  return (
    <ContentShell title="Plugins">
      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : error || oauthError ? (
        <div>
          <p className="text-muted-foreground text-xs mb-4">
            {error || "Could not finish that connection. Please try again."}
          </p>
          <Button variant="ghost" size="sm" onClick={loadPlugins} className="text-xs">
            Try again
          </Button>
        </div>
      ) : plugins.length === 0 ? (
        <p className="text-muted-foreground text-xs">No plugins available yet.</p>
      ) : (
        <div className="space-y-8">
          {connected.length > 0 && (
            <PluginSection title="Active" plugins={connected} />
          )}
          {attention.length > 0 && (
            <PluginSection title="Needs attention" plugins={attention} />
          )}
          {available.length > 0 && (
            <PluginSection title="Available" plugins={available} />
          )}
        </div>
      )}
    </ContentShell>
  );
}

function PluginSection({ title, plugins }: { title: string; plugins: PluginInfo[] }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/60 font-medium mb-3">
        {title}
      </p>
      <div className="space-y-0.5">
        {plugins.map((plugin) => (
          <PluginRow key={plugin.name} plugin={plugin} />
        ))}
      </div>
    </div>
  );
}

function PluginRow({ plugin }: { plugin: PluginInfo }) {
  return (
    <Link
      href={`/plugins/${plugin.name}`}
      className="flex items-center justify-between py-3 group"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2.5">
          <span className="text-[13px] text-foreground font-medium">
            {plugin.display_name}
          </span>
          <StatusDot plugin={plugin} />
        </div>
        <p className="text-[11px] text-muted-foreground mt-0.5 line-clamp-1">
          {toSimpleDescription(plugin.description)}
        </p>
      </div>
      <ChevronRight className="size-4 text-muted-foreground/30 group-hover:text-muted-foreground transition-colors shrink-0 ml-4" />
    </Link>
  );
}

function StatusDot({ plugin }: { plugin: PluginInfo }) {
  if (plugin.needs_reconnect || (plugin.auth_type === "oauth2" && plugin.installed && !plugin.connected)) {
    return <span className="w-1.5 h-1.5 rounded-full bg-amber-400" title="Needs reconnection" />;
  }
  if (plugin.installed && plugin.enabled && plugin.connected) {
    return <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" title="Connected" />;
  }
  if (plugin.installed && !plugin.enabled) {
    return <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40" title="Disabled" />;
  }
  return null;
}

function toSimpleDescription(text: string): string {
  const value = text.trim();
  if (!value) return "Use this plugin with Aether.";
  const clean = value.replace(/oauth|api|sdk|token|webhook/gi, "").replace(/\s+/g, " ").trim();
  return clean || "Use this plugin with Aether.";
}
