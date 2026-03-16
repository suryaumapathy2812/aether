"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import { listPlugins, PluginInfo } from "@/lib/api";

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

  return (
    <ContentShell title="Connections">
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider">
          loading...
        </p>
      ) : error || oauthError ? (
        <div>
          <p className="text-muted-foreground text-xs mb-4">
            {error ? "Could not load your connections right now." : "Could not finish that connection. Please try again."}
          </p>
          <Button variant="aether" size="aether" onClick={loadPlugins}>
            try again
          </Button>
        </div>
      ) : plugins.length === 0 ? (
        <p className="text-muted-foreground text-xs">No connections available yet.</p>
      ) : (
        <div className="space-y-2">
          <p className="text-[12px] text-muted-foreground mb-1">
            Connect your favorite apps so Aether can help across everything in one place.
          </p>
          {plugins.map((plugin) => (
            <PluginRow key={plugin.name} plugin={plugin} />
          ))}
        </div>
      )}
    </ContentShell>
  );
}

function PluginRow({ plugin }: { plugin: PluginInfo }) {
  const shortDescription = toSimpleDescription(plugin.description);

  return (
    <div className="py-4 px-3 rounded-2xl border border-border/60 bg-white/5 flex items-start justify-between gap-4">
      <Link
        href={`/plugins/${plugin.name}`}
        className="block group min-w-0 flex-1"
      >
        <h3 className="text-[14px] text-foreground group-hover:text-secondary-foreground transition-colors duration-300 font-medium mb-1">
          {plugin.display_name}
        </h3>
        <p className="text-[12px] text-muted-foreground leading-relaxed font-normal line-clamp-2">
          {shortDescription}
        </p>
      </Link>

      <div className="flex items-start justify-end shrink-0 pt-0.5">
        <PluginStatus plugin={plugin} />
      </div>
    </div>
  );
}

function PluginStatus({ plugin }: { plugin: PluginInfo }) {
  if (!plugin.installed) {
    return (
      <Link
        href={`/plugins/${plugin.name}`}
        className="text-[11px] tracking-wider px-2.5 py-1 rounded-full bg-secondary/10 text-secondary-foreground hover:bg-secondary/20 transition-colors duration-300"
      >
        set up
      </Link>
    );
  }

  if (plugin.needs_reconnect || (plugin.auth_type === "oauth2" && !plugin.connected)) {
    return (
      <Link href={`/plugins/${plugin.name}`} className="block">
        <span className="text-[11px] tracking-wider px-2 py-1 rounded-full bg-amber-500/10 text-amber-400">
          needs attention
        </span>
      </Link>
    );
  }

  if (plugin.installed && !plugin.enabled) {
    return (
      <Link href={`/plugins/${plugin.name}`} className="block">
        <span className="text-[11px] tracking-wider px-2 py-1 rounded-full bg-muted/50 text-muted-foreground">
          off
        </span>
      </Link>
    );
  }

  return (
    <Link href={`/plugins/${plugin.name}`} className="block">
      <span className="text-[11px] tracking-wider px-2 py-1 rounded-full bg-green-500/10 text-green-400">
        connected
      </span>
    </Link>
  );
}

function toSimpleDescription(text: string): string {
  const value = text.trim();
  if (!value) return "Use this app with Aether.";
  const clean = value.replace(/oauth|api|sdk|token|webhook/gi, "").replace(/\s+/g, " ").trim();
  return clean || "Use this app with Aether.";
}
