"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import ListItem from "@/components/ListItem";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import { listIntegrations, IntegrationInfo } from "@/lib/api";
import { IconSearch } from "@tabler/icons-react";

export default function IntegrationsPage() {
  return (
    <Suspense>
      <IntegrationsContent />
    </Suspense>
  );
}

function IntegrationsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const [integrations, setIntegrations] = useState<IntegrationInfo[]>([]);
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
    loadIntegrations();
  }, [session, isPending, router]);

  async function loadIntegrations() {
    try {
      setLoading(true);
      setError("");
      const data = await listIntegrations();
      setIntegrations(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load integrations");
    } finally {
      setLoading(false);
    }
  }

  const active = useMemo(
    () =>
      integrations.filter(
        (p) => p.installed && p.enabled && p.connected && !p.needs_reconnect
      ),
    [integrations]
  );
  const inactive = useMemo(
    () =>
      integrations.filter(
        (p) =>
          !p.installed ||
          !p.enabled ||
          !p.connected ||
          p.needs_reconnect
      ),
    [integrations]
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
    <ContentShell title="Integrations">
      <div className="flex items-center gap-1 mb-6">
        <button
          onClick={() => setTab("browse")}
          className={`
            px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
            ${tab === "browse"
              ? "bg-accent/80 text-foreground"
              : "text-muted-foreground hover:text-foreground/80 hover:bg-accent/40"
            }
          `}
        >
          Browse
        </button>
        <button
          onClick={() => setTab("installed")}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
            ${tab === "installed"
              ? "bg-accent/80 text-foreground"
              : "text-muted-foreground hover:text-foreground/80 hover:bg-accent/40"
            }
          `}
        >
          Installed
          <span
            className={`
              text-xs tabular-nums min-w-[18px] text-center rounded-full px-1.5 py-0.5
              ${tab === "installed" ? "bg-accent/80 text-foreground/70" : "bg-accent/40 text-muted-foreground/60"}
            `}
          >
            {active.length}
          </span>
        </button>
      </div>

      <div className="relative mb-6">
        <IconSearch className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/40" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={tab === "installed" ? "Search installed..." : "Search integrations..."}
          className="w-full h-9 pl-9 pr-3 text-sm bg-accent/30 border border-border rounded-lg focus:outline-none focus:border-input text-foreground placeholder:text-muted-foreground/40 transition-colors"
        />
      </div>

      {(error || oauthError) && (
        <div className="mb-6">
          <p className="text-muted-foreground text-xs mb-3">
            {error || "Could not finish that connection. Please try again."}
          </p>
          <Button variant="aether" size="sm" onClick={loadIntegrations} className="text-xs">
            Try again
          </Button>
        </div>
      )}

      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : filtered.length === 0 ? (
        <div className="py-12 text-center">
          <p className="text-muted-foreground/60 text-xs">
            {search.trim()
              ? "No integrations match your search."
              : tab === "installed"
                ? "No active integrations yet."
                : "All integrations are active."}
          </p>
          {tab === "installed" && !search.trim() && inactive.length > 0 && (
            <button
              onClick={() => setTab("browse")}
              className="mt-3 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Browse available integrations
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((item) => (
            <ListItem
              key={item.name}
              title={item.display_name}
              description={item.description}
              href={`/integrations/${item.name}`}
              action={
                !item.installed && tab === "browse" ? (
                  <span className="shrink-0 text-sm font-medium text-foreground/70 bg-accent/80 px-3 py-1 rounded-full">
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
