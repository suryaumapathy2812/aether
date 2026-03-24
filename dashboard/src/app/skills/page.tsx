"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import ListItem from "@/components/ListItem";
import { useSession } from "@/lib/auth-client";
import {
  listInstalledSkills,
  searchMarketplaceSkills,
  type MarketplaceSkill,
  type SkillMeta,
} from "@/lib/api";
import { IconSearch } from "@tabler/icons-react";

const DEFAULT_MARKETPLACE_QUERY = "agent";
const SEARCH_DEBOUNCE_MS = 350;

export default function SkillsPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [search, setSearch] = useState("");
  const [results, setResults] = useState<MarketplaceSkill[]>([]);
  const [installed, setInstalled] = useState<SkillMeta[]>([]);
  const [tab, setTab] = useState<"browse" | "installed">("browse");
  const [showTrending, setShowTrending] = useState(true);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    void loadInstalled();
    void runSearch("");
  }, [session, isPending, router]);

  useEffect(() => {
    if (!session) return;
    if (tab !== "browse") return;
    const timer = window.setTimeout(() => {
      void runSearch(search);
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [search, session, tab]);

  const installedNames = useMemo(() => new Set(installed.map((s) => s.name)), [installed]);
  const filteredInstalled = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return installed;
    return installed.filter((skill) => {
      const name = (skill.name || "").toLowerCase();
      const description = (skill.description || "").toLowerCase();
      return name.includes(q) || description.includes(q);
    });
  }, [installed, search]);

  async function loadInstalled() {
    try {
      const data = await listInstalledSkills();
      setInstalled(data.skills || []);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed to load installed skills");
    }
  }

  async function runSearch(nextQuery?: string) {
    const q = (nextQuery ?? search).trim();
    const usingTrendingQuery = q === "";
    const apiQuery = usingTrendingQuery ? DEFAULT_MARKETPLACE_QUERY : q;
    setLoading(true);
    setMessage("");
    setShowTrending(usingTrendingQuery);
    try {
      const data = await searchMarketplaceSkills(apiQuery, 25);
      const items = data.skills || [];
      if (usingTrendingQuery) {
        setResults([...items].sort((a, b) => b.installs - a.installs));
      } else {
        setResults(items);
      }
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed to search marketplace");
    } finally {
      setLoading(false);
    }
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="Skills">
      <div className="flex items-center gap-1 mb-6">
        <button
          onClick={() => setTab("browse")}
          className={`
            px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
            ${
              tab === "browse"
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
            ${
              tab === "installed"
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
            {installed.length}
          </span>
        </button>
      </div>

      <div className="relative mb-6">
        <IconSearch className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/40" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={tab === "browse" ? "Search marketplace..." : "Search installed..."}
          className="w-full h-9 pl-9 pr-3 text-sm bg-accent/30 border border-border rounded-lg focus:outline-none focus:border-input text-foreground placeholder:text-muted-foreground/40 transition-colors"
        />
      </div>

      {tab === "browse" && (
        <div className="mb-5 flex items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground">
            {showTrending ? "Showing top skills by installs." : "Showing search results."}
          </p>
        </div>
      )}

      {message && <p className="text-xs text-muted-foreground mb-4">{message}</p>}

      {tab === "browse" ? (
        <div className="space-y-2">
          {results.map((skill) => (
            <ListItem
              key={skill.id}
              title={skill.name}
              description={skill.source}
              href={`/skills/${encodeURIComponent(skill.name)}`}
              action={
                installedNames.has(skill.name) ? (
                  <span className="shrink-0 text-sm text-muted-foreground bg-muted px-3 py-1 rounded-full">
                    Installed
                  </span>
                ) : undefined
              }
            />
          ))}
          {!loading && results.length === 0 && (
            <p className="text-xs text-muted-foreground">
              {search.trim() ? "No skills match your search." : "No results yet."}
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {filteredInstalled.map((skill) => (
            <ListItem
              key={skill.name}
              title={skill.name || "Unnamed skill"}
              description={skill.description || "No description"}
              href={`/skills/${encodeURIComponent(skill.name)}`}
            />
          ))}
          {filteredInstalled.length === 0 && (
            <p className="text-xs text-muted-foreground">
              {search.trim() ? "No installed skills match your search." : "No installed skills."}
            </p>
          )}
        </div>
      )}
    </ContentShell>
  );
}
