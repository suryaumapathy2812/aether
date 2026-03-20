"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { useSession } from "@/lib/auth-client";
import {
  installSkill,
  listInstalledSkills,
  removeSkill,
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
  const [busySkill, setBusySkill] = useState("");

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

  async function onInstall(skill: MarketplaceSkill) {
    setBusySkill(skill.id);
    setMessage("");
    try {
      await installSkill(skill.source, skill.skill_id || skill.name);
      await loadInstalled();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Install failed");
    } finally {
      setBusySkill("");
    }
  }

  async function onRemove(name: string) {
    setBusySkill(name);
    setMessage("");
    try {
      await removeSkill(name);
      await loadInstalled();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Remove failed");
    } finally {
      setBusySkill("");
    }
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="Skills">
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
          className="w-full h-9 pl-9 pr-3 text-[13px] bg-white/[0.03] border border-white/[0.06] rounded-lg focus:outline-none focus:border-white/[0.12] text-foreground placeholder:text-muted-foreground/40 transition-colors"
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
          {results.map((skill) => {
            const isInstalled = installedNames.has(skill.name);
            return (
                <div
                  key={skill.id}
                  className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg border border-white/[0.06]"
                >
                <div className="min-w-0">
                  <p className="text-sm truncate">{skill.name}</p>
                  <p className="text-[11px] text-muted-foreground truncate">{skill.source}</p>
                </div>
                <button
                  className="h-8 px-3 text-xs border border-white/[0.12] rounded-md hover:bg-white/[0.04] disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={busySkill === skill.id || isInstalled}
                  onClick={() => onInstall(skill)}
                >
                  {isInstalled ? "Installed" : "Install"}
                </button>
              </div>
            );
          })}
          {!loading && results.length === 0 && (
            <p className="text-xs text-muted-foreground">
              {search.trim() ? "No skills match your search." : "No results yet."}
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {filteredInstalled.map((skill) => (
            <div
              key={skill.name}
              className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg border border-white/[0.06]"
            >
              <div className="min-w-0">
                <p className="text-sm truncate">{skill.name || "Unnamed skill"}</p>
                <p className="text-[11px] text-muted-foreground truncate">{skill.description || "No description"}</p>
              </div>
              <button
                className="h-8 px-3 text-xs border border-white/[0.12] rounded-md hover:bg-white/[0.04] disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={busySkill === skill.name}
                onClick={() => onRemove(skill.name)}
              >
                Remove
              </button>
            </div>
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
