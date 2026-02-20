"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  getMemoryFacts,
  getMemorySessions,
  getMemoryConversations,
} from "@/lib/api";

type Tab = "about" | "conversations" | "sessions";

interface Session {
  session_id: string;
  summary: string;
  started_at: number;
  ended_at: number;
  turns: number;
  tools_used: string[];
}

interface Conversation {
  id: number;
  user_message: string;
  assistant_message: string;
  timestamp: number;
}

/**
 * Memory — browse what Aether remembers.
 * Tab-based: about you / conversations / sessions.
 */
export default function MemoryPage() {
  const router = useRouter();
  const { data: session, isPending: sessionPending } = useSession();
  const [tab, setTab] = useState<Tab>("about");
  const [facts, setFacts] = useState<string[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (sessionPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    async function load() {
      try {
        const [factsRes, sessionsRes, convsRes] = await Promise.all([
          getMemoryFacts(),
          getMemorySessions(),
          getMemoryConversations(20),
        ]);
        setFacts(factsRes.facts);
        setSessions(sessionsRes.sessions);
        setConversations(convsRes.conversations);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load memory");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [session, sessionPending, router]);

  if (sessionPending || !session) return null;

  const isEmpty =
    facts.length === 0 && sessions.length === 0 && conversations.length === 0;

  return (
    <PageShell
      title="Memory"
      back="/home"
      centered={loading || !!error || isEmpty}
    >
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider">
          loading...
        </p>
      ) : error ? (
        <p className="text-muted-foreground text-xs">{error}</p>
      ) : isEmpty ? (
        <p className="text-muted-foreground text-xs">
          no memories yet — start a conversation
        </p>
      ) : (
        <div className="space-y-6 max-w-[900px] mx-auto">
          {/* Tabs */}
          <TabBar tab={tab} onTabChange={setTab} />

          {/* Tab content */}
          {tab === "about" && <AboutTab facts={facts} />}
          {tab === "conversations" && (
            <ConversationsTab conversations={conversations} />
          )}
          {tab === "sessions" && <SessionsTab sessions={sessions} />}
        </div>
      )}
    </PageShell>
  );
}

// -- Tab bar --

function TabBar({
  tab,
  onTabChange,
}: {
  tab: Tab;
  onTabChange: (t: Tab) => void;
}) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "about", label: "about you" },
    { id: "conversations", label: "conversations" },
    { id: "sessions", label: "sessions" },
  ];

  return (
    <div className="flex gap-8">
      {tabs.map((t) => (
        <Button
          key={t.id}
          variant="aether-link"
          size="aether-link"
          onClick={() => onTabChange(t.id)}
          className={`text-[11px] tracking-[0.12em] uppercase pb-2 rounded-none border-b transition-all duration-300 ${
            tab === t.id
              ? "text-foreground border-foreground"
              : "text-muted-foreground border-transparent hover:text-secondary-foreground"
          }`}
        >
          {t.label}
        </Button>
      ))}
    </div>
  );
}

// -- About tab (facts) --

function AboutTab({ facts }: { facts: string[] }) {
  if (facts.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">
        nothing learned yet
      </p>
    );
  }

  return (
    <div className="space-y-3 pt-2">
      {facts.map((f, i) => (
        <div
          key={i}
          className="pl-3 border-l-2 border-border animate-[fade-in_0.2s_ease]"
        >
          <p className="text-sm text-secondary-foreground leading-relaxed font-normal max-w-[84ch]">
            {f}
          </p>
        </div>
      ))}
      <p className="text-[10px] text-muted-foreground pt-2">
        {facts.length} {facts.length === 1 ? "fact" : "facts"} remembered
      </p>
    </div>
  );
}

// -- Conversations tab --

function ConversationsTab({
  conversations,
}: {
  conversations: Conversation[];
}) {
  if (conversations.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">
        no conversations yet
      </p>
    );
  }

  const reversed = [...conversations].reverse();

  return (
    <div className="space-y-0 pt-2">
      {reversed.map((c, i) => {
        const prevTimestamp = i > 0 ? reversed[i - 1].timestamp : null;
        const showDateBreak = shouldShowDateBreak(c.timestamp, prevTimestamp);

        return (
          <div key={c.id} className="animate-[fade-in_0.2s_ease]">
            {/* Date separator when day changes */}
            {showDateBreak && (
              <div className="flex items-center gap-3 py-4">
                <Separator className="flex-1" />
                <span className="text-[10px] text-muted-foreground tracking-wider shrink-0">
                  {formatDate(c.timestamp)}
                </span>
                <Separator className="flex-1" />
              </div>
            )}

            <div className="py-3 space-y-2.5">
              {/* User message — right-aligned bubble */}
              <div className="flex flex-col items-end">
                <div className="bg-card border border-border rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[80%]">
                  <p className="text-[13px] text-secondary-foreground leading-relaxed font-normal max-w-[76ch]">
                    {c.user_message}
                  </p>
                </div>
              </div>

              {/* Aether response — left-aligned, no bubble */}
              <div className="flex flex-col items-start">
                <p className="text-[13px] text-foreground leading-relaxed font-normal max-w-[76ch]">
                  {c.assistant_message}
                </p>
                <span className="text-[9px] text-muted-foreground mt-1 tracking-wider">
                  {formatRelativeTime(c.timestamp)}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// -- Sessions tab --

function SessionsTab({ sessions }: { sessions: Session[] }) {
  if (sessions.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">no sessions yet</p>
    );
  }

  return (
    <div className="pt-2">
      {sessions.map((s, i) => (
        <div key={s.session_id} className="animate-[fade-in_0.2s_ease]">
          <div className="py-5">
            <p className="text-sm text-secondary-foreground leading-relaxed font-normal max-w-[84ch]">
              {s.summary}
            </p>

            {/* Metadata row */}
            <div className="flex items-center gap-3 mt-2 flex-wrap">
              <span className="text-[10px] text-muted-foreground tracking-wider">
                {formatRelativeTime(s.ended_at)}
              </span>
              <span className="text-[10px] text-muted-foreground">
                ·
              </span>
              <span className="text-[10px] text-muted-foreground tracking-wider">
                {s.turns} {s.turns === 1 ? "turn" : "turns"}
              </span>

              {/* Tools used */}
              {s.tools_used && s.tools_used.length > 0 && (
                <>
                  <span className="text-[10px] text-muted-foreground">
                    ·
                  </span>
                  <div className="flex gap-1.5 flex-wrap">
                    {s.tools_used.map((tool) => (
                      <Badge
                        key={tool}
                        variant="secondary"
                        className="text-[9px] tracking-wider font-normal px-1.5 py-0 h-4 rounded-sm"
                      >
                        {tool}
                      </Badge>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
          {i < sessions.length - 1 && <Separator />}
        </div>
      ))}
    </div>
  );
}

// -- Time helpers --

function formatRelativeTime(ts: number): string {
  const now = Date.now();
  const then = ts * 1000;
  const diff = now - then;

  const minutes = Math.floor(diff / 60_000);
  const hours = Math.floor(diff / 3_600_000);
  const days = Math.floor(diff / 86_400_000);

  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;

  return new Date(then).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function formatDate(ts: number): string {
  const date = new Date(ts * 1000);
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === now.toDateString()) return "today";
  if (date.toDateString() === yesterday.toDateString()) return "yesterday";

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function shouldShowDateBreak(
  currentTs: number,
  prevTs: number | null
): boolean {
  if (prevTs === null) return true;
  const current = new Date(currentTs * 1000).toDateString();
  const prev = new Date(prevTs * 1000).toDateString();
  return current !== prev;
}
