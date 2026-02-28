"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  exportMemory,
  getDecisions,
  getMemories,
  getMemoryConversations,
  getMemoryFacts,
  getMemoryNotifications,
  getMemorySessions,
} from "@/lib/api";

type Tab =
  | "about"
  | "conversations"
  | "sessions"
  | "memories"
  | "decisions"
  | "notifications";

interface SessionRow {
  session_id: string;
  summary: string;
  started_at: number;
  ended_at: number;
  turns: number;
  tools_used: string[];
}

interface ConversationRow {
  id: number;
  user_message: string;
  assistant_message: string;
  timestamp: number;
}

interface MemoryRow {
  id: number;
  memory: string;
  category: string;
  confidence: number;
  created_at: string;
}

interface DecisionRow {
  id: number;
  decision: string;
  category: string;
  source: string;
  active: boolean;
  confidence: number;
  updated_at: string;
}

interface NotificationRow {
  id?: number;
  text?: string;
  status?: string;
  source?: string;
  delivery_type?: string;
  created_at?: string;
  createdAt?: string;
}

const assistantMarkdownComponents: Components = {
  h1: ({ children }) => <strong className="block mb-1">{children}</strong>,
  h2: ({ children }) => <strong className="block mb-1">{children}</strong>,
  h3: ({ children }) => <strong className="block mb-1">{children}</strong>,
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => <em>{children}</em>,
  code: ({ children, className }) => {
    if (className) {
      return <code className={className}>{children}</code>;
    }

    return (
      <code className="rounded bg-white/8 px-1.5 py-0.5 text-[0.9em] font-mono text-foreground/90">
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-white/6 border border-border/50 p-3 text-[0.85em] font-mono leading-relaxed">
      {children}
    </pre>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="underline underline-offset-2 decoration-foreground/30 hover:decoration-foreground/60 transition-colors"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 ml-4 list-disc space-y-0.5 last:mb-0">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 ml-4 list-decimal space-y-0.5 last:mb-0">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-foreground/20 pl-3 italic text-secondary-foreground">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-border/50" />,
  img: () => null,
};

export default function MemoryPage() {
  const router = useRouter();
  const { data: session, isPending: sessionPending } = useSession();
  const [tab, setTab] = useState<Tab>("about");
  const [facts, setFacts] = useState<string[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [memories, setMemories] = useState<MemoryRow[]>([]);
  const [decisions, setDecisions] = useState<DecisionRow[]>([]);
  const [notifications, setNotifications] = useState<NotificationRow[]>([]);
  const [reliability, setReliability] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (sessionPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    const userId = session.user.id;

    async function load() {
      try {
        const [
          factsRes,
          sessionsRes,
          convsRes,
          memoriesRes,
          decisionsRes,
          notificationsRes,
        ] = await Promise.all([
          getMemoryFacts(userId),
          getMemorySessions(userId),
          getMemoryConversations(userId, 30),
          getMemories(userId, 100),
          getDecisions(userId, undefined, true),
          getMemoryNotifications(userId, 200),
        ]);
        setFacts(factsRes.facts || []);
        setSessions(sessionsRes.sessions || []);
        setConversations(convsRes.conversations || []);
        setMemories(memoriesRes.memories || []);
        setDecisions(decisionsRes.decisions || []);
        setNotifications(
          (notificationsRes.notifications || []) as NotificationRow[],
        );
        setReliability(notificationsRes.reliability || {});
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load memory");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [session, sessionPending, router]);

  async function handleExport() {
    setExporting(true);
    try {
      const res = await exportMemory(session?.user?.id || "");
      const blob = new Blob([JSON.stringify(res.export, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `aether-memory-export-${new Date().toISOString().slice(0, 19)}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  if (sessionPending || !session) return null;

  const isEmpty =
    facts.length === 0 &&
    sessions.length === 0 &&
    conversations.length === 0 &&
    memories.length === 0 &&
    decisions.length === 0 &&
    notifications.length === 0;

  return (
    <PageShell
      title="Memory"
      back="/home"
      centered={loading || !!error || isEmpty}
    >
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider text-center">
          loading...
        </p>
      ) : error ? (
        <p className="text-muted-foreground text-xs text-center">{error}</p>
      ) : isEmpty ? (
        <p className="text-muted-foreground text-xs text-center">
          no memories yet — start a conversation
        </p>
      ) : (
        <div className="space-y-6 max-w-[950px] mx-auto">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <TabBar tab={tab} onTabChange={setTab} />
            <Button
              variant="aether-link"
              size="aether-link"
              onClick={handleExport}
              disabled={exporting}
              className="text-[11px] uppercase tracking-[0.12em]"
            >
              {exporting ? "exporting..." : "export memory"}
            </Button>
          </div>

          {tab === "about" && <FactsTab facts={facts} />}
          {tab === "conversations" && (
            <ConversationsTab conversations={conversations} />
          )}
          {tab === "sessions" && <SessionsTab sessions={sessions} />}
          {tab === "memories" && <MemoriesTab memories={memories} />}
          {tab === "decisions" && <DecisionsTab decisions={decisions} />}
          {tab === "notifications" && (
            <NotificationsTab
              notifications={notifications}
              reliability={reliability}
            />
          )}
        </div>
      )}
    </PageShell>
  );
}

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
    { id: "memories", label: "memories" },
    { id: "decisions", label: "decisions" },
    { id: "notifications", label: "notifications" },
  ];

  return (
    <div className="flex gap-3 sm:gap-6 flex-wrap">
      {tabs.map((t) => (
        <Button
          key={t.id}
          variant="aether-link"
          size="aether-link"
          onClick={() => onTabChange(t.id)}
          className={`text-[11px] tracking-[0.12em] uppercase pb-2 rounded-none border-b transition-all duration-300 shrink-0 ${
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

function FactsTab({ facts }: { facts: string[] }) {
  if (facts.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">nothing learned yet</p>
    );
  }
  return (
    <div className="space-y-3 pt-2">
      {facts.map((fact, i) => (
        <div
          key={i}
          className="pl-3 border-l-2 border-border animate-[fade-in_0.2s_ease]"
        >
          <p className="text-sm text-secondary-foreground leading-relaxed max-w-[84ch]">
            {fact}
          </p>
        </div>
      ))}
      <p className="text-[10px] text-muted-foreground pt-2">
        {facts.length} {facts.length === 1 ? "fact" : "facts"} remembered
      </p>
    </div>
  );
}

function ConversationsTab({
  conversations,
}: {
  conversations: ConversationRow[];
}) {
  if (conversations.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">no conversations yet</p>
    );
  }

  const reversed = [...conversations].reverse();
  return (
    <div className="space-y-0 pt-2">
      {reversed.map((c, i) => {
        const prevTimestamp = i > 0 ? reversed[i - 1].timestamp : null;
        const showDateBreak = shouldShowDateBreak(c.timestamp, prevTimestamp);
        return (
          <div key={`${c.id}-${i}`} className="animate-[fade-in_0.2s_ease]">
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
              <div className="flex flex-col items-end">
                <div className="bg-card border border-border rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[80%]">
                  <p className="text-[13px] text-secondary-foreground leading-relaxed max-w-[76ch]">
                    {c.user_message}
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-start">
                <div className="text-[13px] text-foreground leading-relaxed max-w-[76ch]">
                  <ReactMarkdown components={assistantMarkdownComponents}>
                    {c.assistant_message}
                  </ReactMarkdown>
                </div>
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

function SessionsTab({ sessions }: { sessions: SessionRow[] }) {
  if (sessions.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">no sessions yet</p>
    );
  }

  return (
    <div className="pt-2">
      {sessions.map((s, i) => (
        <div
          key={`${s.session_id}-${i}`}
          className="animate-[fade-in_0.2s_ease]"
        >
          <div className="py-5">
            <p className="text-sm text-secondary-foreground leading-relaxed max-w-[84ch]">
              {s.summary}
            </p>
            <div className="flex items-center gap-3 mt-2 flex-wrap">
              <span className="text-[10px] text-muted-foreground tracking-wider">
                {formatRelativeTime(s.ended_at)}
              </span>
              <span className="text-[10px] text-muted-foreground">·</span>
              <span className="text-[10px] text-muted-foreground tracking-wider">
                {s.turns} {s.turns === 1 ? "turn" : "turns"}
              </span>
              {s.tools_used.length > 0 && (
                <>
                  <span className="text-[10px] text-muted-foreground">·</span>
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

function MemoriesTab({ memories }: { memories: MemoryRow[] }) {
  if (memories.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">
        no episodic memories yet
      </p>
    );
  }
  return (
    <div className="space-y-3 pt-2">
      {memories.map((m) => (
        <div key={m.id} className="rounded-xl border border-border/70 p-3">
          <p className="text-sm text-secondary-foreground leading-relaxed">
            {m.memory}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <Badge variant="secondary" className="text-[9px] tracking-wider">
              {m.category}
            </Badge>
            <span className="text-[10px] text-muted-foreground">
              confidence {Math.round((m.confidence || 1) * 100)}%
            </span>
            <span className="text-[10px] text-muted-foreground">·</span>
            <span className="text-[10px] text-muted-foreground">
              {formatTimestamp(m.created_at)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function DecisionsTab({ decisions }: { decisions: DecisionRow[] }) {
  if (decisions.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">
        no active decisions yet
      </p>
    );
  }
  return (
    <div className="space-y-3 pt-2">
      {decisions.map((d) => (
        <div key={d.id} className="rounded-xl border border-border/70 p-3">
          <p className="text-sm text-secondary-foreground leading-relaxed">
            {d.decision}
          </p>
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            <Badge variant="secondary" className="text-[9px] tracking-wider">
              {d.category}
            </Badge>
            <span className="text-[10px] text-muted-foreground">
              source {d.source}
            </span>
            <span className="text-[10px] text-muted-foreground">·</span>
            <span className="text-[10px] text-muted-foreground">
              {d.active ? "active" : "inactive"}
            </span>
            <span className="text-[10px] text-muted-foreground">·</span>
            <span className="text-[10px] text-muted-foreground">
              {formatTimestamp(d.updated_at)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function NotificationsTab({
  notifications,
  reliability,
}: {
  notifications: NotificationRow[];
  reliability: Record<string, unknown>;
}) {
  return (
    <div className="space-y-4 pt-2">
      <div className="rounded-xl border border-border/70 p-3">
        <p className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
          reliability
        </p>
        <pre className="mt-2 text-[11px] text-secondary-foreground overflow-auto whitespace-pre-wrap">
          {JSON.stringify(reliability, null, 2)}
        </pre>
      </div>

      {notifications.length === 0 ? (
        <p className="text-muted-foreground text-xs">
          no notifications recorded yet
        </p>
      ) : (
        <div className="space-y-2">
          {notifications.map((n, i) => (
            <div
              key={`${n.id ?? i}`}
              className="rounded-xl border border-border/70 p-3"
            >
              <p className="text-sm text-secondary-foreground leading-relaxed">
                {n.text || "(no text)"}
              </p>
              <div className="flex items-center gap-2 mt-2 flex-wrap">
                {n.status && (
                  <Badge
                    variant="secondary"
                    className="text-[9px] tracking-wider"
                  >
                    {n.status}
                  </Badge>
                )}
                {n.delivery_type && (
                  <span className="text-[10px] text-muted-foreground">
                    {n.delivery_type}
                  </span>
                )}
                {n.source && (
                  <span className="text-[10px] text-muted-foreground">
                    source {n.source}
                  </span>
                )}
                <span className="text-[10px] text-muted-foreground">
                  {formatTimestamp(n.created_at || n.createdAt || "")}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatRelativeTime(ts: number): string {
  if (!ts) return "-";
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
  if (!ts) return "unknown day";
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
  prevTs: number | null,
): boolean {
  if (prevTs === null) return true;
  const current = new Date(currentTs * 1000).toDateString();
  const prev = new Date(prevTs * 1000).toDateString();
  return current !== prev;
}

function formatTimestamp(value: string): string {
  if (!value) return "-";
  const ts = Date.parse(value);
  if (Number.isNaN(ts)) return "-";
  return new Date(ts).toLocaleString();
}
