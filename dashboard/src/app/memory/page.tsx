"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  exportMemory,
  getDecisions,
  getEntities,
  getEntityDetails,
  getMemories,
  getMemoryConversations,
  getMemoryFacts,
  type EntityRow,
  type EntityDetails,
} from "@/lib/api";

type Tab = "about" | "conversations" | "entities" | "memories" | "decisions";

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
  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [memories, setMemories] = useState<MemoryRow[]>([]);
  const [decisions, setDecisions] = useState<DecisionRow[]>([]);
  const [entities, setEntities] = useState<EntityRow[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetails | null>(null);
  const [entityLoading, setEntityLoading] = useState(false);
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
        const [factsRes, convsRes, memoriesRes, decisionsRes, entitiesRes] =
          await Promise.all([
            getMemoryFacts(userId),
            getMemoryConversations(userId, 30),
            getMemories(userId, 100),
            getDecisions(userId, undefined, true),
            getEntities(userId),
          ]);
        setFacts(factsRes.facts || []);
        setConversations(convsRes.conversations || []);
        setMemories(memoriesRes.memories || []);
        setDecisions(decisionsRes.decisions || []);
        setEntities(entitiesRes.entities || []);
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
    conversations.length === 0 &&
    entities.length === 0 &&
    memories.length === 0 &&
    decisions.length === 0;

  return (
    <ContentShell
      title="Memory"
      back="/home"
     
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
          {tab === "entities" && (
            <EntitiesTab
              entities={entities}
              selectedEntity={selectedEntity}
              entityLoading={entityLoading}
              onSelectEntity={async (entity) => {
                setEntityLoading(true);
                setSelectedEntity(null);
                try {
                  const details = await getEntityDetails(entity.id);
                  setSelectedEntity(details);
                } catch {
                  setSelectedEntity(null);
                } finally {
                  setEntityLoading(false);
                }
              }}
              onBack={() => setSelectedEntity(null)}
            />
          )}
          {tab === "memories" && <MemoriesTab memories={memories} />}
          {tab === "decisions" && <DecisionsTab decisions={decisions} />}
        </div>
      )}
    </ContentShell>
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
    { id: "entities", label: "entities" },
    { id: "memories", label: "memories" },
    { id: "decisions", label: "decisions" },
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

function EntitiesTab({
  entities,
  selectedEntity,
  entityLoading,
  onSelectEntity,
  onBack,
}: {
  entities: EntityRow[];
  selectedEntity: EntityDetails | null;
  entityLoading: boolean;
  onSelectEntity: (entity: EntityRow) => void;
  onBack: () => void;
}) {
  if (selectedEntity) {
    return (
      <EntityDetailView
        details={selectedEntity}
        onBack={onBack}
      />
    );
  }

  if (entityLoading) {
    return (
      <p className="text-muted-foreground text-xs pt-4">loading entity...</p>
    );
  }

  if (entities.length === 0) {
    return (
      <p className="text-muted-foreground text-xs pt-4">
        no entities discovered yet
      </p>
    );
  }

  // Group entities by type
  const grouped = entities.reduce<Record<string, EntityRow[]>>((acc, e) => {
    const type = e.entity_type || "other";
    if (!acc[type]) acc[type] = [];
    acc[type].push(e);
    return acc;
  }, {});

  const typeOrder = ["person", "project", "organization", "topic", "place", "tool"];
  const sortedTypes = Object.keys(grouped).sort((a, b) => {
    const ai = typeOrder.indexOf(a);
    const bi = typeOrder.indexOf(b);
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
  });

  return (
    <div className="space-y-6 pt-2">
      {sortedTypes.map((type) => (
        <div key={type}>
          <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
            {type}s ({grouped[type].length})
          </p>
          <div className="space-y-2">
            {grouped[type].map((entity) => (
              <button
                key={entity.id}
                onClick={() => onSelectEntity(entity)}
                className="w-full text-left rounded-xl border border-border/70 p-3 hover:border-foreground/20 transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-foreground font-medium truncate">
                      {entity.name}
                    </p>
                    {entity.summary && (
                      <p className="text-xs text-secondary-foreground mt-0.5 line-clamp-2 leading-relaxed">
                        {entity.summary}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {entity.interaction_count > 0 && (
                      <span className="text-[10px] text-muted-foreground">
                        {entity.interaction_count} interactions
                      </span>
                    )}
                  </div>
                </div>
                {entity.aliases.length > 0 && (
                  <div className="flex gap-1.5 mt-2 flex-wrap">
                    {entity.aliases.map((alias, i) => (
                      <Badge
                        key={i}
                        variant="secondary"
                        className="text-[9px] tracking-wider font-normal px-1.5 py-0 h-4 rounded-sm"
                      >
                        {alias}
                      </Badge>
                    ))}
                  </div>
                )}
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-[10px] text-muted-foreground">
                    last seen {formatTimestamp(entity.last_seen_at)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}
      <p className="text-[10px] text-muted-foreground pt-2">
        {entities.length} {entities.length === 1 ? "entity" : "entities"} tracked
      </p>
    </div>
  );
}

function EntityDetailView({
  details,
  onBack,
}: {
  details: EntityDetails;
  onBack: () => void;
}) {
  const { entity, observations, interactions, relations } = details;

  return (
    <div className="space-y-6 pt-2">
      {/* Header with back button */}
      <div className="flex items-center gap-3">
        <Button
          variant="aether-link"
          size="aether-link"
          onClick={onBack}
          className="text-[11px] uppercase tracking-[0.12em]"
        >
          &larr; back
        </Button>
      </div>

      {/* Entity header */}
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-medium text-foreground">{entity.name}</h2>
          <Badge variant="secondary" className="text-[9px] tracking-wider">
            {entity.entity_type}
          </Badge>
        </div>
        {entity.summary && (
          <p className="text-sm text-secondary-foreground mt-1 leading-relaxed">
            {entity.summary}
          </p>
        )}
        {entity.aliases.length > 0 && (
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {entity.aliases.map((alias, i) => (
              <Badge
                key={i}
                variant="secondary"
                className="text-[9px] tracking-wider font-normal px-1.5 py-0 h-4 rounded-sm"
              >
                {alias}
              </Badge>
            ))}
          </div>
        )}
        <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground">
          <span>{entity.interaction_count} interactions</span>
          <span>&middot;</span>
          <span>first seen {formatTimestamp(entity.first_seen_at)}</span>
          <span>&middot;</span>
          <span>last seen {formatTimestamp(entity.last_seen_at)}</span>
        </div>
      </div>

      <Separator />

      {/* Observations */}
      {observations.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
            observations ({observations.length})
          </p>
          <div className="space-y-2">
            {observations.map((obs) => (
              <div
                key={obs.id}
                className="pl-3 border-l-2 border-border"
              >
                <p className="text-sm text-secondary-foreground leading-relaxed">
                  {obs.observation}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="secondary" className="text-[9px] tracking-wider">
                    {obs.category}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">
                    {obs.source}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Relations */}
      {relations.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
            relationships ({relations.length})
          </p>
          <div className="space-y-2">
            {relations.map((rel) => {
              const isSource = rel.source_entity_id === entity.id;
              return (
                <div
                  key={rel.id}
                  className="rounded-xl border border-border/70 p-3"
                >
                  <p className="text-sm text-secondary-foreground">
                    {isSource ? (
                      <>
                        <span className="text-foreground font-medium">{entity.name}</span>
                        {" "}<span className="text-muted-foreground">{rel.relation}</span>{" "}
                        <span className="text-foreground">{rel.target_entity_id}</span>
                      </>
                    ) : (
                      <>
                        <span className="text-foreground">{rel.source_entity_id}</span>
                        {" "}<span className="text-muted-foreground">{rel.relation}</span>{" "}
                        <span className="text-foreground font-medium">{entity.name}</span>
                      </>
                    )}
                  </p>
                  {rel.context && (
                    <p className="text-xs text-muted-foreground mt-1">{rel.context}</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Interactions timeline */}
      {interactions.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
            interactions ({interactions.length})
          </p>
          <div className="space-y-2">
            {interactions.map((inter) => (
              <div
                key={inter.id}
                className="pl-3 border-l-2 border-border"
              >
                <p className="text-sm text-secondary-foreground leading-relaxed">
                  {inter.summary}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="secondary" className="text-[9px] tracking-wider">
                    {inter.source}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">
                    {formatTimestamp(inter.interaction_at)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state for sections */}
      {observations.length === 0 && relations.length === 0 && interactions.length === 0 && (
        <p className="text-muted-foreground text-xs">
          no details recorded yet for this entity
        </p>
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
