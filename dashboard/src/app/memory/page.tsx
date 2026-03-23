"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import ListItem from "@/components/ListItem";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  getEntities,
  getEntityDetails,
  getMemoryItems,
  type EntityRow,
  type EntityDetails,
} from "@/lib/api";

type Tab = "about" | "entities";

interface MemoryItem {
  id: number;
  content: string;
}

export default function MemoryPage() {
  const router = useRouter();
  const { data: session, isPending: sessionPending } = useSession();
  const [tab, setTab] = useState<Tab>("about");
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [entities, setEntities] = useState<EntityRow[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetails | null>(null);
  const [entityLoading, setEntityLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (sessionPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    const userId = session.user.id;

    async function load() {
      try {
        const [itemsRes, entitiesRes] = await Promise.all([
          getMemoryItems(userId, 300, { status: "active" }),
          getEntities(userId),
        ]);
        const raw = itemsRes.items || [];
        setItems(
          raw
            .filter((item) => ["fact", "memory", "summary", "decision"].includes(item.kind))
            .map((item) => ({
              id: item.id,
              content: item.content,
            }))
        );
        setEntities(entitiesRes.entities || []);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load memory");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [session, sessionPending, router]);

  if (sessionPending || !session) return null;

  const isEmpty = items.length === 0 && entities.length === 0;

  return (
    <ContentShell title="Memory">
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider text-center">loading...</p>
      ) : error ? (
        <p className="text-muted-foreground text-xs text-center">{error}</p>
      ) : isEmpty ? (
        <p className="text-muted-foreground text-xs text-center">
          no memories yet — start a conversation
        </p>
      ) : (
        <div className="space-y-6 max-w-[950px] mx-auto">
          <TabBar
            tab={tab}
            onTabChange={setTab}
            counts={{
              about: items.length,
              entities: entities.length,
            }}
          />

          {tab === "about" && <AboutTab items={items} />}
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
        </div>
      )}
    </ContentShell>
  );
}

function TabBar({
  tab,
  onTabChange,
  counts,
}: {
  tab: Tab;
  onTabChange: (t: Tab) => void;
  counts: Record<Tab, number>;
}) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "about", label: "About You" },
    { id: "entities", label: "Entities" },
  ];

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onTabChange(t.id)}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors shrink-0
            ${
              tab === t.id
                ? "bg-white/[0.08] text-foreground"
                : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]"
            }
          `}
        >
          {t.label}
          <span
            className={`
              text-[10px] tabular-nums min-w-[18px] text-center rounded-full px-1.5 py-0.5
              ${tab === t.id ? "bg-white/[0.08] text-foreground/70" : "bg-white/[0.04] text-muted-foreground/60"}
            `}
          >
            {counts[t.id]}
          </span>
        </button>
      ))}
    </div>
  );
}

function AboutTab({ items }: { items: MemoryItem[] }) {
  if (items.length === 0) {
    return <p className="text-muted-foreground text-xs pt-4">nothing learned yet</p>;
  }

  return (
    <div className="space-y-2 pt-2">
      {items.map((item) => (
        <ListItem key={item.id} title={item.content} />
      ))}
      <p className="text-[10px] text-muted-foreground pt-2">
        {items.length} {items.length === 1 ? "item" : "items"} remembered
      </p>
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
    return <EntityDetailView details={selectedEntity} onBack={onBack} />;
  }

  if (entityLoading) {
    return <p className="text-muted-foreground text-xs pt-4">loading entity...</p>;
  }

  if (entities.length === 0) {
    return <p className="text-muted-foreground text-xs pt-4">no entities discovered yet</p>;
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
                    <p className="text-sm text-foreground font-medium truncate">{entity.name}</p>
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

function EntityDetailView({ details, onBack }: { details: EntityDetails; onBack: () => void }) {
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
          <p className="text-sm text-secondary-foreground mt-1 leading-relaxed">{entity.summary}</p>
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
              <div key={obs.id} className="pl-3 border-l-2 border-border">
                <p className="text-sm text-secondary-foreground leading-relaxed">
                  {obs.observation}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="secondary" className="text-[9px] tracking-wider">
                    {obs.category}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">{obs.source}</span>
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
                <div key={rel.id} className="rounded-xl border border-border/70 p-3">
                  <p className="text-sm text-secondary-foreground">
                    {isSource ? (
                      <>
                        <span className="text-foreground font-medium">{entity.name}</span>{" "}
                        <span className="text-muted-foreground">{rel.relation}</span>{" "}
                        <span className="text-foreground">{rel.target_entity_id}</span>
                      </>
                    ) : (
                      <>
                        <span className="text-foreground">{rel.source_entity_id}</span>{" "}
                        <span className="text-muted-foreground">{rel.relation}</span>{" "}
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
              <div key={inter.id} className="pl-3 border-l-2 border-border">
                <p className="text-sm text-secondary-foreground leading-relaxed">{inter.summary}</p>
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
        <p className="text-muted-foreground text-xs">no details recorded yet for this entity</p>
      )}
    </div>
  );
}

function formatTimestamp(value: string): string {
  if (!value) return "-";
  const ts = Date.parse(value);
  if (Number.isNaN(ts)) return "-";
  return new Date(ts).toLocaleString();
}
