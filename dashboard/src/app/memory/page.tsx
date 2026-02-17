"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import {
  isLoggedIn,
  getMemoryFacts,
  getMemorySessions,
  getMemoryConversations,
} from "@/lib/api";

/**
 * Memory — browse what Aether remembers.
 * Compact conversation history with role-based layout.
 */
export default function MemoryPage() {
  const router = useRouter();
  const [facts, setFacts] = useState<string[]>([]);
  const [sessions, setSessions] = useState<
    { summary: string; ended_at: number; turns: number }[]
  >([]);
  const [conversations, setConversations] = useState<
    { user_message: string; assistant_message: string; timestamp: number }[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isLoggedIn()) {
      router.push("/");
      return;
    }

    async function load() {
      try {
        const [factsRes, sessionsRes, convsRes] = await Promise.all([
          getMemoryFacts(),
          getMemorySessions(),
          getMemoryConversations(10),
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
  }, [router]);

  const isEmpty =
    facts.length === 0 && sessions.length === 0 && conversations.length === 0;

  function formatTime(ts: number): string {
    return new Date(ts * 1000).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  return (
    <PageShell
      title="Memory"
      back="/home"
      centered={loading || !!error || isEmpty}
    >
      {loading ? (
        <p className="text-[var(--color-text-muted)] text-xs tracking-wider">
          loading...
        </p>
      ) : error ? (
        <p className="text-[var(--color-text-muted)] text-xs">{error}</p>
      ) : isEmpty ? (
        <p className="text-[var(--color-text-muted)] text-xs">
          no memories yet — start a conversation
        </p>
      ) : (
        <div className="space-y-8">
          {/* ── Facts ── */}
          {facts.length > 0 && (
            <section>
              <h2 className="text-xs tracking-widest text-[var(--color-text-muted)] uppercase mb-3 font-normal">
                known facts
              </h2>
              <div className="space-y-1.5">
                {facts.map((f, i) => (
                  <p
                    key={i}
                    className="text-sm text-[var(--color-text-secondary)] leading-relaxed font-light pl-3 border-l border-[var(--color-border)]"
                  >
                    {f}
                  </p>
                ))}
              </div>
            </section>
          )}

          {/* ── Sessions ── */}
          {sessions.length > 0 && (
            <section>
              <h2 className="text-xs tracking-widest text-[var(--color-text-muted)] uppercase mb-3 font-normal">
                past sessions
              </h2>
              <div className="space-y-2.5">
                {sessions.map((s, i) => (
                  <div
                    key={i}
                    className="border-l-2 border-[var(--color-border)] pl-3 py-0.5"
                  >
                    <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed font-light">
                      {s.summary}
                    </p>
                    <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                      {formatTime(s.ended_at)} · {s.turns} turns
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* ── Recent conversations ── */}
          {conversations.length > 0 && (
            <section>
              <div className="space-y-4">
                {conversations.map((c, i) => (
                  <div
                    key={i}
                    className="last:border-b-0 last:pb-0 space-y-4"
                  >

                    {/* User message — right-aligned */}
                    <div className="flex flex-col items-end mb-2">
                      <div className="bg-surface border border-border rounded-2xl rounded-tr-sm px-3 py-1.5 max-w-[80%]">
                        <p className="text-sm text-text-secondary leading-relaxed font-light p-2!">
                          {c.user_message}
                        </p>
                      </div>
                    </div>

                    {/* Aether response — left-aligned */}
                    <div className="flex flex-col items-start">
                      <p className="text-sm text-[var(--color-text)] leading-relaxed font-light max-w-[85%]">
                        {c.assistant_message}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </PageShell>
  );
}
