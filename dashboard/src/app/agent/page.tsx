"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useSession } from "@/lib/auth-client";
import {
  cancelAgentTask,
  getAgentTaskEvents,
  listAgentTasks,
  rejectAgentTask,
  resumeAgentTask,
  type AgentTask,
  type AgentTaskEvent,
} from "@/lib/api";

type WaitingPrompt = {
  question: string;
  options: string[];
  suggestedDefault: string;
};

function statusLabel(status: string): string {
  if (status === "waiting_input") return "Waiting on your input";
  if (status === "queued") return "Starting";
  if (status === "running") return "Running";
  if (status === "failed") return "Needs attention";
  if (status === "completed") return "Completed";
  if (status === "cancelled") return "Stopped";
  return "Running";
}

function statusClass(status: string): string {
  if (status === "waiting_input") {
    return "text-amber-200 border-amber-400/40 bg-amber-500/10";
  }
  if (status === "running" || status === "queued") {
    return "text-emerald-200 border-emerald-400/40 bg-emerald-500/10";
  }
  return "text-muted-foreground border-border/50 bg-white/[0.03]";
}

function formatRelative(value: string): string {
  const ts = Date.parse(value);
  if (Number.isNaN(ts)) return "";
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
}

function parseWaitingPrompt(events: AgentTaskEvent[]): WaitingPrompt {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (event.Kind !== "status") continue;
    try {
      const payload = JSON.parse(event.PayloadJSON || "{}");
      if (String(payload.status || "") !== "waiting_input") continue;
      const detail =
        payload.payload && typeof payload.payload === "object"
          ? payload.payload
          : {};
      const question =
        String((detail as Record<string, unknown>).question || "").trim() ||
        String(payload.message || "").trim();
      const suggestedDefault = String(
        (detail as Record<string, unknown>).suggested_default || "",
      ).trim();
      const rawOptions = (detail as Record<string, unknown>).options;
      const options = Array.isArray(rawOptions)
        ? rawOptions.map((v) => String(v || "").trim()).filter(Boolean)
        : [];
      return { question, options, suggestedDefault };
    } catch {
      continue;
    }
  }
  return { question: "", options: [], suggestedDefault: "" };
}

function prettyPayload(payloadJSON: string): string {
  try {
    return JSON.stringify(JSON.parse(payloadJSON), null, 2);
  } catch {
    return payloadJSON;
  }
}

export default function AgentPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const userId = session?.user?.id || "";
  const taskFromQuery = (searchParams.get("task") || "").trim();
  const debugMode = (searchParams.get("debug") || "").toLowerCase() === "true";

  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [events, setEvents] = useState<AgentTaskEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [error, setError] = useState("");
  const [reply, setReply] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const selectedFromQueryRef = useRef(false);

  useEffect(() => {
    if (!isPending && !session) router.push("/");
  }, [isPending, router, session]);

  useEffect(() => {
    selectedFromQueryRef.current = false;
  }, [taskFromQuery]);

  async function loadTasks(silent = false): Promise<void> {
    try {
      if (!silent) setLoading(true);
      setError("");
      const [runningRes, waitingRes] = await Promise.all([
        listAgentTasks(userId, "running", 60),
        listAgentTasks(userId, "waiting_input", 60),
      ]);
      const merged = [...runningRes.tasks, ...waitingRes.tasks].sort(
        (a, b) =>
          new Date(b.UpdatedAt).getTime() - new Date(a.UpdatedAt).getTime(),
      );
      setTasks(merged);
      if (!selectedFromQueryRef.current && taskFromQuery) {
        selectedFromQueryRef.current = true;
        if (merged.some((t) => t.ID === taskFromQuery)) {
          setSelectedTaskId(taskFromQuery);
          return;
        }
      }
      if (!selectedTaskId && merged.length > 0) {
        setSelectedTaskId(merged[0].ID);
      }
      if (selectedTaskId && !merged.some((t) => t.ID === selectedTaskId)) {
        setSelectedTaskId(merged[0]?.ID || "");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load agents");
    } finally {
      if (!silent) setLoading(false);
    }
  }

  async function loadEvents(taskId: string): Promise<void> {
    if (!taskId) {
      setEvents([]);
      return;
    }
    try {
      setEventsLoading(true);
      const res = await getAgentTaskEvents(taskId, 300);
      setEvents(res.events || []);
    } catch {
      setEvents([]);
    } finally {
      setEventsLoading(false);
    }
  }

  useEffect(() => {
    if (!session || !userId) return;
    void loadTasks(false);
    const timer = window.setInterval(() => {
      void loadTasks(true);
      if (selectedTaskId) void loadEvents(selectedTaskId);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [selectedTaskId, session, taskFromQuery, userId]);

  useEffect(() => {
    if (selectedTaskId) void loadEvents(selectedTaskId);
  }, [selectedTaskId]);

  const waitingAgents = useMemo(
    () => tasks.filter((t) => t.Status === "waiting_input"),
    [tasks],
  );
  const runningAgents = useMemo(
    () => tasks.filter((t) => t.Status === "running" || t.Status === "queued"),
    [tasks],
  );
  const selectedTask = tasks.find((t) => t.ID === selectedTaskId) || null;
  const prompt = parseWaitingPrompt(events);
  const optionChoices = (prompt.options || []).slice(0, 3);

  async function runAction(action: () => Promise<unknown>) {
    try {
      setError("");
      await action();
      setReply("");
      setRejectReason("");
      await loadTasks(false);
      if (selectedTaskId) await loadEvents(selectedTaskId);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Action failed");
    }
  }

  if (isPending || !session) return null;

  return (
    <PageShell title="Agents" back="/home">
      <div className="space-y-6 max-w-[980px] mx-auto">
        {error ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3">
            <p className="text-[12px] text-red-200">{error}</p>
          </div>
        ) : null}

        {loading ? (
          <p className="text-[12px] text-muted-foreground">Loading agents...</p>
        ) : tasks.length === 0 ? (
          <Empty className="border border-border/60 bg-white/[0.02]">
            <EmptyHeader>
              <EmptyTitle className="text-[16px] text-foreground font-medium">
                No active agents right now
              </EmptyTitle>
              <EmptyDescription className="text-[12px] max-w-[520px]">
                Start in chat to ask Aether to run something. Running agents and
                agents waiting on your input will appear here.
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button
                variant="aether"
                size="aether"
                onClick={() => router.push("/chat")}
              >
                open chat
              </Button>
            </EmptyContent>
          </Empty>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_1.85fr] gap-5">
            <div className="space-y-2">
              {waitingAgents.length > 0 && (
                <p className="text-[11px] tracking-[0.1em] uppercase text-amber-200 pb-1">
                  Waiting on your input
                </p>
              )}
              {waitingAgents.map((task) => (
                <button
                  key={task.ID}
                  onClick={() => setSelectedTaskId(task.ID)}
                  className={`w-full text-left rounded-xl border px-3.5 py-3.5 transition-colors ${
                    selectedTaskId === task.ID
                      ? "border-amber-400/45 bg-amber-500/10"
                      : "border-amber-500/25 bg-amber-500/5 hover:bg-amber-500/10"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-[13px] text-foreground truncate">
                      {task.Title}
                    </p>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full border ${statusClass(task.Status)}`}
                    >
                      {statusLabel(task.Status)}
                    </span>
                  </div>
                  <p className="text-[12px] text-muted-foreground mt-1 line-clamp-2">
                    {task.ResultSummary ||
                      "This agent is waiting for your decision."}
                  </p>
                </button>
              ))}

              {runningAgents.length > 0 && (
                <p className="text-[11px] tracking-[0.1em] uppercase text-emerald-200 pt-2 pb-1">
                  Running now
                </p>
              )}
              {runningAgents.map((task) => (
                <button
                  key={task.ID}
                  onClick={() => setSelectedTaskId(task.ID)}
                  className={`w-full text-left rounded-xl border px-3.5 py-3.5 transition-colors ${
                    selectedTaskId === task.ID
                      ? "border-secondary-foreground/35 bg-white/[0.10]"
                      : "border-border/70 bg-white/[0.03] hover:bg-white/[0.07]"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-[13px] text-foreground truncate">
                      {task.Title}
                    </p>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full border ${statusClass(task.Status)}`}
                    >
                      {statusLabel(task.Status)}
                    </span>
                  </div>
                  <p className="text-[12px] text-muted-foreground mt-1 line-clamp-2">
                    {task.ResultSummary || "Working on your request..."}
                  </p>
                  <p className="text-[11px] text-muted-foreground/80 mt-2">
                    updated {formatRelative(task.UpdatedAt)}
                  </p>
                </button>
              ))}
            </div>

            <div className="space-y-4">
              {!selectedTask ? (
                <div className="rounded-xl border border-border/70 bg-white/[0.03] p-4">
                  <p className="text-[12px] text-muted-foreground">
                    Select an agent to view details.
                  </p>
                </div>
              ) : (
                <>
                  <div className="rounded-2xl border border-border/70 bg-white/[0.03] p-4 space-y-2">
                    <p className="text-[15px] text-foreground">
                      {selectedTask.Title}
                    </p>
                    <p className="text-[12px] text-secondary-foreground">
                      Status: {statusLabel(selectedTask.Status)}
                    </p>
                    <p className="text-[12px] text-muted-foreground leading-relaxed">
                      {selectedTask.Goal}
                    </p>
                    {selectedTask.ResultSummary ? (
                      <div className="rounded-xl border border-border/60 bg-black/10 px-3 py-2 mt-1">
                        <p className="text-[11px] text-muted-foreground">
                          Current update
                        </p>
                        <p className="text-[12px] text-secondary-foreground mt-1">
                          {selectedTask.ResultSummary}
                        </p>
                      </div>
                    ) : null}
                  </div>

                  {selectedTask.Status === "waiting_input" && (
                    <div className="rounded-2xl border border-amber-500/35 bg-amber-500/10 p-4 space-y-4">
                      <h3 className="text-[13px] text-amber-100 font-medium">
                        Agent needs your answer
                      </h3>
                      <p className="text-[12px] text-amber-50/95">
                        {prompt.question ||
                          selectedTask.ResultSummary ||
                          "Please choose how this agent should proceed."}
                      </p>

                      {optionChoices.length > 0 && (
                        <div className="space-y-2">
                          <p className="text-[11px] text-amber-100/85">
                            Suggested replies
                          </p>
                          <div className="grid gap-2">
                            {optionChoices.map((opt) => (
                              <button
                                key={opt}
                                onClick={() =>
                                  void runAction(() =>
                                    resumeAgentTask({
                                      taskId: selectedTask.ID,
                                      userId,
                                      message: opt,
                                    }),
                                  )
                                }
                                className={`text-left rounded-lg border px-3 py-2 text-[12px] transition-colors ${
                                  prompt.suggestedDefault &&
                                  prompt.suggestedDefault === opt
                                    ? "border-amber-300/60 bg-amber-400/15 text-amber-50"
                                    : "border-amber-400/30 bg-black/10 text-amber-100 hover:bg-black/20"
                                }`}
                              >
                                {opt}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="space-y-2">
                        <label className="text-[11px] text-amber-100/90">
                          Type your own answer
                        </label>
                        <Textarea
                          rows={3}
                          value={reply}
                          onChange={(e) => setReply(e.target.value)}
                          placeholder="Type your response"
                        />
                        <Button
                          variant="aether"
                          size="aether"
                          onClick={() =>
                            void runAction(() =>
                              resumeAgentTask({
                                taskId: selectedTask.ID,
                                userId,
                                message: reply.trim() || "Approved. Continue.",
                              }),
                            )
                          }
                        >
                          send response
                        </Button>
                      </div>

                      <div className="space-y-2 pt-1">
                        <label className="text-[11px] text-amber-100/90">
                          Decline for now (optional)
                        </label>
                        <Input
                          value={rejectReason}
                          onChange={(e) => setRejectReason(e.target.value)}
                          placeholder="Reason"
                        />
                        <Button
                          variant="aether-link"
                          size="aether-link"
                          className="text-red-300 hover:text-red-200"
                          onClick={() =>
                            void runAction(() =>
                              rejectAgentTask({
                                taskId: selectedTask.ID,
                                userId,
                                reason: rejectReason || "Rejected by user",
                                nextAction:
                                  "Stop and wait for new instructions.",
                              }),
                            )
                          }
                        >
                          decline request
                        </Button>
                      </div>
                    </div>
                  )}

                  {(selectedTask.Status === "running" ||
                    selectedTask.Status === "queued") && (
                    <div>
                      <Button
                        variant="aether-link"
                        size="aether-link"
                        className="text-red-300 hover:text-red-200"
                        onClick={() =>
                          void runAction(() => cancelAgentTask(selectedTask.ID))
                        }
                      >
                        stop this agent
                      </Button>
                    </div>
                  )}

                  {debugMode && (
                    <div className="rounded-2xl border border-border/70 bg-white/[0.02] p-4">
                      <h3 className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground mb-3">
                        Technical details {eventsLoading ? "(loading...)" : ""}
                      </h3>
                      {events.length === 0 ? (
                        <p className="text-[12px] text-muted-foreground">
                          No events yet
                        </p>
                      ) : (
                        <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
                          {events.map((evt) => (
                            <div
                              key={evt.ID}
                              className="text-[11px] border-b border-border/40 pb-2"
                            >
                              <p className="text-secondary-foreground">
                                {evt.Kind}
                              </p>
                              <pre className="text-muted-foreground break-all whitespace-pre-wrap">
                                {prettyPayload(evt.PayloadJSON)}
                              </pre>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </PageShell>
  );
}
