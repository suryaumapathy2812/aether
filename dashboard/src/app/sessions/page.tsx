"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { useSession } from "@/lib/auth-client";
import { chatRuntime, useChatStatusMap } from "@/lib/chat-runtime";
import {
  listChatSessions,
  createChatSession,
  updateChatSessionTitle,
  archiveChatSession,
  deleteChatSession,
  type ChatSession,
} from "@/lib/api";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import {
  Plus,
  MoreHorizontal,
  Pencil,
  Archive,
  Trash2,
} from "lucide-react";

export default function SessionsPage() {
  return (
    <Suspense>
      <SessionsContent />
    </Suspense>
  );
}

function SessionsContent() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

  const activeSessionId = searchParams.get("s") || "";
  const statusMap = useChatStatusMap();

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
  }, [session, isPending, router]);

  const loadSessions = useCallback(() => {
    if (!session?.user?.id) return;
    setLoading(true);
    listChatSessions(session.user.id, 50)
      .then((res) => setSessions(res.sessions || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [session?.user?.id]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions, pathname]);

  useEffect(() => {
    const userId = session?.user?.id?.trim();
    if (!userId) return;
    void chatRuntime.bootstrapForUser(userId);
  }, [session?.user?.id]);

  async function handleNewChat() {
    if (!session?.user?.id) return;
    try {
      const newSess = await createChatSession(session.user.id);
      setSessions((prev) => [newSess, ...prev]);
      router.push(`/chat?s=${newSess.id}`);
    } catch {
      router.push("/chat");
    }
  }

  function startRename(s: ChatSession) {
    setEditingId(s.id);
    setEditTitle(s.title);
    setTimeout(() => editInputRef.current?.focus(), 50);
  }

  async function commitRename() {
    if (!editingId) return;
    const trimmed = editTitle.trim();
    if (trimmed) {
      await updateChatSessionTitle(editingId, trimmed).catch(() => {});
      setSessions((prev) =>
        prev.map((s) => (s.id === editingId ? { ...s, title: trimmed } : s))
      );
    }
    setEditingId(null);
  }

  async function handleArchive(id: string) {
    await archiveChatSession(id).catch(() => {});
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) router.push("/chat");
  }

  async function handleDelete(id: string) {
    await deleteChatSession(id).catch(() => {});
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) router.push("/chat");
  }

  if (isPending || !session) return null;

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const todaySessions = sessions.filter((s) => new Date(s.updated_at) >= today);
  const yesterdaySessions = sessions.filter((s) => {
    const updatedAt = new Date(s.updated_at);
    return updatedAt >= yesterday && updatedAt < today;
  });
  const olderSessions = sessions.filter(
    (s) => new Date(s.updated_at) < yesterday
  );

  return (
    <ContentShell
      title="Sessions"
      action={
        <Button
          variant="aether"
          size="sm"
          onClick={handleNewChat}
          className="h-8 px-3 text-sm"
        >
          <Plus className="size-3.5 mr-1" />
          new
        </Button>
      }
    >
      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : sessions.length === 0 ? (
        <p className="text-muted-foreground/60 text-xs text-center py-12">
          No sessions yet. Start a new chat.
        </p>
      ) : (
        <div className="space-y-6">
          {todaySessions.length > 0 && (
            <SessionGroup
              label="Today"
              sessions={todaySessions}
              activeSessionId={activeSessionId}
              editingId={editingId}
              editTitle={editTitle}
              editInputRef={editInputRef}
              statusMap={statusMap}
              onEditTitleChange={setEditTitle}
              onSelect={(id) => router.push(`/chat?s=${id}`)}
              onRename={startRename}
              onCommitRename={commitRename}
              onCancelRename={() => setEditingId(null)}
              onArchive={handleArchive}
              onDelete={handleDelete}
            />
          )}
          {yesterdaySessions.length > 0 && (
            <SessionGroup
              label="Yesterday"
              sessions={yesterdaySessions}
              activeSessionId={activeSessionId}
              editingId={editingId}
              editTitle={editTitle}
              editInputRef={editInputRef}
              statusMap={statusMap}
              onEditTitleChange={setEditTitle}
              onSelect={(id) => router.push(`/chat?s=${id}`)}
              onRename={startRename}
              onCommitRename={commitRename}
              onCancelRename={() => setEditingId(null)}
              onArchive={handleArchive}
              onDelete={handleDelete}
            />
          )}
          {olderSessions.length > 0 && (
            <SessionGroup
              label="Previous"
              sessions={olderSessions}
              activeSessionId={activeSessionId}
              editingId={editingId}
              editTitle={editTitle}
              editInputRef={editInputRef}
              statusMap={statusMap}
              onEditTitleChange={setEditTitle}
              onSelect={(id) => router.push(`/chat?s=${id}`)}
              onRename={startRename}
              onCommitRename={commitRename}
              onCancelRename={() => setEditingId(null)}
              onArchive={handleArchive}
              onDelete={handleDelete}
            />
          )}
        </div>
      )}
    </ContentShell>
  );
}

// ── Session group ──

function SessionGroup({
  label,
  sessions,
  activeSessionId,
  editingId,
  editTitle,
  editInputRef,
  statusMap,
  onEditTitleChange,
  onSelect,
  onRename,
  onCommitRename,
  onCancelRename,
  onArchive,
  onDelete,
}: {
  label: string;
  sessions: ChatSession[];
  activeSessionId: string;
  editingId: string | null;
  editTitle: string;
  editInputRef: React.RefObject<HTMLInputElement | null>;
  statusMap: Record<string, string>;
  onEditTitleChange: (v: string) => void;
  onSelect: (id: string) => void;
  onRename: (s: ChatSession) => void;
  onCommitRename: () => void;
  onCancelRename: () => void;
  onArchive: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.15em] text-muted-foreground/50 font-medium mb-2">
        {label}
      </p>
      <div className="space-y-2">
        {sessions.map((s) => {
          const isActive = s.id === activeSessionId;
          const isEditing = s.id === editingId;
          const isRunning = statusMap[s.id] === "streaming";

          return (
            <div
              key={s.id}
              className="group/item flex items-center gap-3 px-3 py-2.5 rounded-lg border border-border hover:bg-accent/30 transition-colors"
            >
              {isEditing ? (
                <input
                  ref={editInputRef}
                  value={editTitle}
                  onChange={(e) => onEditTitleChange(e.target.value)}
                  onBlur={onCommitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onCommitRename();
                    if (e.key === "Escape") onCancelRename();
                  }}
                  className="h-7 min-w-0 flex-1 bg-transparent text-sm font-medium text-foreground outline-none"
                />
              ) : (
                <>
                  <button
                    onClick={() => onSelect(s.id)}
                    onDoubleClick={() => onRename(s)}
                    className="min-w-0 flex-1 text-left cursor-pointer"
                  >
                    <span className="text-sm font-medium text-foreground flex items-center gap-2">
                      {s.title || "New chat"}
                      {isRunning && (
                        <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                      )}
                    </span>
                  </button>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button className="shrink-0 p-1.5 mr-1 rounded text-muted-foreground/30 hover:text-muted-foreground opacity-0 group-hover/item:opacity-100 transition-all">
                        <MoreHorizontal className="size-3.5" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-36">
                      <DropdownMenuItem onClick={() => onRename(s)} className="text-xs">
                        <Pencil className="mr-2 size-3" />
                        Rename
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onArchive(s.id)} className="text-xs">
                        <Archive className="mr-2 size-3" />
                        Archive
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onClick={() => onDelete(s.id)}
                        className="text-xs text-red-400 focus:text-red-400"
                      >
                        <Trash2 className="mr-2 size-3" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
