"use client";

import { useEffect, useState, useCallback, useRef, Suspense } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import { useSession } from "@/lib/auth-client";
import { useShortcutsContext } from "@/components/KeyboardShortcutsProvider";
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  MessageCircle,
  Brain,
  Zap,
  Smartphone,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Plus,
  MoreHorizontal,
  Pencil,
  Archive,
  Trash2,
} from "lucide-react";
import { useChatStatusMap } from "@/lib/chat-runtime";

export default function Sidebar() {
  return (
    <Suspense>
      <SidebarInner />
    </Suspense>
  );
}

function SidebarInner() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session } = useSession();
  const { setSidebarToggle } = useShortcutsContext();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

  const activeSessionId = searchParams.get("s") || "";
  const statusMap = useChatStatusMap();

  const toggle = useCallback(() => setCollapsed((c) => !c), []);

  useEffect(() => {
    setSidebarToggle(toggle);
  }, [toggle, setSidebarToggle]);

  const loadSessions = useCallback(() => {
    if (!session?.user?.id) return;
    listChatSessions(session.user.id, 30)
      .then((res) => setSessions(res.sessions || []))
      .catch(() => {});
  }, [session?.user?.id]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions, pathname, activeSessionId]);

  async function handleNewChat() {
    if (!session?.user?.id) return;
    try {
      const newSess = await createChatSession(session.user.id);
      setSessions((prev) => [newSess, ...prev]);
      router.push(`/chat?s=${newSess.id}`);
    } catch {
      router.push("/chat");
    }
    setMobileOpen(false);
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
    if (activeSessionId === id) {
      router.push("/chat");
    }
  }

  async function handleDelete(id: string) {
    await deleteChatSession(id).catch(() => {});
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      router.push("/chat");
    }
  }

  if (pathname === "/") return null;

  const navItems = [
    { href: "/plugins", label: "Plugins", icon: Zap },
    { href: "/devices", label: "Devices", icon: Smartphone },
    { href: "/memory", label: "Memory", icon: Brain },
    { href: "/account", label: "Settings", icon: Settings },
  ];

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const todaySessions = sessions.filter((s) => new Date(s.updated_at) >= today);
  const yesterdaySessions = sessions.filter((s) => {
    const d = new Date(s.updated_at);
    return d >= yesterday && d < today;
  });
  const olderSessions = sessions.filter((s) => new Date(s.updated_at) < yesterday);

  const renderSession = (s: ChatSession) => {
    const isActive = s.id === activeSessionId;
    const isEditing = s.id === editingId;
    const isRunning = statusMap[s.id] === "streaming";

    return (
      <div
        key={s.id}
        className={cn(
          "group flex items-center gap-1 rounded-md transition-colors",
          isActive ? "bg-white/[0.06]" : "hover:bg-white/[0.04]"
        )}
      >
        {isEditing ? (
          <input
            ref={editInputRef}
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") setEditingId(null);
            }}
            className="flex-1 min-w-0 px-3 py-1.5 text-[13px] bg-transparent text-foreground outline-none"
          />
        ) : (
          <button
            onClick={() => {
              router.push(`/chat?s=${s.id}`);
              setMobileOpen(false);
            }}
            onDoubleClick={() => startRename(s)}
            className="flex-1 min-w-0 text-left px-3 py-1.5 text-[13px] text-muted-foreground truncate"
          >
            {s.title || "New chat"}
          </button>
        )}

        {isRunning && !isEditing && (
          <span className="shrink-0 mr-1 inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
        )}

        {!isEditing && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="shrink-0 w-6 h-6 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground/50 hover:text-muted-foreground">
                <MoreHorizontal className="size-3.5" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-36">
              <DropdownMenuItem onClick={() => startRename(s)} className="text-xs">
                <Pencil className="size-3 mr-2" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleArchive(s.id)} className="text-xs">
                <Archive className="size-3 mr-2" />
                Archive
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => handleDelete(s.id)} className="text-xs text-red-400 focus:text-red-400">
                <Trash2 className="size-3 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
    );
  };

  const renderCollapsedSession = (s: ChatSession) => {
    const isActive = s.id === activeSessionId;
    const isRunning = statusMap[s.id] === "streaming";
    const label = (s.title || "New chat").trim();
    const glyph = label.charAt(0).toUpperCase() || "C";

    return (
      <Tooltip key={s.id}>
        <TooltipTrigger asChild>
          <button
            onClick={() => {
              router.push(`/chat?s=${s.id}`);
              setMobileOpen(false);
            }}
            className={cn(
              "w-8 h-8 rounded-md flex items-center justify-center text-[11px] font-medium transition-colors",
              isActive
                ? "bg-white/[0.1] text-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-white/[0.06]"
            )}
            aria-label={label}
          >
            <span className="relative inline-flex items-center justify-center">
              {glyph}
              {isRunning && (
                <span className="absolute -top-1 -right-1 h-1.5 w-1.5 rounded-full bg-emerald-400" />
              )}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="right">{label}</TooltipContent>
      </Tooltip>
    );
  };

  const renderGroup = (label: string, items: ChatSession[]) => {
    if (items.length === 0) return null;
    return (
      <div className="mb-3">
        <p className="px-3 pb-1.5 text-[10px] uppercase tracking-[0.12em] text-muted-foreground/50 font-medium">
          {label}
        </p>
        <div className="space-y-0.5">{items.map(renderSession)}</div>
      </div>
    );
  };

  return (
    <>
      <button
        onClick={() => setMobileOpen(true)}
        className="md:hidden fixed top-5 left-5 z-50 w-8 h-8 flex items-center justify-center rounded-lg bg-black/40 backdrop-blur-sm border border-white/[0.08] text-muted-foreground"
        style={{ display: pathname === "/" ? "none" : undefined }}
      >
        <MessageCircle className="size-3.5" />
      </button>

      {mobileOpen && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={cn(
          "h-full shrink-0 bg-white/[0.02] border-r border-white/[0.06] transition-all duration-200",
          "hidden md:block",
          collapsed ? "w-[60px]" : "w-[240px]",
          mobileOpen && "!block fixed inset-y-0 left-0 z-50 w-[260px] bg-[#111111]"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className={cn("flex items-center px-4 pt-5 pb-4 min-w-0", collapsed ? "justify-center" : "justify-between")}>
            {!collapsed && (
              <span className="logo-wordmark text-[11px] text-muted-foreground font-medium truncate">
                aether
              </span>
            )}
            <button
              onClick={toggle}
              className="hidden md:flex items-center justify-center w-7 h-7 rounded-md text-muted-foreground/50 hover:text-muted-foreground hover:bg-white/[0.04] transition-colors"
            >
              {collapsed ? <PanelLeft className="size-4" /> : <PanelLeftClose className="size-4" />}
            </button>
          </div>

          {/* New Chat */}
          <div className="px-3 pb-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={handleNewChat}
                  className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] font-medium transition-colors",
                    "text-foreground/80 hover:bg-white/[0.06]",
                    collapsed && "justify-center px-0"
                  )}
                  aria-label="New Chat"
                >
                  <Plus className="size-4 shrink-0" />
                  {!collapsed && "New Chat"}
                </button>
              </TooltipTrigger>
              {collapsed && <TooltipContent side="right">New Chat</TooltipContent>}
            </Tooltip>
          </div>

          {/* Sessions */}
          {!collapsed && sessions.length > 0 && (
            <div className="flex-1 overflow-y-auto px-3 pt-2">
              {renderGroup("Today", todaySessions)}
              {renderGroup("Yesterday", yesterdaySessions)}
              {renderGroup("Previous", olderSessions)}
            </div>
          )}
          {collapsed && (
            <div className="flex-1 overflow-y-auto px-3 pt-2">
              <div className="space-y-1.5">
                {sessions.slice(0, 12).map(renderCollapsedSession)}
              </div>
            </div>
          )}

          {/* Nav */}
          <div className="mt-auto border-t border-white/[0.06] px-3 py-3 space-y-0.5">
            {navItems.map((item) => (
              <Tooltip key={item.href}>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                      "flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] transition-colors",
                      pathname.startsWith(item.href)
                        ? "text-foreground/90 bg-white/[0.06]"
                        : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]",
                      collapsed && "justify-center px-0"
                    )}
                    aria-label={item.label}
                  >
                    <item.icon className="size-4 shrink-0" />
                    {!collapsed && item.label}
                  </Link>
                </TooltipTrigger>
                {collapsed && <TooltipContent side="right">{item.label}</TooltipContent>}
              </Tooltip>
            ))}
          </div>
        </div>
      </aside>
    </>
  );
}
