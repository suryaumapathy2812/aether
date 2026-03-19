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
  Sparkles,
  PanelLeftClose,
  PanelLeft,
  Plus,
  MoreHorizontal,
  Pencil,
  Archive,
  Trash2,
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { chatRuntime, useChatStatusMap } from "@/lib/chat-runtime";

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

  // Hide main content when mobile sidebar is open
  useEffect(() => {
    const main = document.getElementById("app-main");
    if (!main) return;
    if (mobileOpen) {
      main.style.display = "none";
    } else {
      main.style.display = "";
    }
    return () => { main.style.display = ""; };
  }, [mobileOpen]);

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

  const topNavItems = [
    { href: "/devices", label: "Devices", icon: Smartphone },
    { href: "/memory", label: "Memory", icon: Brain },
    { href: "/plugins", label: "Plugins", icon: Zap },
    { href: "/skills", label: "Skills", icon: Sparkles },
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
            className="flex-1 min-w-0 text-left px-3 py-2.5 text-[13px] text-foreground/60 truncate"
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
              <button className="shrink-0 w-6 h-6 flex items-center justify-center rounded opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity text-muted-foreground/50 hover:text-muted-foreground">
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
        <p className="px-3 pb-1.5 text-[10px] uppercase tracking-[0.12em] text-foreground/40 font-medium">
          {label}
        </p>
        <div className="space-y-0.5">{items.map(renderSession)}</div>
      </div>
    );
  };

    return (
    <>
      {/* ── Mobile: header + full-screen sidebar ── */}
      {pathname !== "/" && (
        <div className="md:hidden shrink-0 h-12 flex items-center justify-between px-4 bg-[#0a0a0a] border-b border-white/[0.06]">
          <span className="logo-wordmark text-[11px] text-foreground/70 font-medium">aether</span>
          <button
            onClick={() => setMobileOpen((o) => !o)}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-foreground/70"
          >
            {mobileOpen ? <PanelLeftClose className="size-4" /> : <PanelLeft className="size-4" />}
          </button>
        </div>
      )}

      {mobileOpen && (
        <aside className="md:hidden flex-1 min-h-0 overflow-y-auto bg-[#0a0a0a]">
          <div className="flex flex-col h-full">
            {/* New Chat */}
            <div className="px-3 pt-3 pb-2">
              <button
                onClick={handleNewChat}
                className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] font-medium text-foreground hover:bg-white/[0.06] transition-colors"
                aria-label="New Chat"
              >
                <Plus className="size-4 shrink-0" />
                New Chat
              </button>
            </div>

            {/* Top Nav */}
            <div className="px-3 pb-2 space-y-0.5">
              {topNavItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] transition-colors",
                    pathname.startsWith(item.href)
                      ? "text-foreground bg-white/[0.06]"
                      : "text-foreground/70 hover:text-foreground hover:bg-white/[0.04]"
                  )}
                >
                  <item.icon className="size-4 shrink-0" />
                  {item.label}
                </Link>
              ))}
            </div>

            {/* Sessions */}
            {sessions.length > 0 && (
              <div className="flex-1 overflow-y-auto px-3 pt-2">
                {renderGroup("Today", todaySessions)}
                {renderGroup("Yesterday", yesterdaySessions)}
                {renderGroup("Previous", olderSessions)}
              </div>
            )}

            {/* Profile */}
            <div className="mt-auto border-t border-white/[0.06] px-3 py-3">
              <Link
                href="/account"
                onClick={() => setMobileOpen(false)}
                className={cn(
                  "flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] transition-colors",
                  pathname.startsWith("/account")
                    ? "text-foreground bg-white/[0.06]"
                    : "text-foreground/70 hover:text-foreground hover:bg-white/[0.04]"
                )}
              >
                <Avatar size="sm">
                  {session?.user?.image && (
                    <AvatarImage src={session.user.image} alt={session?.user?.name || ""} />
                  )}
                  <AvatarFallback>
                    {(session?.user?.name || session?.user?.email || "U").charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <span className="truncate">{session?.user?.name || "Profile"}</span>
              </Link>
            </div>
          </div>
        </aside>
      )}

      {/* ── Desktop: standard sidebar ── */}
      <aside
        className={cn(
          "h-full shrink-0 bg-white/[0.02] border-r border-white/[0.06] transition-all duration-200",
          "hidden md:block",
          collapsed ? "w-[60px]" : "w-[240px]"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header — hidden on mobile since the fixed top bar handles it */}
          <div className={cn("hidden md:flex items-center px-4 pt-5 pb-4 min-w-0", collapsed ? "justify-center" : "justify-between")}>
            {!collapsed && (
              <span className="logo-wordmark text-[11px] text-foreground/70 font-medium truncate">
                aether
              </span>
            )}
            <button
              onClick={toggle}
              className="flex items-center justify-center w-7 h-7 rounded-md text-foreground/40 hover:text-foreground/70 hover:bg-white/[0.04] transition-colors"
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
                    "w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-colors",
                    "text-foreground hover:bg-white/[0.06]",
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

          {/* Top Nav */}
          <div className={cn("px-3 pb-2 space-y-0.5", collapsed && "flex flex-col items-center")}>
            {topNavItems.map((item) => (
              <Tooltip key={item.href}>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                    "flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] transition-colors",
                    pathname.startsWith(item.href)
                      ? "text-foreground bg-white/[0.06]"
                      : "text-foreground/70 hover:text-foreground hover:bg-white/[0.04]",
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

          {/* Profile */}
          <div className="mt-auto border-t border-white/[0.06] px-3 py-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  href="/account"
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] transition-colors",
                    pathname.startsWith("/account")
                      ? "text-foreground bg-white/[0.06]"
                      : "text-foreground/70 hover:text-foreground hover:bg-white/[0.04]",
                    collapsed && "justify-center px-0"
                  )}
                  aria-label="Profile"
                >
                  <Avatar size="sm">
                    {session?.user?.image && (
                      <AvatarImage src={session.user.image} alt={session?.user?.name || ""} />
                    )}
                    <AvatarFallback>
                      {(session?.user?.name || session?.user?.email || "U").charAt(0).toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  {!collapsed && (
                    <span className="truncate">{session?.user?.name || "Profile"}</span>
                  )}
                </Link>
              </TooltipTrigger>
              {collapsed && (
                <TooltipContent side="right">
                  {session?.user?.name || "Profile"}
                </TooltipContent>
              )}
            </Tooltip>
          </div>
        </div>
      </aside>
    </>
  );
}
