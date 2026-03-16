"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import { useSession } from "@/lib/auth-client";
import { useShortcutsContext } from "@/components/KeyboardShortcutsProvider";
import { listChatSessions, createChatSession, type ChatSession } from "@/lib/api";
import {
  MessageCircle,
  Brain,
  Zap,
  Smartphone,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Plus,
} from "lucide-react";

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session } = useSession();
  const { setSidebarToggle } = useShortcutsContext();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);

  const activeSessionId = searchParams.get("s") || "";

  const toggle = useCallback(() => setCollapsed((c) => !c), []);

  useEffect(() => {
    setSidebarToggle(toggle);
  }, [toggle, setSidebarToggle]);

  // Load sessions
  useEffect(() => {
    if (!session?.user?.id) return;
    listChatSessions(session.user.id, 30)
      .then((res) => setSessions(res.sessions || []))
      .catch(() => {});
  }, [session?.user?.id, pathname]);

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

  // Hide on login page
  if (pathname === "/") return null;

  const navItems = [
    { href: "/plugins", label: "Plugins", icon: Zap },
    { href: "/devices", label: "Devices", icon: Smartphone },
    { href: "/memory", label: "Memory", icon: Brain },
    { href: "/account", label: "Settings", icon: Settings },
  ];

  // Group sessions by date
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);

  const todaySessions = sessions.filter((s) => new Date(s.updated_at) >= today);
  const yesterdaySessions = sessions.filter((s) => {
    const d = new Date(s.updated_at);
    return d >= yesterday && d < today;
  });
  const olderSessions = sessions.filter((s) => new Date(s.updated_at) < yesterday);

  const sidebarContent = (
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
        <button
          onClick={handleNewChat}
          className={cn(
            "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] font-medium transition-colors",
            "text-foreground/80 hover:bg-white/[0.06]",
            collapsed && "justify-center px-0"
          )}
        >
          <Plus className="size-4 shrink-0" />
          {!collapsed && "New Chat"}
        </button>
      </div>

      {/* Sessions */}
      {!collapsed && sessions.length > 0 && (
        <div className="flex-1 overflow-y-auto px-3 pt-2">
          <SessionGroup label="Today" sessions={todaySessions} activeId={activeSessionId} onSelect={(id) => { router.push(`/chat?s=${id}`); setMobileOpen(false); }} />
          <SessionGroup label="Yesterday" sessions={yesterdaySessions} activeId={activeSessionId} onSelect={(id) => { router.push(`/chat?s=${id}`); setMobileOpen(false); }} />
          <SessionGroup label="Previous" sessions={olderSessions} activeId={activeSessionId} onSelect={(id) => { router.push(`/chat?s=${id}`); setMobileOpen(false); }} />
        </div>
      )}
      {collapsed && <div className="flex-1" />}

      {/* Nav */}
      <div className="mt-auto border-t border-white/[0.06] px-3 py-3 space-y-0.5">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            onClick={() => setMobileOpen(false)}
            className={cn(
              "flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] transition-colors",
              pathname.startsWith(item.href)
                ? "text-foreground/90 bg-white/[0.06]"
                : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]",
              collapsed && "justify-center px-0"
            )}
          >
            <item.icon className="size-4 shrink-0" />
            {!collapsed && item.label}
          </Link>
        ))}
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile hamburger */}
      <button
        onClick={() => setMobileOpen(true)}
        className="md:hidden fixed top-5 left-5 z-50 w-8 h-8 flex items-center justify-center rounded-lg bg-black/40 backdrop-blur-sm border border-white/[0.08] text-muted-foreground"
        style={{ display: pathname === "/" ? "none" : undefined }}
      >
        <MessageCircle className="size-3.5" />
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "h-full shrink-0 bg-white/[0.02] border-r border-white/[0.06] transition-all duration-200",
          "hidden md:block",
          collapsed ? "w-[60px]" : "w-[240px]",
          mobileOpen && "!block fixed inset-y-0 left-0 z-50 w-[260px] bg-[#111111]"
        )}
      >
        {sidebarContent}
      </aside>
    </>
  );
}

function SessionGroup({
  label,
  sessions,
  activeId,
  onSelect,
}: {
  label: string;
  sessions: ChatSession[];
  activeId: string;
  onSelect: (id: string) => void;
}) {
  if (sessions.length === 0) return null;
  return (
    <div className="mb-3">
      <p className="px-3 pb-1.5 text-[10px] uppercase tracking-[0.12em] text-muted-foreground/50 font-medium">
        {label}
      </p>
      <div className="space-y-0.5">
        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={cn(
              "w-full text-left px-3 py-1.5 rounded-md text-[13px] transition-colors truncate",
              s.id === activeId
                ? "text-foreground bg-white/[0.06]"
                : "text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04]"
            )}
          >
            {s.title || "New chat"}
          </button>
        ))}
      </div>
    </div>
  );
}
