"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { useSession } from "@/lib/auth-client";
import { useShortcutsContext } from "@/components/KeyboardShortcutsProvider";
import { getMemoryConversations } from "@/lib/api";
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

interface ConversationItem {
  id: number;
  user_message: string;
  timestamp: number;
}

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { data: session } = useSession();
  const { setSidebarToggle } = useShortcutsContext();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);

  const toggle = useCallback(() => setCollapsed((c) => !c), []);

  useEffect(() => {
    setSidebarToggle(toggle);
  }, [toggle, setSidebarToggle]);

  useEffect(() => {
    if (!session?.user?.id) return;
    getMemoryConversations(session.user.id, 20)
      .then((res) => {
        const startOfToday = new Date();
        startOfToday.setHours(0, 0, 0, 0);
        const todaySec = Math.floor(startOfToday.getTime() / 1000);
        const items = (res.conversations || [])
          .filter((c: ConversationItem) => (c.timestamp || 0) >= todaySec && c.user_message?.trim())
          .sort((a: ConversationItem, b: ConversationItem) => (b.timestamp || 0) - (a.timestamp || 0))
          .slice(0, 12);
        setConversations(items);
      })
      .catch(() => {});
  }, [session?.user?.id]);

  // Hide on login page
  if (pathname === "/") return null;

  const navItems = [
    { href: "/plugins", label: "Plugins", icon: Zap },
    { href: "/devices", label: "Devices", icon: Smartphone },
    { href: "/memory", label: "Memory", icon: Brain },
    { href: "/account", label: "Settings", icon: Settings },
  ];

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
          onClick={() => {
            router.push("/chat");
            setMobileOpen(false);
          }}
          className={cn(
            "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] font-medium transition-colors",
            "text-foreground/80 hover:bg-white/[0.06]",
            pathname === "/chat" && "bg-white/[0.06]",
            collapsed && "justify-center px-0"
          )}
        >
          <Plus className="size-4 shrink-0" />
          {!collapsed && "New Chat"}
        </button>
      </div>

      {/* Conversations */}
      {!collapsed && conversations.length > 0 && (
        <div className="flex-1 overflow-y-auto px-3 pt-2">
          <p className="px-3 pb-2 text-[10px] uppercase tracking-[0.12em] text-muted-foreground/60 font-medium">
            Today
          </p>
          <div className="space-y-0.5">
            {conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => {
                  router.push("/chat");
                  setMobileOpen(false);
                }}
                className="w-full text-left px-3 py-1.5 rounded-md text-[13px] text-muted-foreground hover:text-foreground/80 hover:bg-white/[0.04] transition-colors truncate"
              >
                {c.user_message.length > 40 ? c.user_message.slice(0, 40) + "..." : c.user_message}
              </button>
            ))}
          </div>
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
          // Desktop
          "hidden md:block",
          collapsed ? "w-[60px]" : "w-[240px]",
          // Mobile
          mobileOpen && "!block fixed inset-y-0 left-0 z-50 w-[260px] bg-[#111111]"
        )}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
