"use client";

import { Suspense, useEffect, useRef, useState, useCallback } from "react";
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Brain,
  Zap,
  Smartphone,
  Sparkles,
  Plus,
  MoreHorizontal,
  Pencil,
  Archive,
  Trash2,
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Sidebar as SidebarShell,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { chatRuntime, useChatStatusMap } from "@/lib/chat-runtime";
import { Separator } from "./ui/separator";

type AppSidebarProps = {
  variant?: "sidebar" | "floating" | "inset";
};

export default function Sidebar({ variant = "floating" }: AppSidebarProps) {
  return (
    <Suspense>
      <SidebarInner variant={variant} />
    </Suspense>
  );
}

function SidebarInner({ variant }: AppSidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session } = useSession();
  const { setSidebarToggle } = useShortcutsContext();
  const { state, isMobile, setOpenMobile, toggleSidebar } = useSidebar();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

  const collapsed = state === "collapsed";
  const activeSessionId = searchParams.get("s") || "";
  const statusMap = useChatStatusMap();

  useEffect(() => {
    setSidebarToggle(toggleSidebar);
  }, [setSidebarToggle, toggleSidebar]);

  const closeMobileSidebar = useCallback(() => {
    if (isMobile) {
      setOpenMobile(false);
    }
  }, [isMobile, setOpenMobile]);

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

    closeMobileSidebar();
  }

  function startRename(sessionToRename: ChatSession) {
    setEditingId(sessionToRename.id);
    setEditTitle(sessionToRename.title);
    setTimeout(() => editInputRef.current?.focus(), 50);
  }

  async function commitRename() {
    if (!editingId) return;

    const trimmed = editTitle.trim();
    if (trimmed) {
      await updateChatSessionTitle(editingId, trimmed).catch(() => {});
      setSessions((prev) =>
        prev.map((sessionItem) =>
          sessionItem.id === editingId ? { ...sessionItem, title: trimmed } : sessionItem,
        ),
      );
    }

    setEditingId(null);
  }

  async function handleArchive(id: string) {
    await archiveChatSession(id).catch(() => {});
    setSessions((prev) => prev.filter((sessionItem) => sessionItem.id !== id));

    if (activeSessionId === id) {
      router.push("/chat");
    }
  }

  async function handleDelete(id: string) {
    await deleteChatSession(id).catch(() => {});
    setSessions((prev) => prev.filter((sessionItem) => sessionItem.id !== id));

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

  const todaySessions = sessions.filter((sessionItem) => new Date(sessionItem.updated_at) >= today);
  const yesterdaySessions = sessions.filter((sessionItem) => {
    const updatedAt = new Date(sessionItem.updated_at);
    return updatedAt >= yesterday && updatedAt < today;
  });
  const olderSessions = sessions.filter(
    (sessionItem) => new Date(sessionItem.updated_at) < yesterday,
  );

  const renderSession = (sessionItem: ChatSession) => {
    const isActive = sessionItem.id === activeSessionId;
    const isEditing = sessionItem.id === editingId;
    const isRunning = statusMap[sessionItem.id] === "streaming";

    return (
      <SidebarMenuItem
        key={sessionItem.id}
        className={cn(
          "group/menu-item rounded-md transition-colors",
          isActive ? "bg-white/[0.06]" : "hover:bg-white/[0.04]",
        )}
      >
        {isEditing ? (
          <input
            ref={editInputRef}
            value={editTitle}
            onChange={(event) => setEditTitle(event.target.value)}
            onBlur={commitRename}
            onKeyDown={(event) => {
              if (event.key === "Enter") void commitRename();
              if (event.key === "Escape") setEditingId(null);
            }}
            className="h-8 min-w-0 w-full bg-transparent px-3 text-sm font-bold text-foreground outline-none cursor-pointer"
          />
        ) : (
          <SidebarMenuButton
            isActive={isActive}
            tooltip={sessionItem.title || "New chat"}
            onDoubleClick={() => startRename(sessionItem)}
            onClick={() => {
              router.push(`/chat?s=${sessionItem.id}`);
              closeMobileSidebar();
            }}
            className={cn(
              "h-8 px-3 text-sm font-bold cursor-pointer",
              isActive
                ? "bg-transparent text-foreground hover:bg-transparent hover:text-foreground"
                : "text-foreground/60 hover:bg-transparent hover:text-foreground",
            )}
          >
            <span>{sessionItem.title || "New chat"}</span>
          </SidebarMenuButton>
        )}

        {!isEditing && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <SidebarMenuAction
                showOnHover
                className=" text-muted-foreground/50 hover:bg-white/6 hover:text-muted-foreground"
              >
                {isRunning ? (
                  <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
                ) : (
                  <MoreHorizontal className="size-3.5" />
                )}
              </SidebarMenuAction>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-36">
              <DropdownMenuItem onClick={() => startRename(sessionItem)} className="text-xs">
                <Pencil className="mr-2 size-3" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleArchive(sessionItem.id)} className="text-xs">
                <Archive className="mr-2 size-3" />
                Archive
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => handleDelete(sessionItem.id)}
                className="text-xs text-red-400 focus:text-red-400"
              >
                <Trash2 className="mr-2 size-3" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </SidebarMenuItem>
    );
  };

  const renderCollapsedSession = (sessionItem: ChatSession) => {
    const isActive = sessionItem.id === activeSessionId;
    const isRunning = statusMap[sessionItem.id] === "streaming";
    const label = (sessionItem.title || "New chat").trim();
    const glyph = label.charAt(0).toUpperCase() || "C";

    return (
      <Tooltip key={sessionItem.id}>
        <TooltipTrigger asChild>
          <SidebarMenuButton
            isActive={isActive}
            onClick={() => {
              router.push(`/chat?s=${sessionItem.id}`);
              closeMobileSidebar();
            }}
            className={cn(
              "justify-center text-xs font-bold tracking-tight cursor-pointer",
              isActive
                ? "bg-white/[0.1] text-foreground hover:bg-white/[0.1]"
                : "text-muted-foreground hover:bg-white/[0.06] hover:text-foreground",
            )}
            aria-label={label}
          >
            <span className="relative inline-flex items-center justify-center">
              <span>{glyph}</span>
              {isRunning && (
                <span className="absolute -right-1 -top-1 h-1.5 w-1.5 rounded-full bg-emerald-400" />
              )}
            </span>
          </SidebarMenuButton>
        </TooltipTrigger>
        <TooltipContent side="right">{label}</TooltipContent>
      </Tooltip>
    );
  };

  const renderSessionGroup = (label: string, items: ChatSession[]) => {
    if (items.length === 0) return null;

    return (
      <SidebarGroup className="mb-3">
        <SidebarGroupLabel>{label}</SidebarGroupLabel>
        <SidebarGroupContent className="space-y-0.5">
          <SidebarMenu className="gap-y-1.5">{items.map(renderSession)}</SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  };

  return (
    <>
      <div className="border-b border-white/[0.06] bg-[#0a0a0a] md:hidden">
        <div className="flex h-12 items-center justify-between px-4">
          <span className="logo-wordmark text-base font-bold text-foreground/70">aether</span>
          <SidebarTrigger className="h-8 w-8 rounded-lg text-foreground/70 hover:bg-white/[0.04] hover:text-foreground" />
        </div>
      </div>

      <SidebarShell collapsible="icon" variant={variant}>
        <SidebarHeader className="flex items-center justify-between px-4 pb-3 pt-4 md:hidden">
          <span className="logo-wordmark text-base font-bold text-foreground/70">aether</span>
          <SidebarTrigger className="h-8 w-8 rounded-lg text-foreground/70 hover:bg-white/[0.04] hover:text-foreground" />
        </SidebarHeader>

        <SidebarHeader
          className={cn(
            "hidden min-w-0 items-center px-2 pb-4 pt-5 md:flex flex-row",
            collapsed ? "justify-center" : "justify-between",
          )}
        >
          {!collapsed && (
            <span className="logo-wordmark truncate text-sm font-bold text-foreground/70">
              aether
            </span>
          )}
          <SidebarTrigger className="h-7 w-7 rounded-md text-foreground/40 hover:bg-white/[0.04] hover:text-foreground/70" />
        </SidebarHeader>

        <SidebarContent className="p-0">
          <SidebarGroup className="pt-3 md:pt-0">
            <SidebarGroupContent>
              <SidebarMenu className="space-y-1.5">
                <SidebarMenuItem>
                  <SidebarMenuButton
                    onClick={handleNewChat}
                    tooltip="New Chat"
                    className={cn(
                      "h-auto px-3 text-sm font-bold text-foreground",
                      collapsed && "justify-center px-2",
                    )}
                    aria-label="New Chat"
                  >
                    <Plus className="size-4 shrink-0" />
                    {!collapsed && <span>New Chat</span>}
                  </SidebarMenuButton>
                </SidebarMenuItem>

                {topNavItems.map((item) => (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      asChild
                      isActive={pathname.startsWith(item.href)}
                      tooltip={item.label}
                      className={cn(
                        "px-3 py-2.5 text-sm font-bold",
                        pathname.startsWith(item.href)
                          ? "bg-white/[0.06] text-foreground hover:bg-white/[0.06]"
                          : "text-foreground/70 hover:bg-white/[0.04] hover:text-foreground",
                        collapsed && "justify-center px-2",
                      )}
                    >
                      <Link href={item.href} onClick={closeMobileSidebar} aria-label={item.label}>
                        <item.icon className="size-4 shrink-0" />
                        {!collapsed && <span>{item.label}</span>}
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          <Separator />

          {!collapsed && sessions.length > 0 && (
            <SidebarGroup className="flex-1 pt-2">
              <SidebarGroupContent>
                {renderSessionGroup("Today", todaySessions)}
                {renderSessionGroup("Yesterday", yesterdaySessions)}
                {renderSessionGroup("Previous", olderSessions)}
              </SidebarGroupContent>
            </SidebarGroup>
          )}

          {collapsed && (
            <SidebarGroup className="flex-1 pt-2">
              <SidebarGroupContent>
                <SidebarMenu>{sessions.slice(0, 12).map(renderCollapsedSession)}</SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          )}
        </SidebarContent>

        <SidebarFooter className="mt-auto border-t border-white/6 py-4">
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                asChild
                isActive={pathname.startsWith("/account")}
                tooltip={session?.user?.name || "Profile"}
                className={cn(
                  "text-sm font-bold",
                  pathname.startsWith("/account")
                    ? "bg-white/6 text-foreground hover:bg-white/6"
                    : "text-foreground/70 hover:bg-white/4 hover:text-foreground",
                  collapsed && "justify-center",
                )}
              >
                <Link href="/account" onClick={closeMobileSidebar} aria-label="Profile">
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
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
      </SidebarShell>
    </>
  );
}
