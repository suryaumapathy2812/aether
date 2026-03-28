import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "@tanstack/react-router";
import { useTheme } from "#/components/ThemeProvider";
import { useSession } from "#/lib/auth-client";
import {
  listChatSessions,
  type ChatSession,
} from "#/lib/api";
import { getRecentChatSessionId } from "#/lib/recent-chat";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
  CommandSeparator,
} from "#/components/ui/command";
import {
  IconMessageCircle,
  IconSettings,
  IconPlugConnected,
  IconBrain,
  IconDeviceMobile,
  IconPlus,
  IconMessage,
  IconKeyboard,
  IconSun,
  IconMoon,
  IconSearch,
  IconSparkles,
} from "@tabler/icons-react";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onToggleSessions?: () => void;
  onNewChat?: () => void;
  onOpenShortcutsHelp?: () => void;
}

export default function CommandPalette({
  open,
  onOpenChange,
  onToggleSessions,
  onNewChat,
  onOpenShortcutsHelp,
}: CommandPaletteProps) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { resolvedTheme, setTheme } = useTheme();
  const { data: session } = useSession();
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);

  const isChatRoute = pathname === "/chat";
  const recentChatSessionId = session?.user?.id
    ? getRecentChatSessionId(session.user.id)
    : null;

  const openRecentChat = useCallback(() => {
    const fallbackSessionId = chatSessions[0]?.id || null;
    const targetSessionId =
      recentChatSessionId &&
      chatSessions.some((chatSession) => chatSession.id === recentChatSessionId)
        ? recentChatSessionId
        : fallbackSessionId;

    if (targetSessionId) {
      navigate({ to: "/chat", search: { s: targetSessionId } });
      return;
    }
    navigate({ to: "/chat" });
  }, [chatSessions, navigate, recentChatSessionId]);

  useEffect(() => {
    if (!open || !session?.user?.id) return;
    let cancelled = false;
    listChatSessions(session.user.id, 20)
      .then((res) => {
        if (cancelled) return;
        setChatSessions(res.sessions || []);
      })
      .catch(() => {
        if (!cancelled) setChatSessions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [open, session?.user?.id]);

  const runAction = useCallback(
    (action: () => void) => {
      onOpenChange(false);
      // Small delay so dialog closes before navigation
      setTimeout(action, 50);
    },
    [onOpenChange]
  );

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup heading="Actions">
          <CommandItem
            onSelect={() =>
              runAction(() => {
                if (isChatRoute) {
                  onNewChat?.();
                  return;
                }
                openRecentChat();
              })
            }
            value={isChatRoute ? "new chat create conversation" : "return to chat recent last conversation"}
          >
            {isChatRoute ? (
              <IconPlus className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            ) : (
              <IconMessageCircle className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            )}
            {isChatRoute ? "New Chat" : "Return to Chat"}
            <CommandShortcut>{isChatRoute ? "⌘N" : "⌘C"}</CommandShortcut>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => onToggleSessions?.())}
          >
            <IconMessage className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Sessions
            <CommandShortcut>⌘B</CommandShortcut>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => onOpenShortcutsHelp?.())}
          >
            <IconKeyboard className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Keyboard Shortcuts
            <span className="ml-auto text-sm text-muted-foreground/60">?</span>
          </CommandItem>
          <CommandItem
            onSelect={() =>
              runAction(() =>
                setTheme(resolvedTheme === "dark" ? "light" : "dark")
              )
            }
          >
            {resolvedTheme === "dark" ? (
              <IconSun className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            ) : (
              <IconMoon className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            )}
            Toggle Theme
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Navigate">
          <CommandItem onSelect={() => runAction(openRecentChat)}>
            <IconMessageCircle className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Chat
            <CommandShortcut>⌘C</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/integrations" }))}>
            <IconPlugConnected className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Connections
            <CommandShortcut>⌘P</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/devices" }))}>
            <IconDeviceMobile className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Devices
            <CommandShortcut>⌘D</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/memory" }))}>
            <IconBrain className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Memory
            <CommandShortcut>⌘M</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/skills" }))}>
            <IconSparkles className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Skills
            <CommandShortcut>⌘I</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/account" }))}>
            <IconSettings className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Settings
            <CommandShortcut>⌘S</CommandShortcut>
          </CommandItem>
        </CommandGroup>

        {chatSessions.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Chat Sessions">
              {chatSessions.map((chatSession) => (
                <CommandItem
                  key={chatSession.id}
                  value={`${chatSession.title} ${chatSession.updated_at} chat session conversation`}
                  onSelect={() =>
                    runAction(() =>
                      navigate({ to: "/chat", search: { s: chatSession.id } })
                    )
                  }
                >
                  <IconSearch className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate">{chatSession.title || "Untitled chat"}</div>
                    <div className="text-xs text-muted-foreground/60">
                      {new Date(chatSession.updated_at).toLocaleDateString()}
                    </div>
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
