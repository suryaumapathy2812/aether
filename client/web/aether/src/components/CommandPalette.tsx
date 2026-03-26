import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useTheme } from "#/components/ThemeProvider";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
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
  const { theme, setTheme } = useTheme();

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
            onSelect={() => runAction(() => onNewChat?.())}
          >
            <IconPlus className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            New Chat
            <span className="ml-auto text-sm text-muted-foreground/60">⌘N</span>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => onToggleSessions?.())}
          >
            <IconMessage className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Sessions
            <span className="ml-auto text-sm text-muted-foreground/60">⌘B</span>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => onOpenShortcutsHelp?.())}
          >
            <IconKeyboard className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Keyboard Shortcuts
            <span className="ml-auto text-sm text-muted-foreground/60">?</span>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => setTheme(theme === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? (
              <IconSun className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            ) : (
              <IconMoon className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            )}
            Toggle Theme
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Navigate">
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/chat" }))}>
            <IconMessageCircle className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Chat
            <span className="ml-auto text-sm text-muted-foreground/60">G C</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/integrations" }))}>
            <IconPlugConnected className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Connections
            <span className="ml-auto text-sm text-muted-foreground/60">G P</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/devices" }))}>
            <IconDeviceMobile className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Devices
            <span className="ml-auto text-sm text-muted-foreground/60">G D</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/memory" }))}>
            <IconBrain className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Memory
            <span className="ml-auto text-sm text-muted-foreground/60">G M</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate({ to: "/account" }))}>
            <IconSettings className="size-4 mr-2 text-muted-foreground" strokeWidth={1.5} />
            Settings
            <span className="ml-auto text-sm text-muted-foreground/60">G S</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
