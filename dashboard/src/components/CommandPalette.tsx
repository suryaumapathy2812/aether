"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import {
  MessageCircle,
  Settings,
  Zap,
  Brain,
  Smartphone,
  Plus,
  PanelLeftClose,
  Keyboard,
} from "lucide-react";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onToggleSidebar?: () => void;
  onNewChat?: () => void;
  onOpenShortcutsHelp?: () => void;
}

export default function CommandPalette({
  open,
  onOpenChange,
  onToggleSidebar,
  onNewChat,
  onOpenShortcutsHelp,
}: CommandPaletteProps) {
  const router = useRouter();

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
            <Plus className="size-4 mr-2 text-muted-foreground" />
            New Chat
            <span className="ml-auto text-[11px] text-muted-foreground/60">⌘N</span>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => onToggleSidebar?.())}
          >
            <PanelLeftClose className="size-4 mr-2 text-muted-foreground" />
            Toggle Sidebar
            <span className="ml-auto text-[11px] text-muted-foreground/60">⌘B</span>
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => onOpenShortcutsHelp?.())}
          >
            <Keyboard className="size-4 mr-2 text-muted-foreground" />
            Keyboard Shortcuts
            <span className="ml-auto text-[11px] text-muted-foreground/60">?</span>
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Navigate">
          <CommandItem onSelect={() => runAction(() => router.push("/chat"))}>
            <MessageCircle className="size-4 mr-2 text-muted-foreground" />
            Chat
            <span className="ml-auto text-[11px] text-muted-foreground/60">G C</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => router.push("/plugins"))}>
            <Zap className="size-4 mr-2 text-muted-foreground" />
            Connections
            <span className="ml-auto text-[11px] text-muted-foreground/60">G P</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => router.push("/devices"))}>
            <Smartphone className="size-4 mr-2 text-muted-foreground" />
            Devices
            <span className="ml-auto text-[11px] text-muted-foreground/60">G D</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => router.push("/memory"))}>
            <Brain className="size-4 mr-2 text-muted-foreground" />
            Memory
            <span className="ml-auto text-[11px] text-muted-foreground/60">G M</span>
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => router.push("/account"))}>
            <Settings className="size-4 mr-2 text-muted-foreground" />
            Settings
            <span className="ml-auto text-[11px] text-muted-foreground/60">G S</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
