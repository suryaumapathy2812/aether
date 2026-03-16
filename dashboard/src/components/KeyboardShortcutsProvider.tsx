"use client";

import { useState, useCallback, createContext, useContext } from "react";
import { useRouter } from "next/navigation";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import CommandPalette from "@/components/CommandPalette";
import ShortcutsHelp from "@/components/ShortcutsHelp";

interface ShortcutsContext {
  toggleSidebar: () => void;
  setSidebarToggle: (fn: () => void) => void;
}

const ShortcutsCtx = createContext<ShortcutsContext>({
  toggleSidebar: () => {},
  setSidebarToggle: () => {},
});

export const useShortcutsContext = () => useContext(ShortcutsCtx);

export default function KeyboardShortcutsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [sidebarToggleFn, setSidebarToggleFn] = useState<(() => void) | null>(null);

  const toggleSidebar = useCallback(() => {
    sidebarToggleFn?.();
  }, [sidebarToggleFn]);

  const newChat = useCallback(() => {
    router.push("/chat");
  }, [router]);

  useKeyboardShortcuts({
    onOpenCommandPalette: () => setPaletteOpen(true),
    onToggleSidebar: toggleSidebar,
    onNewChat: newChat,
    onOpenShortcutsHelp: () => setHelpOpen(true),
  });

  return (
    <ShortcutsCtx.Provider
      value={{
        toggleSidebar,
        setSidebarToggle: (fn) => setSidebarToggleFn(() => fn),
      }}
    >
      {children}
      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        onToggleSidebar={toggleSidebar}
        onNewChat={newChat}
        onOpenShortcutsHelp={() => {
          setPaletteOpen(false);
          setTimeout(() => setHelpOpen(true), 100);
        }}
      />
      <ShortcutsHelp open={helpOpen} onOpenChange={setHelpOpen} />
    </ShortcutsCtx.Provider>
  );
}
