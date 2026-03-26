import { useState, useCallback, createContext, useContext } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useKeyboardShortcuts } from "#/hooks/useKeyboardShortcuts";
import CommandPalette from "#/components/CommandPalette";
import ShortcutsHelp from "#/components/ShortcutsHelp";

interface ShortcutsContext {
  toggleSessions: () => void;
  setSessionsToggle: (fn: () => void) => void;
}

const ShortcutsCtx = createContext<ShortcutsContext>({
  toggleSessions: () => {},
  setSessionsToggle: () => {},
});

export const useShortcutsContext = () => useContext(ShortcutsCtx);

export default function KeyboardShortcutsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const navigate = useNavigate();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  const newChat = useCallback(() => {
    navigate({ to: "/chat" });
  }, [navigate]);

  const openSessions = useCallback(() => {
    navigate({ to: "/sessions" });
  }, [navigate]);

  useKeyboardShortcuts({
    onOpenCommandPalette: () => setPaletteOpen(true),
    onToggleSessions: openSessions,
    onNewChat: newChat,
    onOpenShortcutsHelp: () => setHelpOpen(true),
  });

  return (
    <ShortcutsCtx.Provider
      value={{
        toggleSessions: openSessions,
        setSessionsToggle: () => {},
      }}
    >
      {children}
      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        onToggleSessions={openSessions}
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
