import { useState, createContext, useContext } from "react";
import { useKeyboardShortcuts } from "#/hooks/useKeyboardShortcuts";
import CommandPalette from "#/components/CommandPalette";

interface ShortcutsContext {
  openCommandPalette: () => void;
}

const ShortcutsCtx = createContext<ShortcutsContext>({
  openCommandPalette: () => {},
});

export const useShortcutsContext = () => useContext(ShortcutsCtx);

export default function KeyboardShortcutsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [paletteOpen, setPaletteOpen] = useState(false);

  useKeyboardShortcuts({
    onOpenCommandPalette: () => setPaletteOpen(true),
  });

  return (
    <ShortcutsCtx.Provider
      value={{
        openCommandPalette: () => setPaletteOpen(true),
      }}
    >
      {children}
      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
      />
    </ShortcutsCtx.Provider>
  );
}
