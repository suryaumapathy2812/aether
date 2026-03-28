import { useEffect, useCallback } from "react";
import { useRouterState } from "@tanstack/react-router";
import { matchesCommandPaletteShortcut } from "#/lib/shortcuts";

interface ShortcutHandlers {
  onOpenCommandPalette?: () => void;
}

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || (el as HTMLElement).isContentEditable;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const typing = isInputFocused();

      if (matchesCommandPaletteShortcut(e)) {
        e.preventDefault();
        handlers.onOpenCommandPalette?.();
        return;
      }

      if (typing) return;

      if (e.key === "Escape") {
        (document.activeElement as HTMLElement)?.blur();
        return;
      }

      if (e.key === "/" && pathname === "/chat") {
        e.preventDefault();
        const textarea = document.querySelector<HTMLTextAreaElement>(
          "[data-slot='prompt-input-textarea']"
        ) || document.querySelector<HTMLTextAreaElement>("textarea");
        textarea?.focus();
        return;
      }

    },
    [handlers, pathname]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleKeyDown]);
}
