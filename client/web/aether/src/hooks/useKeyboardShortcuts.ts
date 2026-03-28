import { useEffect, useCallback } from "react";
import { useRouterState } from "@tanstack/react-router";

interface ShortcutHandlers {
  onToggleSessions?: () => void;
  onNewChat?: () => void;
  onOpenCommandPalette?: () => void;
  onOpenShortcutsHelp?: () => void;
  onOpenChat?: () => void;
  onOpenDevices?: () => void;
  onOpenMemory?: () => void;
  onOpenIntegrations?: () => void;
  onOpenSkills?: () => void;
  onOpenAccount?: () => void;
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
      const meta = e.metaKey || e.ctrlKey;
      const typing = isInputFocused();
      const key = e.key.toLowerCase();

      if (meta && key === "k") {
        e.preventDefault();
        handlers.onOpenCommandPalette?.();
        return;
      }

      if (meta && key === "b") {
        e.preventDefault();
        handlers.onToggleSessions?.();
        return;
      }

      if (meta && key === "n") {
        e.preventDefault();
        handlers.onNewChat?.();
        return;
      }

      if (meta && key === "c") {
        e.preventDefault();
        handlers.onOpenChat?.();
        return;
      }

      if (meta && key === "d") {
        e.preventDefault();
        handlers.onOpenDevices?.();
        return;
      }

      if (meta && key === "m") {
        e.preventDefault();
        handlers.onOpenMemory?.();
        return;
      }

      if (meta && key === "p") {
        e.preventDefault();
        handlers.onOpenIntegrations?.();
        return;
      }

      if (meta && key === "i") {
        e.preventDefault();
        handlers.onOpenSkills?.();
        return;
      }

      if (meta && key === "s") {
        e.preventDefault();
        handlers.onOpenAccount?.();
        return;
      }

      if (typing) return;

      if (e.key === "Escape") {
        (document.activeElement as HTMLElement)?.blur();
        return;
      }

      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        handlers.onOpenShortcutsHelp?.();
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
