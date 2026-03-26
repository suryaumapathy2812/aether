import { useEffect, useRef, useCallback } from "react";
import { useNavigate, useRouterState } from "@tanstack/react-router";

interface ShortcutHandlers {
  onToggleSessions?: () => void;
  onNewChat?: () => void;
  onOpenCommandPalette?: () => void;
  onOpenShortcutsHelp?: () => void;
}

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || (el as HTMLElement).isContentEditable;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const pendingG = useRef(false);
  const gTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      const typing = isInputFocused();

      if (meta && e.key === "k") {
        e.preventDefault();
        handlers.onOpenCommandPalette?.();
        return;
      }

      if (meta && e.key === "b") {
        e.preventDefault();
        handlers.onToggleSessions?.();
        return;
      }

      if (meta && e.key === "n") {
        e.preventDefault();
        handlers.onNewChat?.();
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

      if (e.key === "g" && !pendingG.current) {
        pendingG.current = true;
        clearTimeout(gTimer.current);
        gTimer.current = setTimeout(() => {
          pendingG.current = false;
        }, 800);
        return;
      }

      if (pendingG.current) {
        pendingG.current = false;
        clearTimeout(gTimer.current);
        switch (e.key) {
          case "c":
            navigate({ to: "/chat" });
            break;
          case "s":
            navigate({ to: "/account" });
            break;
          case "p":
            navigate({ to: "/integrations" });
            break;
          case "d":
            navigate({ to: "/devices" });
            break;
          case "m":
            navigate({ to: "/memory" });
            break;
        }
        return;
      }
    },
    [handlers, navigate, pathname]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      clearTimeout(gTimer.current);
    };
  }, [handleKeyDown]);
}
