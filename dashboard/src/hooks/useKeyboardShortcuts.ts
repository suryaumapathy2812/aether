"use client";

import { useEffect, useRef, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";

interface ShortcutHandlers {
  onToggleSidebar?: () => void;
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

/**
 * Global keyboard shortcuts for Aether.
 *
 * ⌘K — command palette (always)
 * ⌘B — toggle sidebar (always)
 * ⌘N — new chat (always)
 * /  — focus chat input (when not typing)
 * ?  — shortcuts help (when not typing)
 * Escape — blur input
 * G then C/S/P/D/M — navigation sequences (when not typing)
 */
export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  const router = useRouter();
  const pathname = usePathname();
  const pendingG = useRef(false);
  const gTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      const typing = isInputFocused();

      // ⌘K — command palette
      if (meta && e.key === "k") {
        e.preventDefault();
        handlers.onOpenCommandPalette?.();
        return;
      }

      // ⌘B — toggle sidebar
      if (meta && e.key === "b") {
        e.preventDefault();
        handlers.onToggleSidebar?.();
        return;
      }

      // ⌘N — new chat
      if (meta && e.key === "n") {
        e.preventDefault();
        handlers.onNewChat?.();
        return;
      }

      // Everything below only works when NOT typing in an input
      if (typing) return;

      // Escape — blur active input
      if (e.key === "Escape") {
        (document.activeElement as HTMLElement)?.blur();
        return;
      }

      // ? — shortcuts help
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        handlers.onOpenShortcutsHelp?.();
        return;
      }

      // / — focus chat input
      if (e.key === "/" && pathname === "/chat") {
        e.preventDefault();
        const textarea = document.querySelector<HTMLTextAreaElement>(
          "[data-slot='prompt-input-textarea']"
        ) || document.querySelector<HTMLTextAreaElement>("textarea");
        textarea?.focus();
        return;
      }

      // G-sequence navigation
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
            router.push("/chat");
            break;
          case "s":
            router.push("/account");
            break;
          case "p":
            router.push("/plugins");
            break;
          case "d":
            router.push("/devices");
            break;
          case "m":
            router.push("/memory");
            break;
        }
        return;
      }
    },
    [handlers, router, pathname]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      clearTimeout(gTimer.current);
    };
  }, [handleKeyDown]);
}
