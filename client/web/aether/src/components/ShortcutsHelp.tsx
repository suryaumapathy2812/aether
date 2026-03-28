import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "#/components/ui/dialog";
import { getCommandPaletteShortcutKeys } from "#/lib/shortcuts";

interface ShortcutsHelpProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const sections = [
  {
    title: "Global",
    shortcuts: [
      { keys: getCommandPaletteShortcutKeys().join(" "), label: "Open command panel" },
    ],
  },
  {
    title: "Navigation",
    shortcuts: [
      { keys: "Use Command Panel", label: "Open chat, sessions, devices, memory, skills, and settings" },
    ],
  },
  {
    title: "Chat",
    shortcuts: [
      { keys: "/", label: "Focus input" },
      { keys: "Escape", label: "Unfocus input" },
      { keys: "Enter", label: "Send message" },
    ],
  },
];

export default function ShortcutsHelp({ open, onOpenChange }: ShortcutsHelpProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[420px] bg-card border-border">
        <DialogHeader>
          <DialogTitle className="text-base font-medium">Keyboard Shortcuts</DialogTitle>
        </DialogHeader>
        <div className="space-y-6 pt-2">
          {sections.map((section) => (
            <div key={section.title}>
              <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground/60 font-medium mb-2.5">
                {section.title}
              </p>
              <div className="space-y-1.5">
                {section.shortcuts.map((s) => (
                  <div
                    key={s.keys}
                    className="flex items-center justify-between py-1"
                  >
                    <span className="text-sm text-foreground/80">{s.label}</span>
                    <div className="flex items-center gap-1">
                      {s.keys.split(" ").map((k, i) => (
                        <span key={i}>
                          {k === "→" ? (
                            <span className="text-sm text-muted-foreground/40 mx-0.5">then</span>
                          ) : (
                            <kbd className="inline-flex items-center justify-center min-w-[24px] h-[22px] px-1.5 rounded bg-secondary text-sm text-muted-foreground font-mono">
                              {k}
                            </kbd>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
