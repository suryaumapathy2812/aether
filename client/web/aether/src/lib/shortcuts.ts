export function isMacShortcutPlatform(): boolean {
  if (typeof navigator === "undefined") return true;
  return /Mac|iPhone|iPad|iPod/i.test(navigator.platform);
}

export function getCommandPaletteShortcutKeys(): readonly string[] {
  return isMacShortcutPlatform() ? ["⌘", "K"] : ["Ctrl", "K"];
}

export function getCommandPaletteShortcutLabel(): string {
  return getCommandPaletteShortcutKeys().join(" ");
}

export function matchesCommandPaletteShortcut(event: KeyboardEvent): boolean {
  const key = event.key.toLowerCase();
  if (key !== "k") return false;

  if (isMacShortcutPlatform()) {
    return event.metaKey && !event.ctrlKey && !event.altKey;
  }

  return event.ctrlKey && !event.metaKey && !event.altKey;
}
