const RECENT_CHAT_KEY_PREFIX = "aether.recent-chat";

function storageKey(userId: string): string {
  return `${RECENT_CHAT_KEY_PREFIX}:${userId}`;
}

export function getRecentChatSessionId(userId: string): string | null {
  if (typeof window === "undefined") return null;
  const normalized = userId.trim();
  if (!normalized) return null;
  const value = localStorage.getItem(storageKey(normalized))?.trim();
  return value || null;
}

export function setRecentChatSessionId(userId: string, sessionId: string): void {
  if (typeof window === "undefined") return;
  const normalizedUserId = userId.trim();
  const normalizedSessionId = sessionId.trim();
  if (!normalizedUserId || !normalizedSessionId) return;
  localStorage.setItem(storageKey(normalizedUserId), normalizedSessionId);
}
