/**
 * Orchestrator API client.
 *
 * Auth is handled by better-auth (cookie-based sessions via /api/auth/*).
 * The session token is obtained from better-auth's useSession()/getSession()
 * and passed to the orchestrator as an Authorization: Bearer header.
 *
 * Why not cookies? Dashboard (localhost:3000) and orchestrator (localhost:9000)
 * are different origins — cross-origin cookies don't work without SameSite=None + Secure.
 * Instead, we read the session token from better-auth's client-side session data
 * and send it explicitly.
 */

const ORCHESTRATOR_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:9000";

// ── Session token management ──
// Set by components that have access to useSession() data.
// All API calls read from here.

let _sessionToken: string | null = null;

export function setSessionToken(token: string | null) {
  _sessionToken = token;
}

export function getSessionToken(): string | null {
  return _sessionToken;
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const token = _sessionToken;
  if (!token) {
    throw new Error("Not authenticated");
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  const res = await fetch(`${ORCHESTRATOR_URL}${path}`, {
    headers,
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// ── Devices ──

export async function listDevices() {
  return api<
    { id: string; name: string; device_type: string; paired_at: string }[]
  >("/devices");
}

export async function confirmPairing(code: string) {
  return api<{ device_id: string; device_token: string }>("/pair/confirm", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}

// ── Services (API keys) ──

export async function listApiKeys() {
  return api<{ provider: string; preview: string }[]>("/services/keys");
}

export async function saveApiKey(provider: string, keyValue: string) {
  return api("/services/keys", {
    method: "POST",
    body: JSON.stringify({ provider, key_value: keyValue }),
  });
}

export async function deleteApiKey(provider: string) {
  return api(`/services/keys/${provider}`, {
    method: "DELETE",
  });
}

// ── Memory ──

export async function getMemoryFacts() {
  return api<{ facts: string[] }>("/memory/facts");
}

export async function getMemorySessions() {
  return api<{
    sessions: {
      session_id: string;
      summary: string;
      started_at: number;
      ended_at: number;
      turns: number;
      tools_used: string[];
    }[];
  }>("/memory/sessions");
}

export async function getMemoryConversations(limit = 20) {
  return api<{
    conversations: {
      id: number;
      user_message: string;
      assistant_message: string;
      timestamp: number;
    }[];
  }>(`/memory/conversations?limit=${limit}`);
}

// ── WebSocket URL ──

export function getWsUrl(): string {
  const token = _sessionToken || "";
  const base = ORCHESTRATOR_URL.replace("http", "ws");
  return `${base}/ws?token=${token}`;
}
