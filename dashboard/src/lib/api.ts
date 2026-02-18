/**
 * Orchestrator API client.
 *
 * With Caddy reverse proxy, the orchestrator is on the same origin as the dashboard.
 * Caddy routes /agents/*, /pair/*, /devices, /services/*, /memory/*, /health, /ws
 * to the orchestrator. Everything else (including /api/auth/*) goes to Next.js.
 *
 * Auth: session token from better-auth sent as Authorization: Bearer header.
 * This works both same-origin (via Caddy) and cross-origin (direct access).
 */

const ORCHESTRATOR_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "";

// ── Session token management ──
// Set by SessionSync component which reads from useSession().
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
  // Same-origin: use current page's protocol/host
  // Cross-origin: use ORCHESTRATOR_URL
  if (ORCHESTRATOR_URL) {
    const base = ORCHESTRATOR_URL.replace("http", "ws");
    return `${base}/ws?token=${token}`;
  }
  // Same-origin via Caddy — derive WS URL from current page
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws?token=${token}`;
  }
  // SSR fallback
  return `ws://localhost:3000/ws?token=${token}`;
}
