/**
 * Orchestrator API client.
 * All dashboard ↔ orchestrator communication goes through here.
 */

const ORCHESTRATOR_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:9000";

function tokenParam(): string {
  if (typeof window === "undefined") return "";
  const token = localStorage.getItem("aether_token");
  return token ? `?token=${token}` : "";
}

async function api<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${ORCHESTRATOR_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// ── Auth ──

export async function signup(email: string, password: string, name: string) {
  const data = await api<{ user_id: string; token: string }>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
  localStorage.setItem("aether_token", data.token);
  localStorage.setItem("aether_user_id", data.user_id);
  return data;
}

export async function login(email: string, password: string) {
  const data = await api<{ user_id: string; token: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  localStorage.setItem("aether_token", data.token);
  localStorage.setItem("aether_user_id", data.user_id);
  return data;
}

export async function getMe() {
  return api<{ id: string; email: string; name: string }>(
    `/auth/me${tokenParam()}`
  );
}

export function logout() {
  localStorage.removeItem("aether_token");
  localStorage.removeItem("aether_user_id");
}

export function isLoggedIn(): boolean {
  if (typeof window === "undefined") return false;
  return !!localStorage.getItem("aether_token");
}

// ── Devices ──

export async function listDevices() {
  return api<
    { id: string; name: string; device_type: string; paired_at: string }[]
  >(`/devices${tokenParam()}`);
}

export async function confirmPairing(code: string) {
  return api<{ device_id: string; device_token: string }>(
    `/pair/confirm${tokenParam()}`,
    {
      method: "POST",
      body: JSON.stringify({ code }),
    }
  );
}

// ── Services (API keys) ──

export async function listApiKeys() {
  return api<{ provider: string; preview: string }[]>(
    `/services/keys${tokenParam()}`
  );
}

export async function saveApiKey(provider: string, keyValue: string) {
  return api(`/services/keys${tokenParam()}`, {
    method: "POST",
    body: JSON.stringify({ provider, key_value: keyValue }),
  });
}

export async function deleteApiKey(provider: string) {
  return api(`/services/keys/${provider}${tokenParam()}`, {
    method: "DELETE",
  });
}

// ── Memory ──

export async function getMemoryFacts() {
  return api<{ facts: string[] }>(`/memory/facts${tokenParam()}`);
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
  }>(`/memory/sessions${tokenParam()}`);
}

export async function getMemoryConversations(limit = 20) {
  return api<{
    conversations: {
      id: number;
      user_message: string;
      assistant_message: string;
      timestamp: number;
    }[];
  }>(`/memory/conversations${tokenParam()}&limit=${limit}`);
}

// ── WebSocket URL ──

export function getWsUrl(): string {
  const token = localStorage.getItem("aether_token") || "";
  const base = ORCHESTRATOR_URL.replace("http", "ws");
  return `${base}/ws?token=${token}`;
}
