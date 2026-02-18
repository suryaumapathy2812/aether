/**
 * Orchestrator API client.
 *
 * With Caddy reverse proxy, the orchestrator is on the same origin as the dashboard.
 * Caddy routes /api/* to the orchestrator (except /api/auth/* which stays with Next.js).
 * Everything else (pages, static assets) goes to Next.js.
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
  >("/api/devices");
}

export async function confirmPairing(code: string) {
  return api<{ device_id: string; device_token: string }>("/api/pair/confirm", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}

// ── Services (API keys) ──

export async function listApiKeys() {
  return api<{ provider: string; preview: string }[]>("/api/services/keys");
}

export async function saveApiKey(provider: string, keyValue: string) {
  return api("/api/services/keys", {
    method: "POST",
    body: JSON.stringify({ provider, key_value: keyValue }),
  });
}

export async function deleteApiKey(provider: string) {
  return api(`/api/services/keys/${provider}`, {
    method: "DELETE",
  });
}

// ── Memory ──

export async function getMemoryFacts() {
  return api<{ facts: string[] }>("/api/memory/facts");
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
  }>("/api/memory/sessions");
}

export async function getMemoryConversations(limit = 20) {
  return api<{
    conversations: {
      id: number;
      user_message: string;
      assistant_message: string;
      timestamp: number;
    }[];
  }>(`/api/memory/conversations?limit=${limit}`);
}

// ── Preferences ──

export interface UserPreferences {
  stt_provider: string | null;
  stt_model: string | null;
  stt_language: string | null;
  llm_provider: string | null;
  llm_model: string | null;
  tts_provider: string | null;
  tts_model: string | null;
  tts_voice: string | null;
  base_style: string | null;
  custom_instructions: string | null;
}

export async function getPreferences() {
  return api<UserPreferences>("/api/preferences");
}

export async function updatePreferences(prefs: Partial<UserPreferences>) {
  return api<{ status: string }>("/api/preferences", {
    method: "PUT",
    body: JSON.stringify(prefs),
  });
}

// ── Plugins ──

export interface PluginInfo {
  name: string;
  display_name: string;
  description: string;
  auth_type: string;
  auth_provider: string;
  config_fields: { key: string; label: string; type: string; required: boolean }[];
  installed: boolean;
  plugin_id: string | null;
  enabled: boolean;
  connected: boolean;
}

export async function listPlugins() {
  return api<PluginInfo[]>("/api/plugins");
}

export async function installPlugin(name: string) {
  return api<{ plugin_id: string; status: string }>(`/api/plugins/${name}/install`, {
    method: "POST",
  });
}

export async function enablePlugin(name: string) {
  return api<{ status: string }>(`/api/plugins/${name}/enable`, { method: "POST" });
}

export async function disablePlugin(name: string) {
  return api<{ status: string }>(`/api/plugins/${name}/disable`, { method: "POST" });
}

export async function savePluginConfig(name: string, config: Record<string, string>) {
  return api(`/api/plugins/${name}/config`, {
    method: "POST",
    body: JSON.stringify({ config }),
  });
}

export async function getPluginConfig(name: string) {
  return api<Record<string, string>>(`/api/plugins/${name}/config`);
}

export async function uninstallPlugin(name: string) {
  return api(`/api/plugins/${name}`, { method: "DELETE" });
}

export function getOAuthStartUrl(pluginName: string): string {
  // This is a redirect endpoint — browser navigates directly (not a fetch call).
  // Browser redirects can't send Authorization headers, so we pass the
  // session token as a query param (same pattern as WebSocket connections).
  const base = ORCHESTRATOR_URL || "";
  const token = _sessionToken || "";
  return `${base}/api/plugins/${pluginName}/oauth/start?token=${encodeURIComponent(token)}`;
}

// ── WebSocket URL ──

export function getWsUrl(): string {
  const token = _sessionToken || "";
  // Same-origin: use current page's protocol/host
  // Cross-origin: use ORCHESTRATOR_URL
  if (ORCHESTRATOR_URL) {
    const base = ORCHESTRATOR_URL.replace("http", "ws");
    return `${base}/api/ws?token=${token}`;
  }
  // Same-origin via Caddy — derive WS URL from current page
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/ws?token=${token}`;
  }
  // SSR fallback
  return `ws://localhost:3000/api/ws?token=${token}`;
}
