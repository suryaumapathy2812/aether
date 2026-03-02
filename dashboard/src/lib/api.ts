/**
 * Agent / Orchestrator API client.
 *
 * Browser-based calls use same-origin URLs which Traefik proxies to the orchestrator.
 * In production: /api/* routes → Traefik → orchestrator
 * In development: /api/* routes → Caddy → orchestrator
 *
 * Auth: session token from better-auth sent as Authorization: Bearer header.
 */

// ── Base URL ──
// Client-side calls go through the Next.js proxy at /api/go.
// This keeps the actual upstream URL server-side only.
const API_BASE = "/api/go";

// ── Session token management ──
// Set by SessionSync component which reads from useSession().

let _sessionToken: string | null = null;

export function setSessionToken(token: string | null) {
  _sessionToken = token;
}

export function getSessionToken(): string | null {
  return _sessionToken;
}

// ── fetchWithAuth ──
// Central fetch wrapper. Every API call to the agent/orchestrator
// should use this. Attaches the auth token and base URL automatically.

export async function fetchWithAuth(
  path: string,
  options?: RequestInit
): Promise<Response> {
  const token = _sessionToken;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  // Merge caller-provided headers (if any) with auth headers.
  const callerHeaders = options?.headers;
  if (callerHeaders) {
    if (callerHeaders instanceof Headers) {
      callerHeaders.forEach((v, k) => {
        headers[k] = v;
      });
    } else if (Array.isArray(callerHeaders)) {
      for (const [k, v] of callerHeaders) {
        headers[k] = v;
      }
    } else {
      Object.assign(headers, callerHeaders);
    }
  }

  return fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
}

// ── Orchestrator Direct API ──
// These functions connect directly to the orchestrator (via same-origin proxy in production).
// In production, Traefik proxies /api/* to the orchestrator, so we use relative URLs.
// This avoids hardcoding hostnames that differ between dev/prod.

function getOrchestratorBaseUrl(): string {
  if (typeof window !== "undefined") {
    return ""; // Same-origin - Traefik proxies /api/* to orchestrator in production
  }
  // SSR fallback
  return process.env.AGENT_BASE_URL || "http://localhost:4000";
}

export async function orchestratorFetch(
  path: string,
  options?: RequestInit
): Promise<Response> {
  const token = getSessionToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const baseUrl = getOrchestratorBaseUrl();
  return fetch(`${baseUrl}${path}`, { ...options, headers });
}

export function orchestratorWs(path: string, token?: string): WebSocket {
  const baseUrl = getOrchestratorBaseUrl();
  const protocol =
    typeof window !== "undefined"
      ? window.location.protocol === "https:"
        ? "wss:"
        : "ws:"
      : "ws:";
  const host =
    typeof window !== "undefined" ? window.location.host : "localhost:3000";

  const params = token ? `?token=${encodeURIComponent(token)}` : "";
  return new WebSocket(`${protocol}//${host}${baseUrl}${path}${params}`);
}

// ── Typed JSON helper built on fetchWithAuth ──

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetchWithAuth(path, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(extractErrorMessage(err, res.statusText));
  }
  return res.json();
}

function extractErrorMessage(err: unknown, fallback: string): string {
  if (!err || typeof err !== "object") return fallback || "Request failed";
  const body = err as {
    detail?: unknown;
    error?: unknown;
    message?: unknown;
  };

  if (typeof body.detail === "string" && body.detail.trim()) return body.detail;
  if (typeof body.message === "string" && body.message.trim()) return body.message;
  if (typeof body.error === "string" && body.error.trim()) return body.error;
  if (body.error && typeof body.error === "object") {
    const nested = body.error as { message?: unknown; detail?: unknown };
    if (typeof nested.message === "string" && nested.message.trim()) {
      return nested.message;
    }
    if (typeof nested.detail === "string" && nested.detail.trim()) {
      return nested.detail;
    }
  }

  return fallback || "Request failed";
}

// ── Observability ──

export interface LatencyMetrics {
  chat?: { ttft_p95_ms?: number | null };
  voice?: { ttft_p95_ms?: number | null; tts_p95_ms?: number | null };
  kernel?: { enqueue_delay_p95_ms?: number | null };
  services?: {
    notification_delivery_p95_ms?: number | null;
    delegation_duration_p95_ms?: number | null;
  };
}

export async function getLatencyMetrics() {
  return api<LatencyMetrics>("/api/metrics/latency");
}

// ── Memory ──

export async function getMemoryFacts(userId: string) {
  return api<{ facts: string[] }>(`/api/memory/facts?user_id=${encodeURIComponent(userId)}`);
}

export async function getMemorySessions(userId: string) {
  const res = await api<{
    sessions: {
      session_id?: string;
      summary?: string;
      started_at?: number;
      ended_at?: number;
      turns?: number;
      tools_used?: string[];
      SessionID?: string;
      Summary?: string;
      StartedAt?: string;
      EndedAt?: string;
      Turns?: number;
      ToolsUsed?: string[];
    }[];
  }>(`/api/memory/sessions?user_id=${encodeURIComponent(userId)}`);
  return {
    sessions: (res.sessions || []).map((s) => ({
      session_id: s.session_id || s.SessionID || "",
      summary: s.summary || s.Summary || "",
      started_at: s.started_at || toEpoch(s.StartedAt),
      ended_at: s.ended_at || toEpoch(s.EndedAt),
      turns: s.turns ?? s.Turns ?? 0,
      tools_used: s.tools_used || s.ToolsUsed || [],
    })),
  };
}

export async function getMemoryConversations(userId: string, limit = 20) {
  const res = await api<{
    conversations: {
      id?: number;
      user_message?: string;
      assistant_message?: string;
      timestamp?: number;
      ID?: number;
      UserMessage?: string;
      AssistantMessage?: string;
      Timestamp?: string;
      user_content?: ChatContentPart[];
      UserContent?: ChatContentPart[];
    }[];
  }>(`/api/memory/conversations?user_id=${encodeURIComponent(userId)}&limit=${limit}`);
  return {
    conversations: (res.conversations || []).map((c, idx) => ({
      id: c.id ?? c.ID ?? idx,
      user_message: c.user_message || c.UserMessage || "",
      assistant_message: c.assistant_message || c.AssistantMessage || "",
      user_content: c.user_content || c.UserContent || [],
      timestamp: c.timestamp || toEpoch(c.Timestamp),
    })),
  };
}

export async function getMemories(userId: string, limit = 100, category?: string) {
  const params = new URLSearchParams({ user_id: userId, limit: String(limit) });
  if (category && category.trim()) {
    params.set("category", category.trim());
  }
  const res = await api<{
    memories: {
      id?: number;
      memory?: string;
      category?: string;
      confidence?: number;
      created_at?: string;
      expires_at?: string | null;
      ID?: number;
      Memory?: string;
      Category?: string;
      Confidence?: number;
      CreatedAt?: string;
      ExpiresAt?: string | null;
    }[];
  }>(`/api/memory/memories?${params.toString()}`);
  return {
    memories: (res.memories || []).map((m) => ({
      id: m.id ?? m.ID ?? 0,
      memory: m.memory || m.Memory || "",
      category: m.category || m.Category || "episodic",
      confidence: m.confidence ?? m.Confidence ?? 1,
      created_at: m.created_at || m.CreatedAt || "",
      expires_at: m.expires_at ?? m.ExpiresAt ?? null,
    })),
  };
}

export async function getDecisions(userId: string, category?: string, activeOnly = true) {
  const params = new URLSearchParams({ user_id: userId, active_only: String(activeOnly) });
  if (category && category.trim()) {
    params.set("category", category.trim());
  }
  const res = await api<{
    decisions: {
      id?: number;
      decision?: string;
      category?: string;
      source?: string;
      active?: boolean;
      confidence?: number;
      created_at?: string;
      updated_at?: string;
      ID?: number;
      Decision?: string;
      Category?: string;
      Source?: string;
      Active?: boolean;
      Confidence?: number;
      CreatedAt?: string;
      UpdatedAt?: string;
    }[];
  }>(`/api/memory/decisions?${params.toString()}`);
  return {
    decisions: (res.decisions || []).map((d) => ({
      id: d.id ?? d.ID ?? 0,
      decision: d.decision || d.Decision || "",
      category: d.category || d.Category || "preference",
      source: d.source || d.Source || "extracted",
      active: d.active ?? d.Active ?? true,
      confidence: d.confidence ?? d.Confidence ?? 1,
      created_at: d.created_at || d.CreatedAt || "",
      updated_at: d.updated_at || d.UpdatedAt || "",
    })),
  };
}

export async function getMemoryNotifications(userId: string, limit = 200) {
  const res = await api<{
    notifications: Array<
      Record<string, unknown> & {
        id?: number;
        text?: string;
        status?: string;
        source?: string;
        delivery_type?: string;
        created_at?: string;
        ID?: number;
        Text?: string;
        Status?: string;
        Source?: string;
        DeliveryType?: string;
        CreatedAt?: string;
      }
    >;
    reliability: Record<string, unknown>;
  }>(`/api/memory/notifications?user_id=${encodeURIComponent(userId)}&limit=${limit}`);
  return {
    notifications: (res.notifications || []).map((n) => ({
      ...n,
      id: n.id ?? n.ID,
      text: n.text || n.Text,
      status: n.status || n.Status,
      source: n.source || n.Source,
      delivery_type: n.delivery_type || n.DeliveryType,
      created_at: n.created_at || n.CreatedAt,
    })),
    reliability: res.reliability || {},
  };
}

export async function exportMemory(userId: string) {
  return api<{ export: Record<string, unknown> }>(`/api/memory/export?user_id=${encodeURIComponent(userId)}`);
}

function toEpoch(value?: string): number {
  if (!value) return 0;
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) return 0;
  return Math.floor(ms / 1000);
}

// ── Plugins ──

export interface PluginInfo {
  name: string;
  display_name: string;
  description: string;
  auth_type: string;
  auth_provider: string;
  config_fields: {
    key: string;
    label: string;
    type: string;
    required: boolean;
    description?: string;
  }[];
  installed: boolean;
  plugin_id: string | null;
  enabled: boolean;
  connected: boolean;
  needs_reconnect: boolean;
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
  return api<{ status: string; auto_enabled?: boolean }>(`/api/plugins/${name}/config`, {
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

// ── Chat completions ──

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string | ChatContentPart[];
}

export interface MediaRef {
  bucket?: string;
  key: string;
  url?: string;
  mime?: string;
  size?: number;
  file_name?: string;
  format?: string;
}

export type ChatContentPart =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "image_url";
      image_url: {
        url: string;
      };
    }
  | {
      type: "input_audio";
      input_audio: {
        data: string;
        format: string;
      };
    }
  | {
      type: "image_ref";
      media: MediaRef;
    }
  | {
      type: "audio_ref";
      media: MediaRef;
    };

export async function initMediaUpload(input: {
  user_id: string;
  session_id: string;
  file_name: string;
  content_type: string;
  size: number;
  kind: "image" | "audio";
}) {
  return api<{
    bucket: string;
    object_key: string;
    upload_url: string;
    headers: Record<string, string>;
    expires_at: number;
  }>("/v1/media/upload/init", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function completeMediaUpload(input: {
  user_id: string;
  bucket?: string;
  object_key: string;
  file_name: string;
  content_type: string;
  size: number;
  kind: "image" | "audio";
}) {
  return api<{
    media: MediaRef;
  }>("/v1/media/upload/complete", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: string;
  }[];
}

export interface ConversationTurnEvent {
  id?: string;
  object?: string;
  created?: number;
  phase?: "start" | "ack" | "act" | "answer" | "error" | "done";
  event?: string;
  text?: string;
  error?: string;
  payload?: Record<string, unknown>;
}

export async function chatCompletions(input: {
  model?: string;
  messages: ChatMessage[];
  user: string;
  temperature?: number;
  max_tokens?: number;
}): Promise<ChatCompletionResponse> {
  const body: Record<string, unknown> = {
    messages: input.messages,
    stream: false,
    user: input.user,
    temperature: input.temperature,
    max_tokens: input.max_tokens,
  };
  if (input.model && input.model.trim()) {
    body.model = input.model.trim();
  }
  return api<ChatCompletionResponse>("/v1/chat/completions", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function streamConversationTurn(input: {
  messages: ChatMessage[];
  user: string;
  session?: string;
  temperature?: number;
  max_tokens?: number;
  onEvent: (event: ConversationTurnEvent) => void;
  signal?: AbortSignal;
}): Promise<void> {
  const body: Record<string, unknown> = {
    messages: input.messages,
    user: input.user,
    session: input.session || "",
    temperature: input.temperature,
    max_tokens: input.max_tokens,
  };
  const res = await fetchWithAuth("/v1/conversations/turn", {
    method: "POST",
    body: JSON.stringify(body),
    signal: input.signal,
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(extractErrorMessage(err, res.statusText));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n\n");
    while (idx !== -1) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      idx = buffer.indexOf("\n\n");
      const lines = chunk.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") return;
        if (!payload) continue;
        try {
          input.onEvent(JSON.parse(payload) as ConversationTurnEvent);
        } catch {
          // Ignore malformed events
        }
      }
    }
  }
}

// ── Agent tasks ──

export type AgentTaskStatus =
  | "queued"
  | "running"
  | "waiting_input"
  | "completed"
  | "failed"
  | "cancelled"
  | "timed_out";

export interface AgentTask {
  ID: string;
  UserID: string;
  SessionID: string;
  Title: string;
  Goal: string;
  Status: AgentTaskStatus;
  Priority: number;
  MaxSteps: number;
  StepCount: number;
  CancelRequested: boolean;
  LastError: string;
  ResultSummary: string;
  CreatedAt: string;
  UpdatedAt: string;
}

export interface AgentTaskEvent {
  ID: number;
  TaskID: string;
  Kind: string;
  PayloadJSON: string;
  CreatedAt: string;
}

export interface AgentJob {
  ID: string;
  Module: string;
  JobType: string;
  Status: string;
  LastError: string;
  RunAt: string;
  NextRunAt: string;
  CreatedAt: string;
}

export async function listAgentTasks(userId: string, status = "", limit = 50) {
  const statusParam = status ? `&status=${encodeURIComponent(status)}` : "";
  return api<{ tasks: AgentTask[]; count: number }>(
    `/v1/agent/tasks?user_id=${encodeURIComponent(userId)}&limit=${limit}${statusParam}`
  );
}

export async function listAgentJobs(module = "", limit = 50) {
  const moduleParam = module ? `&module=${encodeURIComponent(module)}` : "";
  return api<{ jobs: AgentJob[]; count: number }>(
    `/v1/agent/jobs?limit=${limit}${moduleParam}`
  );
}

export async function getAgentTask(taskId: string) {
  return api<{ task: AgentTask }>(`/v1/agent/tasks/${encodeURIComponent(taskId)}`);
}

export async function getAgentTaskEvents(taskId: string, limit = 200) {
  return api<{ task_id: string; events: AgentTaskEvent[]; count: number }>(
    `/v1/agent/tasks/${encodeURIComponent(taskId)}/events?limit=${limit}`
  );
}

export async function cancelAgentTask(taskId: string) {
  return api<{ task_id: string; status: string }>(
    `/v1/agent/tasks/${encodeURIComponent(taskId)}/cancel`,
    { method: "POST" }
  );
}

export async function approveAgentTask(input: {
  taskId: string;
  userId: string;
  decision?: string;
  reason?: string;
  instructions?: string;
}) {
  return api<{ task_id: string; status: AgentTaskStatus; decision: string }>(
    `/v1/agent/tasks/${encodeURIComponent(input.taskId)}/approve`,
    {
      method: "POST",
      body: JSON.stringify({
        user_id: input.userId,
        decision: input.decision || "approved",
        reason: input.reason || "",
        instructions: input.instructions || "",
      }),
    }
  );
}

export async function rejectAgentTask(input: {
  taskId: string;
  userId: string;
  reason: string;
  nextAction?: string;
}) {
  return api<{ task_id: string; status: AgentTaskStatus; decision: string }>(
    `/v1/agent/tasks/${encodeURIComponent(input.taskId)}/reject`,
    {
      method: "POST",
      body: JSON.stringify({
        user_id: input.userId,
        reason: input.reason,
        next_action: input.nextAction || "Stop and wait for new instructions.",
      }),
    }
  );
}

export async function resumeAgentTask(input: {
  taskId: string;
  userId: string;
  message?: string;
}) {
  return api<{ task_id: string; status: AgentTaskStatus }>(
    `/v1/agent/tasks/${encodeURIComponent(input.taskId)}/resume`,
    {
      method: "POST",
      body: JSON.stringify({
        user_id: input.userId,
        message: input.message || "Human approved. Continue.",
      }),
    }
  );
}

// ── OAuth / WebSocket URLs ──

export function getOAuthStartUrl(pluginName: string): string {
  const token = _sessionToken || "";
  const qs = token ? `?token=${encodeURIComponent(token)}` : "";
  return `/plugins/${pluginName}/oauth/start${qs}`;
}
