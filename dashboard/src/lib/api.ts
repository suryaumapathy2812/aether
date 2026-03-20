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
  return process.env.ORCHESTRATOR_BASE_URL || process.env.AGENT_BASE_URL || "http://localhost:4000";
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

// ── Devices ──

export interface Device {
  id: string;
  name: string;
  device_type: string;
  plugin_name: string;
  paired_at: string;
  last_seen?: string;
}

export async function listDevices(): Promise<Device[]> {
  const res = await orchestratorFetch("/api/devices");
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail || "Failed to list devices");
  }
  const data = await res.json();
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.devices)) return data.devices;
  return [];
}

export interface RegisterTelegramRequest {
  bot_token: string;
  allowed_chat_ids?: string;
  name?: string;
}

export interface RegisterTelegramResponse {
  status: string;
  device_id: string;
  webhook_url: string;
}

export async function registerTelegramDevice(
  botToken: string,
  allowedChatIDs?: string,
  name?: string
): Promise<RegisterTelegramResponse> {
  const res = await orchestratorFetch("/api/devices/telegram", {
    method: "POST",
    body: JSON.stringify({
      bot_token: botToken,
      allowed_chat_ids: allowedChatIDs,
      name,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail || "Failed to register Telegram device");
  }
  return res.json();
}

export async function deleteDevice(deviceId: string): Promise<void> {
  const res = await orchestratorFetch(`/api/devices/${deviceId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail || "Failed to delete device");
  }
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

// ── Entities ──

export interface EntityRow {
  id: string;
  entity_type: string;
  name: string;
  aliases: string[];
  summary: string;
  properties: Record<string, unknown>;
  first_seen_at: string;
  last_seen_at: string;
  interaction_count: number;
  created_at: string;
  updated_at: string;
}

export async function getEntities(userId: string, entityType?: string, limit = 50) {
  const params = new URLSearchParams({ user_id: userId, limit: String(limit) });
  if (entityType && entityType.trim()) {
    params.set("type", entityType.trim());
  }
  const res = await api<{
    entities: Array<Record<string, unknown> & {
      id?: string; ID?: string;
      entity_type?: string; EntityType?: string;
      name?: string; Name?: string;
      aliases?: string[]; Aliases?: string[];
      summary?: string; Summary?: string;
      properties?: Record<string, unknown>; Properties?: Record<string, unknown>;
      first_seen_at?: string; FirstSeenAt?: string;
      last_seen_at?: string; LastSeenAt?: string;
      interaction_count?: number; InteractionCount?: number;
      created_at?: string; CreatedAt?: string;
      updated_at?: string; UpdatedAt?: string;
    }>;
    count: number;
  }>(`/api/memory/entities?${params.toString()}`);
  return {
    entities: (res.entities || []).map((e) => ({
      id: e.id || e.ID || "",
      entity_type: e.entity_type || e.EntityType || "",
      name: e.name || e.Name || "",
      aliases: e.aliases || e.Aliases || [],
      summary: e.summary || e.Summary || "",
      properties: e.properties || e.Properties || {},
      first_seen_at: e.first_seen_at || e.FirstSeenAt || "",
      last_seen_at: e.last_seen_at || e.LastSeenAt || "",
      interaction_count: e.interaction_count ?? e.InteractionCount ?? 0,
      created_at: e.created_at || e.CreatedAt || "",
      updated_at: e.updated_at || e.UpdatedAt || "",
    })) as EntityRow[],
    count: res.count || 0,
  };
}

export interface EntityObservationRow {
  id: number;
  entity_id: string;
  observation: string;
  category: string;
  confidence: number;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface EntityInteractionRow {
  id: number;
  entity_id: string;
  summary: string;
  source: string;
  source_ref: string;
  interaction_at: string;
  created_at: string;
}

export interface EntityRelationRow {
  id: number;
  source_entity_id: string;
  relation: string;
  target_entity_id: string;
  context: string;
  confidence: number;
  created_at: string;
  updated_at: string;
}

export interface EntityDetails {
  entity: EntityRow;
  observations: EntityObservationRow[];
  interactions: EntityInteractionRow[];
  relations: EntityRelationRow[];
}

export async function getEntityDetails(entityId: string) {
  const res = await api<{
    entity: Record<string, unknown>;
    observations: Array<Record<string, unknown>>;
    interactions: Array<Record<string, unknown>>;
    relations: Array<Record<string, unknown>>;
  }>(`/api/memory/entities/${encodeURIComponent(entityId)}`);

  const e = res.entity || {};
  const entity: EntityRow = {
    id: (e.id || e.ID || "") as string,
    entity_type: (e.entity_type || e.EntityType || "") as string,
    name: (e.name || e.Name || "") as string,
    aliases: (e.aliases || e.Aliases || []) as string[],
    summary: (e.summary || e.Summary || "") as string,
    properties: (e.properties || e.Properties || {}) as Record<string, unknown>,
    first_seen_at: (e.first_seen_at || e.FirstSeenAt || "") as string,
    last_seen_at: (e.last_seen_at || e.LastSeenAt || "") as string,
    interaction_count: (e.interaction_count ?? e.InteractionCount ?? 0) as number,
    created_at: (e.created_at || e.CreatedAt || "") as string,
    updated_at: (e.updated_at || e.UpdatedAt || "") as string,
  };

  const observations: EntityObservationRow[] = (res.observations || []).map((o: Record<string, unknown>) => ({
    id: (o.id ?? o.ID ?? 0) as number,
    entity_id: (o.entity_id || o.EntityID || "") as string,
    observation: (o.observation || o.Observation || "") as string,
    category: (o.category || o.Category || "trait") as string,
    confidence: (o.confidence ?? o.Confidence ?? 1) as number,
    source: (o.source || o.Source || "extracted") as string,
    created_at: (o.created_at || o.CreatedAt || "") as string,
    updated_at: (o.updated_at || o.UpdatedAt || "") as string,
  }));

  const interactions: EntityInteractionRow[] = (res.interactions || []).map((i: Record<string, unknown>) => ({
    id: (i.id ?? i.ID ?? 0) as number,
    entity_id: (i.entity_id || i.EntityID || "") as string,
    summary: (i.summary || i.Summary || "") as string,
    source: (i.source || i.Source || "") as string,
    source_ref: (i.source_ref || i.SourceRef || "") as string,
    interaction_at: (i.interaction_at || i.InteractionAt || "") as string,
    created_at: (i.created_at || i.CreatedAt || "") as string,
  }));

  const relations: EntityRelationRow[] = (res.relations || []).map((r: Record<string, unknown>) => ({
    id: (r.id ?? r.ID ?? 0) as number,
    source_entity_id: (r.source_entity_id || r.SourceEntityID || "") as string,
    relation: (r.relation || r.Relation || "") as string,
    target_entity_id: (r.target_entity_id || r.TargetEntityID || "") as string,
    context: (r.context || r.Context || "") as string,
    confidence: (r.confidence ?? r.Confidence ?? 1) as number,
    created_at: (r.created_at || r.CreatedAt || "") as string,
    updated_at: (r.updated_at || r.UpdatedAt || "") as string,
  }));

  return { entity, observations, interactions, relations } as EntityDetails;
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

// ── Skills ──

export interface MarketplaceSkill {
  id: string;
  skill_id: string;
  name: string;
  installs: number;
  source: string;
}

export interface SkillMeta {
  name: string;
  description: string;
  location: string;
  source: "builtin" | "user" | "external";
}

export async function searchMarketplaceSkills(query: string, limit = 10) {
  return api<{ query: string; search_type: string; skills: MarketplaceSkill[]; count: number }>(
    `/api/skills/marketplace/search?q=${encodeURIComponent(query)}&limit=${limit}`
  );
}

export async function listInstalledSkills() {
  return api<{ skills: SkillMeta[]; count: number }>("/api/skills/installed");
}

export async function installSkill(source: string, skillName?: string) {
  return api<{ installed: SkillMeta; remote_url: string }>("/api/skills/install", {
    method: "POST",
    body: JSON.stringify({ source, skill_name: skillName }),
  });
}

export async function removeSkill(name: string) {
  return api<{ removed: boolean; name: string }>("/api/skills/remove", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

// ── Chat sessions ──

export interface ChatSession {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export async function listChatSessions(userId: string, limit = 50) {
  return api<{ sessions: ChatSession[] }>(
    `/v1/sessions?user_id=${encodeURIComponent(userId)}&limit=${limit}`
  );
}

export async function getChatSessionStatuses(userId: string) {
  return api<{ statuses: Record<string, "idle" | "streaming" | "error"> }>(
    `/v1/sessions/status?user_id=${encodeURIComponent(userId)}`
  );
}

export async function createChatSession(userId: string, title = "New chat") {
  return api<ChatSession>("/v1/sessions", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, title }),
  });
}

export async function getChatSession(sessionId: string) {
  return api<{ session: ChatSession; messages: Array<{ id: number; role?: string; content: unknown; created_at?: string }> }>(
    `/v1/sessions/${encodeURIComponent(sessionId)}`
  );
}

export async function updateChatSessionTitle(sessionId: string, title: string) {
  return api<ChatSession>(`/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function archiveChatSession(sessionId: string) {
  return api<ChatSession>(`/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify({ archive: true }),
  });
}

export async function deleteChatSession(sessionId: string) {
  return api<{ deleted: boolean }>(`/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
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





// ── OAuth / WebSocket URLs ──

export function getOAuthStartUrl(pluginName: string): string {
  const token = _sessionToken || "";
  const qs = token ? `?token=${encodeURIComponent(token)}` : "";
  return `/plugins/${pluginName}/oauth/start${qs}`;
}

// ── Channels ──

export interface ChannelInfo {
  id: string;
  user_id: string;
  channel_type: string;
  channel_id: string;
  bot_token?: string;
  display_name: string;
  config: Record<string, string>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface TelegramConnectRequest {
  user_id?: string;
  bot_token: string;
  chat_id: string;
}

export interface TelegramBotInfo {
  id: number;
  first_name: string;
  username: string;
  name: string;
}

export async function listChannels(userId?: string): Promise<ChannelInfo[]> {
  const params = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  const data = await api<{ channels: ChannelInfo[] }>(`/api/channels${params}`);
  return data.channels ?? [];
}

export async function connectTelegram(request: TelegramConnectRequest): Promise<{
  success: boolean;
  channel: ChannelInfo;
  bot_info: TelegramBotInfo;
}> {
  return api("/api/channels/telegram/connect", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function disconnectChannel(channelId: string): Promise<{ success: boolean }> {
  return api("/api/channels/telegram/disconnect", {
    method: "POST",
    body: JSON.stringify({ channel_id: channelId }),
  });
}

export async function enableChannel(channelId: string): Promise<{ success: boolean }> {
  return api(`/api/channels/${encodeURIComponent(channelId)}/enable`, {
    method: "POST",
  });
}

export async function disableChannel(channelId: string): Promise<{ success: boolean }> {
  return api(`/api/channels/${encodeURIComponent(channelId)}/disable`, {
    method: "POST",
  });
}

export async function sendChannelMessage(channelId: string, text: string): Promise<{ success: boolean }> {
  return api(`/api/channels/${encodeURIComponent(channelId)}/send`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export async function claimPairingCode(code: string): Promise<{ status: string }> {
  return api<{ status: string }>("/api/pair/claim", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}

// ── Questions (ask_user tool) ──

export async function replyToQuestion(questionId: string, answers: string[]) {
  return api<{ ok: boolean }>(`/v1/questions/${encodeURIComponent(questionId)}/reply`, {
    method: "POST",
    body: JSON.stringify({ answers }),
  });
}

export async function rejectQuestion(questionId: string) {
  return api<{ ok: boolean }>(`/v1/questions/${encodeURIComponent(questionId)}/reject`, {
    method: "POST",
  });
}
