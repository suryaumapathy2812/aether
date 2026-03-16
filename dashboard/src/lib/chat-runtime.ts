import { useSyncExternalStore } from "react";
import type { UIMessage } from "ai";
import { getChatSession, getChatSessionStatuses, getSessionToken } from "@/lib/api";

export type SessionStatus = "idle" | "streaming" | "error";

type SessionState = {
  messages: UIMessage[];
  status: SessionStatus;
  error: string | null;
  loaded: boolean;
  loading: boolean;
};

type NormalizedGlobalEvent =
  | { kind: "session-status"; sessionID: string; status: SessionStatus; version: number }
  | { kind: "session-created"; sessionID: string; version: number }
  | { kind: "session-updated"; sessionID: string; archived?: boolean; version: number }
  | { kind: "session-deleted"; sessionID: string; version: number }
  | { kind: "message-updated"; sessionID: string; messageID: string; role?: string; version: number }
  | { kind: "message-removed"; sessionID: string; messageID: string; version: number }
  | {
      kind: "part-updated";
      sessionID: string;
      messageID: string;
      partID: string;
      delta?: string;
      text?: string;
      version: number;
    }
  | { kind: "part-removed"; sessionID: string; messageID: string; partID: string; version: number };

type SessionSnapshot = SessionState;

type StreamEvent = Record<string, unknown> & { type?: string };

type SessionAction =
  | { type: "history-loading" }
  | { type: "history-loaded"; messages: UIMessage[] }
  | { type: "history-error"; error: string }
  | { type: "stream-start" }
  | { type: "stream-end" }
  | { type: "stream-error"; error: string }
  | { type: "append-user"; text: string; messageID: string }
  | { type: "append-assistant"; messageID: string }
  | { type: "assistant-text-delta"; messageID: string; delta: string }
  | { type: "assistant-reasoning-delta"; messageID: string; delta: string }
  | {
      type: "assistant-tool-input";
      messageID: string;
      toolName: string;
      toolCallID: string;
      input: unknown;
    }
  | {
      type: "assistant-tool-output";
      messageID: string;
      toolCallID: string;
      output?: unknown;
      errorText?: string;
      failed: boolean;
    }
  | { type: "session-status"; status: SessionStatus }
  | { type: "message-upsert-from-event"; messageID: string; role: "assistant" | "user"; text?: string }
  | { type: "message-remove"; messageID: string }
  | { type: "part-upsert-delta"; messageID: string; partID: string; delta?: string; text?: string }
  | { type: "part-remove"; messageID: string; partID: string };

const RUNNING_SESSIONS_KEY = "aether.chat.runtime.running.v1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function textFromStoredMessageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!isRecord(content)) return "";

  const inner = content.content;
  if (typeof inner === "string") return inner;
  if (!Array.isArray(inner)) return "";

  return inner
    .map((part) => {
      if (!isRecord(part)) return "";
      if (part.type !== "text") return "";
      return typeof part.text === "string" ? part.text : "";
    })
    .join("");
}

function buildHistoryMessages(
  rows: Array<{ id: number; role?: string; content: unknown }>
): UIMessage[] {
  const restored: UIMessage[] = [];
  for (const row of rows) {
    const role =
      typeof row.role === "string"
        ? row.role
        : isRecord(row.content) && typeof row.content.role === "string"
          ? row.content.role
          : "";
    const text = textFromStoredMessageContent(row.content);
    if (!text.trim()) continue;
    if (role === "user" || role === "assistant") {
      restored.push({
        id: `msg-${row.id}-${role}`,
        role,
        parts: [{ type: "text", text }],
      });
    }
  }
  return restored;
}

function toTurnPayload(messages: UIMessage[]): Array<{ role: string; content: string }> {
  return messages.map((message) => {
    const text = message.parts
      .filter(
        (part): part is Extract<(typeof message.parts)[number], { type: "text" }> =>
          part.type === "text"
      )
      .map((part) => part.text)
      .join("");
    return { role: message.role, content: text };
  });
}

function parseEventVersion(payload: Record<string, unknown>): number {
  const candidates = [payload.sequence, payload.seq, payload.version, payload.updatedAt, payload.updated_at, payload.timestamp];
  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
      const n = Number(value);
      if (Number.isFinite(n)) return n;
      const parsed = Date.parse(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return Date.now();
}

function normalizeStatus(value: unknown): SessionStatus {
  const text = String(value || "").toLowerCase();
  if (text === "busy" || text === "running" || text === "streaming") return "streaming";
  if (text === "error" || text === "failed" || text === "blocked") return "error";
  return "idle";
}

function normalizeGlobalEvent(eventType: string, payload: Record<string, unknown>): NormalizedGlobalEvent | null {
  const normalizedType = String(eventType || "").toLowerCase();
  const sessionID = typeof payload.sessionID === "string" ? payload.sessionID : typeof payload.session_id === "string" ? payload.session_id : "";
  if (!sessionID) return null;
  const version = parseEventVersion(payload);

  if (normalizedType === "session.status") {
    const raw = isRecord(payload.status) ? payload.status.type : payload.status;
    return { kind: "session-status", sessionID, status: normalizeStatus(raw), version };
  }
  if (normalizedType === "session.created") {
    return { kind: "session-created", sessionID, version };
  }
  if (normalizedType === "session.updated") {
    return {
      kind: "session-updated",
      sessionID,
      archived: Boolean(payload.archived),
      version,
    };
  }
  if (normalizedType === "session.deleted") {
    return { kind: "session-deleted", sessionID, version };
  }
  if (normalizedType === "message.updated") {
    const messageID = typeof payload.messageID === "string" ? payload.messageID : "msg-latest";
    const role = typeof payload.role === "string" ? payload.role : undefined;
    return { kind: "message-updated", sessionID, messageID, role, version };
  }
  if (normalizedType === "message.removed") {
    const messageID = typeof payload.messageID === "string" ? payload.messageID : "";
    if (!messageID) return null;
    return { kind: "message-removed", sessionID, messageID, version };
  }
  if (normalizedType === "message.part.updated" || normalizedType === "message.part.delta") {
    const messageID = typeof payload.messageID === "string" ? payload.messageID : "assistant-current";
    const partID = typeof payload.partID === "string" ? payload.partID : "text";
    const delta = typeof payload.delta === "string" ? payload.delta : undefined;
    const text = typeof payload.text === "string" ? payload.text : undefined;
    return { kind: "part-updated", sessionID, messageID, partID, delta, text, version };
  }
  if (normalizedType === "message.part.removed") {
    const messageID = typeof payload.messageID === "string" ? payload.messageID : "";
    const partID = typeof payload.partID === "string" ? payload.partID : "";
    if (!messageID || !partID) return null;
    return { kind: "part-removed", sessionID, messageID, partID, version };
  }
  return null;
}

function nextState(current: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case "history-loading":
      return { ...current, loading: true, error: null };
    case "history-loaded":
      return {
        ...current,
        loading: false,
        loaded: true,
        messages: action.messages,
        error: null,
        status: current.status === "streaming" ? "streaming" : "idle",
      };
    case "history-error":
      return { ...current, loading: false, error: action.error, status: "error" };
    case "stream-start":
      return { ...current, loaded: true, loading: false, status: "streaming", error: null };
    case "stream-end":
      return {
        ...current,
        status: current.status === "error" ? "error" : "idle",
        error: current.status === "error" ? current.error : null,
      };
    case "stream-error":
      return { ...current, status: "error", error: action.error };
    case "append-user": {
      const msg: UIMessage = {
        id: action.messageID,
        role: "user",
        parts: [{ type: "text", text: action.text }],
      };
      return { ...current, messages: [...current.messages, msg] };
    }
    case "append-assistant": {
      const msg: UIMessage = {
        id: action.messageID,
        role: "assistant",
        parts: [],
      };
      return { ...current, messages: [...current.messages, msg] };
    }
    case "assistant-text-delta": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID || message.role !== "assistant") return message;
        const parts = [...message.parts];
        const last = parts[parts.length - 1];
        if (last && last.type === "text") {
          parts[parts.length - 1] = { ...last, text: `${last.text}${action.delta}` };
        } else {
          parts.push({ type: "text", text: action.delta });
        }
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "assistant-reasoning-delta": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID || message.role !== "assistant") return message;
        const parts = [
          ...message.parts,
          { type: "reasoning", text: action.delta } as UIMessage["parts"][number],
        ];
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "assistant-tool-input": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID || message.role !== "assistant") return message;
        const parts = [
          ...message.parts,
          {
            type: `tool-${action.toolName}`,
            state: "input-available",
            toolCallId: action.toolCallID,
            input: action.input,
          } as UIMessage["parts"][number],
        ];
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "assistant-tool-output": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID || message.role !== "assistant") return message;
        const parts = message.parts.map((part) => {
          if (!String(part.type).startsWith("tool-")) return part;
          const candidate = part as Record<string, unknown>;
          if (String(candidate.toolCallId || "") !== action.toolCallID) return part;
          const nextPart: Record<string, unknown> = {
            ...candidate,
            state: action.failed ? "output-error" : "output-available",
          };
          if (action.failed) {
            nextPart.errorText = action.errorText || "Tool failed";
          } else {
            nextPart.output = action.output;
          }
          return nextPart as UIMessage["parts"][number];
        });
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "session-status":
      return { ...current, status: action.status };
    case "message-upsert-from-event": {
      const idx = current.messages.findIndex((message) => message.id === action.messageID);
      if (idx === -1) {
        const next: UIMessage = {
          id: action.messageID,
          role: action.role,
          parts: action.text ? [{ type: "text", text: action.text }] : [],
        };
        return { ...current, messages: [...current.messages, next] };
      }
      const existing = current.messages[idx];
      const nextMessages = [...current.messages];
      if (action.text) {
        nextMessages[idx] = { ...existing, role: action.role, parts: [{ type: "text", text: action.text }] };
      } else {
        nextMessages[idx] = { ...existing, role: action.role };
      }
      return { ...current, messages: nextMessages };
    }
    case "message-remove":
      return { ...current, messages: current.messages.filter((message) => message.id !== action.messageID) };
    case "part-upsert-delta": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID) return message;
        const parts = [...message.parts];
        const existingIndex = parts.findIndex((part) => String((part as Record<string, unknown>).partId || "") === action.partID);
        const incomingText = action.text ?? action.delta ?? "";
        if (existingIndex === -1) {
          parts.push({ type: "text", text: incomingText, partId: action.partID } as UIMessage["parts"][number]);
          return { ...message, parts };
        }
        const part = parts[existingIndex] as Record<string, unknown>;
        const currentText = typeof part.text === "string" ? part.text : "";
        const nextText = action.text != null ? action.text : `${currentText}${action.delta || ""}`;
        parts[existingIndex] = { ...part, text: nextText } as UIMessage["parts"][number];
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "part-remove": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID) return message;
        const parts = message.parts.filter((part) => String((part as Record<string, unknown>).partId || "") !== action.partID);
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    default:
      return current;
  }
}

class ChatRuntimeStore {
  private sessions = new Map<string, SessionState>();
  private eventVersions = new Map<string, number>();
  private inflight = new Map<string, AbortController>();
  private listeners = new Set<() => void>();
  private bootstrappedUsers = new Set<string>();
  private statusMapCache: Record<string, SessionStatus> = {};
  private statusMapDirty = true;
  private ws: WebSocket | null = null;
  private wsReconnectTimer: number | null = null;
  private wsBackoffMs = 500;

  constructor() {
    this.recoverPersistedRunningSessions();
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    this.ensureGlobalSubscription();
    return () => {
      this.listeners.delete(listener);
      if (this.listeners.size === 0) {
        this.teardownGlobalSubscription();
      }
    };
  };

  private notify() {
    for (const listener of this.listeners) listener();
  }

  private ensureSession(sessionId: string): SessionState {
    const found = this.sessions.get(sessionId);
    if (found) return found;
    const created: SessionState = {
      messages: [],
      status: "idle",
      error: null,
      loaded: false,
      loading: false,
    };
    this.sessions.set(sessionId, created);
    this.statusMapDirty = true;
    return created;
  }

  private dispatch(sessionId: string, action: SessionAction) {
    const current = this.ensureSession(sessionId);
    const next = nextState(current, action);
    this.sessions.set(sessionId, next);
    this.statusMapDirty = true;
    this.notify();
  }

  private static readonly emptySnapshot: SessionSnapshot = {
    messages: [],
    status: "idle",
    error: null,
    loaded: false,
    loading: false,
  };

  getSnapshot = (sessionId: string): SessionSnapshot => {
    const state = this.sessions.get(sessionId);
    return state ?? ChatRuntimeStore.emptySnapshot;
  };

  getStatusMap = (): Record<string, SessionStatus> => {
    if (!this.statusMapDirty) {
      return this.statusMapCache;
    }
    const next: Record<string, SessionStatus> = {};
    for (const [sessionId, state] of this.sessions.entries()) {
      next[sessionId] = state.status;
    }
    this.statusMapCache = next;
    this.statusMapDirty = false;
    return this.statusMapCache;
  };

  async bootstrapForUser(userId: string): Promise<void> {
	const normalizedUserID = userId.trim();
	if (!normalizedUserID) return;
	if (this.bootstrappedUsers.has(normalizedUserID)) return;
	this.bootstrappedUsers.add(normalizedUserID);
	try {
		const response = await getChatSessionStatuses(normalizedUserID);
		const statuses = response.statuses || {};
		for (const [sessionID, status] of Object.entries(statuses)) {
			if (!sessionID) continue;
			this.dispatch(sessionID, { type: "session-status", status: status || "idle" });
		}
	} catch {
		// Non-fatal: websocket stream and local state still work.
	}
  }

  async loadHistory(sessionId: string): Promise<void> {
    if (!sessionId) return;
    const current = this.ensureSession(sessionId);
    if (current.loaded || current.loading) return;
    this.dispatch(sessionId, { type: "history-loading" });
    try {
      const res = await getChatSession(sessionId);
      this.dispatch(sessionId, {
        type: "history-loaded",
        messages: buildHistoryMessages(res.messages || []),
      });
    } catch {
      this.dispatch(sessionId, { type: "history-error", error: "Failed to load chat history" });
    }
  }

  async sendMessage(input: { sessionId: string; userId: string; text: string }): Promise<void> {
    const sessionId = input.sessionId.trim();
    const text = input.text.trim();
    if (!sessionId || !text) return;

    const userMessageID = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const assistantMessageID = `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const session = this.ensureSession(sessionId);
    const requestMessages = [...session.messages, { id: userMessageID, role: "user", parts: [{ type: "text", text }] } as UIMessage];

    this.dispatch(sessionId, { type: "append-user", text, messageID: userMessageID });
    this.dispatch(sessionId, { type: "append-assistant", messageID: assistantMessageID });
    this.dispatch(sessionId, { type: "stream-start" });
    this.markSessionRunning(sessionId, true);

    const controller = new AbortController();
    this.inflight.set(sessionId, controller);

    try {
      const token = getSessionToken();
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const response = await fetch("/api/go/v1/conversations/turn", {
        method: "POST",
        credentials: "include",
        headers,
        body: JSON.stringify({
          messages: toTurnPayload(requestMessages),
          user: input.userId,
          session: sessionId,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Stream request failed (${response.status})`);
      }

      await this.consumeSSE(response.body, (event) => {
        this.reduceStreamEvent(sessionId, assistantMessageID, event);
      });

      const next = this.ensureSession(sessionId);
      if (next.status !== "error") {
        this.dispatch(sessionId, { type: "stream-end" });
      }
    } catch (error) {
      if (controller.signal.aborted) {
        this.dispatch(sessionId, { type: "stream-end" });
      } else {
        this.dispatch(sessionId, {
          type: "stream-error",
          error: error instanceof Error ? error.message : "Failed to stream response",
        });
      }
    } finally {
      this.inflight.delete(sessionId);
      this.markSessionRunning(sessionId, false);
      const current = this.ensureSession(sessionId);
      if (current.status === "streaming") {
        this.dispatch(sessionId, { type: "stream-end" });
      }
    }
  }

  private reduceStreamEvent(sessionId: string, assistantMessageID: string, event: StreamEvent) {
    const eventType = String(event.type || "");
    if (eventType === "text-delta") {
      const delta = typeof event.delta === "string" ? event.delta : "";
      if (delta) this.dispatch(sessionId, { type: "assistant-text-delta", messageID: assistantMessageID, delta });
      return;
    }
    if (eventType === "reasoning-delta") {
      const delta = typeof event.delta === "string" ? event.delta : "";
      if (delta) this.dispatch(sessionId, { type: "assistant-reasoning-delta", messageID: assistantMessageID, delta });
      return;
    }
    if (eventType === "tool-input-available") {
      const toolName = typeof event.toolName === "string" ? event.toolName : "tool";
      const toolCallID = typeof event.toolCallId === "string" ? event.toolCallId : "";
      this.dispatch(sessionId, {
        type: "assistant-tool-input",
        messageID: assistantMessageID,
        toolName,
        toolCallID,
        input: event.input,
      });
      return;
    }
    if (eventType === "tool-output-available" || eventType === "tool-output-error") {
      const toolCallID = typeof event.toolCallId === "string" ? event.toolCallId : "";
      this.dispatch(sessionId, {
        type: "assistant-tool-output",
        messageID: assistantMessageID,
        toolCallID,
        output: event.output,
        errorText: typeof event.errorText === "string" ? event.errorText : undefined,
        failed: eventType === "tool-output-error",
      });
      return;
    }
    if (eventType === "error") {
      this.dispatch(sessionId, {
        type: "stream-error",
        error: typeof event.errorText === "string" ? event.errorText : "Stream error",
      });
    }
  }

  private async consumeSSE(body: ReadableStream<Uint8Array>, onEvent: (event: StreamEvent) => void) {
    const decoder = new TextDecoder();
    const reader = body.getReader();
    let buffered = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });

      let splitIndex = buffered.indexOf("\n\n");
      while (splitIndex !== -1) {
        const rawEvent = buffered.slice(0, splitIndex);
        buffered = buffered.slice(splitIndex + 2);

        const payload = rawEvent
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("\n");

        if (payload && payload !== "[DONE]") {
          try {
            onEvent(JSON.parse(payload) as StreamEvent);
          } catch {
            // Ignore malformed payload.
          }
        }

        splitIndex = buffered.indexOf("\n\n");
      }
    }
  }

  private ensureGlobalSubscription() {
    if (typeof window === "undefined") return;
    if (this.ws) return;
    if (this.listeners.size === 0) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsURL = `${protocol}//${window.location.host}/api/ws/notifications`;
    try {
      this.ws = new WebSocket(wsURL);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.wsBackoffMs = 500;
    };

    this.ws.onmessage = (message) => {
      try {
        const envelope = JSON.parse(String(message.data || "{}")) as {
          type?: string;
          payload?: Record<string, unknown>;
        };
        this.reduceGlobalEvent(envelope.type || "", envelope.payload || {});
      } catch {
        // Ignore malformed websocket messages.
      }
    };

    this.ws.onclose = () => {
      this.ws = null;
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      if (this.ws) this.ws.close();
    };
  }

  private reduceGlobalEvent(eventType: string, payload: Record<string, unknown>) {
    const normalized = normalizeGlobalEvent(eventType, payload);
    if (!normalized) return;
    if (this.shouldIgnoreStaleGlobalEvent(normalized.sessionID, normalized.version)) {
      return;
    }

    switch (normalized.kind) {
      case "session-status":
        this.dispatch(normalized.sessionID, { type: "session-status", status: normalized.status });
        if (normalized.status === "streaming") {
          this.markSessionRunning(normalized.sessionID, true);
        }
        if (normalized.status === "idle") {
          this.markSessionRunning(normalized.sessionID, false);
        }
        return;

      case "session-created":
      case "session-updated":
        this.ensureSession(normalized.sessionID);
        return;

      case "session-deleted":
        this.sessions.delete(normalized.sessionID);
        this.eventVersions.delete(normalized.sessionID);
        this.statusMapDirty = true;
        this.markSessionRunning(normalized.sessionID, false);
        this.notify();
        return;

      case "message-updated":
        this.dispatch(normalized.sessionID, {
          type: "message-upsert-from-event",
          messageID: normalized.messageID,
          role: normalized.role === "user" ? "user" : "assistant",
        });
        return;

      case "message-removed":
        this.dispatch(normalized.sessionID, { type: "message-remove", messageID: normalized.messageID });
        return;

      case "part-updated":
        this.dispatch(normalized.sessionID, {
          type: "part-upsert-delta",
          messageID: normalized.messageID,
          partID: normalized.partID,
          delta: normalized.delta,
          text: normalized.text,
        });
        return;

      case "part-removed":
        this.dispatch(normalized.sessionID, {
          type: "part-remove",
          messageID: normalized.messageID,
          partID: normalized.partID,
        });
        return;

      default:
        return;
    }
  }

  private shouldIgnoreStaleGlobalEvent(sessionID: string, version: number): boolean {
    const current = this.eventVersions.get(sessionID);
    if (current != null && version <= current) {
      return true;
    }
    this.eventVersions.set(sessionID, version);
    return false;
  }

  private teardownGlobalSubscription() {
    if (this.wsReconnectTimer !== null && typeof window !== "undefined") {
      window.clearTimeout(this.wsReconnectTimer);
      this.wsReconnectTimer = null;
    }
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
  }

  private scheduleReconnect() {
    if (typeof window === "undefined") return;
    if (this.listeners.size === 0) return;
    if (this.wsReconnectTimer !== null) return;

    const delay = this.wsBackoffMs;
    this.wsBackoffMs = Math.min(this.wsBackoffMs * 2, 5000);
    this.wsReconnectTimer = window.setTimeout(() => {
      this.wsReconnectTimer = null;
      this.ensureGlobalSubscription();
    }, delay);
  }

  private markSessionRunning(sessionId: string, running: boolean) {
    if (typeof window === "undefined") return;
    const set = this.getPersistedRunningSet();
    if (running) set.add(sessionId);
    else set.delete(sessionId);
    this.savePersistedRunningSet(set);
  }

  private recoverPersistedRunningSessions() {
    if (typeof window === "undefined") return;
    const set = this.getPersistedRunningSet();
    if (set.size === 0) return;
    for (const sessionId of set) {
      this.dispatch(sessionId, { type: "session-status", status: "streaming" });
      void this.loadHistory(sessionId).finally(() => {
        if (!this.inflight.has(sessionId)) {
          this.dispatch(sessionId, { type: "session-status", status: "idle" });
          this.markSessionRunning(sessionId, false);
        }
      });
    }
  }

  private getPersistedRunningSet(): Set<string> {
    if (typeof window === "undefined") return new Set<string>();
    try {
      const raw = window.localStorage.getItem(RUNNING_SESSIONS_KEY);
      if (!raw) return new Set<string>();
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return new Set<string>();
      return new Set(parsed.filter((item): item is string => typeof item === "string" && item.trim() !== ""));
    } catch {
      return new Set<string>();
    }
  }

  private savePersistedRunningSet(set: Set<string>) {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(RUNNING_SESSIONS_KEY, JSON.stringify(Array.from(set)));
    } catch {
      // Ignore storage failures.
    }
  }
}

const runtimeStore = new ChatRuntimeStore();

export function useChatSessionRuntime(sessionId: string): SessionSnapshot {
  return useSyncExternalStore(
    runtimeStore.subscribe,
    () => runtimeStore.getSnapshot(sessionId),
    () => runtimeStore.getSnapshot(sessionId)
  );
}

export function useChatStatusMap(): Record<string, SessionStatus> {
  return useSyncExternalStore(
    runtimeStore.subscribe,
    () => runtimeStore.getStatusMap(),
    () => runtimeStore.getStatusMap()
  );
}

export const chatRuntime = {
  bootstrapForUser: (userId: string) => runtimeStore.bootstrapForUser(userId),
  loadHistory: (sessionId: string) => runtimeStore.loadHistory(sessionId),
  sendMessage: (input: { sessionId: string; userId: string; text: string }) =>
    runtimeStore.sendMessage(input),
};
