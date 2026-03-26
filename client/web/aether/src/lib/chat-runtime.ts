import { useSyncExternalStore } from "react";
import type { UIMessage, FileUIPart } from "ai";
import {
  ensureDirectAgentConnection,
  directAgentWs,
  getChatSession,
  getPendingQuestions,
  getChatSessionStatuses,
  getSessionToken,
  orchestratorWs,
  initMediaUpload,
  completeMediaUpload,
} from "#/lib/api";

export type SessionStatus = "idle" | "streaming" | "error";

export type QuestionField = {
  name: string;
  label: string;
  type: string;
  required?: boolean;
  placeholder?: string;
  options?: string[];
};

export type SessionState = {
  messages: UIMessage[];
  status: SessionStatus;
  error: string | null;
  loaded: boolean;
  loading: boolean;
  loopState: string | null;
  loopReason: string | null;
  questionRequest: {
    id: string;
    sessionId: string;
    toolCallId: string;
    question: string;
    header: string;
    kind: "choice" | "confirm" | "form";
    options: Array<{ label: string; description?: string }>;
    allowCustom: boolean;
    fields: QuestionField[];
    submitLabel: string;
  } | null;
};

type NormalizedGlobalEvent =
  | { kind: "session-status"; sessionID: string; status: SessionStatus; version: number }
  | { kind: "session-created"; sessionID: string; version: number }
  | { kind: "session-updated"; sessionID: string; archived?: boolean; version: number }
  | { kind: "session-deleted"; sessionID: string; version: number }
  | {
      kind: "message-updated";
      sessionID: string;
      messageID: string;
      role?: string;
      version: number;
    }
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
  | { kind: "part-removed"; sessionID: string; messageID: string; partID: string; version: number }
  | {
      kind: "question-asked";
      sessionID: string;
      version: number;
      request: NonNullable<SessionState["questionRequest"]>;
    }
  | { kind: "question-cleared"; sessionID: string; version: number };

type SessionSnapshot = SessionState;

type TurnMode = "text" | "voice";

type ConversationEventType =
  | "session.start"
  | "session.ready"
  | "session.stop"
  | "turn.start"
  | "turn.accepted"
  | "turn.input.text"
  | "turn.input.media"
  | "turn.input.audio.chunk"
  | "turn.commit"
  | "turn.cancel"
  | "turn.cancelled"
  | "assistant.text.delta"
  | "assistant.tool-input-available"
  | "assistant.tool-output-available"
  | "assistant.tool-output-error"
  | "assistant.done"
  | "error"
  | "ack";

type ConversationEnvelope = {
  v: number;
  type: ConversationEventType;
  event_id: string;
  session_id: string;
  turn_id: string;
  seq: number;
  ts: number;
  payload?: Record<string, unknown>;
};

type PendingTurn = {
  sessionId: string;
  turnId: string;
  assistantMessageId: string;
  mode: TurnMode;
  done: Promise<void>;
  completed: boolean;
  resolve: () => void;
  reject: (error: Error) => void;
};

type SessionReadyWaiter = {
  resolve: () => void;
  reject: (error: Error) => void;
};

type VoiceChunkInput = {
  sessionId: string;
  turnId: string;
  chunkBase64: string;
  mimeType?: string;
};

type SessionAction =
  | { type: "history-loading" }
  | { type: "history-loaded"; messages: UIMessage[] }
  | { type: "history-error"; error: string }
  | { type: "stream-start" }
  | { type: "stream-end" }
  | { type: "stream-error"; error: string }
  | { type: "append-user"; text: string; messageID: string; files?: FileUIPart[] }
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
      metadata?: unknown;
      errorText?: string;
      failed: boolean;
    }
  | { type: "session-status"; status: SessionStatus }
  | { type: "loop-state-update"; loopState: string; loopReason: string }
  | {
      type: "message-upsert-from-event";
      messageID: string;
      role: "assistant" | "user";
      text?: string;
    }
  | { type: "message-remove"; messageID: string }
  | { type: "part-upsert-delta"; messageID: string; partID: string; delta?: string; text?: string }
  | { type: "part-remove"; messageID: string; partID: string }
  | { type: "question-asked"; request: SessionState["questionRequest"] }
  | { type: "question-answered" };

const RUNNING_SESSIONS_KEY = "aether.chat.runtime.running.v1";
const WS_CONVERSATION_PATH = "/agent/v1/ws/conversation";
const WS_CONVERSATION_VERSION = 1;
const SESSION_CONTROL_TURN_ID = "session";
const SESSION_READY_TIMEOUT_MS = 10_000;
const DEFAULT_VOICE_USER_TEXT = "[voice instruction]";

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

function normalizeQuestionRequestPayload(
  payload: Record<string, unknown>,
  sessionID: string,
): NonNullable<SessionState["questionRequest"]> {
  const input = isRecord(payload.input) ? (payload.input as Record<string, unknown>) : payload;
  const rawKind = typeof input.kind === "string" ? input.kind.trim().toLowerCase() : "";
  const kind = rawKind === "confirm" || rawKind === "form" ? rawKind : "choice";
  const toolCallID =
    typeof input.tool_call_id === "string"
      ? input.tool_call_id
      : typeof payload.toolCallId === "string"
        ? payload.toolCallId
        : typeof payload.tool_call_id === "string"
          ? payload.tool_call_id
          : "";

  return {
    id:
      typeof input.question_id === "string"
        ? input.question_id
        : typeof input.id === "string"
          ? input.id
          : toolCallID,
    sessionId: sessionID,
    toolCallId: toolCallID,
    question: typeof input.question === "string" ? input.question : "",
    header: typeof input.header === "string" ? input.header : "Question",
    kind,
    options: Array.isArray(input.options)
      ? (input.options as unknown[]).filter(isRecord).map((o) => ({
          label: String(o.label || ""),
          description: typeof o.description === "string" ? o.description : undefined,
        }))
      : [],
    allowCustom: input.allow_custom !== false,
    fields: Array.isArray(input.fields)
      ? (input.fields as unknown[])
          .filter(isRecord)
          .map((field) => ({
            name: typeof field.name === "string" ? field.name : "",
            label:
              typeof field.label === "string"
                ? field.label
                : typeof field.name === "string"
                  ? field.name
                  : "Field",
            type: typeof field.type === "string" ? field.type : "text",
            required: field.required === true,
            placeholder: typeof field.placeholder === "string" ? field.placeholder : undefined,
            options: Array.isArray(field.options)
              ? field.options.filter((value): value is string => typeof value === "string")
              : undefined,
          }))
          .filter((field) => field.name.trim().length > 0)
      : [],
    submitLabel: typeof input.submit_label === "string" ? input.submit_label : "Submit",
  };
}

function parseStoredToolCalls(content: unknown): Array<{
  toolCallId: string;
  toolName: string;
  input: unknown;
}> {
  if (!isRecord(content) || !Array.isArray(content.tool_calls)) {
    return [];
  }

  const calls: Array<{ toolCallId: string; toolName: string; input: unknown }> = [];
  for (const entry of content.tool_calls) {
    if (!isRecord(entry)) continue;
    const toolCallId = typeof entry.id === "string" ? entry.id : "";
    const fn = isRecord(entry.function) ? entry.function : null;
    const toolName = fn && typeof fn.name === "string" ? fn.name : "tool";
    let input: unknown = undefined;
    if (fn && typeof fn.arguments === "string") {
      try {
        input = JSON.parse(fn.arguments);
      } catch {
        input = fn.arguments;
      }
    }
    calls.push({ toolCallId, toolName, input });
  }
  return calls;
}

function ensureHistoryAssistantMessage(
  restored: UIMessage[],
  current: UIMessage | null,
  rowID: number,
): UIMessage {
  if (current) return current;
  const created: UIMessage = {
    id: `msg-${rowID}-assistant`,
    role: "assistant",
    parts: [],
  };
  restored.push(created);
  return created;
}

function buildHistoryMessages(
  rows: Array<{ id: number; role?: string; content: unknown }>,
): UIMessage[] {
  const restored: UIMessage[] = [];
  let currentAssistant: UIMessage | null = null;

  for (const row of rows) {
    const role =
      typeof row.role === "string"
        ? row.role
        : isRecord(row.content) && typeof row.content.role === "string"
          ? row.content.role
          : "";
    const storedContent = isRecord(row.content) ? row.content : null;

    if (role === "user") {
      currentAssistant = null;
    }

    const text = textFromStoredMessageContent(row.content);
    if (role === "user") {
      if (!text.trim()) continue;
      restored.push({
        id: `msg-${row.id}-user`,
        role: "user",
        parts: [{ type: "text", text }],
      });
      continue;
    }

    if (role === "assistant") {
      const toolCalls = parseStoredToolCalls(row.content);
      if (toolCalls.length === 0 && !text.trim()) {
        continue;
      }
      if (toolCalls.length > 0) {
        currentAssistant = ensureHistoryAssistantMessage(restored, currentAssistant, row.id);
        for (const call of toolCalls) {
          currentAssistant.parts.push({
            type: `tool-${call.toolName}`,
            state: "input-available",
            toolCallId: call.toolCallId,
            input: call.input,
          } as UIMessage["parts"][number]);
        }
      }

      if (text.trim()) {
        currentAssistant = ensureHistoryAssistantMessage(restored, currentAssistant, row.id);
        currentAssistant.parts.push({ type: "text", text } as UIMessage["parts"][number]);
      }
      continue;
    }

    if (role === "tool" && storedContent) {
      const toolCallId = typeof storedContent.tool_call_id === "string" ? storedContent.tool_call_id : "";
      const rawMetadata = isRecord(storedContent.metadata) ? storedContent.metadata : undefined;
      const contentText = typeof storedContent.content === "string" ? storedContent.content : text;
      if (!toolCallId && !contentText.trim()) {
        continue;
      }
      const failed = contentText.startsWith("[tool_error] ");
      const output = failed ? undefined : contentText;
      const errorText = failed ? contentText.replace(/^\[tool_error\]\s*/, "") : undefined;

      currentAssistant = ensureHistoryAssistantMessage(restored, currentAssistant, row.id);
      const partIndex = currentAssistant.parts.findIndex((part) => {
        if (!String(part.type).startsWith("tool-")) return false;
        return String((part as Record<string, unknown>).toolCallId || "") === toolCallId;
      });

      if (partIndex >= 0) {
        const candidate = currentAssistant.parts[partIndex] as Record<string, unknown>;
        currentAssistant.parts[partIndex] = {
          ...candidate,
          state: failed ? "output-error" : "output-available",
          ...(failed ? { errorText } : { output }),
          ...(rawMetadata ? { metadata: rawMetadata } : {}),
        } as UIMessage["parts"][number];
      } else {
        currentAssistant.parts.push({
          type: "tool-unknown",
          state: failed ? "output-error" : "output-available",
          toolCallId,
          input: {},
          ...(failed ? { errorText } : { output }),
          ...(rawMetadata ? { metadata: rawMetadata } : {}),
        } as unknown as UIMessage["parts"][number]);
      }
    }
  }
  return restored;
}

function randomID(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function envelopeFromUnknown(value: unknown): ConversationEnvelope | null {
  if (!isRecord(value)) return null;
  const type = String(value.type || "") as ConversationEventType;
  const eventID = typeof value.event_id === "string" ? value.event_id : "";
  const sessionID = typeof value.session_id === "string" ? value.session_id : "";
  const turnID = typeof value.turn_id === "string" ? value.turn_id : "";
  const seq = typeof value.seq === "number" && Number.isFinite(value.seq) ? value.seq : 0;
  const ts = typeof value.ts === "number" && Number.isFinite(value.ts) ? value.ts : Date.now();
  const payload = isRecord(value.payload) ? value.payload : undefined;

  if (!type || !eventID || !sessionID) return null;
  return {
    v: typeof value.v === "number" && Number.isFinite(value.v) ? value.v : WS_CONVERSATION_VERSION,
    type,
    event_id: eventID,
    session_id: sessionID,
    turn_id: turnID,
    seq,
    ts,
    payload,
  };
}

function parseEventVersion(payload: Record<string, unknown>): number {
  const candidates = [
    payload.sequence,
    payload.seq,
    payload.version,
    payload.updatedAt,
    payload.updated_at,
    payload.timestamp,
  ];
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

function normalizeGlobalEvent(
  eventType: string,
  payload: Record<string, unknown>,
): NormalizedGlobalEvent | null {
  const normalizedType = String(eventType || "").toLowerCase();
  const sessionID =
    typeof payload.sessionID === "string"
      ? payload.sessionID
      : typeof payload.session_id === "string"
        ? payload.session_id
        : "";
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
    const messageID =
      typeof payload.messageID === "string" ? payload.messageID : "assistant-current";
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
  if (normalizedType === "question.asked") {
    return {
      kind: "question-asked",
      sessionID,
      version,
      request: normalizeQuestionRequestPayload(payload, sessionID),
    };
  }
  if (normalizedType === "question.replied" || normalizedType === "question.rejected") {
    return { kind: "question-cleared", sessionID, version };
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
      return {
        ...current,
        loaded: true,
        loading: false,
        status: "streaming",
        error: null,
        loopState: null,
        loopReason: null,
      };
    case "stream-end":
      return {
        ...current,
        status: current.status === "error" ? "error" : "idle",
        error: current.status === "error" ? current.error : null,
        loopState: null,
        loopReason: null,
      };
    case "stream-error": {
      // Remove empty assistant messages on error
      const filteredMessages = current.messages.filter((msg) => {
        if (msg.role !== "assistant") return true;
        // Keep assistant messages that have content
        if (msg.parts.length === 0) return false;
        if (msg.parts.length === 1 && msg.parts[0].type === "text" && !msg.parts[0].text.trim()) return false;
        return true;
      });
      return { ...current, status: "error", error: action.error, messages: filteredMessages };
    }
    case "append-user": {
      const parts: UIMessage["parts"] = [];
      if (action.files && action.files.length > 0) {
        for (const file of action.files) {
          parts.push(file as UIMessage["parts"][number]);
        }
      }
      if (action.text.trim()) {
        parts.push({ type: "text", text: action.text });
      }
      if (parts.length === 0) {
        parts.push({ type: "text", text: " " });
      }
      const msg: UIMessage = {
        id: action.messageID,
        role: "user",
        parts,
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
          if (action.metadata !== undefined) {
            nextPart.metadata = action.metadata;
          }
          return nextPart as UIMessage["parts"][number];
        });
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "session-status":
      return { ...current, status: action.status };
    case "loop-state-update":
      return { ...current, loopState: action.loopState, loopReason: action.loopReason };
    case "message-upsert-from-event": {
      const idx = current.messages.findIndex((message) => message.id === action.messageID);
      if (idx === -1) {
        // Only create new message if there's text content
        if (!action.text) {
          return current;
        }
        const next: UIMessage = {
          id: action.messageID,
          role: action.role,
          parts: [{ type: "text", text: action.text }],
        };
        return { ...current, messages: [...current.messages, next] };
      }
      const existing = current.messages[idx];
      const nextMessages = [...current.messages];
      if (action.text) {
        nextMessages[idx] = {
          ...existing,
          role: action.role,
          parts: [{ type: "text", text: action.text }],
        };
      } else {
        nextMessages[idx] = { ...existing, role: action.role };
      }
      return { ...current, messages: nextMessages };
    }
    case "message-remove":
      return {
        ...current,
        messages: current.messages.filter((message) => message.id !== action.messageID),
      };
    case "part-upsert-delta": {
      const messages = current.messages.map((message) => {
        if (message.id !== action.messageID) return message;
        const parts = [...message.parts];
        const existingIndex = parts.findIndex(
          (part) => String((part as Record<string, unknown>).partId || "") === action.partID,
        );
        const incomingText = action.text ?? action.delta ?? "";
        if (existingIndex === -1) {
          parts.push({
            type: "text",
            text: incomingText,
            partId: action.partID,
          } as UIMessage["parts"][number]);
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
        const parts = message.parts.filter(
          (part) => String((part as Record<string, unknown>).partId || "") !== action.partID,
        );
        return { ...message, parts };
      });
      return { ...current, messages };
    }
    case "question-asked":
      return { ...current, questionRequest: action.request };
    case "question-answered":
      return { ...current, questionRequest: null };
    default:
      return current;
  }
}

class ChatRuntimeStore {
  private sessions = new Map<string, SessionState>();
  private eventVersions = new Map<string, number>();
  private inflight = new Map<string, PendingTurn>();
  private listeners = new Set<() => void>();
  private bootstrappedUsers = new Set<string>();
  private statusMapCache: Record<string, SessionStatus> = {};
  private statusMapDirty = true;
  private globalWS: WebSocket | null = null;
  private globalWSReconnectTimer: number | null = null;
  private globalWSBackoffMs = 500;
  private conversationWS: WebSocket | null = null;
  private conversationWSConnectPromise: Promise<void> | null = null;
  private conversationWSReconnectTimer: number | null = null;
  private conversationWSBackoffMs = 500;
  private conversationSeqByTurn = new Map<string, number>();
  private pendingAckEventIDs = new Set<string>();
  private processedServerEventIDs = new Set<string>();
  private pendingSessionReady = new Map<string, SessionReadyWaiter>();
  private readySessions = new Set<string>();

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
      loopState: null,
      loopReason: null,
      questionRequest: null,
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
    loopState: null,
    loopReason: null,
    questionRequest: null,
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
      try {
        const pending = await getPendingQuestions(sessionId);
        const first = pending.questions?.[0];
        if (first && isRecord(first)) {
          this.dispatch(sessionId, {
            type: "question-asked",
            request: normalizeQuestionRequestPayload(first, sessionId),
          });
        } else {
          this.dispatch(sessionId, { type: "question-answered" });
        }
      } catch {
        // Ignore pending question load failures; history is still usable.
      }
    } catch {
      this.dispatch(sessionId, { type: "history-error", error: "Failed to load chat history" });
    }
  }

  async sendMessage(input: { 
    sessionId: string; 
    userId: string; 
    text: string;
    files?: FileUIPart[];
  }): Promise<void> {
    const sessionId = input.sessionId.trim();
    const userId = input.userId.trim();
    const text = input.text.trim();
    
    if (input.files && input.files.length > 0) {
      await this.sendMediaTurn({ sessionId, userId, text, files: input.files });
    } else {
      await this.sendTextTurn({ sessionId, userId, text });
    }
  }

  private async sendMediaTurn(input: {
    sessionId: string;
    userId: string;
    text: string;
    files: FileUIPart[];
  }): Promise<void> {
    const turn = await this.beginTurn({
      sessionId: input.sessionId,
      userId: input.userId,
      mode: "text",
      userText: input.text || "[media]",
      files: input.files,
    });

    try {
      // 1. Upload files to S3 and collect media refs
      const mediaRefs: Array<{ kind: string; media: Record<string, unknown> }> = [];

      for (const file of input.files) {
        if (!file.url) continue;
        
        // Convert blob URL to data URL if needed
        let dataUrl = file.url;
        if (file.url.startsWith("blob:")) {
          const response = await fetch(file.url);
          const blob = await response.blob();
          const reader = new FileReader();
          dataUrl = await new Promise<string>((resolve) => {
            reader.onloadend = () => resolve(reader.result as string);
            reader.readAsDataURL(blob);
          });
        }

        // Determine media type from file.mediaType or data URL prefix
        let mediaType = file.mediaType;
        if (!mediaType && dataUrl.startsWith("data:")) {
          const match = dataUrl.match(/^data:([^;]+);/);
          if (match) {
            mediaType = match[1];
          }
        }
        
        const isImage = mediaType?.startsWith("image/") ?? false;
        const kind = isImage ? "image" : "audio";
        const contentType = mediaType || (isImage ? "image/png" : "audio/webm");

        if (dataUrl.startsWith("data:")) {
          // Convert data URL to blob first so we know the size
          const base64Data = dataUrl.split(",")[1];
          const binaryString = atob(base64Data);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          const blob = new Blob([bytes], { type: contentType });

          // Upload to S3
          const initResult = await initMediaUpload({
            user_id: input.userId,
            session_id: input.sessionId,
            file_name: file.filename || "file",
            content_type: contentType,
            size: blob.size,
            kind,
          });

          const uploadResponse = await fetch(initResult.upload_url, {
            method: "PUT",
            body: blob,
            headers: initResult.headers,
          });

          if (!uploadResponse.ok) {
            throw new Error(`Failed to upload file: ${uploadResponse.statusText}`);
          }

          const completeResult = await completeMediaUpload({
            user_id: input.userId,
            bucket: initResult.bucket,
            object_key: initResult.object_key,
            file_name: file.filename || "file",
            content_type: contentType,
            size: blob.size,
            kind,
          });

          mediaRefs.push({
            kind,
            media: completeResult.media as unknown as Record<string, unknown>,
          });
        }
      }

      // 2. Send media refs over WebSocket
      if (mediaRefs.length > 0) {
        this.sendConversationEnvelope({
          type: "turn.input.media",
          sessionId: turn.sessionId,
          turnId: turn.turnId,
          payload: { media_refs: mediaRefs },
        });
      }

      // 3. Send text if present
      if (input.text.trim()) {
        this.sendConversationEnvelope({
          type: "turn.input.text",
          sessionId: turn.sessionId,
          turnId: turn.turnId,
          payload: { text: input.text },
        });
      }

      // 4. Commit turn
      this.sendConversationEnvelope({
        type: "turn.commit",
        sessionId: turn.sessionId,
        turnId: turn.turnId,
      });

      await turn.done;
    } catch (error) {
      const pending = this.inflight.get(turn.sessionId);
      if (pending && pending.turnId === turn.turnId) {
        this.failTurn(
          turn.sessionId,
          error instanceof Error ? error : new Error("Failed to stream response"),
        );
      }
      throw error;
    }
  }

  async startVoiceTurn(input: {
    sessionId: string;
    userId: string;
    textHint?: string;
  }): Promise<{ turnId: string }> {
    const sessionId = input.sessionId.trim();
    const userId = input.userId.trim();
    if (!sessionId || !userId) {
      throw new Error("sessionId and userId are required");
    }
    const turn = await this.beginTurn({
      sessionId,
      userId,
      mode: "voice",
      userText: input.textHint?.trim() || DEFAULT_VOICE_USER_TEXT,
    });
    turn.done.catch(() => {
      // Voice flow may be controlled by external commit/cancel handlers.
    });
    return { turnId: turn.turnId };
  }

  async sendVoiceChunk(input: VoiceChunkInput): Promise<void> {
    const sessionId = input.sessionId.trim();
    const turnId = input.turnId.trim();
    const chunkBase64 = input.chunkBase64.trim();
    if (!sessionId || !turnId || !chunkBase64) return;
    const inflight = this.inflight.get(sessionId);
    if (!inflight || inflight.turnId !== turnId || inflight.mode !== "voice") {
      throw new Error("No active voice turn for session");
    }
    this.sendConversationEnvelope({
      type: "turn.input.audio.chunk",
      sessionId,
      turnId,
      payload: {
        audio: chunkBase64,
        mime_type: input.mimeType || "audio/webm",
      },
    });
  }

  async commitVoiceTurn(input: { sessionId: string; turnId: string }): Promise<void> {
    const sessionId = input.sessionId.trim();
    const turnId = input.turnId.trim();
    if (!sessionId || !turnId) return;
    const inflight = this.inflight.get(sessionId);
    if (!inflight || inflight.turnId !== turnId || inflight.mode !== "voice") {
      throw new Error("No active voice turn for session");
    }
    this.sendConversationEnvelope({
      type: "turn.commit",
      sessionId,
      turnId,
    });
    const current = this.inflight.get(sessionId);
    if (!current || current.turnId !== turnId) return;
    await current.done;
  }

  async cancelTurn(sessionId: string): Promise<void> {
    const normalized = sessionId.trim();
    if (!normalized) return;
    const inflight = this.inflight.get(normalized);
    if (!inflight) return;

    try {
      this.sendConversationEnvelope({
        type: "turn.cancel",
        sessionId: normalized,
        turnId: inflight.turnId,
      });
    } catch {
      // Ignore send failures during cancellation.
    }

    this.finishTurn(normalized);
  }

  private async sendTextTurn(input: {
    sessionId: string;
    userId: string;
    text: string;
  }): Promise<void> {
    const turn = await this.beginTurn({
      sessionId: input.sessionId,
      userId: input.userId,
      mode: "text",
      userText: input.text,
    });

    try {
      this.sendConversationEnvelope({
        type: "turn.input.text",
        sessionId: turn.sessionId,
        turnId: turn.turnId,
        payload: { text: input.text },
      });
      this.sendConversationEnvelope({
        type: "turn.commit",
        sessionId: turn.sessionId,
        turnId: turn.turnId,
      });

      await turn.done;
    } catch (error) {
      const pending = this.inflight.get(turn.sessionId);
      if (pending && pending.turnId === turn.turnId) {
        this.failTurn(
          turn.sessionId,
          error instanceof Error ? error : new Error("Failed to stream response"),
        );
      }
      throw error;
    }
  }

  private async beginTurn(input: {
    sessionId: string;
    userId: string;
    mode: TurnMode;
    userText: string;
    files?: FileUIPart[];
  }): Promise<{ sessionId: string; turnId: string; done: Promise<void> }> {
    const sessionId = input.sessionId.trim();
    if (!sessionId) {
      throw new Error("sessionId is required");
    }
    if (this.inflight.has(sessionId)) {
      throw new Error("A turn is already in progress for this session");
    }

    const userMessageID = randomID("user");
    const assistantMessageID = randomID("assistant");
    const turnId = randomID("turn");

    this.dispatch(sessionId, {
      type: "append-user",
      text: input.userText,
      messageID: userMessageID,
      files: input.files,
    });
    this.dispatch(sessionId, { type: "append-assistant", messageID: assistantMessageID });
    this.dispatch(sessionId, { type: "stream-start" });
    this.markSessionRunning(sessionId, true);

    let resolveTurn: () => void = () => {};
    let rejectTurn: (error: Error) => void = () => {};
    const done = new Promise<void>((resolve, reject) => {
      resolveTurn = resolve;
      rejectTurn = reject;
    });

    this.inflight.set(sessionId, {
      sessionId,
      turnId,
      assistantMessageId: assistantMessageID,
      mode: input.mode,
      done,
      completed: false,
      resolve: resolveTurn,
      reject: rejectTurn,
    });

    try {
      await this.ensureConversationConnection();
      await this.ensureConversationSessionReady(sessionId, input.userId);
      this.sendConversationEnvelope({
        type: "turn.start",
        sessionId,
        turnId,
        payload: { mode: input.mode },
      });
    } catch (error) {
      this.failTurn(
        sessionId,
        error instanceof Error ? error : new Error("Failed to connect to conversation service"),
      );
      throw error;
    }

    return { sessionId, turnId, done };
  }

  private finishTurn(sessionId: string) {
    const inflight = this.inflight.get(sessionId);
    if (!inflight || inflight.completed) return;
    inflight.completed = true;
    this.inflight.delete(sessionId);
    this.markSessionRunning(sessionId, false);
    const current = this.ensureSession(sessionId);
    if (current.status === "streaming") {
      this.dispatch(sessionId, { type: "stream-end" });
    }
    inflight.resolve();
  }

  private failTurn(sessionId: string, error: Error) {
    const inflight = this.inflight.get(sessionId);
    if (inflight && !inflight.completed) {
      inflight.completed = true;
      this.inflight.delete(sessionId);
      this.markSessionRunning(sessionId, false);
      this.dispatch(sessionId, {
        type: "stream-error",
        error: error.message || "Failed to stream response",
      });
      inflight.reject(error);
      return;
    }

    this.markSessionRunning(sessionId, false);
    this.dispatch(sessionId, {
      type: "stream-error",
      error: error.message || "Failed to stream response",
    });
  }

  private nextSeq(turnId: string): number {
    const next = (this.conversationSeqByTurn.get(turnId) || 0) + 1;
    this.conversationSeqByTurn.set(turnId, next);
    return next;
  }

  private sendConversationEnvelope(input: {
    type: ConversationEventType;
    sessionId: string;
    turnId: string;
    payload?: Record<string, unknown>;
  }) {
    const socket = this.conversationWS;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("Conversation websocket is not connected");
    }
    const envelope: ConversationEnvelope = {
      v: WS_CONVERSATION_VERSION,
      type: input.type,
      event_id: randomID("evt"),
      session_id: input.sessionId,
      turn_id: input.turnId,
      seq: this.nextSeq(input.turnId),
      ts: Date.now(),
      payload: input.payload,
    };
    socket.send(JSON.stringify(envelope));
    if (input.type !== "ack") {
      this.pendingAckEventIDs.add(envelope.event_id);
    }
  }

  private async ensureConversationConnection(): Promise<void> {
    if (typeof window === "undefined") {
      throw new Error("Conversation websocket requires browser runtime");
    }
    if (this.conversationWS && this.conversationWS.readyState === WebSocket.OPEN) {
      return;
    }
    if (this.conversationWSConnectPromise) {
      return this.conversationWSConnectPromise;
    }

    this.conversationWSConnectPromise = new Promise<void>((resolve, reject) => {
      const token = getSessionToken() || undefined;
      let socket: WebSocket;
      let settled = false;
      let fallbackPending = false;
      const resolveOnce = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      const rejectOnce = (error: Error) => {
        if (settled) return;
        settled = true;
        reject(error);
      };
      const connectProxyFallback = () => {
        if (fallbackPending || settled) return;
        fallbackPending = true;
        try {
          attachSocket(orchestratorWs(WS_CONVERSATION_PATH, token), false);
        } catch {
          this.conversationWSConnectPromise = null;
          rejectOnce(new Error("Failed to create conversation websocket"));
        }
      };

      const attachSocket = (nextSocket: WebSocket, allowFallback: boolean) => {
        socket = nextSocket;
        this.conversationWS = socket;

        socket.onopen = () => {
          this.conversationWSBackoffMs = 500;
          this.conversationWSConnectPromise = null;
          resolveOnce();
        };

        socket.onmessage = (message) => {
          this.handleConversationSocketMessage(message.data);
        };

        socket.onclose = () => {
          if (!settled && allowFallback) {
            this.conversationWS = null;
            connectProxyFallback();
            return;
          }
          this.conversationWS = null;
          this.conversationWSConnectPromise = null;
          this.readySessions.clear();
          this.conversationSeqByTurn.clear();
          this.rejectSessionReadyWaiters(new Error("Conversation websocket closed"));
          this.failAllTurns(new Error("Conversation disconnected"));
          rejectOnce(new Error("Conversation websocket closed"));
          this.scheduleConversationReconnect();
        };

        socket.onerror = () => {
          if (!settled && allowFallback) {
            if (
              socket.readyState === WebSocket.OPEN ||
              socket.readyState === WebSocket.CONNECTING
            ) {
              socket.close();
            }
            connectProxyFallback();
            return;
          }
          rejectOnce(new Error("Conversation websocket error"));
          if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            socket.close();
          }
        };
      };

      void ensureDirectAgentConnection()
        .then((direct) => {
          try {
            attachSocket(
              direct
                ? directAgentWs("/agent/v1/ws/conversation", direct)
                : orchestratorWs(WS_CONVERSATION_PATH, token),
              Boolean(direct),
            );
          } catch {
            this.conversationWSConnectPromise = null;
            rejectOnce(new Error("Failed to create conversation websocket"));
          }
        })
        .catch(() => {
          try {
            attachSocket(orchestratorWs(WS_CONVERSATION_PATH, token), false);
          } catch {
            this.conversationWSConnectPromise = null;
            rejectOnce(new Error("Failed to create conversation websocket"));
          }
        });
    });

    return this.conversationWSConnectPromise;
  }

  private scheduleConversationReconnect() {
    if (typeof window === "undefined") return;
    if (this.conversationWSReconnectTimer !== null) return;
    if (this.inflight.size === 0 && this.listeners.size === 0) return;

    const delay = this.conversationWSBackoffMs;
    this.conversationWSBackoffMs = Math.min(this.conversationWSBackoffMs * 2, 5000);
    this.conversationWSReconnectTimer = window.setTimeout(() => {
      this.conversationWSReconnectTimer = null;
      void this.ensureConversationConnection().catch(() => {
        this.scheduleConversationReconnect();
      });
    }, delay);
  }

  private async ensureConversationSessionReady(sessionId: string, userId: string): Promise<void> {
    if (this.readySessions.has(sessionId)) return;

    const pending = new Promise<void>((resolve, reject) => {
      this.pendingSessionReady.set(sessionId, { resolve, reject });
      window.setTimeout(() => {
        const waiter = this.pendingSessionReady.get(sessionId);
        if (waiter) {
          this.pendingSessionReady.delete(sessionId);
          waiter.reject(new Error("Timed out waiting for conversation session.ready"));
        }
      }, SESSION_READY_TIMEOUT_MS);
    });

    this.sendConversationEnvelope({
      type: "session.start",
      sessionId,
      turnId: SESSION_CONTROL_TURN_ID,
      payload: { user_id: userId },
    });

    await pending;
  }

  private rejectSessionReadyWaiters(error: Error) {
    for (const [, waiter] of this.pendingSessionReady.entries()) {
      waiter.reject(error);
    }
    this.pendingSessionReady.clear();
  }

  private failAllTurns(error: Error) {
    const sessionIds = Array.from(this.inflight.keys());
    for (const sessionId of sessionIds) {
      this.failTurn(sessionId, error);
    }
  }

  private handleConversationSocketMessage(raw: unknown) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(String(raw || "{}"));
    } catch {
      return;
    }

    const envelope = envelopeFromUnknown(parsed);
    if (!envelope) return;

    if (this.processedServerEventIDs.has(envelope.event_id)) {
      return;
    }
    this.processedServerEventIDs.add(envelope.event_id);
    if (this.processedServerEventIDs.size > 1000) {
      const [first] = this.processedServerEventIDs;
      if (first) this.processedServerEventIDs.delete(first);
    }

    if (envelope.type !== "ack") {
      try {
        this.sendConversationEnvelope({
          type: "ack",
          sessionId: envelope.session_id,
          turnId: envelope.turn_id || SESSION_CONTROL_TURN_ID,
          payload: { event_id: envelope.event_id },
        });
      } catch {
        // Ignore ack send failures.
      }
    }

    this.reduceConversationEvent(envelope);
  }

  private reduceConversationEvent(envelope: ConversationEnvelope) {
    const sessionId = envelope.session_id;
    const inflight = this.inflight.get(sessionId);

    switch (envelope.type) {
      case "ack": {
        const ackEventID =
          typeof envelope.payload?.event_id === "string" ? envelope.payload.event_id : "";
        if (ackEventID) {
          this.pendingAckEventIDs.delete(ackEventID);
        }
        return;
      }
      case "session.ready": {
        this.readySessions.add(sessionId);
        const waiter = this.pendingSessionReady.get(sessionId);
        if (waiter) {
          this.pendingSessionReady.delete(sessionId);
          waiter.resolve();
        }
        return;
      }
      case "turn.accepted":
        return;
      case "assistant.text.delta": {
        if (!inflight) return;
        const delta =
          typeof envelope.payload?.delta === "string"
            ? envelope.payload.delta
            : typeof envelope.payload?.text === "string"
              ? envelope.payload.text
              : "";
        if (!delta) return;
        this.dispatch(sessionId, {
          type: "assistant-text-delta",
          messageID: inflight.assistantMessageId,
          delta,
        });
        return;
      }
      case "assistant.tool-input-available": {
        if (!inflight) return;
        const toolName =
          typeof envelope.payload?.toolName === "string" ? envelope.payload.toolName : "tool";
        const toolCallID =
          typeof envelope.payload?.toolCallId === "string" ? envelope.payload.toolCallId : "";
        this.dispatch(sessionId, {
          type: "assistant-tool-input",
          messageID: inflight.assistantMessageId,
          toolName,
          toolCallID,
          input: envelope.payload?.input,
        });
        return;
      }
      case "assistant.tool-output-available": {
        if (!inflight) return;
        const toolCallID =
          typeof envelope.payload?.toolCallId === "string" ? envelope.payload.toolCallId : "";
        this.dispatch(sessionId, {
          type: "assistant-tool-output",
          messageID: inflight.assistantMessageId,
          toolCallID,
          output: envelope.payload?.output,
          metadata: envelope.payload?.metadata,
          failed: false,
        });
        return;
      }
      case "assistant.tool-output-error": {
        if (!inflight) return;
        const toolCallID =
          typeof envelope.payload?.toolCallId === "string" ? envelope.payload.toolCallId : "";
        const errorText =
          typeof envelope.payload?.errorText === "string"
            ? envelope.payload.errorText
            : "Tool failed";
        this.dispatch(sessionId, {
          type: "assistant-tool-output",
          messageID: inflight.assistantMessageId,
          toolCallID,
          metadata: envelope.payload?.metadata,
          failed: true,
          errorText,
        });
        return;
      }
      case "assistant.done": {
        this.finishTurn(sessionId);
        return;
      }
      case "turn.cancelled": {
        this.finishTurn(sessionId);
        return;
      }
      case "error": {
        const message =
          typeof envelope.payload?.message === "string"
            ? envelope.payload.message
            : typeof envelope.payload?.error === "string"
              ? envelope.payload.error
              : "Conversation error";
        this.failTurn(sessionId, new Error(message));
        return;
      }
      default:
        return;
    }
  }

  private ensureGlobalSubscription() {
    if (typeof window === "undefined") return;
    if (this.globalWS) return;
    if (this.listeners.size === 0) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsURL = `${protocol}//${window.location.host}/agent/v1/ws/notifications`;
    try {
      this.globalWS = new WebSocket(wsURL);
    } catch {
      this.scheduleGlobalReconnect();
      return;
    }

    this.globalWS.onopen = () => {
      this.globalWSBackoffMs = 500;
    };

    this.globalWS.onmessage = (message) => {
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

    this.globalWS.onclose = () => {
      this.globalWS = null;
      this.scheduleGlobalReconnect();
    };

    this.globalWS.onerror = () => {
      if (this.globalWS) this.globalWS.close();
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
        this.dispatch(normalized.sessionID, {
          type: "message-remove",
          messageID: normalized.messageID,
        });
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

      case "question-asked":
        this.dispatch(normalized.sessionID, {
          type: "question-asked",
          request: normalized.request,
        });
        return;

      case "question-cleared":
        this.dispatch(normalized.sessionID, { type: "question-answered" });
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
    if (this.globalWSReconnectTimer !== null && typeof window !== "undefined") {
      window.clearTimeout(this.globalWSReconnectTimer);
      this.globalWSReconnectTimer = null;
    }
    if (this.globalWS) {
      const ws = this.globalWS;
      this.globalWS = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }

    if (this.conversationWSReconnectTimer !== null && typeof window !== "undefined") {
      window.clearTimeout(this.conversationWSReconnectTimer);
      this.conversationWSReconnectTimer = null;
    }
    if (this.conversationWS) {
      const ws = this.conversationWS;
      this.conversationWS = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    }
    this.readySessions.clear();
    this.conversationSeqByTurn.clear();
    this.pendingAckEventIDs.clear();
    this.processedServerEventIDs.clear();
    this.rejectSessionReadyWaiters(new Error("Conversation websocket closed"));
  }

  private scheduleGlobalReconnect() {
    if (typeof window === "undefined") return;
    if (this.listeners.size === 0) return;
    if (this.globalWSReconnectTimer !== null) return;

    const delay = this.globalWSBackoffMs;
    this.globalWSBackoffMs = Math.min(this.globalWSBackoffMs * 2, 5000);
    this.globalWSReconnectTimer = window.setTimeout(() => {
      this.globalWSReconnectTimer = null;
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
      return new Set(
        parsed.filter((item): item is string => typeof item === "string" && item.trim() !== ""),
      );
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
    () => runtimeStore.getSnapshot(sessionId),
  );
}

export function useChatStatusMap(): Record<string, SessionStatus> {
  return useSyncExternalStore(
    runtimeStore.subscribe,
    () => runtimeStore.getStatusMap(),
    () => runtimeStore.getStatusMap(),
  );
}

export const chatRuntime = {
  bootstrapForUser: (userId: string) => runtimeStore.bootstrapForUser(userId),
  loadHistory: (sessionId: string) => runtimeStore.loadHistory(sessionId),
  sendMessage: (input: { sessionId: string; userId: string; text: string; files?: FileUIPart[] }) =>
    runtimeStore.sendMessage(input),
  startVoiceTurn: (input: { sessionId: string; userId: string; textHint?: string }) =>
    runtimeStore.startVoiceTurn(input),
  sendVoiceChunk: (input: VoiceChunkInput) => runtimeStore.sendVoiceChunk(input),
  commitVoiceTurn: (input: { sessionId: string; turnId: string }) =>
    runtimeStore.commitVoiceTurn(input),
  cancelTurn: (sessionId: string) => runtimeStore.cancelTurn(sessionId),
};
