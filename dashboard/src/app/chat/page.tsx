"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { getWsUrl } from "@/lib/api";
import StatusOrb from "@/components/StatusOrb";
import type { AgentStatus } from "@/hooks/useAgentStatus";

/**
 * Chat — text conversation with Aether.
 * Header, scrollable messages, input pinned at bottom.
 * Reconnects automatically with exponential backoff on disconnect.
 */

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_RECONNECT_DELAY = 1000; // 1 second
const MAX_RECONNECT_DELAY = 30000; // 30 seconds

/** Notification data from the agent */
interface NotificationData {
  event_id?: string;
  plugin?: string;
  level: "speak" | "nudge" | "batch";
  text: string;
  actions?: string[];
  items?: Array<{
    event_id: string;
    plugin: string;
    text: string;
    actions?: string[];
  }>;
}

/** Map the chat page's WS status string to an AgentStatus for the orb. */
function wsStatusToOrb(status: string): AgentStatus {
  if (status === "connected") return "connected";
  if (status === "listening...") return "listening";
  if (status === "thinking...") return "thinking";
  if (status.startsWith("reconnecting") || status.startsWith("connecting")) return "unknown";
  if (status.startsWith("disconnected") || status === "auth failed") return "disconnected";
  return "connected";
}

export default function ChatPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef(0);
  const unmounted = useRef(false);
  const [messages, setMessages] = useState<{ role: string; text: string }[]>(
    []
  );
  const [notifications, setNotifications] = useState<NotificationData[]>([]);
  const [status, setStatus] = useState("connecting...");
  const [isStreaming, setIsStreaming] = useState(false);  // Track if response is streaming
  const [input, setInput] = useState("");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const connect = useCallback(() => {
    if (unmounted.current) return;

    // Clean up previous connection
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      try { wsRef.current.close(); } catch { /* ignore */ }
    }

    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttempts.current = 0;
      setStatus("connected");
    };

    ws.onclose = (e) => {
      if (unmounted.current) return;

      console.log("[Chat] WebSocket closed:", e.code, e.reason);

      // Auth errors — don't reconnect
      if (e.code === 4001) {
        setStatus("auth failed");
        router.push("/");
        return;
      }

      // Schedule reconnect with exponential backoff
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current++;
        const delay = Math.min(
          BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttempts.current - 1),
          MAX_RECONNECT_DELAY
        );
        console.log("[Chat] Reconnecting in", delay, "ms (attempt", reconnectAttempts.current, ")");
        setStatus(`reconnecting (${reconnectAttempts.current}/${MAX_RECONNECT_ATTEMPTS})...`);
        reconnectTimer.current = setTimeout(connect, delay);
      } else {
        setStatus("disconnected — refresh to retry");
      }
    };

    ws.onerror = (error) => {
      console.log("[Chat] WebSocket error:", error);
      // onclose will fire after this, which handles reconnection
    };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        console.log("[Chat] ← Received:", msg.type, msg.data ? `: "${msg.data?.substring?.(0, 50)}..."` : "");
        if (msg.type === "text_chunk") {
          setIsStreaming(true);  // Mark that response is streaming
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "assistant") {
              return [
                ...prev.slice(0, -1),
                { role: "assistant", text: last.text + msg.data },
              ];
            }
            return [...prev, { role: "assistant", text: msg.data }];
          });
        } else if (msg.type === "stream_end") {
          setIsStreaming(false);  // Response complete
          setStatus("connected");
        } else if (msg.type === "status") {
          setStatus(msg.data || "");
        } else if (msg.type === "stream_end") {
          setStatus("connected");
        } else if (msg.type === "error") {
          setStatus(msg.message || "error");
        } else if (msg.type === "ping") {
          // Respond to orchestrator ping (keep-alive)
          console.log("[Chat] → Sent: pong");
          ws.send(JSON.stringify({ type: "pong" }));
        } else if (msg.type === "notification") {
          // Handle plugin event notifications
          const data = msg.data as NotificationData;
          setNotifications((prev) => [...prev, data]);
          
          // Play audio cue based on level
          if (data.level === "speak") {
            // Urgent — could play a sound
            console.log("🔔 Urgent notification:", data.text);
          } else if (data.level === "nudge") {
            // Soft notification
            console.log("🔔 Notification:", data.text);
          } else if (data.level === "batch") {
            // Batched notification
            console.log("🔔 Batch notification:", data.text);
          }
        }
      } catch {
        /* ignore non-JSON messages */
      }
    };
  }, [router]);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    unmounted.current = false;
    connect();

    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [router, connect, session, isPending]);

  function sendText() {
    if (!input.trim() || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;
    if (isStreaming) {
      // Don't allow sending while response is still streaming
      console.log("[Chat] Waiting for response to complete before sending...");
      return;
    }
    const text = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    const payload = JSON.stringify({ type: "text", data: text });
    console.log("[Chat] → Sent:", payload.substring(0, 100));
    wsRef.current.send(payload);
    setStatus("thinking...");
  }

  function sendFeedback(notification: NotificationData, action: "engaged" | "dismissed" | "muted") {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    wsRef.current.send(JSON.stringify({
      type: "notification_feedback",
      data: {
        event_id: notification.event_id || "",
        plugin: notification.plugin || "unknown",
        sender: "", // Could extract from notification
        action,
      },
    }));
    
    // Remove the notification from the list after feedback
    setNotifications((prev) => prev.filter((n) => n !== notification));
  }

  function dismissNotification(notification: NotificationData) {
    setNotifications((prev) => prev.filter((n) => n !== notification));
  }

  if (isPending) return null;

  return (
    <div className="h-dvh flex flex-col w-full">
      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 pt-8 pb-3 shrink-0">
        <button
          onClick={() => router.push("/home")}
          className="w-8 h-8 flex items-center justify-center -ml-2 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors duration-300"
          aria-label="Go back"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 18 18"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M11 14L6 9L11 4" />
          </svg>
        </button>
        <span className="text-[11px] tracking-[0.18em] uppercase text-[var(--color-text-secondary)] font-normal">
          Chat
        </span>
        {/* Status orb */}
        <div className="w-8 flex items-center justify-center">
          <StatusOrb status={wsStatusToOrb(status)} />
        </div>
      </header>

      <div className="mx-6 h-px bg-[var(--color-border)] shrink-0" />

      {/* ── Notifications ── */}
      {notifications.length > 0 && (
        <div className="shrink-0 px-6 pt-4 pb-2 space-y-2">
          {notifications.map((n, i) => (
            <div
              key={i}
              className={`p-4 rounded-xl border animate-[fade-in_0.3s_ease] ${
                n.level === "speak"
                  ? "bg-red-950/30 border-red-900/50"
                  : n.level === "batch"
                  ? "bg-amber-950/30 border-amber-900/50"
                  : "bg-blue-950/30 border-blue-900/50"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[10px] uppercase tracking-wider ${
                      n.level === "speak"
                        ? "text-red-400"
                        : n.level === "batch"
                        ? "text-amber-400"
                        : "text-blue-400"
                    }`}>
                      {n.level === "speak" ? "🔔 Urgent" : n.level === "batch" ? "📋 Updates" : "💡 Notice"}
                    </span>
                    {n.plugin && (
                      <span className="text-[9px] text-[var(--color-text-muted)] uppercase">
                        {n.plugin}
                      </span>
                    )}
                  </div>
                  <p className="text-[14px] leading-[1.5] text-[var(--color-text)]">
                    {n.text}
                  </p>
                  {n.items && n.items.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {n.items.slice(0, 3).map((item, j) => (
                        <p key={j} className="text-[12px] text-[var(--color-text-muted)]">
                          • {item.text}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => dismissNotification(n)}
                  className="text-[var(--color-text-muted)] hover:text-[var(--color-text)] text-lg leading-none shrink-0"
                >
                  ×
                </button>
              </div>
              {/* Feedback buttons */}
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => sendFeedback(n, "engaged")}
                  className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] hover:text-green-400 transition-colors"
                >
                  ✓ Got it
                </button>
                <button
                  onClick={() => sendFeedback(n, "muted")}
                  className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] hover:text-red-400 transition-colors"
                >
                  Mute
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Messages ── */}
      <div className="flex-1 overflow-y-auto min-h-0 px-6 pt-5">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-[var(--color-text-muted)] text-xs">
              type a message to begin
            </p>
          </div>
        ) : (
          <div className="space-y-5 pb-2">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`animate-[fade-in_0.25s_ease] ${
                  m.role === "user" ? "flex flex-col items-end" : ""
                }`}
              >
                <span className="text-[9px] text-[var(--color-text-muted)] uppercase tracking-[0.12em] mb-1.5 block">
                  {m.role === "user" ? "you" : "aether"}
                </span>
                <div
                  className={`text-[14px] leading-[1.65] font-light max-w-[85%] ${
                    m.role === "user"
                      ? "text-[var(--color-text-secondary)] bg-[var(--color-surface)] border border-[var(--color-border)] rounded-2xl rounded-tr-sm px-4 py-2.5"
                      : "text-[var(--color-text)]"
                  }`}
                >
                  {m.text}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* ── Input ── */}
      <div className="shrink-0 px-6 pt-3 pb-6">
        <div className="flex items-center gap-3 bg-[var(--color-surface)] rounded-full px-5 py-3 border border-[var(--color-border)]">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendText()}
            placeholder={isStreaming ? "waiting..." : "type a message..."}
            disabled={isStreaming}
            className="input-inline flex-1 disabled:opacity-50"
          />
          <button
            onClick={sendText}
            disabled={!input.trim() || isStreaming}
            className="text-[10px] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors duration-300 tracking-[0.1em] uppercase disabled:opacity-20 disabled:cursor-not-allowed shrink-0"
          >
            send
          </button>
        </div>
      </div>
    </div>
  );
}
