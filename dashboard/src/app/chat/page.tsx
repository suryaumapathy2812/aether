"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { getWsUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import type { AgentStatus } from "@/hooks/useAgentStatus";

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;

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
  const wsConnected = useRef(false);
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([]);
  const [notifications, setNotifications] = useState<NotificationData[]>([]);
  const [status, setStatus] = useState("connecting...");
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState("");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auth guard — separate from WS lifecycle
  useEffect(() => {
    if (!isPending && !session) {
      router.push("/");
    }
  }, [isPending, session, router]);

  // WS lifecycle — runs once on mount, independent of session changes
  useEffect(() => {
    unmounted.current = false;

    function connect() {
      if (unmounted.current) return;

      // Don't connect if already connected
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

      // Clean up any existing dead connection
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
        wsConnected.current = true;
        // Tell backend this is a text-only session — skip STT/TTS
        ws.send(JSON.stringify({ type: "session_config", mode: "text" }));
        setStatus("connected");
      };

      ws.onclose = (e) => {
        wsConnected.current = false;
        if (unmounted.current) return;

        if (e.code === 4001) {
          setStatus("auth failed");
          router.push("/");
          return;
        }

        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          const delay = Math.min(
            BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttempts.current - 1),
            MAX_RECONNECT_DELAY
          );
          setStatus(`reconnecting (${reconnectAttempts.current}/${MAX_RECONNECT_ATTEMPTS})...`);
          reconnectTimer.current = setTimeout(connect, delay);
        } else {
          setStatus("disconnected — refresh to retry");
        }
      };

      ws.onerror = () => {
        // onclose handles reconnection
      };

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);

          if (msg.type === "text_chunk") {
            setIsStreaming(true);
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
            setIsStreaming(false);
            setStatus("connected");
          } else if (msg.type === "status") {
            setStatus(msg.data || "");
          } else if (msg.type === "error") {
            setStatus(msg.message || "error");
          } else if (msg.type === "ping") {
            ws.send(JSON.stringify({ type: "pong" }));
          } else if (msg.type === "notification") {
            const data = msg.data as NotificationData;
            setNotifications((prev) => [...prev, data]);
          }
        } catch {
          /* ignore non-JSON messages */
        }
      };
    }

    connect();

    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on intentional close
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function sendText() {
    if (!input.trim() || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;
    if (isStreaming) return;

    const text = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    wsRef.current.send(JSON.stringify({ type: "text", data: text }));
    setStatus("thinking...");
  }

  function sendFeedback(notification: NotificationData, action: "engaged" | "muted") {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({
      type: "notification_feedback",
      data: {
        event_id: notification.event_id || "",
        plugin: notification.plugin || "unknown",
        sender: "",
        action,
      },
    }));

    setNotifications((prev) => prev.filter((n) => n !== notification));
  }

  function dismissNotification(notification: NotificationData) {
    setNotifications((prev) => prev.filter((n) => n !== notification));
  }

  if (isPending) return null;

  return (
    <div className="h-dvh flex flex-col w-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 pt-8 pb-3 shrink-0">
        <Button
          variant="aether-ghost"
          size="icon"
          onClick={() => router.push("/home")}
          className="w-8 h-8 -ml-2"
          aria-label="Go back"
        >
          <ChevronLeft className="size-[18px]" strokeWidth={1.5} />
        </Button>
        <span className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal">
          Chat
        </span>
        <div className="w-8 flex items-center justify-center">
          <StatusOrb status={wsStatusToOrb(status)} />
        </div>
      </header>

      <Separator className="mx-6 shrink-0 w-auto" />

      {/* Notifications */}
      {notifications.length > 0 && (
        <div className="shrink-0 px-6 pt-4 pb-2 space-y-2">
          {notifications.map((n, i) => (
            <Card
              key={i}
              className={`animate-[fade-in_0.3s_ease] rounded-xl py-4 ${n.level === "speak"
                  ? "bg-red-950/30 border-red-900/50"
                  : n.level === "batch"
                    ? "bg-amber-950/30 border-amber-900/50"
                    : "bg-blue-950/30 border-blue-900/50"
                }`}
            >
              <CardContent className="space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge
                        variant="ghost"
                        className={`text-[10px] uppercase tracking-wider px-0 py-0 h-auto font-normal ${n.level === "speak"
                            ? "text-red-400"
                            : n.level === "batch"
                              ? "text-amber-400"
                              : "text-blue-400"
                          }`}
                      >
                        {n.level === "speak" ? "Urgent" : n.level === "batch" ? "Updates" : "Notice"}
                      </Badge>
                      {n.plugin && (
                        <span className="text-[9px] text-muted-foreground uppercase">
                          {n.plugin}
                        </span>
                      )}
                    </div>
                    <p className="text-[14px] leading-[1.5] text-foreground">
                      {n.text}
                    </p>
                    {n.items && n.items.length > 0 && (
                      <div className="mt-2 space-y-1">
                        {n.items.slice(0, 3).map((item, j) => (
                          <p key={j} className="text-[12px] text-muted-foreground">
                            • {item.text}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                  <Button
                    variant="aether-ghost"
                    size="aether-link"
                    onClick={() => dismissNotification(n)}
                    className="text-lg leading-none shrink-0"
                  >
                    ×
                  </Button>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="aether-link"
                    size="aether-link"
                    onClick={() => sendFeedback(n, "engaged")}
                    className="text-[10px] uppercase tracking-wider hover:text-green-400"
                  >
                    ✓ Got it
                  </Button>
                  <Button
                    variant="aether-link"
                    size="aether-link"
                    onClick={() => sendFeedback(n, "muted")}
                    className="text-[10px] uppercase tracking-wider hover:text-red-400"
                  >
                    Mute
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto min-h-0 px-6 pt-5">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground text-xs">
              type a message to begin
            </p>
          </div>
        ) : (
          <div className="space-y-5 pb-2">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`animate-[fade-in_0.25s_ease] ${m.role === "user" ? "flex flex-col items-end" : ""
                  }`}
              >
                <span className="text-[9px] text-muted-foreground uppercase tracking-[0.12em] mb-1.5 block">
                  {m.role === "user" ? "you" : "aether"}
                </span>
                <div
                  className={`text-[14px] leading-[1.65] font-light max-w-[85%] ${m.role === "user"
                      ? "text-secondary-foreground bg-card border border-border rounded-2xl rounded-tr-sm px-4 py-2.5"
                      : "text-foreground"
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

      {/* Input */}
      <div className="flex border border-border mx-6 mb-6">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendText()}
          placeholder={isStreaming ? "waiting..." : "type a message..."}
          disabled={isStreaming}
          className="flex-1 bg-transparent border-0 rounded-none shadow-none px-3 py-2 text-[13px] font-light focus-visible:ring-0 disabled:opacity-50 h-auto"
        />
        <Button
          variant="aether-ghost"
          onClick={sendText}
          disabled={!input.trim() || isStreaming}
          className="text-[10px] tracking-[0.1em] uppercase shrink-0 px-4"
        >
          send
        </Button>
      </div>
    </div>
  );
}
