"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { isLoggedIn, getWsUrl } from "@/lib/api";

/**
 * Chat — text conversation with Aether.
 * Header, scrollable messages, input pinned at bottom.
 */
export default function ChatPage() {
  const router = useRouter();
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<{ role: string; text: string }[]>(
    []
  );
  const [status, setStatus] = useState("connecting...");
  const [input, setInput] = useState("");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isLoggedIn()) {
      router.push("/");
      return;
    }

    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => setStatus("connected");
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("connection failed");

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "text_chunk") {
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
        } else if (msg.type === "status") {
          setStatus(msg.data || "");
        } else if (msg.type === "stream_end") {
          setStatus("connected");
        }
      } catch {
        /* ignore */
      }
    };

    return () => ws.close();
  }, [router]);

  function sendText() {
    if (!input.trim() || !wsRef.current) return;
    const text = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    wsRef.current.send(JSON.stringify({ type: "text", data: text }));
    setStatus("thinking...");
  }

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
        {/* Status indicator */}
        <span className="text-[9px] text-[var(--color-text-muted)] tracking-wider w-8 text-right">
          {status === "connected" ? "·" : status === "thinking..." ? "···" : ""}
        </span>
      </header>

      <div className="mx-6 h-px bg-[var(--color-border)] shrink-0" />

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
            placeholder="type a message..."
            className="input-inline flex-1"
          />
          <button
            onClick={sendText}
            disabled={!input.trim()}
            className="text-[10px] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors duration-300 tracking-[0.1em] uppercase disabled:opacity-20 disabled:cursor-not-allowed shrink-0"
          >
            send
          </button>
        </div>
      </div>
    </div>
  );
}
