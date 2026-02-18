"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { ChevronLeft } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { getSessionToken } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import type { AgentStatus } from "@/hooks/useAgentStatus";

function statusFromChat(
  chatStatus: string,
  error: Error | undefined
): AgentStatus {
  if (error) return "disconnected";
  if (chatStatus === "streaming" || chatStatus === "submitted")
    return "thinking";
  return "connected";
}

/** Extract plain text from a UIMessage's parts array. */
function getMessageText(parts: Array<{ type: string; text?: string }>): string {
  return parts
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text)
    .join("");
}

const chatTransport = new TextStreamChatTransport({
  api: "/api/chat",
  headers: (): Record<string, string> => {
    const token = getSessionToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  },
});

export default function ChatPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [input, setInput] = useState("");

  const { messages, sendMessage, status, error } = useChat({
    transport: chatTransport,
  });

  const isLoading = status === "streaming" || status === "submitted";

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auth guard
  useEffect(() => {
    if (!isPending && !session) {
      router.push("/");
    }
  }, [isPending, session, router]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage({ text: input.trim() });
    setInput("");
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
          <StatusOrb status={statusFromChat(status, error)} />
        </div>
      </header>

      <Separator className="mx-6 shrink-0 w-auto" />

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
            {messages.map((m) => (
              <div
                key={m.id}
                className={`animate-[fade-in_0.25s_ease] ${
                  m.role === "user" ? "flex flex-col items-end" : ""
                }`}
              >
                <span className="text-[9px] text-muted-foreground uppercase tracking-[0.12em] mb-1.5 block">
                  {m.role === "user" ? "you" : "aether"}
                </span>
                <div
                  className={`text-[14px] leading-[1.65] font-light max-w-[85%] ${
                    m.role === "user"
                      ? "text-secondary-foreground bg-card border border-border rounded-2xl rounded-tr-sm px-4 py-2.5"
                      : "text-foreground"
                  }`}
                >
                  {getMessageText(m.parts as Array<{ type: string; text?: string }>)}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="px-6 py-2">
          <p className="text-[11px] text-red-400">
            Connection error — try again
          </p>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex border border-border mx-6 mb-6">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isLoading ? "thinking..." : "type a message..."}
          disabled={isLoading}
          className="flex-1 bg-transparent border-0 rounded-none shadow-none px-3 py-2 text-[13px] font-light focus-visible:ring-0 disabled:opacity-50 h-auto"
        />
        <Button
          type="submit"
          variant="aether-ghost"
          disabled={!input.trim() || isLoading}
          className="text-[10px] tracking-[0.1em] uppercase shrink-0 px-4"
        >
          send
        </Button>
      </form>
    </div>
  );
}
