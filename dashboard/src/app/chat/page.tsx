"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { useRealtime, type RealtimeConnectionState } from "@/components/RealtimeProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import ChatMessage from "@/components/ChatMessage";
import type { AgentStatus } from "@/hooks/useAgentStatus";

function orbStatus(state: RealtimeConnectionState): AgentStatus {
  if (state === "thinking") return "thinking";
  if (state === "listening") return "listening";
  if (state === "connected") return "connected";
  return "disconnected";
}

export default function ChatPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const { connState, errorText, messages, sendText } = useRealtime();

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const isLoading = connState === "thinking";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isPending && !session) router.push("/");
  }, [isPending, router, session]);

  function handleSubmit(e: React.FormEvent): void {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    const sent = sendText(text);
    if (sent) {
      setInput("");
    }
  }

  if (isPending) return null;

  return (
    <div className="h-full flex flex-col w-full px-6 sm:px-8">
      <header className="flex items-center justify-between pt-7 sm:pt-8 pb-4 shrink-0">
        <Button
          variant="aether-ghost"
          size="icon"
          onClick={() => router.push("/home")}
          className="w-8 h-8 min-w-[44px] min-h-[44px] -ml-2"
          aria-label="Go back"
        >
          <ChevronLeft className="size-[18px]" strokeWidth={1.5} />
        </Button>
        <span className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal">
          Chat
        </span>
        <div className="w-8 flex items-center justify-center">
          <StatusOrb status={orbStatus(connState)} size={10} />
        </div>
      </header>

      <Separator className="shrink-0 w-auto opacity-80" />

      <div className="flex-1 overflow-y-auto min-h-0 pt-6">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground text-xs">type a message to begin</p>
          </div>
        ) : (
          <div className="space-y-6 pb-3">
            {messages.map((m) => (
              <ChatMessage key={m.id} role={m.role} text={m.text} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {errorText && (
        <div className="py-2">
          <p className="text-[11px] text-red-400">{errorText}</p>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="flex items-center border border-border rounded-full bg-white/6 pb-safe mt-4 mb-2"
      >
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isLoading ? "thinking..." : "type a message..."}
          disabled={isLoading}
          className="flex-1 bg-transparent border-0 rounded-none shadow-none px-4 py-3 text-base md:text-[13px] font-medium focus-visible:ring-0 disabled:opacity-50 h-auto leading-none"
          style={{ fontSize: "16px" }}
        />
        <Button
          type="submit"
          variant="aether-ghost"
          disabled={!input.trim() || isLoading}
          className="text-[10px] tracking-[0.1em] uppercase shrink-0 px-4 self-center"
        >
          send
        </Button>
      </form>
    </div>
  );
}
