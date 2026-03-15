"use client";

import { useEffect, useMemo, useRef, useState, Fragment } from "react";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { useChat } from "@ai-sdk/react";
import { getMemoryConversations } from "@/lib/api";
import { createAetherTransport } from "@/lib/aether-transport";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
  MessageActions,
  MessageAction,
} from "@/components/ai-elements/message";
import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputSubmit,
  PromptInputBody,
  PromptInputFooter,
  PromptInputTools,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { CopyIcon, MessageSquare } from "lucide-react";
import type { UIMessage } from "ai";

export default function ChatPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && !session) router.push("/");
  }, [isPending, router, session]);

  if (isPending || !session) return null;

  return <ChatView session={session} />;
}

function ChatView({ session }: { session: { user: { id: string; name?: string | null } } }) {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const historyLoadedRef = useRef(false);

  const userId = session.user.id;

  const transport = useMemo(
    () => createAetherTransport({ userId, sessionId: userId }),
    [userId]
  );

  const { messages, setMessages, sendMessage, status, error } = useChat({
    transport,
    messages: initialMessages.length > 0 ? initialMessages : undefined,
  });

  // Load today's conversation history.
  useEffect(() => {
    if (historyLoadedRef.current) return;
    historyLoadedRef.current = true;

    async function loadTodayHistory(): Promise<void> {
      setLoadingHistory(true);
      try {
        const uid = userId;
        if (!uid) return;
        const res = await getMemoryConversations(uid, 200);
        const startOfToday = new Date();
        startOfToday.setHours(0, 0, 0, 0);
        const startSec = Math.floor(startOfToday.getTime() / 1000);

        const todayTurns = (res.conversations || [])
          .filter((c) => (c.timestamp || 0) >= startSec)
          .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

        const restored: UIMessage[] = [];
        for (const turn of todayTurns) {
          const userText = (turn.user_message || "").trim();
          const assistantText = (turn.assistant_message || "").trim();
          if (userText) {
            restored.push({
              id: `hist-${turn.id}-user`,
              role: "user",
              parts: [{ type: "text", text: userText }],
            });
          }
          if (assistantText) {
            restored.push({
              id: `hist-${turn.id}-assistant`,
              role: "assistant",
              parts: [{ type: "text", text: assistantText }],
            });
          }
        }

        if (restored.length > 0) {
          setInitialMessages(restored);
        }
      } catch {
        // Ignore history load errors; chat should remain usable.
      } finally {
        setLoadingHistory(false);
      }
    }

    void loadTodayHistory();
  }, [userId]);

  function handleSubmit(message: PromptInputMessage) {
    const text = message.text?.trim();
    if (!text) return;
    sendMessage({ text });
    setInput("");
  }

  const isStreaming = status === "streaming" || status === "submitted";

  return (
    <div className="h-full flex flex-col w-full px-6 sm:px-8 pb-4">
      <header className="flex items-center justify-between pt-7 sm:pt-8 pb-4 shrink-0">
        <div className="w-8 min-w-[44px] -ml-2 flex items-center justify-start">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.push("/home")}
            className="w-8 h-8 min-w-[44px] min-h-[44px] md:hidden"
            aria-label="Go back"
          >
            <ChevronLeft className="size-[18px]" strokeWidth={1.5} />
          </Button>
        </div>
        <span className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal">
          Chat
        </span>
        <div className="w-8 flex items-center justify-center">
          <StatusOrb status={isStreaming ? "thinking" : "connected"} size={10} />
        </div>
      </header>

      <Separator className="shrink-0 w-auto opacity-80" />

      <Conversation className="flex-1 min-h-0">
        <ConversationContent className="gap-6 px-0 pt-6">
          {loadingHistory ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground text-xs">loading today&apos;s chat...</p>
            </div>
          ) : messages.length === 0 ? (
            <ConversationEmptyState
              icon={<MessageSquare className="size-8 text-muted-foreground/50" />}
              title="What can I help you with?"
              description="Type a message to begin"
            />
          ) : (
            messages.map((message, messageIndex) => (
              <Message from={message.role} key={message.id} className="max-w-full">
                <MessageContent>
                  {message.parts.map((part, i) => {
                    const key = `${message.id}-${i}`;
                    switch (part.type) {
                      case "text":
                        return (
                          <Fragment key={key}>
                            <MessageResponse>{part.text}</MessageResponse>
                            {message.role === "assistant" &&
                              messageIndex === messages.length - 1 &&
                              !isStreaming && (
                                <MessageActions>
                                  <MessageAction
                                    onClick={() =>
                                      navigator.clipboard.writeText(part.text)
                                    }
                                    label="Copy"
                                  >
                                    <CopyIcon className="size-3" />
                                  </MessageAction>
                                </MessageActions>
                              )}
                          </Fragment>
                        );
                      default:
                        if (part.type.startsWith("tool-")) {
                          const toolPart = part as { type: string; state: string; toolCallId: string; input?: unknown; output?: unknown; errorText?: string };
                          const toolName = toolPart.type.replace("tool-", "");
                          return (
                            <Tool key={key}>
                              <ToolHeader
                                type={toolPart.type as "tool-invocation"}
                                state={toolPart.state as "input-available"}
                                title={toolName}
                              />
                              <ToolContent>
                                <ToolInput input={toolPart.input} />
                                {toolPart.state === "output-available" && (
                                  <ToolOutput
                                    output={toolPart.output}
                                    errorText={toolPart.errorText}
                                  />
                                )}
                              </ToolContent>
                            </Tool>
                          );
                        }
                        return null;
                      case "reasoning":
                        return (
                          <div
                            key={key}
                            className="text-xs text-muted-foreground italic border-l-2 border-muted pl-3 py-1"
                          >
                            {part.text}
                          </div>
                        );
                    }
                  })}
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      {error && (
        <div className="py-2">
          <p className="text-[11px] text-red-400">{error.message}</p>
        </div>
      )}

      <PromptInput
        onSubmit={handleSubmit}
        className="mt-2 shrink-0"
      >
        <PromptInputBody>
          <PromptInputTextarea
            value={input}
            onChange={(e) => setInput(e.currentTarget.value)}
            placeholder="Type a message..."
            className="min-h-[44px]"
          />
        </PromptInputBody>
        <PromptInputFooter>
          <PromptInputTools />
          <PromptInputSubmit
            status={isStreaming ? "streaming" : "ready"}
            disabled={!input.trim() && !isStreaming}
          />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
}
