"use client";

import { Suspense, useEffect, useMemo, useRef, useState, Fragment } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useChat } from "@ai-sdk/react";
import { getChatSession, createChatSession } from "@/lib/api";
import { createAetherTransport } from "@/lib/aether-transport";
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
import { CopyIcon, Sparkles } from "lucide-react";
import type { UIMessage } from "ai";

export default function ChatPage() {
  return (
    <Suspense>
      <ChatPageInner />
    </Suspense>
  );
}

function ChatPageInner() {
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
  const searchParams = useSearchParams();
  const [input, setInput] = useState("");
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [sessionId, setSessionId] = useState(searchParams.get("s") || "");
  const lastLoadedSession = useRef("");

  const userId = session.user.id;

  const transport = useMemo(
    () => createAetherTransport({ userId, sessionId }),
    [userId, sessionId]
  );

  const { messages, setMessages, sendMessage, status, error } = useChat({
    transport,
  });

  // Load session messages when session ID changes
  useEffect(() => {
    const sid = searchParams.get("s") || "";
    if (sid === lastLoadedSession.current) return;
    setSessionId(sid);
    lastLoadedSession.current = sid;

    if (!sid) {
      setMessages([]);
      return;
    }

    setLoadingHistory(true);
    getChatSession(sid)
      .then((res) => {
        const restored: UIMessage[] = [];
        for (const msg of res.messages || []) {
          const content = msg.content as Record<string, unknown>;
          const role = content.role as string;
          if (role === "user") {
            const text = (content.content as string) || "";
            if (text.trim()) {
              restored.push({
                id: `msg-${msg.id}-user`,
                role: "user",
                parts: [{ type: "text", text }],
              });
            }
          } else if (role === "assistant") {
            const text = (content.content as string) || "";
            if (text.trim()) {
              restored.push({
                id: `msg-${msg.id}-assistant`,
                role: "assistant",
                parts: [{ type: "text", text }],
              });
            }
          }
          // tool_calls and tool results are part of the context but
          // not displayed in restored history (they ran in the past)
        }
        setMessages(restored);
      })
      .catch(() => setMessages([]))
      .finally(() => setLoadingHistory(false));
  }, [searchParams, setMessages]);

  async function handleSubmit(message: PromptInputMessage) {
    const text = message.text?.trim();
    if (!text) return;

    // Auto-create session on first message if none exists
    if (!sessionId) {
      try {
        const newSess = await createChatSession(userId, text.slice(0, 60));
        setSessionId(newSess.id);
        lastLoadedSession.current = newSess.id;
        router.replace(`/chat?s=${newSess.id}`, { scroll: false });
      } catch {
        // Continue without session — backend will auto-create
      }
    }

    sendMessage({ text });
    setInput("");
  }

  const isStreaming = status === "streaming" || status === "submitted";

  return (
    <div className="h-full flex flex-col">
      <div className="absolute top-5 right-5 z-10">
        <StatusOrb status={isStreaming ? "thinking" : "connected"} size={8} />
      </div>

      <Conversation className="flex-1 min-h-0">
        <ConversationContent className="gap-5 px-6 pt-8 pb-4 max-w-[720px] mx-auto w-full">
          {loadingHistory ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground/60 text-xs">loading...</p>
            </div>
          ) : messages.length === 0 ? (
            <ConversationEmptyState
              icon={<Sparkles className="size-6 text-muted-foreground/30" />}
              title="What can I help you with?"
              description=""
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
                          const inputObj = toolPart.input as Record<string, unknown> | undefined;
                          const subtitle = inputObj
                            ? (inputObj.query || inputObj.message_id || inputObj.name || inputObj.summary || "") as string
                            : "";
                          return (
                            <Tool key={key}>
                              <ToolHeader
                                type={toolPart.type as "tool-invocation"}
                                state={toolPart.state as "input-available"}
                                title={subtitle ? `${toolName} ${subtitle}` : toolName}
                              />
                              <ToolContent>
                                <ToolInput input={toolPart.input} />
                                {(toolPart.state === "output-available" || toolPart.state === "output-error") && (
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
                            className="text-xs text-muted-foreground/60 italic border-l border-white/[0.08] pl-3 py-1"
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
        <div className="max-w-[720px] mx-auto w-full px-6 py-1">
          <p className="text-[11px] text-red-400/80">{error.message}</p>
        </div>
      )}

      <div className="max-w-[720px] mx-auto w-full px-6 pb-5 pt-2">
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputBody>
            <PromptInputTextarea
              value={input}
              onChange={(e) => setInput(e.currentTarget.value)}
              placeholder="Message aether..."
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
    </div>
  );
}
