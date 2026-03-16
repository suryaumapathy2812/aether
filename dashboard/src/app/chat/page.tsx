"use client";

import { Suspense, useEffect, useRef, useState, Fragment } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { createChatSession } from "@/lib/api";
import { chatRuntime, useChatSessionRuntime } from "@/lib/chat-runtime";
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
import { IconCopy, IconSparkles } from "@tabler/icons-react";

export default function ChatPage() {
  return (
    <Suspense>
      <ChatPageInner />
    </Suspense>
  );
}

function ChatPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const sessionId = searchParams.get("s") || "";

  useEffect(() => {
    if (!isPending && !session) router.push("/");
  }, [isPending, router, session]);

  if (isPending || !session) return null;

  // Keep key remount so route changes reset local input state.
  return <ChatView key={sessionId || "new"} session={session} sessionId={sessionId} />;
}

function ChatView({ session, sessionId: initialSessionId }: { session: { user: { id: string; name?: string | null } }; sessionId: string }) {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(initialSessionId);
  const creatingSessionRef = useRef(false);

  const userId = session.user.id;
  const { messages, status, error, loading } = useChatSessionRuntime(sessionId);

  useEffect(() => {
    if (!sessionId) return;
    void chatRuntime.loadHistory(sessionId);
  }, [sessionId]);

  async function handleSubmit(message: PromptInputMessage) {
    const text = message.text?.trim();
    if (!text) return;

    // Auto-create session on first message if none exists
    if (!sessionId) {
      if (creatingSessionRef.current) return;
      creatingSessionRef.current = true;
      try {
        const newSess = await createChatSession(userId, text.slice(0, 60));
        setSessionId(newSess.id);
        router.replace(`/chat?s=${newSess.id}`, { scroll: false });
        await chatRuntime.sendMessage({ sessionId: newSess.id, userId, text });
        return;
      } catch {
        // Continue without session — backend will auto-create
      } finally {
        creatingSessionRef.current = false;
      }
    }

    await chatRuntime.sendMessage({ sessionId, userId, text });
    setInput("");
  }

  const isStreaming = status === "streaming";

  return (
    <div className="h-full flex flex-col">
      <div className="absolute top-5 right-5 z-10">
        <StatusOrb status={isStreaming ? "thinking" : "connected"} size={8} />
      </div>

      <Conversation className="flex-1 min-h-0">
        <ConversationContent className="gap-5 px-6 pt-8 pb-4 max-w-[720px] mx-auto w-full">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground/60 text-xs">loading...</p>
            </div>
          ) : messages.length === 0 ? (
            <ConversationEmptyState
              icon={<IconSparkles className="size-6 text-muted-foreground/30" />}
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
                                    <IconCopy className="size-3" />
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
          <p className="text-[11px] text-red-400/80">{error}</p>
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
