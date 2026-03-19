"use client";

import { Suspense, useEffect, useRef, useState, Fragment } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { createChatSession, replyToQuestion, rejectQuestion } from "@/lib/api";
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
import { QuestionDock } from "@/components/ai-elements/question-dock";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputSubmit,
  PromptInputButton,
  PromptInputBody,
  PromptInputFooter,
  PromptInputTools,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { IconCopy, IconSparkles, IconX, IconPlayerPause } from "@tabler/icons-react";
import { AudioLines } from "lucide-react";

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

  return <ChatView key={sessionId || "new"} session={session} sessionId={sessionId} />;
}

function ChatView({ session, sessionId: initialSessionId }: { session: { user: { id: string; name?: string | null } }; sessionId: string }) {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [inputMode, setInputMode] = useState<"text" | "voice">("text");
  const [isRecordingVoice, setIsRecordingVoice] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const creatingSessionRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const voiceTurnIdRef = useRef<string | null>(null);
  const recordingTimerRef = useRef<number | null>(null);
  const voiceChunkQueueRef = useRef<Promise<void>>(Promise.resolve());

  const userId = session.user.id;
  const { messages, status, error, loading, loopState, questionRequest } = useChatSessionRuntime(sessionId);

  useEffect(() => {
    if (!sessionId) return;
    void chatRuntime.bootstrapForUser(userId);
    void chatRuntime.loadHistory(sessionId);
  }, [sessionId, userId]);

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current !== null) {
        window.clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
      mediaRecorderRef.current?.stop();
      mediaRecorderRef.current = null;
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      voiceTurnIdRef.current = null;
      setIsRecordingVoice(false);
    };
  }, []);

  // ── Text submit ──────────────────────────────────────────────────────
  async function handleSubmit(message: PromptInputMessage) {
    const text = message.text?.trim();
    if (!text) return;
    setInput("");

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
        setInput(text);
        return;
      } finally {
        creatingSessionRef.current = false;
      }
    }

    try {
      await chatRuntime.sendMessage({ sessionId, userId, text });
    } catch {
      setInput(text);
    }
  }

  // ── Voice helpers ────────────────────────────────────────────────────
  async function ensureSessionForVoice(): Promise<string> {
    if (sessionId) return sessionId;
    if (creatingSessionRef.current) throw new Error("Session is already being created");
    creatingSessionRef.current = true;
    try {
      const newSess = await createChatSession(userId, "Voice chat");
      setSessionId(newSess.id);
      router.replace(`/chat?s=${newSess.id}`, { scroll: false });
      return newSess.id;
    } finally {
      creatingSessionRef.current = false;
    }
  }

  function blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = typeof reader.result === "string" ? reader.result : "";
        const commaIndex = result.indexOf(",");
        resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
      };
      reader.onerror = () => reject(new Error("Failed to read audio chunk"));
      reader.readAsDataURL(blob);
    });
  }

  async function startVoiceRecording(): Promise<void> {
    if (isStreaming || isRecordingVoice) return;
    const activeSessionId = await ensureSessionForVoice();
    const turn = await chatRuntime.startVoiceTurn({ sessionId: activeSessionId, userId });
    voiceTurnIdRef.current = turn.turnId;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (event: BlobEvent) => {
      if (!event.data || event.data.size === 0) return;
      const turnId = voiceTurnIdRef.current;
      if (!turnId) return;
      voiceChunkQueueRef.current = voiceChunkQueueRef.current
        .then(async () => {
          const chunkBase64 = await blobToBase64(event.data);
          if (!chunkBase64) return;
          await chatRuntime.sendVoiceChunk({
            sessionId: activeSessionId,
            turnId,
            chunkBase64,
            mimeType: event.data.type || "audio/webm",
          });
        })
        .catch(() => {});
    };

    recorder.onstop = () => {
      const turnId = voiceTurnIdRef.current;
      const targetSessionId = sessionId || activeSessionId;
      voiceTurnIdRef.current = null;
      if (!turnId || !targetSessionId) return;
      void voiceChunkQueueRef.current
        .then(() => chatRuntime.commitVoiceTurn({ sessionId: targetSessionId, turnId }))
        .catch(async () => { await chatRuntime.cancelTurn(targetSessionId); });
    };

    recorder.start(250);
    setIsRecordingVoice(true);
    setRecordingSeconds(0);
    if (recordingTimerRef.current !== null) window.clearInterval(recordingTimerRef.current);
    recordingTimerRef.current = window.setInterval(() => {
      setRecordingSeconds((prev) => prev + 1);
    }, 1000);
  }

  function stopVoiceRecording(): void {
    if (!isRecordingVoice) return;
    setIsRecordingVoice(false);
    if (recordingTimerRef.current !== null) {
      window.clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    setRecordingSeconds(0);
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
  }

  function closeVoiceMode(): void {
    stopVoiceRecording();
    setInputMode("text");
  }

  // ── Derived state ────────────────────────────────────────────────────
  const isStreaming = status === "streaming";

  const latestUserMessage = [...messages].reverse().find((m) => m.role === "user");
  const latestAssistantMessage = [...messages].reverse().find((m) => m.role === "assistant");
  const latestUserText = latestUserMessage?.parts.find((p) => p.type === "text")?.text || "";
  const latestAssistantText = latestAssistantMessage?.parts.find((p) => p.type === "text")?.text || "";

  const voiceState: "idle" | "recording" | "thinking" =
    isRecordingVoice ? "recording" : isStreaming ? "thinking" : "idle";

  const loopLabel = isStreaming && loopState && loopState !== "running" && loopState !== "stopped"
    ? loopState === "retrying" ? "retrying..."
    : loopState === "recovering" ? "recovering..."
    : loopState === "blocked" ? "blocked"
    : loopState === "compacting" ? "compacting context..."
    : null
    : null;

  return (
    <>
      {/* ── Voice mode overlay ──────────────────────────────────────── */}
      {inputMode === "voice" && (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
          {/* Center: visualizer + status + transcript */}
          <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8">
            {/* Orb / bars */}
            <div className="flex items-center justify-center h-28">
              {voiceState === "recording" ? (
                <div className="flex items-center gap-[5px]">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="w-[7px] rounded-full bg-white/80"
                      style={{
                        height: "44px",
                        animation: "aether-vbar 1s ease-in-out infinite",
                        animationDelay: `${i * 0.12}s`,
                        transformOrigin: "center",
                      }}
                    />
                  ))}
                </div>
              ) : (
                <div
                  className={[
                    "h-20 w-20 rounded-full transition-all duration-700",
                    voiceState === "thinking"
                      ? "bg-white/20 animate-pulse"
                      : "bg-white/[0.08]",
                  ].join(" ")}
                />
              )}
            </div>

            {/* Status label */}
            <p className="text-[13px] text-white/40 tracking-wide select-none">
              {voiceState === "recording"
                ? `Listening · ${recordingSeconds}s`
                : voiceState === "thinking"
                  ? "Thinking..."
                  : "Tap to speak"}
            </p>

            {/* Transcript */}
            {(latestAssistantText || (latestUserText && latestUserText !== "[voice instruction]")) && (
              <div className="max-w-md w-full text-center space-y-2 overflow-y-auto max-h-[40vh] px-2">
                {latestUserText && latestUserText !== "[voice instruction]" && (
                  <p className="text-white/30 text-sm leading-relaxed line-clamp-2">{latestUserText}</p>
                )}
                {latestAssistantText && (
                  <div className="text-white/70 text-left">
                    <MessageResponse>{latestAssistantText}</MessageResponse>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Bottom controls */}
          <div className="pb-10 flex items-center justify-center gap-4">
            {/* Close */}
            <button
              type="button"
              onClick={closeVoiceMode}
              className="h-14 w-14 rounded-full bg-white/[0.07] flex items-center justify-center text-white/50 hover:bg-white/10 hover:text-white/70 transition-colors"
            >
              <IconX className="size-6" />
            </button>

            {/* Waveform (start) / Pause (stop) */}
            {voiceState === "recording" ? (
              <button
                type="button"
                onClick={stopVoiceRecording}
                className="h-14 w-14 rounded-full bg-white/[0.12] flex items-center justify-center text-white hover:bg-white/[0.18] transition-colors"
              >
                <IconPlayerPause className="size-6" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => { void startVoiceRecording(); }}
                disabled={isStreaming}
                className="h-14 w-14 rounded-full bg-white/[0.12] flex items-center justify-center text-white hover:bg-white/[0.18] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <AudioLines className="size-6" />
              </button>
            )}
          </div>

          {/* Bar animation keyframes */}
          <style dangerouslySetInnerHTML={{ __html: `
            @keyframes aether-vbar {
              0%, 100% { transform: scaleY(0.3); }
              50% { transform: scaleY(1); }
            }
          ` }} />
        </div>
      )}

      {/* ── Main chat view ──────────────────────────────────────────── */}
      <div className="h-full flex flex-col">
        <div className="absolute top-5 right-5 z-10 flex items-center gap-2">
          {loopLabel && (
            <span className="text-[10px] text-muted-foreground/60 animate-pulse">{loopLabel}</span>
          )}
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
                        case "text": {
                          const isVoiceMsg = message.role === "user" && part.text === "[voice instruction]";
                          return (
                            <Fragment key={key}>
                              {isVoiceMsg ? (
                                <div className="flex items-center gap-1.5 text-muted-foreground/50">
                                  <AudioLines className="size-3.5" />
                                  <span className="text-sm">Voice message</span>
                                </div>
                              ) : (
                                <MessageResponse>{part.text}</MessageResponse>
                              )}
                              {message.role === "assistant" &&
                                messageIndex === messages.length - 1 &&
                                !isStreaming && (
                                  <MessageActions>
                                    <MessageAction
                                      onClick={() => navigator.clipboard.writeText(part.text)}
                                      label="Copy"
                                    >
                                      <IconCopy className="size-3" />
                                    </MessageAction>
                                  </MessageActions>
                                )}
                            </Fragment>
                          );
                        }
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
                                    <ToolOutput output={toolPart.output} errorText={toolPart.errorText} />
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
          {questionRequest ? (
            <QuestionDock
              request={questionRequest}
              onSubmit={async (answers) => {
                try { await replyToQuestion(questionRequest.id, answers); } catch { /* ignore */ }
              }}
              onDismiss={async () => {
                try { await rejectQuestion(questionRequest.id); } catch { /* ignore */ }
              }}
            />
          ) : (
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
                {!input.trim() && !isStreaming ? (
                  <PromptInputButton
                    onClick={() => setInputMode("voice")}
                    tooltip="Voice mode"
                    size="icon-sm"
                    variant="default"
                    className="rounded-full"
                  >
                    <AudioLines className="size-4" />
                  </PromptInputButton>
                ) : (
                  <PromptInputSubmit
                    status={isStreaming ? "streaming" : "ready"}
                    disabled={!input.trim() && !isStreaming}
                    onStop={() => {
                      if (!sessionId) return;
                      void chatRuntime.cancelTurn(sessionId);
                    }}
                  />
                )}
              </PromptInputFooter>
            </PromptInput>
          )}
        </div>
      </div>
    </>
  );
}
