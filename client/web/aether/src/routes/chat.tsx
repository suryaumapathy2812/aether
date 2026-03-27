import { createFileRoute } from "@tanstack/react-router";
import { Suspense, useEffect, useRef, useState, Fragment } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useSession } from "#/lib/auth-client";
import {
  createChatSession,
  replyToQuestion,
  rejectQuestion,
  type QuestionReplyPayload,
} from "#/lib/api";
import { chatRuntime, useChatSessionRuntime } from "#/lib/chat-runtime";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "#/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "#/components/ai-elements/message";
import { QuestionDock } from "#/components/ai-elements/question-dock";
import {
  getToolRenderer,
  type ToolPartRecord,
} from "#/components/ai-elements/tool-renderer";
import {
  PromptInput,
  PromptInputProvider,
  PromptInputTextarea,
  PromptInputSubmit,
  PromptInputButton,
  PromptInputBody,
  PromptInputFooter,
  PromptInputTools,
  type PromptInputMessage,
  usePromptInputAttachments,
} from "#/components/ai-elements/prompt-input";
import {
  Attachments,
  Attachment,
  AttachmentPreview,
  AttachmentInfo,
  AttachmentRemove,
} from "#/components/ai-elements/attachments";
import {
  IconSparkles,
  IconX,
  IconPlayerPause,
  IconPaperclip,
} from "@tabler/icons-react";
import { AudioLines } from "lucide-react";
import { z } from "zod";
import { setRecentChatSessionId } from "#/lib/recent-chat";

const SUPPORTED_UPLOAD_TYPES = [
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
  "image/gif",
  "audio/wav",
  "audio/x-wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/ogg",
  "audio/flac",
  "audio/aac",
  "audio/x-aiff",
  "audio/aiff",
  "audio/mp4",
  "audio/m4a",
  "audio/webm",
].join(",");

const chatSearchSchema = z.object({
  s: z.string().optional().catch(undefined),
});

export const Route = createFileRoute("/chat")({
  validateSearch: chatSearchSchema,
  component: ChatPage,
});

type ChatSearch = z.infer<typeof chatSearchSchema>;

function ChatPage() {
  return (
    <Suspense>
      <ChatPageInner />
    </Suspense>
  );
}

function ChatPageInner() {
  const navigate = useNavigate();
  const { s: sessionId = "" } = Route.useSearch();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && !session) navigate({ to: "/" });
  }, [isPending, navigate, session]);

  useEffect(() => {
    const userId = session?.user?.id?.trim();
    if (!userId || !sessionId) return;
    setRecentChatSessionId(userId, sessionId);
  }, [session?.user?.id, sessionId]);

  if (isPending || !session) return null;

  return (
    <ChatView
      key={sessionId || "new"}
      session={session}
      sessionId={sessionId}
    />
  );
}

function ChatPromptInput({
  input,
  setInput,
  isStreaming,
  sessionId,
  setInputMode,
  onSubmit,
}: {
  input: string;
  setInput: (v: string) => void;
  isStreaming: boolean;
  sessionId: string;
  setInputMode: (mode: "text" | "voice") => void;
  onSubmit: (message: PromptInputMessage) => Promise<void>;
}) {
  const attachments = usePromptInputAttachments();

  return (
    <PromptInput
      accept={SUPPORTED_UPLOAD_TYPES}
      globalDrop
      multiple
      onSubmit={onSubmit}
    >
      <PromptInputBody>
        {attachments.files.length > 0 && (
          <div className="mb-3 w-full">
            <Attachments variant="inline" className="justify-start">
              {attachments.files.map((file) => (
                <Attachment
                  key={file.id}
                  data={file}
                  onRemove={() => attachments.remove(file.id)}
                >
                  <AttachmentPreview />
                  <AttachmentInfo className="max-w-40" />
                  <AttachmentRemove />
                </Attachment>
              ))}
            </Attachments>
          </div>
        )}
        <PromptInputTextarea
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          placeholder="Message aether..."
          className="min-h-11"
        />
      </PromptInputBody>
      <PromptInputFooter>
        <PromptInputTools>
          <PromptInputButton
            onClick={() => attachments.openFileDialog()}
            tooltip="Add photos or files"
            size="icon-sm"
            variant="ghost"
            className="rounded-full"
          >
            <IconPaperclip className="size-4" />
          </PromptInputButton>
        </PromptInputTools>
        {!input.trim() && !isStreaming && attachments.files.length === 0 ? (
          <PromptInputButton
            onClick={() => setInputMode("voice")}
            tooltip="Voice mode"
            size="icon-sm"
            variant="default"
            className="rounded-full"
          >
            <AudioLines className="size-5" />
          </PromptInputButton>
        ) : (
          <PromptInputSubmit
            status={isStreaming ? "streaming" : "ready"}
            disabled={
              !input.trim() && attachments.files.length === 0 && !isStreaming
            }
            size="icon-sm"
            className={isStreaming ? "rounded-lg" : "rounded-full"}
            onStop={() => {
              if (!sessionId) return;
              void chatRuntime.cancelTurn(sessionId);
            }}
          />
        )}
      </PromptInputFooter>
    </PromptInput>
  );
}

function ChatView({
  session,
  sessionId: initialSessionId,
}: {
  session: { user: { id: string; name?: string | null } };
  sessionId: string;
}) {
  const navigate = useNavigate();
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
  const { messages, status, error, loading, questionRequest } =
    useChatSessionRuntime(sessionId);

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

  async function handleSubmit(message: PromptInputMessage) {
    const text = message.text?.trim();
    const files = message.files;
    if (!text && (!files || files.length === 0)) return;
    setInput("");

    if (!sessionId) {
      if (creatingSessionRef.current) return;
      creatingSessionRef.current = true;
      try {
        const title =
          text.slice(0, 60) ||
          (files && files.length > 0 ? "Media message" : "");
        const newSess = await createChatSession(userId, title);
        setSessionId(newSess.id);
        navigate({ to: "/chat", search: { s: newSess.id }, replace: true });
        await chatRuntime.sendMessage({
          sessionId: newSess.id,
          userId,
          text: text || "",
          files,
        });
        return;
      } catch {
        setInput(text || "");
        return;
      } finally {
        creatingSessionRef.current = false;
      }
    }

    try {
      await chatRuntime.sendMessage({
        sessionId,
        userId,
        text: text || "",
        files,
      });
    } catch {
      setInput(text || "");
    }
  }

  async function ensureSessionForVoice(): Promise<string> {
    if (sessionId) return sessionId;
    if (creatingSessionRef.current)
      throw new Error("Session is already being created");
    creatingSessionRef.current = true;
    try {
      const newSess = await createChatSession(userId, "Voice chat");
      setSessionId(newSess.id);
      navigate({ to: "/chat", search: { s: newSess.id }, replace: true });
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
    const turn = await chatRuntime.startVoiceTurn({
      sessionId: activeSessionId,
      userId,
    });
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
        .then(() =>
          chatRuntime.commitVoiceTurn({ sessionId: targetSessionId, turnId }),
        )
        .catch(async () => {
          await chatRuntime.cancelTurn(targetSessionId);
        });
    };

    recorder.start(250);
    setIsRecordingVoice(true);
    setRecordingSeconds(0);
    if (recordingTimerRef.current !== null)
      window.clearInterval(recordingTimerRef.current);
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

  const isStreaming = status === "streaming";
  const hasInlineQuestion = Boolean(
    questionRequest?.toolCallId &&
      messages.some(
        (message) =>
          message.role === "assistant" &&
          message.parts.some((part) => {
            if (!String(part.type).startsWith("tool-")) return false;
            const candidate = part as Record<string, unknown>;
            return (
              String(candidate.toolCallId || "") ===
                questionRequest.toolCallId &&
              String(candidate.state || "") === "input-available"
            );
          }),
      ),
  );

  async function handleQuestionSubmit(payload: QuestionReplyPayload) {
    if (!questionRequest) return;
    await replyToQuestion(questionRequest.id, payload);
  }

  async function handleQuestionDismiss() {
    if (!questionRequest) return;
    await rejectQuestion(questionRequest.id);
  }

  const latestUserMessage = [...messages]
    .reverse()
    .find((m) => m.role === "user");
  const latestAssistantMessage = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const latestUserText =
    latestUserMessage?.parts.find((p) => p.type === "text")?.text || "";
  const latestAssistantText =
    latestAssistantMessage?.parts.find((p) => p.type === "text")?.text || "";

  const voiceState: "idle" | "recording" | "thinking" = isRecordingVoice
    ? "recording"
    : isStreaming
      ? "thinking"
      : "idle";

  return (
    <>
      {inputMode === "voice" && (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
          <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8">
            <div className="flex items-center justify-center h-28">
              {voiceState === "recording" ? (
                <div className="flex items-center gap-1.25">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="w-1.75 rounded-full bg-white"
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
                      ? "bg-white/30 animate-pulse"
                      : "bg-white/75",
                  ].join(" ")}
                />
              )}
            </div>

            <p className="text-sm text-white/70 tracking-wide select-none">
              {voiceState === "recording"
                ? `Listening · ${recordingSeconds}s`
                : voiceState === "thinking"
                  ? "Thinking..."
                  : "Tap to speak"}
            </p>

            {(latestAssistantText ||
              (latestUserText && latestUserText !== "[voice instruction]")) && (
              <div className="max-w-md w-full text-center space-y-2 overflow-y-auto max-h-[40vh] px-2">
                {latestUserText && latestUserText !== "[voice instruction]" && (
                  <p className="text-white/50 text-sm leading-relaxed line-clamp-2">
                    {latestUserText}
                  </p>
                )}
                {latestAssistantText && (
                  <div className="text-white/90 text-left">
                    <MessageResponse>{latestAssistantText}</MessageResponse>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="pb-10 flex items-center justify-center gap-4">
            <button
              type="button"
              onClick={closeVoiceMode}
              className="h-14 w-14 rounded-full bg-red-500/15 flex items-center justify-center text-red-400 hover:bg-red-500/25 hover:text-red-300 transition-colors"
            >
              <IconX className="size-6" />
            </button>

            {voiceState === "recording" ? (
              <button
                type="button"
                onClick={stopVoiceRecording}
                className="h-14 w-14 rounded-full bg-white/15 flex items-center justify-center text-white hover:bg-white/20 transition-colors"
              >
                <IconPlayerPause className="size-6" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  void startVoiceRecording();
                }}
                disabled={isStreaming}
                className="h-14 w-14 rounded-full bg-white/15 flex items-center justify-center text-white hover:bg-white/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <AudioLines className="size-6" />
              </button>
            )}
          </div>

          <style
            dangerouslySetInnerHTML={{
              __html: `
            @keyframes aether-vbar {
              0%, 100% { transform: scaleY(0.3); }
              50% { transform: scaleY(1); }
            }
          `,
            }}
          />
        </div>
      )}

      <div className="flex h-full min-h-0 flex-col overflow-hidden">
        <Conversation className="flex-1 min-h-0">
          <ConversationContent className="gap-10 px-6 pt-20 pb-10 max-w-180 mx-auto w-full">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground/60 text-xs">loading...</p>
              </div>
            ) : messages.length === 0 ? (
              <ConversationEmptyState
                icon={
                  <IconSparkles className="size-6 text-muted-foreground/30" />
                }
                title="What can I help you with?"
                description=""
              />
            ) : (
              messages.map((message) => (
                <Message
                  from={message.role}
                  key={message.id}
                  className="max-w-full"
                >
                  <MessageContent>
                    {message.parts.map((part, i) => {
                      const key = `${message.id}-${i}`;
                      switch (part.type) {
                        case "text": {
                          const isVoiceMsg =
                            message.role === "user" &&
                            part.text === "[voice instruction]";
                          return (
                            <Fragment key={key}>
                              {isVoiceMsg ? (
                                <div className="flex items-center gap-1.5 text-muted-foreground">
                                  <AudioLines className="size-3.5" />
                                  <span className="text-sm">Voice message</span>
                                </div>
                              ) : (
                                <MessageResponse>{part.text}</MessageResponse>
                              )}
                            </Fragment>
                          );
                        }
                        case "file": {
                          const filePart = part as {
                            url: string;
                            mediaType?: string;
                            filename?: string;
                          };
                          if (
                            filePart.mediaType?.startsWith("image/") &&
                            filePart.url
                          ) {
                            return (
                              <div key={key} className="mt-2">
                                <img
                                  src={filePart.url}
                                  alt={filePart.filename || "Attached image"}
                                  className="max-w-full rounded-lg max-h-96 object-contain"
                                />
                              </div>
                            );
                          }
                          if (
                            filePart.mediaType?.startsWith("audio/") &&
                            filePart.url
                          ) {
                            return (
                              <div key={key} className="mt-2 w-full max-w-md">
                                <audio
                                  controls
                                  src={filePart.url}
                                  className="w-full"
                                />
                              </div>
                            );
                          }
                          return null;
                        }
                        default:
                          if (part.type.startsWith("tool-")) {
                            const toolPart = part as ToolPartRecord;
                            const toolName = toolPart.type.replace("tool-", "");
                            const Renderer = getToolRenderer(toolName, toolPart.metadata);
                            return (
                              <Renderer
                                key={key}
                                part={toolPart}
                                questionRequest={questionRequest}
                                onQuestionSubmit={handleQuestionSubmit}
                                onQuestionDismiss={handleQuestionDismiss}
                              />
                            );
                          }
                          return null;
                        case "reasoning":
                          return (
                            <div
                              key={key}
                              className="text-xs text-muted-foreground/60 italic border-l border-white/8 pl-3 py-1"
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
          <div className="max-w-180 mx-auto w-full px-6 py-1">
            <p className="text-sm text-red-400/80">{error}</p>
          </div>
        )}

        <div className="max-w-180 mx-auto w-full px-6 pb-5 pt-2">
          {questionRequest && !hasInlineQuestion ? (
            <QuestionDock
              request={questionRequest}
              onSubmit={handleQuestionSubmit}
              onDismiss={handleQuestionDismiss}
            />
          ) : questionRequest && hasInlineQuestion ? (
            <div className="rounded-lg border border-border bg-accent/20 px-3 py-2 text-xs text-muted-foreground">
              Respond to the inline prompt above to continue.
            </div>
          ) : (
            <PromptInputProvider initialInput={input}>
              <ChatPromptInput
                input={input}
                setInput={setInput}
                isStreaming={isStreaming}
                sessionId={sessionId}
                setInputMode={setInputMode}
                onSubmit={handleSubmit}
              />
            </PromptInputProvider>
          )}
        </div>
      </div>
    </>
  );
}
