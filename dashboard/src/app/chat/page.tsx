"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import {
  chatCompletions,
  completeMediaUpload,
  initMediaUpload,
  type ChatContentPart,
  getMemoryConversations,
  type ChatMessage,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import ChatMessageView from "@/components/ChatMessage";
import { AIChatComposer } from "@/components/ai-input";

interface UiMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  content: string | ChatContentPart[];
}

interface PendingAttachment {
  id: string;
  name: string;
  kind: "image" | "audio";
  previewUrl: string;
  contentPart: ChatContentPart;
}

const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const MAX_AUDIO_BYTES = 12 * 1024 * 1024;
const MAX_TOTAL_ATTACHMENT_BYTES = 20 * 1024 * 1024;

function ext(name: string): string {
  const idx = name.lastIndexOf(".");
  if (idx < 0 || idx === name.length - 1) return "";
  return name.slice(idx + 1).toLowerCase();
}

function formatCodeFromMimeOrName(type: string, name: string): string {
  const t = type.toLowerCase();
  if (t.includes("wav")) return "wav";
  if (t.includes("mpeg") || t.includes("mp3")) return "mp3";
  if (t.includes("ogg")) return "ogg";
  if (t.includes("flac")) return "flac";
  if (t.includes("aac")) return "aac";
  if (t.includes("aiff")) return "aiff";
  if (t.includes("m4a") || t.includes("mp4")) return "m4a";
  if (t.includes("webm")) return "webm";
  const fromExt = ext(name);
  return fromExt || "wav";
}

function randomId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function pendingImageUrl(a: PendingAttachment): string | null {
  if (a.kind !== "image") return null;
  return a.previewUrl || null;
}

function summarizeUserContent(content: string | ChatContentPart[]): string {
  if (typeof content === "string") {
    return content;
  }
  const text = content
    .filter((p): p is Extract<ChatContentPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text.trim())
    .filter(Boolean)
    .join("\n");
  const imageCount = content.filter((p) => p.type === "image_ref" || p.type === "image_url").length;
  const audioCount = content.filter((p) => p.type === "audio_ref" || p.type === "input_audio").length;
  const mediaParts: string[] = [];
  if (imageCount > 0) mediaParts.push(`${imageCount} image${imageCount === 1 ? "" : "s"}`);
  if (audioCount > 0) mediaParts.push(`${audioCount} audio${audioCount === 1 ? "" : "s"}`);
  if (!text && mediaParts.length === 0) return "";
  if (!text) return `[media: ${mediaParts.join(", ")}]`;
  if (mediaParts.length === 0) return text;
  return `${text}\n[media: ${mediaParts.join(", ")}]`;
}

export default function ChatPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [uploadingMedia, setUploadingMedia] = useState(false);
  const [pendingAttachments, setPendingAttachments] = useState<PendingAttachment[]>(
    [],
  );
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const historyLoadedRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isPending && !session) router.push("/");
  }, [isPending, router, session]);

  useEffect(() => {
    if (!session || historyLoadedRef.current) return;
    historyLoadedRef.current = true;
    const currentSession = session;

    async function loadTodayHistory(): Promise<void> {
      setLoadingHistory(true);
      try {
        const userId = currentSession.user.id || "";
        if (!userId) return;
        const res = await getMemoryConversations(userId, 200);
        const startOfToday = new Date();
        startOfToday.setHours(0, 0, 0, 0);
        const startSec = Math.floor(startOfToday.getTime() / 1000);

        const todayTurns = (res.conversations || [])
          .filter((c) => (c.timestamp || 0) >= startSec)
          .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

        const restored: UiMessage[] = [];
        for (const turn of todayTurns) {
          const userText = (turn.user_message || "").trim();
          const userContent = Array.isArray(turn.user_content)
            ? turn.user_content
            : [];
          const assistantText = (turn.assistant_message || "").trim();
          if (userText || userContent.length > 0) {
            restored.push({
              id: `hist-${turn.id}-user`,
              role: "user",
              text: userText || "media",
              content: userContent.length > 0 ? userContent : userText,
            });
          }
          if (assistantText) {
            restored.push({
              id: `hist-${turn.id}-assistant`,
              role: "assistant",
              text: assistantText,
              content: assistantText,
            });
          }
        }

        if (restored.length > 0) {
          setMessages(restored);
        }
      } catch {
        // Ignore history load errors; chat should remain usable.
      } finally {
        setLoadingHistory(false);
      }
    }

    void loadTodayHistory();
  }, [session]);

  async function handlePickMedia(
    e: React.ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const selected = Array.from(e.target.files || []);
    if (selected.length === 0) return;

    setErrorText(null);
    setUploadingMedia(true);

    let totalBytes = pendingAttachments.reduce((sum, a) => {
      const media = (a.contentPart as { media?: { size?: number } }).media;
      return sum + (media?.size || 0);
    }, 0);

    const next: PendingAttachment[] = [];
    for (const file of selected) {
      const isImage = file.type.startsWith("image/");
      const isAudio = file.type.startsWith("audio/");
      if (!isImage && !isAudio) {
        setErrorText(`Unsupported file type: ${file.name}`);
        continue;
      }

      const maxBytes = isImage ? MAX_IMAGE_BYTES : MAX_AUDIO_BYTES;
      if (file.size > maxBytes) {
        const limitMb = Math.floor(maxBytes / (1024 * 1024));
        setErrorText(`${file.name} exceeds ${limitMb}MB limit`);
        continue;
      }

      if (totalBytes + file.size > MAX_TOTAL_ATTACHMENT_BYTES) {
        setErrorText("Total attachment size exceeds 20MB limit");
        continue;
      }

      try {
        const init = await initMediaUpload({
          user_id: session?.user?.id || "default",
          session_id: session?.user?.id || "chat",
          file_name: file.name,
          content_type: file.type,
          size: file.size,
          kind: isImage ? "image" : "audio",
        });

        const putRes = await fetch(init.upload_url, {
          method: "PUT",
          headers: init.headers,
          body: file,
        });
        if (!putRes.ok) {
          throw new Error("upload failed");
        }

        const complete = await completeMediaUpload({
          user_id: session?.user?.id || "default",
          bucket: init.bucket,
          object_key: init.object_key,
          file_name: file.name,
          content_type: file.type,
          size: file.size,
          kind: isImage ? "image" : "audio",
        });

        const previewUrl = URL.createObjectURL(file);
        if (isImage) {
          next.push({
            id: randomId("img"),
            name: file.name,
            kind: "image",
            previewUrl,
            contentPart: {
              type: "image_ref",
              media: complete.media,
            },
          });
        } else {
          next.push({
            id: randomId("aud"),
            name: file.name,
            kind: "audio",
            previewUrl,
            contentPart: {
              type: "audio_ref",
              media: {
                ...complete.media,
                format:
                  complete.media.format ||
                  formatCodeFromMimeOrName(file.type, file.name),
              },
            },
          });
        }
        totalBytes += file.size;
      } catch {
        setErrorText(`Failed to read ${file.name}`);
      }
    }

    if (next.length > 0) {
      setPendingAttachments((prev) => [...prev, ...next]);
    }

    setUploadingMedia(false);

    e.target.value = "";
  }

  function removeAttachment(id: string): void {
    setPendingAttachments((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((a) => a.id !== id);
    });
  }

  async function send(): Promise<void> {
    const text = input.trim();
    if ((!text && pendingAttachments.length === 0) || loading || uploadingMedia) return;

    const content: string | ChatContentPart[] =
      pendingAttachments.length > 0
        ? [
            ...(text ? [{ type: "text", text } as ChatContentPart] : []),
            ...pendingAttachments.map((a) => a.contentPart),
          ]
        : text;

    const attachmentSummary =
      pendingAttachments.length > 0
        ? ` (${pendingAttachments.length} attachment${pendingAttachments.length === 1 ? "" : "s"})`
        : "";
    const displayText = text || `sent media${attachmentSummary}`;

    const userMessage: UiMessage = {
      id: randomId("user"),
      role: "user",
      text: displayText,
      content,
    };
    const next = [...messages, userMessage];
    setMessages(next);
    setInput("");
    pendingAttachments.forEach((a) => {
      if (a.previewUrl) URL.revokeObjectURL(a.previewUrl);
    });
    setPendingAttachments([]);
    setErrorText(null);
    setLoading(true);

    try {
      const latestUserIndex = next.length - 1;
      const modelMessages: ChatMessage[] = next.map((m, idx) => {
        if (m.role === "assistant") {
          return { role: m.role, content: m.text };
        }
        if (idx === latestUserIndex) {
          return { role: m.role, content: m.content };
        }
        return {
          role: m.role,
          content: summarizeUserContent(m.content),
        };
      });
      const response = await chatCompletions({
        user: session?.user?.id || "",
        messages: modelMessages,
      });
      const assistant = response.choices?.[0]?.message?.content?.trim() || "";
      setMessages((prev) => [
        ...prev,
        {
          id: randomId("assistant"),
          role: "assistant",
          text: assistant || "(empty response)",
          content: assistant || "(empty response)",
        },
      ]);
    } catch (e: unknown) {
      setErrorText(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  if (isPending || !session) return null;

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
          <StatusOrb status={loading ? "thinking" : "connected"} size={10} />
        </div>
      </header>

      <Separator className="shrink-0 w-auto opacity-80" />

      <div className="flex-1 overflow-y-auto min-h-0 pt-6">
        {loadingHistory ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground text-xs">loading today&apos;s chat...</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground text-xs">type a message to begin</p>
          </div>
        ) : (
          <div className="space-y-6 pb-3">
            {messages.map((m) => (
              <ChatMessageView
                key={m.id}
                role={m.role}
                text={m.text}
                content={m.content}
              />
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

      <AIChatComposer
        value={input}
        onChange={setInput}
        onSubmit={() => {
          void send();
        }}
        onRemoveAttachment={removeAttachment}
        onFileChange={(e) => {
          void handlePickMedia(e);
        }}
        fileInputRef={fileInputRef}
        pendingAttachments={pendingAttachments.map((a) => ({
          id: a.id,
          name: a.name,
          kind: a.kind,
          previewUrl: pendingImageUrl(a) || undefined,
        }))}
        disabled={uploadingMedia}
        uploading={uploadingMedia}
        placeholder="type a message..."
      />
    </div>
  );
}
