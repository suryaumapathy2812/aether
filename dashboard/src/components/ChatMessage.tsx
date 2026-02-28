"use client";

import { memo, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import { Pause, Play } from "lucide-react";
import type { ChatContentPart } from "@/lib/api";
import { LiveWaveform } from "@/components/ui/live-waveform";
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

interface ChatMessageProps {
  role: "user" | "assistant";
  text: string;
  content?: string | ChatContentPart[];
}

function audioMimeFromFormat(format: string): string {
  const v = format.toLowerCase();
  if (v === "mp3") return "audio/mpeg";
  if (v === "m4a") return "audio/mp4";
  if (v === "ogg") return "audio/ogg";
  if (v === "flac") return "audio/flac";
  if (v === "aac") return "audio/aac";
  if (v === "aiff") return "audio/aiff";
  if (v === "webm") return "audio/webm";
  return "audio/wav";
}

function UserAudioAttachment({ src }: { src: string }) {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onPause = () => setPlaying(false);
    const onPlay = () => setPlaying(true);
    const onEnded = () => setPlaying(false);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("ended", onEnded);
    return () => {
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("ended", onEnded);
    };
  }, []);

  async function toggle(): Promise<void> {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      try {
        await audio.play();
      } catch {
        setPlaying(false);
      }
      return;
    }
    audio.pause();
  }

  return (
    <div className="flex items-center gap-2 rounded-xl border border-border/70 bg-black/25 px-2.5 py-2">
      <button
        type="button"
        onClick={() => void toggle()}
        className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/10 text-foreground hover:bg-white/16 transition-colors"
        aria-label={playing ? "Pause audio" : "Play audio"}
      >
        {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
      </button>
      <div className="h-8 flex-1 min-w-[170px]">
        <LiveWaveform
          active={false}
          processing={playing}
          mode="static"
          height={32}
          barWidth={2}
          barGap={1.5}
          barRadius={2}
          className="w-full"
        />
      </div>
      <audio ref={audioRef} preload="none" src={src} className="hidden" />
    </div>
  );
}

function UserImageAttachment({ src, alt }: { src: string; alt: string }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          className="block overflow-hidden rounded-xl border border-border/70 bg-black/20 transition-transform hover:scale-[1.01]"
          aria-label="Open image preview"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt={alt} className="h-36 w-full object-cover" loading="lazy" />
        </button>
      </DialogTrigger>
      <DialogContent
        showCloseButton={false}
        className="h-[90vh] w-[94vw] max-w-[94vw] border-0 bg-transparent p-0 shadow-none flex items-center justify-center"
      >
        <DialogTitle className="sr-only">Image preview</DialogTitle>
        <div className="flex h-full w-full items-center justify-center">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt}
            className="max-h-[86vh] w-auto max-w-[92vw] rounded-2xl border border-border/70 object-contain"
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}

const mdComponents: Components = {
  h1: ({ children }) => <strong className="block mb-1">{children}</strong>,
  h2: ({ children }) => <strong className="block mb-1">{children}</strong>,
  h3: ({ children }) => <strong className="block mb-1">{children}</strong>,
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => <em>{children}</em>,
  code: ({ children, className }) => {
    if (className) return <code className={className}>{children}</code>;
    return (
      <code className="rounded bg-white/8 px-1.5 py-0.5 text-[0.9em] font-mono text-foreground/90">
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-white/6 border border-border/50 p-3 text-[0.85em] font-mono leading-relaxed">
      {children}
    </pre>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="underline underline-offset-2 decoration-foreground/30 hover:decoration-foreground/60 transition-colors"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 ml-4 list-disc space-y-0.5 last:mb-0">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 ml-4 list-decimal space-y-0.5 last:mb-0">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-foreground/20 pl-3 italic text-secondary-foreground">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-border/50" />,
  img: () => null,
};

const ChatMessage = memo(
  function ChatMessage({ role, text, content: messageContent }: ChatMessageProps) {
    if (role === "assistant") {
      return (
        <div className="animate-[fade-in_0.25s_ease]">
          <span className="text-[9px] text-muted-foreground uppercase tracking-[0.12em] mb-1.5 block">
            aether
          </span>
          <div className="text-[14px] leading-[1.65] font-normal max-w-[85%] text-secondary-foreground">
            <ReactMarkdown components={mdComponents}>{text}</ReactMarkdown>
          </div>
        </div>
      );
    }

    const parts = Array.isArray(messageContent) ? messageContent : [];
    const textParts = parts
      .filter((p): p is Extract<ChatContentPart, { type: "text" }> => p.type === "text")
      .map((p) => p.text.trim())
      .filter(Boolean);
    const imageParts = parts.filter(
      (
        p
      ): p is
        | Extract<ChatContentPart, { type: "image_url" }>
        | Extract<ChatContentPart, { type: "image_ref" }> =>
        (p.type === "image_url" && Boolean(p.image_url?.url)) ||
        (p.type === "image_ref" && Boolean(p.media?.url))
    );
    const audioParts = parts.filter(
      (
        p
      ): p is
        | Extract<ChatContentPart, { type: "input_audio" }>
        | Extract<ChatContentPart, { type: "audio_ref" }> =>
        (p.type === "input_audio" && Boolean(p.input_audio?.data)) ||
        (p.type === "audio_ref" && Boolean(p.media?.url))
    );

    const hasParts = parts.length > 0;
    const plainText = hasParts ? textParts.join("\n") : text;

    return (
      <div className="animate-[fade-in_0.25s_ease] flex flex-col items-end">
        <span className="text-[9px] text-muted-foreground uppercase tracking-[0.12em] mb-1.5 block">
          you
        </span>
        {(imageParts.length > 0 || audioParts.length > 0) && (
          <div className="mb-2 flex w-full flex-col items-end gap-2.5">
            {imageParts.length > 0 && (
              imageParts.length === 1 ? (
                <div className="w-fit max-w-[min(74vw,440px)]">
                  {imageParts.map((p, idx) => {
                    const src = p.type === "image_url" ? p.image_url.url : p.media.url || "";
                    return (
                      <UserImageAttachment
                        key={`img-${idx}`}
                        src={src}
                        alt={`attachment ${idx + 1}`}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-2 w-full max-w-[min(74vw,440px)]">
                  {imageParts.map((p, idx) => {
                    const src = p.type === "image_url" ? p.image_url.url : p.media.url || "";
                    return (
                      <UserImageAttachment
                        key={`img-${idx}`}
                        src={src}
                        alt={`attachment ${idx + 1}`}
                      />
                    );
                  })}
                </div>
              )
            )}

            {audioParts.length > 0 && (
              <div className="space-y-2 w-full max-w-[min(74vw,440px)]">
                {audioParts.map((p, idx) => {
                  const src =
                    p.type === "audio_ref"
                      ? p.media.url || ""
                      : `data:${audioMimeFromFormat(p.input_audio.format)};base64,${p.input_audio.data}`;
                  return <UserAudioAttachment key={`aud-${idx}`} src={src} />;
                })}
              </div>
            )}
          </div>
        )}

        {plainText && (
          <div className="text-[14px] leading-[1.65] font-normal max-w-[85%] text-foreground bg-white/7 border border-border/70 rounded-2xl rounded-tr-sm px-4 py-2.5">
            <p className="whitespace-pre-wrap">{plainText}</p>
          </div>
        )}
      </div>
    );
  },
  (prev, next) =>
    prev.role === next.role &&
    prev.text === next.text &&
    prev.content === next.content
);

ChatMessage.displayName = "ChatMessage";

export default ChatMessage;
