"use client";

import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

// ── Types ────────────────────────────────────────────────

interface ChatMessageProps {
  role: "user" | "assistant";
  text: string;
}

// ── Markdown component overrides ─────────────────────────
// Minimal set: bold, italic, inline code, code blocks, links, lists.
// No images, no headings (keeps the chat aesthetic clean).

const mdComponents: Components = {
  // Strip headings down to bold text — chat doesn't need h1-h6
  h1: ({ children }) => <strong className="block mb-1">{children}</strong>,
  h2: ({ children }) => <strong className="block mb-1">{children}</strong>,
  h3: ({ children }) => <strong className="block mb-1">{children}</strong>,

  // Paragraphs with spacing
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,

  // Bold / italic
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => <em>{children}</em>,

  // Inline code
  code: ({ children, className }) => {
    // If it has a language className, it's a fenced code block (rendered inside <pre>)
    if (className) {
      return <code className={className}>{children}</code>;
    }
    // Inline code
    return (
      <code className="rounded bg-white/8 px-1.5 py-0.5 text-[0.9em] font-mono text-foreground/90">
        {children}
      </code>
    );
  },

  // Fenced code blocks
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-white/6 border border-border/50 p-3 text-[0.85em] font-mono leading-relaxed">
      {children}
    </pre>
  ),

  // Links
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

  // Lists
  ul: ({ children }) => (
    <ul className="mb-2 ml-4 list-disc space-y-0.5 last:mb-0">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 ml-4 list-decimal space-y-0.5 last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,

  // Block quotes
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-foreground/20 pl-3 italic text-secondary-foreground">
      {children}
    </blockquote>
  ),

  // Horizontal rules
  hr: () => <hr className="my-3 border-border/50" />,

  // Suppress images in chat
  img: () => null,
};

// ── Component ────────────────────────────────────────────

/**
 * A single chat message bubble.
 *
 * - User messages render as plain text (no markdown parsing needed).
 * - Assistant messages render through react-markdown.
 * - Wrapped in React.memo so completed messages never re-render
 *   during streaming — only the last (active) message updates.
 */
const ChatMessage = memo(
  function ChatMessage({ role, text }: ChatMessageProps) {
    // Memoize the rendered content so react-markdown only re-parses
    // when the text actually changes (not on parent re-renders).
    const content = useMemo(() => {
      if (role === "user") {
        return <>{text}</>;
      }
      return (
        <ReactMarkdown components={mdComponents}>{text}</ReactMarkdown>
      );
    }, [role, text]);

    return (
      <div
        className={`animate-[fade-in_0.25s_ease] ${
          role === "user" ? "flex flex-col items-end" : ""
        }`}
      >
        <span className="text-[9px] text-muted-foreground uppercase tracking-[0.12em] mb-1.5 block">
          {role === "user" ? "you" : "aether"}
        </span>
        <div
          className={`text-[14px] leading-[1.65] font-normal max-w-[85%] ${
            role === "user"
              ? "text-foreground bg-white/7 border border-border/70 rounded-2xl rounded-tr-sm px-4 py-2.5"
              : "text-secondary-foreground"
          }`}
        >
          {content}
        </div>
      </div>
    );
  },
  // Custom comparator: skip re-render if role and text are identical.
  // During streaming, only the last message's text changes — all
  // previous messages stay stable and never re-render.
  (prev, next) => prev.role === next.role && prev.text === next.text
);

ChatMessage.displayName = "ChatMessage";

export default ChatMessage;
