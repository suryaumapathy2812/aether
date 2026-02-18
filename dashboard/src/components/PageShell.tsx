"use client";

import { useRouter } from "next/navigation";
import StatusOrb from "@/components/StatusOrb";
import { useAgentStatus } from "@/hooks/useAgentStatus";

/**
 * Page shell — consistent layout for all inner pages.
 * Mobile-width container (430px max), consistent horizontal padding throughout.
 */
export default function PageShell({
  title,
  children,
  back,
  onClose,
  centered = false,
}: {
  title: string;
  children: React.ReactNode;
  back?: string;
  onClose?: () => void;
  centered?: boolean;
}) {
  const router = useRouter();
  const agentStatus = useAgentStatus();

  return (
    <div className="min-h-screen flex flex-col w-full px-6">
      {/* Header */}
      <header className="flex items-center justify-between pt-8 pb-3 shrink-0">
        {back ? (
          <button
            onClick={() => router.push(back)}
            className="w-8 h-8 flex items-center justify-center -ml-2 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors duration-300"
            aria-label="Go back"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11 14L6 9L11 4" />
            </svg>
          </button>
        ) : (
          <div className="w-8" />
        )}

        <h1 className="text-[11px] tracking-[0.18em] uppercase text-[var(--color-text-secondary)] font-normal">
          {title}
        </h1>

        {onClose ? (
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center -mr-2 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors duration-300"
            aria-label="Close"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <path d="M3 3L11 11M11 3L3 11" />
            </svg>
          </button>
        ) : (
          <div className="w-8 flex items-center justify-center">
            <StatusOrb status={agentStatus} />
          </div>
        )}
      </header>

      <div className="h-px bg-[var(--color-border)] shrink-0" />

      {/* Content */}
      <div
        className={`flex-1 flex flex-col pt-6 pb-4 ${
          centered
            ? "items-center justify-center"
            : ""
        }`}
      >
        {children}
      </div>

      {/* Brand */}
      <div className="text-center py-5 shrink-0">
        <span className="text-[10px] tracking-[0.3em] text-[var(--color-text-muted)] italic font-light">
          aether
        </span>
      </div>
    </div>
  );
}
