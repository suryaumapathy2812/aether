"use client";

import { useRouter } from "next/navigation";
import { ChevronLeft, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import { useAgentStatus } from "@/hooks/useAgentStatus";

/**
 * Page shell — consistent layout for all inner pages.
 * Mobile-width container (430px max), consistent horizontal padding throughout.
 * Built on shadcn Button + Separator primitives.
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
          <Button
            variant="aether-ghost"
            size="icon"
            onClick={() => router.push(back)}
            className="w-8 h-8 -ml-2"
            aria-label="Go back"
          >
            <ChevronLeft className="size-[18px]" strokeWidth={1.5} />
          </Button>
        ) : (
          <div className="w-8" />
        )}

        <h1 className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal">
          {title}
        </h1>

        {onClose ? (
          <Button
            variant="aether-ghost"
            size="icon"
            onClick={onClose}
            className="w-8 h-8 -mr-2"
            aria-label="Close"
          >
            <X className="size-[14px]" strokeWidth={1.5} />
          </Button>
        ) : (
          <div className="w-8 flex items-center justify-center">
            <StatusOrb status={agentStatus} />
          </div>
        )}
      </header>

      <Separator className="shrink-0" />

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
        <span className="text-[10px] tracking-[0.3em] text-muted-foreground italic font-light">
          aether
        </span>
      </div>
    </div>
  );
}
