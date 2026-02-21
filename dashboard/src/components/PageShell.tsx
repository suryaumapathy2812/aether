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
    <div className="h-full flex flex-col w-full px-6 sm:px-8">
      {/* Header */}
      <header className="flex items-center justify-between pt-7 sm:pt-8 pb-4 shrink-0">
        {back ? (
          <Button
            variant="aether-ghost"
            size="icon"
            onClick={() => router.push(back)}
            className="w-8 h-8 min-w-[44px] min-h-[44px] -ml-2"
            aria-label="Go back"
          >
            <ChevronLeft className="size-[18px]" strokeWidth={1.5} />
          </Button>
        ) : (
          <div className="w-8 min-w-[44px]" />
        )}

        <h1 className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal">
          {title}
        </h1>

        {onClose ? (
          <Button
            variant="aether-ghost"
            size="icon"
            onClick={onClose}
            className="w-8 h-8 min-w-[44px] min-h-[44px] -mr-2"
            aria-label="Close"
          >
            <X className="size-[14px]" strokeWidth={1.5} />
          </Button>
        ) : (
          <div className="w-8 min-w-[44px] flex items-center justify-center">
            <StatusOrb status={agentStatus} size={10} />
          </div>
        )}
      </header>

      <Separator className="shrink-0 opacity-80" />

      {/* Content */}
      <div
        className={`flex-1 min-h-0 pt-8 ${centered ? "flex items-center justify-center pb-6" : "overflow-y-auto"}`}
      >
        <div className={`w-full ${centered ? "max-w-[560px]" : "pb-6"}`}>
          {children}
        </div>
      </div>

      {/* Brand */}
      <div className="text-center py-5 sm:py-6 shrink-0">
        <span className="logo-wordmark text-[10px] text-muted-foreground font-medium">
          aether
        </span>
      </div>
    </div>
  );
}
