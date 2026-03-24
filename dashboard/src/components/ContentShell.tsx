"use client";

import { useRouter } from "next/navigation";
import { IconChevronLeft } from "@tabler/icons-react";

/**
 * ContentShell — simple content wrapper for non-chat pages.
 * Centered, max-width, with optional title, back button, and header action.
 */
export default function ContentShell({
  title,
  children,
  back,
  action,
}: {
  title: string;
  children: React.ReactNode;
  back?: string;
  action?: React.ReactNode;
}) {
  const router = useRouter();

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[720px] mx-auto px-6 pt-20 pb-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            {back && (
              <button
                onClick={() => router.push(back)}
                className="w-7 h-7 flex items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-accent/40 transition-colors"
              >
                <IconChevronLeft className="size-4" />
              </button>
            )}
            <h1 className="text-2xl font-semibold text-foreground tracking-tight">{title}</h1>
          </div>
          {action}
        </div>

        {children}
      </div>
    </div>
  );
}
