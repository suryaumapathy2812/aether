"use client";

import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";

/**
 * ContentShell — simple content wrapper for non-chat pages.
 * Centered, max-width, with optional title and back button.
 */
export default function ContentShell({
  title,
  children,
  back,
}: {
  title: string;
  children: React.ReactNode;
  back?: string;
}) {
  const router = useRouter();

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[720px] mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          {back && (
            <button
              onClick={() => router.push(back)}
              className="w-7 h-7 flex items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-white/[0.04] transition-colors"
            >
              <ChevronLeft className="size-4" />
            </button>
          )}
          <h1 className="text-[13px] font-medium text-foreground/80">{title}</h1>
        </div>

        {children}
      </div>
    </div>
  );
}
