"use client";

import PageShell from "@/components/PageShell";

/**
 * Agent page — placeholder while agent runtime is being rethought.
 * Navigation link is hidden from home; page kept for future use.
 */
export default function AgentPage() {
  return (
    <PageShell title="Agent">
      <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground text-sm">
        <p>Agent runtime is being redesigned.</p>
        <p className="mt-1 text-xs">This page will return with a better experience.</p>
      </div>
    </PageShell>
  );
}
