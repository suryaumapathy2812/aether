"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import { useSession } from "@/lib/auth-client";

/**
 * Devices — temporarily disabled.
 *
 * Device management (pairing, Telegram bot) requires the orchestrator.
 * This page will be re-enabled once the orchestrator is wired up.
 */
export default function DevicesPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && !session) {
      router.push("/");
    }
  }, [session, isPending, router]);

  if (isPending || !session) return null;

  return (
    <PageShell title="Devices" back="/home" centered>
      <div className="text-center">
        <p className="text-muted-foreground text-xs mb-2">
          device management is not available yet
        </p>
        <p className="text-muted-foreground text-[10px] leading-relaxed max-w-[260px] mx-auto">
          pairing and Telegram bot setup require the orchestrator, which is not
          connected in this mode.
        </p>
      </div>
    </PageShell>
  );
}
