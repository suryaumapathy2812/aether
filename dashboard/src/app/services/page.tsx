"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import { useSession } from "@/lib/auth-client";
import { getLatencyMetrics, type LatencyMetrics } from "@/lib/api";

export default function ServicesPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [latency, setLatency] = useState<LatencyMetrics | null>(null);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    getLatencyMetrics().then(setLatency).catch(() => {});
  }, [session, isPending, router]);

  if (isPending || !session) return null;

  return (
    <PageShell title="Services" back="/home">
      <div>
        <h2 className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal mb-4">
          Observability
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <MetricRow
            label="Chat TTFT p95"
            value={latency?.chat?.ttft_p95_ms}
            unit="ms"
          />
          <MetricRow
            label="Voice TTS p95"
            value={latency?.voice?.tts_p95_ms}
            unit="ms"
          />
          <MetricRow
            label="Notification Delivery p95"
            value={latency?.services?.notification_delivery_p95_ms}
            unit="ms"
          />
          <MetricRow
            label="Delegation Duration p95"
            value={latency?.services?.delegation_duration_p95_ms}
            unit="ms"
          />
        </div>
      </div>
    </PageShell>
  );
}

function MetricRow({
  label,
  value,
  unit,
}: {
  label: string;
  value: number | null | undefined;
  unit: string;
}) {
  return (
    <div className="rounded-xl border border-border/50 bg-black/10 px-3 py-2">
      <p className="text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
        {label}
      </p>
      <p className="text-[14px] text-secondary-foreground mt-1">
        {typeof value === "number" ? `${Math.round(value)} ${unit}` : "--"}
      </p>
    </div>
  );
}
