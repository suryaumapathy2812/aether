"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { useSession } from "@/lib/auth-client";
import {
  listApiKeys,
  saveApiKey,
  deleteApiKey,
  getLatencyMetrics,
  type LatencyMetrics,
} from "@/lib/api";

const PROVIDERS = [
  { id: "openai", label: "OpenAI" },
  { id: "deepgram", label: "Deepgram" },
  { id: "elevenlabs", label: "ElevenLabs" },
  { id: "sarvam", label: "Sarvam" },
];

/**
 * Services — API key management.
 * Each provider shown as a row; tap to edit inline.
 */
export default function ServicesPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [editing, setEditing] = useState<string | null>(null);
  const [keyValue, setKeyValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [latency, setLatency] = useState<LatencyMetrics | null>(null);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    listApiKeys()
      .then((list) => {
        const map: Record<string, string> = {};
        list.forEach((k) => (map[k.provider] = k.preview));
        setKeys(map);
      })
      .catch(() => {});

    getLatencyMetrics().then(setLatency).catch(() => {});
  }, [session, isPending, router]);

  async function handleSave(provider: string) {
    setSaving(true);
    try {
      await saveApiKey(provider, keyValue);
      keys[provider] = keyValue.slice(0, 8) + "...";
      setKeys({ ...keys });
      setEditing(null);
      setKeyValue("");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(provider: string) {
    await deleteApiKey(provider);
    delete keys[provider];
    setKeys({ ...keys });
  }

  if (isPending || !session) return null;

  return (
    <PageShell title="Services" back="/home">
      <div className="space-y-1">
        {PROVIDERS.map((p) => (
          <div key={p.id}>
            <div className="py-5">
              {editing === p.id ? (
                <div className="animate-[fade-in_0.2s_ease]">
                  <MinimalInput
                    label={p.label}
                    type="password"
                    value={keyValue}
                    onChange={setKeyValue}
                    placeholder={`${p.label} API Key`}
                  />
                  <div className="flex items-center gap-4 mt-1">
                    <Button
                      variant="aether"
                      size="aether"
                      onClick={() => handleSave(p.id)}
                      disabled={saving || !keyValue.trim()}
                    >
                      {saving ? "..." : "save"}
                    </Button>
                    {keys[p.id] && (
                      <Button
                        variant="aether-link"
                        size="aether-link"
                        onClick={() => handleDelete(p.id)}
                        className="min-h-[44px] px-2"
                      >
                        remove
                      </Button>
                    )}
                    <Button
                      variant="aether-link"
                      size="aether-link"
                      onClick={() => {
                        setEditing(null);
                        setKeyValue("");
                      }}
                      className="min-h-[44px] px-2"
                    >
                      cancel
                    </Button>
                  </div>
                </div>
              ) : (
                <Button
                  variant="aether-ghost"
                  size="aether-link"
                  onClick={() => {
                    setEditing(p.id);
                    setKeyValue("");
                  }}
                  className="w-full flex items-center justify-between group"
                >
                  <span className="text-[14px] text-secondary-foreground group-hover:text-foreground transition-colors duration-300 font-normal">
                    {p.label}
                  </span>
                  <span className="text-[10px] text-muted-foreground tracking-wider">
                    {keys[p.id] || "not set"}
                  </span>
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-10 border-t border-border/60 pt-6">
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
