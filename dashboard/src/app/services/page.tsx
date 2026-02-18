"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { useSession } from "@/lib/auth-client";
import { listApiKeys, saveApiKey, deleteApiKey } from "@/lib/api";

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
      <div>
        {PROVIDERS.map((p) => (
          <div
            key={p.id}
            className="py-4 border-b border-[var(--color-border)] last:border-b-0"
          >
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
                  <button
                    onClick={() => handleSave(p.id)}
                    disabled={saving || !keyValue.trim()}
                    className="btn text-xs disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    {saving ? "..." : "save"}
                  </button>
                  {keys[p.id] && (
                    <button
                      onClick={() => handleDelete(p.id)}
                      className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors duration-300"
                    >
                      remove
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setEditing(null);
                      setKeyValue("");
                    }}
                    className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors duration-300"
                  >
                    cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => {
                  setEditing(p.id);
                  setKeyValue("");
                }}
                className="w-full flex items-center justify-between group"
              >
                <span className="text-[14px] text-[var(--color-text-secondary)] group-hover:text-[var(--color-text)] transition-colors duration-300 font-light">
                  {p.label}
                </span>
                <span className="text-[10px] text-[var(--color-text-muted)] tracking-wider">
                  {keys[p.id] || "not set"}
                </span>
              </button>
            )}
          </div>
        ))}
      </div>
    </PageShell>
  );
}
