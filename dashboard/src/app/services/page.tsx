"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
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
    </PageShell>
  );
}
