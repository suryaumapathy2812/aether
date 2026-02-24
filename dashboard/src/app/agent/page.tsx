"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSession } from "@/lib/auth-client";
import { getPreferences, updatePreferences, type UserPreferences } from "@/lib/api";

// System defaults
const DEFAULTS: UserPreferences = {
  llm_model: null,
  tts_provider: null,
  tts_model: null,
  tts_voice: null,
  base_style: "default",
  custom_instructions: "",
};

const STYLES = [
  { id: "default", label: "Default" },
  { id: "concise", label: "Concise" },
  { id: "detailed", label: "Detailed" },
  { id: "friendly", label: "Friendly" },
  { id: "professional", label: "Professional" },
];

/**
 * Agent -- voice, model, and personality configuration.
 * Auto-saves on every change. Shows system defaults until user customizes.
 */
export default function AgentPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  const [prefs, setPrefs] = useState<UserPreferences>({ ...DEFAULTS });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    getPreferences()
      .then((serverPrefs) => {
        const merged = { ...DEFAULTS };
        for (const key of Object.keys(serverPrefs) as (keyof UserPreferences)[]) {
          if (serverPrefs[key] !== null && serverPrefs[key] !== undefined) {
            (merged as Record<string, string | null>)[key] = serverPrefs[key];
          }
        }
        setPrefs(merged);
      })
      .catch(() => {});
  }, [session, isPending, router]);

  const save = useCallback(
    async (updates: Partial<UserPreferences>) => {
      const newPrefs = { ...prefs, ...updates };
      setPrefs(newPrefs);
      setSaving(true);
      setSaved(false);
      try {
        await updatePreferences(updates);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
      } catch {
        // silent
      } finally {
        setSaving(false);
      }
    },
    [prefs]
  );

  if (isPending || !session) return null;

  return (
    <PageShell title="Agent" back="/home">
      <div className="space-y-10 max-w-[920px] mx-auto">
        {/* Save status */}
        <div className="h-4 flex items-center justify-end">
          {saving && (
            <span className="text-[10px] tracking-[0.15em] text-muted-foreground animate-pulse">
              saving...
            </span>
          )}
          {saved && !saving && (
            <span className="text-[10px] tracking-[0.15em] text-secondary-foreground animate-[fade-in_0.2s_ease]">
              saved
            </span>
          )}
        </div>

        <Section title="Personality">
          <Picker
            label="Base style and tone"
            value={prefs.base_style}
            options={STYLES}
            onChange={(v) => save({ base_style: v })}
          />
          <div className="mt-5">
            <Label className="block text-[10px] tracking-[0.15em] uppercase text-muted-foreground mb-2 font-normal">
              Custom instructions
            </Label>
            <Textarea
              value={prefs.custom_instructions || ""}
              onChange={(e) =>
                setPrefs({ ...prefs, custom_instructions: e.target.value })
              }
              onBlur={() => {
                if (prefs.custom_instructions !== null) {
                  save({ custom_instructions: prefs.custom_instructions });
                }
              }}
              placeholder="Tell Aether how to behave..."
              rows={10}
              className="bg-black/20 border-0 rounded-2xl shadow-none px-3.5 py-3 text-[15px] font-normal leading-relaxed resize-none focus-visible:ring-0 focus-visible:bg-black/25 transition-colors duration-300"
            />
            <p className="text-[10px] text-muted-foreground mt-1 font-normal">
              saves when you tap outside
            </p>
          </div>
        </Section>

      </div>
    </PageShell>
  );
}

// -- Section --

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
       <h2 className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal mb-5">
         {title}
       </h2>
       <div className="space-y-2">{children}</div>
    </div>
  );
}

// -- Picker (shadcn Select) --

function Picker({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string | null;
  options: { id: string; label: string; description?: string; recommended?: boolean }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 py-3.5">
      <span className="text-[13px] text-muted-foreground font-normal shrink-0">
        {label}
      </span>
      <Select value={value || undefined} onValueChange={onChange}>
        <SelectTrigger className="w-auto h-auto gap-1.5 border-0 bg-transparent shadow-none px-0 py-0 text-[13px] text-secondary-foreground font-normal hover:text-foreground focus:ring-0 focus-visible:ring-0 transition-colors duration-300">
          <SelectValue placeholder="--" />
        </SelectTrigger>
        <SelectContent className="min-w-[min(220px,calc(100vw-2rem))] max-h-[280px] bg-card border-border">
          {options.map((opt) => (
            <SelectItem
              key={opt.id}
              value={opt.id}
              className="text-[13px] font-normal"
            >
              <span>{opt.label}</span>
              {opt.id === value && (
                <span className="ml-2 text-[9px] tracking-wider uppercase text-muted-foreground">
                  current
                </span>
              )}
              {opt.recommended && opt.id !== value && (
                <span className="ml-2 text-[9px] tracking-wider uppercase text-muted-foreground">
                  recommended
                </span>
              )}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
