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

// -- Types for config JSONs --

interface STTModel {
  description: string;
  streaming: boolean;
  recommended: boolean;
  languages: string[];
}

interface STTProvider {
  display_name: string;
  models: Record<string, STTModel>;
}

interface LLMModel {
  provider: string;
  description: string;
  recommended: boolean;
}

// Flat model list — all models route through OpenRouter.
// Provider is display-only (shown as a label in the picker).

interface TTSModel {
  description: string;
  streaming: boolean;
  recommended: boolean;
  voices: string[];
}

interface TTSProvider {
  display_name: string;
  models: Record<string, TTSModel>;
}

interface STTConfig {
  providers: Record<string, STTProvider>;
}

interface LLMConfig {
  models: Record<string, LLMModel>;
}

interface TTSConfig {
  providers: Record<string, TTSProvider>;
}

// System defaults
const DEFAULTS: UserPreferences = {
  stt_provider: "deepgram",
  stt_model: "nova-3",
  stt_language: "en",
  llm_model: "openai/gpt-4o",
  tts_provider: "openai",
  tts_model: "tts-1",
  tts_voice: "nova",
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

  const [sttConfig, setSttConfig] = useState<STTConfig | null>(null);
  const [llmConfig, setLlmConfig] = useState<LLMConfig | null>(null);
  const [ttsConfig, setTtsConfig] = useState<TTSConfig | null>(null);

  const [prefs, setPrefs] = useState<UserPreferences>({ ...DEFAULTS });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    fetch("/stt.json").then((r) => r.json()).then(setSttConfig).catch(() => {});
    fetch("/llm.json").then((r) => r.json()).then(setLlmConfig).catch(() => {});
    fetch("/tts.json").then((r) => r.json()).then(setTtsConfig).catch(() => {});

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

  // -- Derived options from config JSONs --

  const sttProviders = sttConfig
    ? Object.entries(sttConfig.providers).map(([id, p]) => ({
        id,
        label: p.display_name,
      }))
    : [];

  const sttModels =
    sttConfig && prefs.stt_provider
      ? Object.entries(
          sttConfig.providers[prefs.stt_provider]?.models || {}
        ).map(([id, m]) => ({ id, label: id, description: m.description, recommended: m.recommended }))
      : [];

  // Flat model list — all models route through OpenRouter.
  // Show "provider · model" as the label for clarity.
  const llmModels = llmConfig
    ? Object.entries(llmConfig.models).map(([id, m]) => ({
        id,
        label: `${m.provider} · ${id.includes("/") ? id.split("/").pop() : id}`,
        description: m.description,
        recommended: m.recommended,
      }))
    : [];

  const ttsProviders = ttsConfig
    ? Object.entries(ttsConfig.providers).map(([id, p]) => ({
        id,
        label: p.display_name,
      }))
    : [];

  const ttsModels =
    ttsConfig && prefs.tts_provider
      ? Object.entries(
          ttsConfig.providers[prefs.tts_provider]?.models || {}
        ).map(([id, m]) => ({ id, label: id, description: m.description, recommended: m.recommended }))
      : [];

  const ttsVoices =
    ttsConfig && prefs.tts_provider && prefs.tts_model
      ? ttsConfig.providers[prefs.tts_provider]?.models[prefs.tts_model]
          ?.voices || []
      : [];

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

        <Section title="STT">
          <Picker
            label="STT Provider"
            value={prefs.stt_provider}
            options={sttProviders}
            onChange={(v) => save({ stt_provider: v, stt_model: null })}
          />
          {sttModels.length > 0 && (
            <Picker
              label="STT Model"
              value={prefs.stt_model}
              options={sttModels}
              onChange={(v) => save({ stt_model: v })}
            />
          )}
        </Section>

        <Section title="TTS">
          <Picker
            label="TTS Provider"
            value={prefs.tts_provider}
            options={ttsProviders}
            onChange={(v) =>
              save({ tts_provider: v, tts_model: null, tts_voice: null })
            }
          />
          {ttsModels.length > 0 && (
            <Picker
              label="TTS Model"
              value={prefs.tts_model}
              options={ttsModels}
              onChange={(v) => save({ tts_model: v, tts_voice: null })}
            />
          )}
          {ttsVoices.length > 0 && (
            <Picker
              label="Voice"
              value={prefs.tts_voice}
              options={ttsVoices.map((v) => ({ id: v, label: v }))}
              onChange={(v) => save({ tts_voice: v })}
            />
          )}
        </Section>

        <Section title="LLM">
          <Picker
            label="Model"
            value={prefs.llm_model}
            options={llmModels}
            onChange={(v) => save({ llm_model: v })}
          />
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
