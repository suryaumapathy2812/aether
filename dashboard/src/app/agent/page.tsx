"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import { useSession } from "@/lib/auth-client";
import { getPreferences, updatePreferences, type UserPreferences } from "@/lib/api";

// ── Types for config JSONs ──

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
  description: string;
  recommended: boolean;
}

interface LLMProvider {
  display_name: string;
  models: Record<string, LLMModel>;
}

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
  providers: Record<string, LLMProvider>;
}

interface TTSConfig {
  providers: Record<string, TTSProvider>;
}

// System defaults — what the agent runs with when user hasn't customized
const DEFAULTS: UserPreferences = {
  stt_provider: "deepgram",
  stt_model: "nova-3",
  stt_language: "en",
  llm_provider: "openai",
  llm_model: "gpt-4o",
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
 * Agent — voice, model, and personality configuration.
 * Auto-saves on every change. Shows system defaults until user customizes.
 */
export default function AgentPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  // Config JSONs (static reference data)
  const [sttConfig, setSttConfig] = useState<STTConfig | null>(null);
  const [llmConfig, setLlmConfig] = useState<LLMConfig | null>(null);
  const [ttsConfig, setTtsConfig] = useState<TTSConfig | null>(null);

  // User preferences — merged with defaults so nothing shows "not set"
  const [prefs, setPrefs] = useState<UserPreferences>({ ...DEFAULTS });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Load config JSONs + user preferences
  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }

    // Load static configs
    fetch("/stt.json").then((r) => r.json()).then(setSttConfig).catch(() => {});
    fetch("/llm.json").then((r) => r.json()).then(setLlmConfig).catch(() => {});
    fetch("/tts.json").then((r) => r.json()).then(setTtsConfig).catch(() => {});

    // Load user preferences — merge with defaults (null values → default)
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

  // Save handler — auto-saves on every change
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

  // ── Derived options from config JSONs ──

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

  const llmProviders = llmConfig
    ? Object.entries(llmConfig.providers).map(([id, p]) => ({
        id,
        label: p.display_name,
      }))
    : [];

  const llmModels =
    llmConfig && prefs.llm_provider
      ? Object.entries(
          llmConfig.providers[prefs.llm_provider]?.models || {}
        ).map(([id, m]) => ({ id, label: id, description: m.description, recommended: m.recommended }))
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
      <div className="space-y-8">
        {/* ── Save status (sticky top) ── */}
        <div className="h-4 flex items-center justify-end">
          {saving && (
            <span className="text-[10px] tracking-[0.15em] text-[var(--color-text-muted)] animate-pulse">
              saving...
            </span>
          )}
          {saved && !saving && (
            <span className="text-[10px] tracking-[0.15em] text-[var(--color-text-secondary)] animate-[fade-in_0.2s_ease]">
              saved
            </span>
          )}
        </div>

        {/* ── Voice ── */}
        <Section title="Voice">
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

        {/* ── Model ── */}
        <Section title="Model">
          <Picker
            label="LLM Provider"
            value={prefs.llm_provider}
            options={llmProviders}
            onChange={(v) => save({ llm_provider: v, llm_model: null })}
          />
          {llmModels.length > 0 && (
            <Picker
              label="Model"
              value={prefs.llm_model}
              options={llmModels}
              onChange={(v) => save({ llm_model: v })}
            />
          )}
        </Section>

        {/* ── Personality ── */}
        <Section title="Personality">
          <Picker
            label="Base style and tone"
            value={prefs.base_style}
            options={STYLES}
            onChange={(v) => save({ base_style: v })}
          />
          <div className="mt-4">
            <label className="block text-[10px] tracking-[0.15em] uppercase text-[var(--color-text-muted)] mb-2 font-normal">
              Custom instructions
            </label>
            <textarea
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
              rows={4}
              className="w-full bg-transparent border-b border-[var(--color-border)] text-[var(--color-text)] font-light text-[15px] leading-relaxed outline-none resize-none py-2 placeholder:text-[var(--color-text-muted)] focus:border-[var(--color-text-secondary)] transition-colors duration-300"
            />
            <p className="text-[10px] text-[var(--color-text-muted)] mt-1 font-light">
              saves when you tap outside
            </p>
          </div>
        </Section>
      </div>
    </PageShell>
  );
}

// ── Section ──

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h2 className="text-[11px] tracking-[0.18em] uppercase text-[var(--color-text-secondary)] font-normal mb-4">
        {title}
      </h2>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

// ── Picker (dropdown row) ──

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
  const [open, setOpen] = useState(false);

  const selected = options.find((o) => o.id === value);
  const displayValue = selected?.label || value || "—";

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-3 border-b border-[var(--color-border)] group"
      >
        <span className="text-[13px] text-[var(--color-text-muted)] font-light">
          {label}
        </span>
        <span className="text-[13px] text-[var(--color-text-secondary)] font-light group-hover:text-[var(--color-text)] transition-colors duration-300">
          {displayValue}
          <svg
            className="inline-block ml-1.5 w-3 h-3 opacity-40"
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          >
            <path d={open ? "M3 7.5L6 4.5L9 7.5" : "M3 4.5L6 7.5L9 4.5"} />
          </svg>
        </span>
      </button>

      {open && (
        <>
          {/* Backdrop to close dropdown */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 top-full mt-1 z-50 min-w-[220px] max-h-[280px] overflow-y-auto bg-[var(--color-surface)] border border-[var(--color-border)] animate-[fade-in_0.15s_ease]">
            {options.map((opt) => (
              <button
                key={opt.id}
                onClick={() => {
                  onChange(opt.id);
                  setOpen(false);
                }}
                className={`w-full text-left px-4 py-2.5 text-[13px] font-light transition-colors duration-200 hover:bg-[var(--color-surface-hover)] ${
                  opt.id === value
                    ? "text-[var(--color-text)]"
                    : "text-[var(--color-text-secondary)]"
                }`}
              >
                <span>{opt.label}</span>
                {opt.id === value && (
                  <span className="ml-2 text-[9px] tracking-wider uppercase text-[var(--color-text-muted)]">
                    current
                  </span>
                )}
                {opt.recommended && opt.id !== value && (
                  <span className="ml-2 text-[9px] tracking-wider uppercase text-[var(--color-text-muted)]">
                    recommended
                  </span>
                )}
                {opt.description && (
                  <span className="block text-[10px] text-[var(--color-text-muted)] mt-0.5">
                    {opt.description}
                  </span>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
