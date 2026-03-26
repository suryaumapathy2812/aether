import { useEffect, useRef, useState, useCallback } from "react";
import type { ArrowTemplate } from "@arrow-js/core";
import { cn } from "#/lib/utils";
import {
  IconAlertCircle,
  IconLoader2,
} from "@tabler/icons-react";

// ── Types ────────────────────────────────────────────────────────────────

export type ToolSandboxSource = {
  /** Virtual files for the Arrow sandbox. Must include main.ts or main.js. */
  source: Record<string, string>;
  /** Optional CSS to inject (alternative to main.css in source). */
  css?: string;
  /** Render inside shadow DOM for style isolation. */
  shadowDOM?: boolean;
};

export type ToolSandboxProps = {
  /** The sandbox source definition from tool metadata. */
  sandbox: ToolSandboxSource;
  /** Current tool state. */
  state: string;
  /** Error text if the tool failed. */
  errorText?: string;
  /** Called when the sandbox emits output (user interaction). */
  onOutput?: (payload: unknown) => void;
  /** Optional className for the wrapper. */
  className?: string;
};

// ── Lazy Arrow imports ───────────────────────────────────────────────────
// Arrow uses tagged template literals that must run in the browser.
// We dynamic-import to avoid SSR issues.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ArrowAny = any;

let arrowModules: { html: ArrowAny; sandbox: ArrowAny } | null = null;

async function loadArrow() {
  if (arrowModules) return arrowModules;
  const [core, sb] = await Promise.all([
    import("@arrow-js/core"),
    import("@arrow-js/sandbox"),
  ]);
  arrowModules = {
    html: core.html,
    sandbox: sb.sandbox,
  };
  return arrowModules;
}

// ── Component ────────────────────────────────────────────────────────────

export function ToolSandbox({
  sandbox,
  state,
  errorText,
  onOutput,
  className,
}: ToolSandboxProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Running state
  const isRunning = state === "input-streaming" || state === "input-available";

  // Render sandbox when output is available
  useEffect(() => {
    if (isRunning || state !== "output-available" || !containerRef.current) return;

    let cancelled = false;

    async function render() {
      try {
        const { html, sandbox: arrowSandbox } = await loadArrow();
        if (cancelled || !containerRef.current) return;

        // Clear previous content
        containerRef.current.innerHTML = "";

        const template = html`<div>
          ${arrowSandbox(
            {
              source: sandbox.source,
              shadowDOM: sandbox.shadowDOM ?? true,
              debug: false,
              onError: (err: Error | string) => {
                if (!cancelled) {
                  setError(typeof err === "string" ? err : err.message);
                }
              },
            },
            onOutput ? { output: onOutput } : undefined,
          )}
        </div>`;

        template(containerRef.current);
        setMounted(true);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Sandbox render failed");
        }
      }
    }

    void render();

    return () => {
      cancelled = true;
    };
  }, [state, sandbox, onOutput, isRunning]);

  // ── Render states ────────────────────────────────────────────────────

  if (error || state === "output-error") {
    const message = error || errorText || "Tool execution failed";
    return (
      <div
        className={cn(
          "w-full rounded-xl border border-red-500/20 bg-red-500/[0.05] px-4 py-3",
          "flex items-start gap-2.5",
          className,
        )}
      >
        <IconAlertCircle className="size-4 text-red-400/70 shrink-0 mt-0.5" />
        <div className="text-xs text-red-300/80">{message}</div>
      </div>
    );
  }

  if (isRunning) {
    return (
      <div
        className={cn(
          "w-full rounded-xl border border-border bg-accent/10 px-4 py-3",
          "flex items-center gap-3",
          className,
        )}
      >
        <IconLoader2 className="size-4 text-muted-foreground animate-spin" />
        <span className="text-xs text-muted-foreground">Running...</span>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn("w-full", className)}
      data-tool-sandbox={mounted ? "active" : "loading"}
    />
  );
}
