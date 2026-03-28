import type { ToolSandboxSource } from "./tool-sandbox";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

const FENCED_BLOCK =
  /```(?:(ts|tsx|js|jsx|javascript|typescript))?\s*([\s\S]*?)```/gi;

function normalizeModuleSource(source: string): string {
  return source.replace(/\r\n/g, "\n").trim();
}

function looksLikeArrowModule(source: string): boolean {
  const normalized = source.trim();
  if (!normalized) return false;

  // Must import from @arrow-js/core
  if (!normalized.includes("@arrow-js/core")) return false;

  // Must use at least one Arrow API
  const usesArrowAPI =
    normalized.includes("html`") ||
    normalized.includes("component(") ||
    normalized.includes("reactive(");
  if (!usesArrowAPI) return false;

  // Reject React/JSX code that also mentions arrow
  if (/\bimport\s+React\b/.test(normalized)) return false;
  if (/\bfrom\s+['"]react['"]/.test(normalized)) return false;
  if (/\bJSX\.Element\b/.test(normalized)) return false;
  // Reject JSX syntax: self-closing <Component /> or <Component prop="...">
  if (/<[A-Z][A-Za-z]*\s[^`]*\/>/.test(normalized)) return false;

  // Must have an export default (the sandbox entry point)
  if (!normalized.includes("export default")) return false;

  // Reject if template literals are obviously unterminated:
  // count opening html` vs closing ` (rough heuristic)
  const htmlTagCount = (normalized.match(/html`/g) || []).length;
  const backtickCount = (normalized.match(/`/g) || []).length;
  // Each html` needs at least one closing `, so total backticks must be even
  if (backtickCount % 2 !== 0) return false;

  return true;
}

function fencedBlocks(value: string): string[] {
  const matches: string[] = [];
  for (const match of value.matchAll(FENCED_BLOCK)) {
    const candidate = normalizeModuleSource(match[2] || "");
    if (candidate) matches.push(candidate);
  }
  return matches;
}

export function extractArrowSandboxSource(input: unknown): ToolSandboxSource | null {
  if (isRecord(input)) {
    const sandbox = isRecord(input.sandbox) ? input.sandbox : input;
    const sourceValue = sandbox.source;
    if (isRecord(sourceValue)) {
      const source: Record<string, string> = {};
      for (const [key, value] of Object.entries(sourceValue)) {
        if (typeof value === "string") {
          source[key] = value;
        }
      }
      if ("main.ts" in source || "main.js" in source) {
        return {
          source,
          css: typeof sandbox.css === "string" ? sandbox.css : undefined,
          shadowDOM:
            typeof sandbox.shadowDOM === "boolean" ? sandbox.shadowDOM : undefined,
        };
      }
    }
  }

  if (typeof input !== "string") return null;
  const normalized = normalizeModuleSource(input);
  if (!normalized) return null;

  if (looksLikeArrowModule(normalized)) {
    return {
      source: {
        "main.ts": normalized,
      },
    };
  }

  for (const block of fencedBlocks(normalized)) {
    if (looksLikeArrowModule(block)) {
      return {
        source: {
          "main.ts": block,
        },
      };
    }
  }

  return null;
}
