import type { ToolSandboxSource } from "./tool-sandbox";
import { normalizeSandboxFileSource } from "./tool-sandbox";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

const FENCED_BLOCK =
  /```(?:(ts|tsx|js|jsx|javascript|typescript))?\s*([\s\S]*?)```/gi;

function normalizeModuleSource(source: string): string {
  return source.replace(/\r\n/g, "\n").trim();
}

function ensureArrowNamedImports(source: string, required: string[]): string {
  const importPattern =
    /import\s*\{([^}]*)\}\s*from\s*['"]@arrow-js\/core['"]/;
  const match = source.match(importPattern);
  if (!match) return source;

  const existing = match[1]
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const merged = Array.from(new Set([...existing, ...required])).sort();
  const replacement = `import { ${merged.join(", ")} } from '@arrow-js/core'`;
  return source.replace(importPattern, replacement);
}

function insertHelpersAfterImports(source: string, helpers: string[]): string {
  if (helpers.length === 0) return source;
  const importBlockMatch = source.match(
    /^(?:import[^\n]*\n(?:[ \t]*\n)?)*/u,
  );
  const insertionIndex = importBlockMatch?.[0]?.length ?? 0;
  const prefix = source.slice(0, insertionIndex);
  const suffix = source.slice(insertionIndex);
  const helperBlock = `${helpers.join("\n\n")}\n\n`;
  return `${prefix}${helperBlock}${suffix}`;
}

function rewriteInlineMappedHtmlTemplates(source: string): string {
  const helpers: string[] = [];
  let helperIndex = 0;

  const directReturnPattern =
    /\.map\(\s*(\(?\s*([A-Za-z_$][\w$]*)\s*(?::[^)=]+)?\s*\)?)\s*=>\s*html`([\s\S]*?)`\s*\)/g;
  const blockReturnPattern =
    /\.map\(\s*(\(?\s*([A-Za-z_$][\w$]*)\s*(?::[^)=]+)?\s*\)?)\s*=>\s*\{\s*return\s+html`([\s\S]*?)`\s*;?\s*\}\s*\)/g;
  const destructuredDirectPattern =
    /\.map\(\s*(\(\s*\{[^}]*\}\s*(?::[^)]+)?\))\s*=>\s*html`([\s\S]*?)`\s*\)/g;
  const destructuredBlockPattern =
    /\.map\(\s*(\(\s*\{[^}]*\}\s*(?::[^)]+)?\))\s*=>\s*\{\s*return\s+html`([\s\S]*?)`\s*;?\s*\}\s*\)/g;

  const buildReplacement = (
    _whole: string,
    params: string,
    valueName: string,
    templateBody: string,
  ) => {
    const helperName = `__AETHER_MAP_COMPONENT_${helperIndex++}`;
    const normalizedParams = params.trim();
    helpers.push(
      `const ${helperName} = component(${normalizedParams} => html\`${templateBody}\`)`,
    );
    return `.map(${normalizedParams} => ${helperName}(${valueName}))`;
  };

  const buildDestructuredReplacement = (
    _whole: string,
    params: string,
    templateBody: string,
  ) => {
    const helperName = `__AETHER_MAP_COMPONENT_${helperIndex++}`;
    const normalizedParams = params.trim();
    helpers.push(
      `const ${helperName} = component(${normalizedParams} => html\`${templateBody}\`)`,
    );
    return `.map(__item => ${helperName}(__item))`;
  };

  let rewritten = source.replace(directReturnPattern, buildReplacement);
  rewritten = rewritten.replace(blockReturnPattern, buildReplacement);
  rewritten = rewritten.replace(destructuredDirectPattern, buildDestructuredReplacement);
  rewritten = rewritten.replace(destructuredBlockPattern, buildDestructuredReplacement);

  if (helpers.length === 0) return source;
  rewritten = ensureArrowNamedImports(rewritten, ["component"]);
  return insertHelpersAfterImports(rewritten, helpers);
}

function wrapDefaultComponentModule(source: string): string {
  const normalized = normalizeModuleSource(source);
  if (!normalized) return normalized;

  const defaultComponentPrefix = "export default component(";
  const idx = normalized.indexOf(defaultComponentPrefix);
  if (idx < 0) return normalized;

  const rewritten =
    normalized.slice(0, idx) +
    "const __AETHER_DEFAULT_COMPONENT = component(" +
    normalized.slice(idx + defaultComponentPrefix.length);

  const result =
    rewritten +
    "\n\nexport default html`${__AETHER_DEFAULT_COMPONENT()}`";

  return ensureArrowNamedImports(result, ["html"]);
}

function normalizeArrowModuleSource(source: string): string {
  let normalized = normalizeModuleSource(source);
  if (!normalized) return normalized;
  normalized = rewriteInlineMappedHtmlTemplates(normalized);
  normalized = wrapDefaultComponentModule(normalized);
  return normalized;
}

function normalizeSandboxSourceFile(source: string): string {
  const normalized = normalizeSandboxFileSource(source);
  if (!looksLikeArrowModule(normalized)) return normalized;
  return normalizeArrowModuleSource(normalized);
}

function looksLikeArrowModule(source: string): boolean {
  const normalized = source.trim();
  if (!normalized) return false;

  // Reject markdown-fenced content — let it fall through to fencedBlocks() extraction
  if (normalized.startsWith("```")) return false;

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
          source[key] = normalizeSandboxSourceFile(value);
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
        "main.ts": normalizeArrowModuleSource(normalized),
      },
    };
  }

  for (const block of fencedBlocks(normalized)) {
    if (looksLikeArrowModule(block)) {
      return {
        source: {
          "main.ts": normalizeArrowModuleSource(block),
        },
      };
    }
  }

  return null;
}
