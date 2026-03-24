"use client";

import type { QuestionReplyPayload } from "@/lib/api";
import type { SessionState } from "@/lib/chat-runtime";

import { CodeBlock } from "./code-block";
import { QuestionDock } from "./question-dock";

type ToolPartRecord = {
  type: string;
  state: string;
  toolCallId: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  metadata?: unknown;
};

type ToolProjectionProps = {
  part: ToolPartRecord;
  questionRequest: SessionState["questionRequest"];
  onQuestionSubmit: (payload: QuestionReplyPayload) => void | Promise<void>;
  onQuestionDismiss: () => void | Promise<void>;
};

type SearchResult = {
  title: string;
  url: string;
  snippet?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function formatKey(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function truncateText(value: string, max = 320) {
  const trimmed = value.replace(/\s+/g, " ").trim();
  if (trimmed.length <= max) return trimmed;
  return `${trimmed.slice(0, max).trimEnd()}...`;
}

function tryParseJSONObject(value: unknown): Record<string, unknown> | null {
  if (isRecord(value)) return value;
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function renderSearchResults(metadata: Record<string, unknown>) {
  const results: SearchResult[] = [];
  if (Array.isArray(metadata.results)) {
    for (const item of metadata.results) {
      if (!isRecord(item)) continue;
      const title = typeof item.title === "string" ? item.title : "Untitled";
      const url = typeof item.url === "string" ? item.url : "";
      if (!url) continue;
      results.push({
        title,
        url,
        snippet: typeof item.snippet === "string" ? item.snippet : undefined,
      });
    }
  }

  if (results.length === 0) return null;

  return (
    <div className="space-y-2">
      {results.map((result) => (
        <a
          key={result.url}
          href={result.url}
          target="_blank"
          rel="noreferrer"
          className="block rounded-lg border border-border bg-accent/20 px-3 py-2 transition-colors hover:bg-accent/35"
        >
          <div className="text-sm font-medium text-foreground">{result.title}</div>
          <div className="truncate text-xs text-sky-300/80">{result.url}</div>
          {result.snippet ? <p className="mt-1 text-xs text-muted-foreground">{result.snippet}</p> : null}
        </a>
      ))}
    </div>
  );
}

function renderWebFetch(part: ToolPartRecord, metadata: Record<string, unknown>) {
  if (part.state !== "output-available" || typeof part.output !== "string") return null;
  const url = typeof metadata.url === "string" ? metadata.url : "";
  const contentType = typeof metadata.content_type === "string" ? metadata.content_type : "";
  const preview = truncateText(part.output, 420);
  if (!url && !preview) return null;

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-2">
      <div className="text-xs font-medium text-muted-foreground">Fetched page</div>
      {url ? (
        <a href={url} target="_blank" rel="noreferrer" className="block truncate text-sm text-sky-300/80 hover:underline">
          {url}
        </a>
      ) : null}
      {contentType ? <div className="text-[11px] uppercase tracking-wide text-muted-foreground/80">{contentType}</div> : null}
      {preview ? <p className="text-sm text-foreground whitespace-pre-wrap">{preview}</p> : null}
    </div>
  );
}

function renderWikipedia(metadata: Record<string, unknown>, part: ToolPartRecord) {
  if (part.state !== "output-available") return null;
  const title = typeof metadata.title === "string" ? metadata.title : "Wikipedia result";
  const url = typeof metadata.url === "string" ? metadata.url : "";
  const description = typeof metadata.description === "string" ? metadata.description : "";
  const extract = typeof metadata.extract === "string" ? metadata.extract : typeof part.output === "string" ? part.output : "";
  const preview = truncateText(description || extract, 360);

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-2">
      <div className="text-xs font-medium text-muted-foreground">Wikipedia</div>
      <div className="text-sm font-medium text-foreground">{title}</div>
      {url ? (
        <a href={url} target="_blank" rel="noreferrer" className="block truncate text-xs text-sky-300/80 hover:underline">
          {url}
        </a>
      ) : null}
      {preview ? <p className="text-sm text-foreground whitespace-pre-wrap">{preview}</p> : null}
    </div>
  );
}

function renderLocalSearch(metadata: Record<string, unknown>) {
  if (!Array.isArray(metadata.places) || metadata.places.length === 0) return null;

  const places = metadata.places
    .filter(isRecord)
    .map((place) => ({
      name: typeof place.name === "string" ? place.name : "Place",
      address: typeof place.address === "string" ? place.address : "",
      rating: typeof place.rating === "number" ? place.rating : null,
      userRatingCount:
        typeof place.user_rating_count === "number" ? place.user_rating_count : null,
      mapsUrl: typeof place.maps_url === "string" ? place.maps_url : "",
      phone: typeof place.phone === "string" ? place.phone : "",
    }));

  return (
    <div className="space-y-2">
      {places.map((place, index) => (
        <div key={`${place.name}-${index}`} className="rounded-lg border border-border bg-accent/20 p-3 space-y-1.5">
          <div className="text-sm font-medium text-foreground">{place.name}</div>
          {place.address ? <div className="text-xs text-muted-foreground">{place.address}</div> : null}
          {place.rating != null ? (
            <div className="text-xs text-muted-foreground">
              Rating {place.rating.toFixed(1)}
              {place.userRatingCount != null ? ` (${place.userRatingCount} reviews)` : ""}
            </div>
          ) : null}
          {place.phone ? <div className="text-xs text-muted-foreground">{place.phone}</div> : null}
          {place.mapsUrl ? (
            <a href={place.mapsUrl} target="_blank" rel="noreferrer" className="block truncate text-xs text-sky-300/80 hover:underline">
              Open in Maps
            </a>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function renderWorldTime(metadata: Record<string, unknown>) {
  const timezone = typeof metadata.timezone === "string" ? metadata.timezone : "";
  const iso = typeof metadata.iso === "string" ? metadata.iso : "";
  if (!timezone || !iso) return null;

  const date = new Date(iso);
  const formatted = Number.isNaN(date.getTime())
    ? iso
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "full",
        timeStyle: "long",
        timeZone: timezone,
      }).format(date);

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-1.5">
      <div className="text-xs font-medium text-muted-foreground">World time</div>
      <div className="text-sm font-medium text-foreground">{timezone}</div>
      <div className="text-sm text-foreground">{formatted}</div>
    </div>
  );
}

function renderReadFile(part: ToolPartRecord, metadata: Record<string, unknown>) {
  if (part.state !== "output-available" || typeof part.output !== "string") return null;
  const path = typeof metadata.path === "string" ? metadata.path : undefined;

  return (
    <div className="space-y-2">
      {path ? <div className="text-xs text-muted-foreground">{path}</div> : null}
      <div className="overflow-hidden rounded border border-border bg-accent/20">
        <CodeBlock code={part.output} language="json" />
      </div>
    </div>
  );
}

function renderRunCommand(part: ToolPartRecord, input: Record<string, unknown> | null, metadata: Record<string, unknown>) {
  if (part.state !== "output-available" && part.state !== "output-error") return null;
  const program = typeof input?.program === "string" ? input.program : "";
  const args = asStringArray(input?.args);
  const stderr = typeof metadata.stderr === "string" ? metadata.stderr : "";
  const command = [program, ...args].filter(Boolean).join(" ").trim();
  const stdout = typeof part.output === "string" ? part.output : "";

  return (
    <div className="space-y-2">
      {command ? <div className="text-xs text-muted-foreground font-mono">$ {command}</div> : null}
      {stdout ? (
        <div className="overflow-hidden rounded border border-border bg-accent/20">
          <CodeBlock code={stdout} language="bash" />
        </div>
      ) : null}
      {stderr ? (
        <div className="overflow-hidden rounded border border-red-500/20 bg-red-500/[0.05]">
          <CodeBlock code={stderr} language="bash" />
        </div>
      ) : null}
    </div>
  );
}

export function renderAskUserOutput(part: ToolPartRecord, metadata: Record<string, unknown>) {
  if (part.state !== "output-available") return null;
  const kind = typeof metadata.kind === "string" ? metadata.kind : "choice";

  if (kind === "form") {
    const data = tryParseJSONObject(metadata.data) ?? tryParseJSONObject(part.output);
    if (!data) return null;
    const entries = Object.entries(data).filter(([, value]) => value != null && `${value}`.trim() !== "");
    if (entries.length === 0) return null;
    return (
      <div className="space-y-2 rounded-lg border border-border bg-accent/20 p-3">
        <div className="text-xs font-medium text-muted-foreground">Submitted form</div>
        <div className="space-y-1.5">
          {entries.map(([key, value]) => (
            <div key={key} className="text-sm">
              <span className="text-muted-foreground">{formatKey(key)}:</span>{" "}
              <span className="text-foreground">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const answers = Array.isArray(metadata.answers)
    ? metadata.answers.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    : typeof part.output === "string" && part.output.trim()
      ? [part.output.trim()]
      : [];

  if (answers.length === 0) return null;

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 text-sm text-foreground">
      <div className="text-xs font-medium text-muted-foreground">User response</div>
      <div className="mt-2 flex flex-wrap gap-2">
        {answers.map((answer) => (
          <span key={answer} className="rounded-full border border-border bg-background px-2.5 py-1 text-xs">
            {answer}
          </span>
        ))}
      </div>
    </div>
  );
}

export function hasInteractiveProjection(part: ToolPartRecord): boolean {
  const toolName = part.type.replace(/^tool-/, "");
  if (toolName !== "ask_user") return false;
  const metadata = asRecord(part.metadata) ?? {};
  if (part.state === "input-available") return true;
  if (part.state === "output-available") {
    const kind = typeof metadata.kind === "string" ? metadata.kind : "choice";
    if (kind === "form") return tryParseJSONObject(metadata.data) !== null || tryParseJSONObject(part.output) !== null;
    const answers = Array.isArray(metadata.answers)
      ? metadata.answers.filter((v): v is string => typeof v === "string" && v.trim().length > 0)
      : typeof part.output === "string" && part.output.trim()
        ? [part.output.trim()]
        : [];
    return answers.length > 0;
  }
  return false;
}

export function renderToolProjection({
  part,
  questionRequest,
  onQuestionSubmit,
  onQuestionDismiss,
}: ToolProjectionProps) {
  const toolName = part.type.replace(/^tool-/, "");
  const input = asRecord(part.input);
  const metadata = asRecord(part.metadata) ?? {};

  if (toolName === "ask_user") {
    const request =
      questionRequest && questionRequest.toolCallId === part.toolCallId && part.state === "input-available"
        ? questionRequest
        : null;

    if (request) {
      return <QuestionDock request={request} onSubmit={onQuestionSubmit} onDismiss={onQuestionDismiss} />;
    }

    return renderAskUserOutput(part, metadata);
  }

  if (toolName === "web_search") {
    return renderSearchResults(metadata);
  }

  if (toolName === "web_fetch") {
    return renderWebFetch(part, metadata);
  }

  if (toolName === "read_file") {
    return renderReadFile(part, metadata);
  }

  if (toolName === "run_command") {
    return renderRunCommand(part, input, metadata);
  }

  if (toolName === "wikipedia_search" || toolName === "wikipedia_get_article") {
    return renderWikipedia(metadata, part);
  }

  if (toolName === "local_search") {
    return renderLocalSearch(metadata);
  }

  if (toolName === "world_time") {
    return renderWorldTime(metadata);
  }

  return null;
}
