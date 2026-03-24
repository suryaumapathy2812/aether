"use client";

import { CodeBlock } from "./code-block";
import { getStatusBadge } from "./tool";
import type { ToolRendererProps } from "./tool-renderer";

// ── Helpers ──────────────────────────────────────────────────────────────

function decodeBase64Url(data: string): string {
  const padded = data.replace(/-/g, "+").replace(/_/g, "/");
  try {
    return atob(padded);
  } catch {
    return "";
  }
}

function tryParseJSON(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null ? parsed : null;
  } catch {
    return null;
  }
}

function extractHeaders(
  headers: unknown,
): Record<string, string> {
  const map: Record<string, string> = {};
  if (!Array.isArray(headers)) return map;
  for (const h of headers) {
    if (h && typeof h === "object" && "name" in h && "value" in h) {
      map[String((h as Record<string, unknown>).name)] = String(
        (h as Record<string, unknown>).value,
      );
    }
  }
  return map;
}

function extractBody(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const p = payload as Record<string, unknown>;

  const bodyObj = typeof p.body === "object" && p.body !== null
    ? (p.body as Record<string, unknown>)
    : null;

  // Simple message with body.data
  if (bodyObj && typeof bodyObj.data === "string") {
    const decoded = decodeBase64Url(bodyObj.data);
    if (decoded) return decoded;
  }

  // Multipart — prefer text/plain, fallback to text/html
  if (Array.isArray(p.parts)) {
    const parts = p.parts as Record<string, unknown>[];

    const textPart = parts.find((part) => String(part.mimeType) === "text/plain");
    if (textPart) {
      const partBody = typeof textPart.body === "object" && textPart.body !== null
        ? (textPart.body as Record<string, unknown>)
        : null;
      if (partBody && typeof partBody.data === "string") {
        const decoded = decodeBase64Url(partBody.data);
        if (decoded) return decoded;
      }
    }

    const htmlPart = parts.find((part) => String(part.mimeType) === "text/html");
    if (htmlPart) {
      const partBody = typeof htmlPart.body === "object" && htmlPart.body !== null
        ? (htmlPart.body as Record<string, unknown>)
        : null;
      if (partBody && typeof partBody.data === "string") {
        return decodeBase64Url(partBody.data);
      }
    }
  }

  return "";
}

function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max) + "...";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Renderers ────────────────────────────────────────────────────────────

function InboxCountCard({ data }: { data: Record<string, unknown> }) {
  const messagesTotal = Number(data.messagesTotal) || 0;
  const messagesUnread = Number(data.messagesUnread) || 0;
  const threadsTotal = Number(data.threadsTotal) || 0;
  const threadsUnread = Number(data.threadsUnread) || 0;

  return (
    <div className="grid grid-cols-2 gap-3">
      <Stat label="Total messages" value={messagesTotal} />
      <Stat label="Unread" value={messagesUnread} highlight={messagesUnread > 0} />
      <Stat label="Total threads" value={threadsTotal} />
      <Stat label="Unread threads" value={threadsUnread} />
    </div>
  );
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: number;
  highlight?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3">
      <div
        className={`text-2xl font-semibold ${highlight ? "text-foreground" : "text-foreground/80"}`}
      >
        {value}
      </div>
      <div className="text-xs text-muted-foreground mt-1">{label}</div>
    </div>
  );
}

function MessageListCard({
  data,
  title,
}: {
  data: Record<string, unknown>;
  title: string;
}) {
  const messages = Array.isArray(data.messages)
    ? (data.messages as Record<string, unknown>[])
    : [];
  const estimate = Number(data.resultSizeEstimate) || messages.length;
  const hasToken = typeof data.nextPageToken === "string";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{title}</span>
        <span>
          {messages.length} message{messages.length !== 1 ? "s" : ""}
          {estimate > messages.length ? ` (~${estimate} total)` : ""}
        </span>
      </div>
      {messages.length === 0 ? (
        <div className="text-sm text-muted-foreground/60 py-2">No messages found.</div>
      ) : (
        <div className="space-y-1 max-h-60 overflow-y-auto">
          {messages.map((msg, i) => (
            <div
              key={`${String(msg.id)}-${i}`}
              className="rounded border border-border bg-accent/10 px-3 py-1.5 text-xs font-mono text-muted-foreground truncate"
            >
              {String(msg.id)}
              {typeof msg.threadId === "string" && (
                <span className="text-muted-foreground/40 ml-2">
                  thread: {msg.threadId}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
      {hasToken && (
        <div className="text-[10px] text-muted-foreground/50">More results available</div>
      )}
    </div>
  );
}

function ReadEmailCard({ data }: { data: Record<string, unknown> }) {
  const headers = extractHeaders(data.payload);
  const from = headers["From"] || "Unknown sender";
  const to = headers["To"] || "";
  const cc = headers["Cc"] || "";
  const subject = headers["Subject"] || "(no subject)";
  const date = headers["Date"] || "";
  const snippet = typeof data.snippet === "string" ? data.snippet : "";
  const size = typeof data.sizeEstimate === "number" ? data.sizeEstimate : 0;
  const labels = Array.isArray(data.labelIds)
    ? (data.labelIds as string[])
    : [];
  const body = extractBody(data.payload);

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="space-y-1.5">
        <div className="text-sm font-semibold text-foreground">{subject}</div>
        <div className="flex items-baseline gap-2 text-xs">
          <span className="font-medium text-foreground/80">
            {from.replace(/<[^>]+>/g, "").trim()}
          </span>
          {date && (
            <span className="text-muted-foreground">
              {new Date(date).toLocaleString(undefined, {
                month: "short",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
              })}
            </span>
          )}
        </div>
        {to && (
          <div className="text-[11px] text-muted-foreground">
            To: {truncate(to.replace(/<[^>]+>/g, "").trim(), 80)}
          </div>
        )}
        {cc && (
          <div className="text-[11px] text-muted-foreground">
            Cc: {truncate(cc.replace(/<[^>]+>/g, "").trim(), 80)}
          </div>
        )}
      </div>

      {/* Labels + size */}
      <div className="flex flex-wrap items-center gap-1.5">
        {labels.map((label) => (
          <span
            key={label}
            className="rounded-full border border-border bg-accent/30 px-2 py-0.5 text-[10px] text-muted-foreground"
          >
            {label}
          </span>
        ))}
        {size > 0 && (
          <span className="text-[10px] text-muted-foreground/50 ml-auto">
            {formatBytes(size)}
          </span>
        )}
      </div>

      {/* Body */}
      {body ? (
        <div className="rounded-lg border border-border bg-accent/10 p-3 text-sm text-foreground/90 whitespace-pre-wrap break-words max-h-80 overflow-y-auto">
          {body}
        </div>
      ) : snippet ? (
        <div className="text-sm text-muted-foreground italic">{snippet}</div>
      ) : (
        <div className="text-sm text-muted-foreground/60">No body content.</div>
      )}
    </div>
  );
}

function SentConfirmationCard({
  data,
  message,
}: {
  data: Record<string, unknown>;
  message: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-1.5">
      <div className="text-sm text-foreground">{message}</div>
      {typeof data.id === "string" && (
        <div className="text-[11px] text-muted-foreground font-mono">
          ID: {data.id}
        </div>
      )}
      {typeof data.threadId === "string" && (
        <div className="text-[11px] text-muted-foreground font-mono">
          Thread: {data.threadId}
        </div>
      )}
    </div>
  );
}

function LabelActionCard({
  data,
  message,
}: {
  data: Record<string, unknown>;
  message: string;
}) {
  const labels = Array.isArray(data.labelIds)
    ? (data.labelIds as string[])
    : [];

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-1.5">
      <div className="text-sm text-foreground">{message}</div>
      {labels.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {labels.map((label) => (
            <span
              key={label}
              className="rounded-full border border-border bg-background px-2 py-0.5 text-[10px] text-muted-foreground"
            >
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

function getToolName(type: string): string {
  return type.replace("tool-", "");
}

function parseOutput(output: unknown): Record<string, unknown> | null {
  if (typeof output !== "string") return null;
  // For send/label tools, output is "Success message\n{json}"
  const jsonStart = output.indexOf("{");
  if (jsonStart === -1) return null;
  try {
    const parsed = JSON.parse(output.slice(jsonStart));
    return typeof parsed === "object" && parsed !== null ? parsed : null;
  } catch {
    return null;
  }
}

function extractSuccessMessage(output: unknown): string {
  if (typeof output !== "string") return "";
  const jsonStart = output.indexOf("{");
  if (jsonStart <= 0) return "";
  return output.slice(0, jsonStart).trim();
}

export function GmailTool({ part }: ToolRendererProps) {
  const toolName = getToolName(part.type);
  const isRunning =
    part.state === "input-streaming" || part.state === "input-available";

  if (isRunning) {
    return (
      <div className="w-full rounded-lg border border-border bg-accent/10 px-3 py-2 flex items-center gap-2">
        {getStatusBadge(part.state as "input-available")}
        <span className="text-xs text-muted-foreground">
          {toolName.replace(/_/g, " ")}
        </span>
      </div>
    );
  }

  if (part.state === "output-error") {
    return (
      <div className="w-full rounded-lg border border-red-500/20 bg-red-500/[0.05] px-3 py-2 text-xs text-red-300/80">
        {part.errorText || "Gmail operation failed"}
      </div>
    );
  }

  if (part.state !== "output-available") return null;

  const data = parseOutput(part.output);
  const successMsg = extractSuccessMessage(part.output);

  if (!data) {
    return (
      <div className="w-full rounded-lg border border-border bg-accent/20 px-3 py-2 text-sm text-foreground">
        {successMsg || "Done"}
      </div>
    );
  }

  switch (toolName) {
    case "inbox_count":
      return (
        <div className="w-full">
          <InboxCountCard data={data} />
        </div>
      );

    case "list_unread":
      return (
        <div className="w-full">
          <MessageListCard data={data} title="Unread messages" />
        </div>
      );

    case "search_email":
      return (
        <div className="w-full">
          <MessageListCard data={data} title="Search results" />
        </div>
      );

    case "read_gmail":
      return (
        <div className="w-full">
          <ReadEmailCard data={data} />
        </div>
      );

    case "send_email":
    case "send_reply":
    case "create_draft":
      return (
        <div className="w-full">
          <SentConfirmationCard data={data} message={successMsg || "Done"} />
        </div>
      );

    case "archive_email":
    case "trash_email":
    case "mark_read":
    case "mark_unread":
    case "add_label":
    case "remove_label":
      return (
        <div className="w-full">
          <LabelActionCard data={data} message={successMsg || "Done"} />
        </div>
      );

    default:
      return (
        <div className="w-full overflow-hidden rounded border border-border bg-accent/20">
          <CodeBlock
            code={JSON.stringify(data, null, 2)}
            language="json"
          />
        </div>
      );
  }
}
