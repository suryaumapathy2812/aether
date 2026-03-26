"use client";

import { cn } from "@/lib/utils";
import {
  IconMail,
  IconMailOpened,
  IconSend,
  IconPaperclip,
  IconArrowLeft,
  IconArrowRight,
  IconClock,
  IconUser,
  IconInbox,
  IconTag,
  IconTrash,
  IconArchive,
  IconCheck,
  IconAlertCircle,
} from "@tabler/icons-react";
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

function extractHeaders(headers: unknown): Record<string, string> {
  const map: Record<string, string> = {};
  if (!Array.isArray(headers)) return map;
  for (const h of headers) {
    if (h && typeof h === "object" && "name" in h && "value" in h) {
      map[String((h as Record<string, unknown>).name).toLowerCase()] = String(
        (h as Record<string, unknown>).value,
      );
    }
  }
  return map;
}

function extractBody(payload: unknown): { text: string; html: string } {
  if (!payload || typeof payload !== "object") return { text: "", html: "" };
  const p = payload as Record<string, unknown>;
  const bodyObj =
    typeof p.body === "object" && p.body !== null
      ? (p.body as Record<string, unknown>)
      : null;

  // Simple message with body.data
  if (bodyObj && typeof bodyObj.data === "string") {
    const decoded = decodeBase64Url(bodyObj.data);
    if (decoded) return { text: decoded, html: "" };
  }

  // Multipart — prefer text/plain, fallback to text/html
  if (Array.isArray(p.parts)) {
    const parts = p.parts as Record<string, unknown>[];
    let text = "";
    let html = "";

    for (const part of parts) {
      const mime = String(part.mimeType || "");
      const partBody =
        typeof part.body === "object" && part.body !== null
          ? (part.body as Record<string, unknown>)
          : null;
      if (!partBody || typeof partBody.data !== "string") continue;

      if (mime === "text/plain" && !text) {
        text = decodeBase64Url(partBody.data);
      } else if (mime === "text/html" && !html) {
        html = decodeBase64Url(partBody.data);
      }

      // Recurse into nested multipart
      if (mime.startsWith("multipart/") && Array.isArray(part.parts)) {
        const nested = extractBody(part);
        if (nested.text && !text) text = nested.text;
        if (nested.html && !html) html = nested.html;
      }
    }

    return { text, html };
  }

  return { text: "", html: "" };
}

function parseEmailAddress(raw: string): { name: string; email: string } {
  const match = raw.match(/^(.+?)\s*<(.+?)>$/);
  if (match) {
    return {
      name: match[1].replace(/^["']|["']$/g, "").trim(),
      email: match[2].trim(),
    };
  }
  return { name: "", email: raw.trim() };
}

function formatEmailAddressList(raw: string): Array<{ name: string; email: string }> {
  if (!raw) return [];
  return raw.split(",").map((s) => parseEmailAddress(s.trim()));
}

function getInitial(nameOrEmail: string): string {
  const trimmed = nameOrEmail.trim();
  if (!trimmed) return "?";
  // If it looks like an email, use the first char before @
  const atIdx = trimmed.indexOf("@");
  if (atIdx > 0) return trimmed[0].toUpperCase();
  return trimmed[0].toUpperCase();
}

function formatDate(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max) + "...";
}

function getToolName(type: string): string {
  return type.replace("tool-", "");
}

function parseOutput(output: unknown): Record<string, unknown> | null {
  if (typeof output !== "string") return null;
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

// ── Avatar ───────────────────────────────────────────────────────────────

function EmailAvatar({
  name,
  email,
  size = "md",
}: {
  name: string;
  email: string;
  size?: "sm" | "md";
}) {
  const initial = getInitial(name || email);
  const dim = size === "sm" ? "size-7 text-xs" : "size-9 text-sm";
  // Deterministic color from email/name
  const hash = (name || email).split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const colors = [
    "bg-rose-500/15 text-rose-400",
    "bg-sky-500/15 text-sky-400",
    "bg-emerald-500/15 text-emerald-400",
    "bg-violet-500/15 text-violet-400",
    "bg-amber-500/15 text-amber-400",
    "bg-teal-500/15 text-teal-400",
    "bg-pink-500/15 text-pink-400",
    "bg-indigo-500/15 text-indigo-400",
  ];
  const colorClass = colors[hash % colors.length];

  return (
    <div
      className={cn(
        "shrink-0 rounded-full flex items-center justify-center font-semibold",
        dim,
        colorClass,
      )}
    >
      {initial}
    </div>
  );
}

// ── Email Preview Card (read_gmail) ──────────────────────────────────────

function EmailPreviewCard({ data }: { data: Record<string, unknown> }) {
  const headers = extractHeaders(data.payload);
  const fromRaw = headers["from"] || "Unknown sender";
  const toRaw = headers["to"] || "";
  const ccRaw = headers["cc"] || "";
  const bccRaw = headers["bcc"] || "";
  const subject = headers["subject"] || "(no subject)";
  const date = headers["date"] || "";
  const messageId = headers["message-id"] || "";
  const snippet = typeof data.snippet === "string" ? data.snippet : "";
  const size = typeof data.sizeEstimate === "number" ? data.sizeEstimate : 0;
  const labels = Array.isArray(data.labelIds)
    ? (data.labelIds as string[])
    : [];
  const threadId = typeof data.threadId === "string" ? data.threadId : "";
  const body = extractBody(data.payload);

  const fromParsed = parseEmailAddress(fromRaw);
  const toList = formatEmailAddressList(toRaw);
  const ccList = formatEmailAddressList(ccRaw);

  // Determine label display
  const systemLabels = labels.filter(
    (l) => !l.startsWith("Label_") && l !== "UNREAD" && l !== "STARRED",
  );
  const userLabels = labels.filter((l) => l.startsWith("Label_"));
  const isUnread = labels.includes("UNREAD");
  const isStarred = labels.includes("STARRED");

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Subject header */}
      <div className="px-4 py-3 border-b border-border/60">
        <div className="flex items-start gap-2">
          {isUnread ? (
            <IconMail className="size-4 text-foreground shrink-0 mt-0.5" />
          ) : (
            <IconMailOpened className="size-4 text-muted-foreground shrink-0 mt-0.5" />
          )}
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-foreground leading-snug">
              {subject}
            </h3>
            {userLabels.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1.5">
                {userLabels.map((label) => (
                  <span
                    key={label}
                    className="inline-flex items-center gap-1 rounded-full bg-accent/60 px-2 py-0.5 text-[10px] text-muted-foreground"
                  >
                    <IconTag className="size-2.5" />
                    {label.replace("Label_", "")}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sender row */}
      <div className="px-4 py-3 border-b border-border/40">
        <div className="flex items-start gap-3">
          <EmailAvatar name={fromParsed.name} email={fromParsed.email} />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-2 flex-wrap">
              <span className="text-sm font-medium text-foreground">
                {fromParsed.name || fromParsed.email}
              </span>
              {fromParsed.name && (
                <span className="text-xs text-muted-foreground">
                  {fromParsed.email}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5 mt-0.5 text-[11px] text-muted-foreground">
              <span>to {toList.length > 0 ? (toList[0].name || toList[0].email) : "me"}</span>
              {toList.length > 1 && (
                <span className="text-muted-foreground/50">
                  +{toList.length - 1}
                </span>
              )}
              {ccList.length > 0 && (
                <span className="text-muted-foreground/50">
                  cc {ccList.length}
                </span>
              )}
            </div>
          </div>
          <div className="shrink-0 flex items-center gap-2">
            {date && (
              <span className="text-[11px] text-muted-foreground whitespace-nowrap">
                {formatDate(date)}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="px-4 py-3">
        {body.text ? (
          <div className="text-sm text-foreground/90 whitespace-pre-wrap break-words leading-relaxed">
            {body.text}
          </div>
        ) : body.html ? (
          <div
            className="text-sm text-foreground/90 leading-relaxed prose prose-sm max-w-none dark:prose-invert"
            dangerouslySetInnerHTML={{ __html: body.html }}
          />
        ) : snippet ? (
          <div className="text-sm text-muted-foreground italic">{snippet}</div>
        ) : (
          <div className="text-sm text-muted-foreground/60">
            No message content.
          </div>
        )}
      </div>

      {/* Footer — labels, size, thread */}
      {(systemLabels.length > 0 || size > 0 || threadId) && (
        <div className="px-4 py-2 border-t border-border/40 flex items-center gap-2 flex-wrap">
          {systemLabels.map((label) => (
            <span
              key={label}
              className="rounded-full bg-accent/40 px-2 py-0.5 text-[10px] text-muted-foreground"
            >
              {label}
            </span>
          ))}
          <span className="ml-auto text-[10px] text-muted-foreground/50 flex items-center gap-2">
            {size > 0 && <span>{formatBytes(size)}</span>}
            {threadId && <span>Thread: {truncate(threadId, 12)}</span>}
          </span>
        </div>
      )}
    </div>
  );
}

// ── Email List Card (list_unread, search_email, inbox_count) ─────────────

function EmailListCard({
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
      <div className="flex items-center justify-between text-xs text-muted-foreground px-1">
        <span className="font-medium">{title}</span>
        <span>
          {messages.length} result{messages.length !== 1 ? "s" : ""}
          {estimate > messages.length ? ` (~${estimate} total)` : ""}
        </span>
      </div>
      {messages.length === 0 ? (
        <div className="rounded-xl border border-border/60 bg-accent/10 px-4 py-6 text-center">
          <IconInbox className="size-5 text-muted-foreground/40 mx-auto mb-2" />
          <p className="text-sm text-muted-foreground/60">No messages found.</p>
        </div>
      ) : (
        <div className="rounded-xl border border-border/60 overflow-hidden divide-y divide-border/40">
          {messages.map((msg, i) => (
            <EmailListRow key={`${String(msg.id)}-${i}`} message={msg} />
          ))}
        </div>
      )}
      {hasToken && (
        <div className="text-[10px] text-muted-foreground/50 text-center">
          More results available
        </div>
      )}
    </div>
  );
}

function EmailListRow({ message }: { message: Record<string, unknown> }) {
  const headers = extractHeaders(message.payload);
  const fromRaw = headers["from"] || "Unknown";
  const subject = headers["subject"] || "(no subject)";
  const snippet = typeof message.snippet === "string" ? message.snippet : "";
  const labels = Array.isArray(message.labelIds)
    ? (message.labelIds as string[])
    : [];
  const isUnread = labels.includes("UNREAD");

  const fromParsed = parseEmailAddress(fromRaw);

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-3 py-2.5 hover:bg-accent/20 transition-colors",
        isUnread && "bg-accent/10",
      )}
    >
      <EmailAvatar
        name={fromParsed.name}
        email={fromParsed.email}
        size="sm"
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span
            className={cn(
              "text-xs truncate",
              isUnread
                ? "font-semibold text-foreground"
                : "font-medium text-foreground/80",
            )}
          >
            {fromParsed.name || fromParsed.email}
          </span>
          {isUnread && (
            <span className="size-1.5 rounded-full bg-sky-400 shrink-0" />
          )}
        </div>
        <p
          className={cn(
            "text-xs mt-0.5 truncate",
            isUnread
              ? "font-medium text-foreground/90"
              : "text-foreground/70",
          )}
        >
          {subject}
        </p>
        {snippet && (
          <p className="text-[11px] text-muted-foreground mt-0.5 truncate">
            {snippet}
          </p>
        )}
      </div>
      {labels.filter((l) => !["UNREAD", "STARRED"].includes(l)).length > 0 && (
        <div className="shrink-0 flex gap-1">
          {labels
            .filter((l) => !["UNREAD", "STARRED"].includes(l))
            .slice(0, 2)
            .map((l) => (
              <span
                key={l}
                className="rounded-full bg-accent/60 px-1.5 py-0.5 text-[9px] text-muted-foreground"
              >
                {l.replace("Label_", "")}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}

// ── Inbox Count Card ─────────────────────────────────────────────────────

function InboxCountCard({ data }: { data: Record<string, unknown> }) {
  const messagesTotal = Number(data.messagesTotal) || 0;
  const messagesUnread = Number(data.messagesUnread) || 0;
  const threadsTotal = Number(data.threadsTotal) || 0;
  const threadsUnread = Number(data.threadsUnread) || 0;

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-3">
      <div className="flex items-center gap-2">
        <IconInbox className="size-4 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Inbox Overview
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <StatBlock label="Messages" value={messagesTotal} />
        <StatBlock
          label="Unread"
          value={messagesUnread}
          highlight={messagesUnread > 0}
        />
        <StatBlock label="Threads" value={threadsTotal} />
        <StatBlock
          label="Unread threads"
          value={threadsUnread}
          highlight={threadsUnread > 0}
        />
      </div>
    </div>
  );
}

function StatBlock({
  label,
  value,
  highlight,
}: {
  label: string;
  value: number;
  highlight?: boolean;
}) {
  return (
    <div className="rounded-lg bg-accent/20 px-3 py-2.5">
      <div
        className={cn(
          "text-xl font-semibold tabular-nums",
          highlight ? "text-foreground" : "text-foreground/70",
        )}
      >
        {value.toLocaleString()}
      </div>
      <div className="text-[11px] text-muted-foreground mt-0.5">{label}</div>
    </div>
  );
}

// ── Sent Confirmation Card ───────────────────────────────────────────────

function SentConfirmationCard({
  data,
  message,
  toolName,
}: {
  data: Record<string, unknown>;
  message: string;
  toolName: string;
}) {
  const headers = extractHeaders(data.payload);
  const toRaw = headers["to"] || "";
  const subject = headers["subject"] || "";
  const toList = formatEmailAddressList(toRaw);

  const isDraft = toolName === "create_draft";
  const actionLabel = isDraft ? "Draft saved" : "Message sent";
  const ActionIcon = isDraft ? IconClock : IconCheck;

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-3">
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "shrink-0 size-8 rounded-full flex items-center justify-center",
            isDraft
              ? "bg-amber-500/10 text-amber-400"
              : "bg-emerald-500/10 text-emerald-400",
          )}
        >
          <ActionIcon className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-foreground">
            {message || actionLabel}
          </p>
          {toList.length > 0 && (
            <div className="flex items-center gap-1.5 mt-1">
              <IconArrowRight className="size-3 text-muted-foreground/50" />
              <span className="text-xs text-muted-foreground">
                {toList.map((t) => t.name || t.email).join(", ")}
              </span>
            </div>
          )}
          {subject && (
            <p className="text-xs text-muted-foreground/70 mt-0.5">
              Re: {subject}
            </p>
          )}
        </div>
      </div>
      {typeof data.id === "string" && (
        <div className="text-[10px] text-muted-foreground/40 font-mono pl-11">
          ID: {truncate(data.id, 20)}
        </div>
      )}
    </div>
  );
}

// ── Label Action Confirmation ────────────────────────────────────────────

function LabelActionCard({
  data,
  toolName,
}: {
  data: Record<string, unknown>;
  toolName: string;
}) {
  const labels = Array.isArray(data.labelIds)
    ? (data.labelIds as string[])
    : [];

  const actionConfig: Record<string, { icon: typeof IconTag; label: string; color: string }> = {
    archive_email: { icon: IconArchive, label: "Archived", color: "text-sky-400 bg-sky-500/10" },
    trash_email: { icon: IconTrash, label: "Moved to trash", color: "text-red-400 bg-red-500/10" },
    mark_read: { icon: IconMailOpened, label: "Marked as read", color: "text-emerald-400 bg-emerald-500/10" },
    mark_unread: { icon: IconMail, label: "Marked as unread", color: "text-sky-400 bg-sky-500/10" },
    add_label: { icon: IconTag, label: "Label added", color: "text-violet-400 bg-violet-500/10" },
    remove_label: { icon: IconTag, label: "Label removed", color: "text-muted-foreground bg-accent/40" },
  };

  const config = actionConfig[toolName] || {
    icon: IconCheck,
    label: "Done",
    color: "text-emerald-400 bg-emerald-500/10",
  };
  const Icon = config.icon;

  return (
    <div className="rounded-xl border border-border bg-card p-3 flex items-center gap-3">
      <div
        className={cn(
          "shrink-0 size-7 rounded-full flex items-center justify-center",
          config.color,
        )}
      >
        <Icon className="size-3.5" />
      </div>
      <div className="min-w-0 flex-1">
        <span className="text-sm text-foreground">{config.label}</span>
        {labels.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {labels.map((label) => (
              <span
                key={label}
                className="rounded-full bg-accent/40 px-1.5 py-0.5 text-[10px] text-muted-foreground"
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Running State ────────────────────────────────────────────────────────

function GmailRunningState({
  toolName,
  state,
}: {
  toolName: string;
  state: string;
}) {
  const friendlyNames: Record<string, string> = {
    inbox_count: "Checking inbox",
    list_unread: "Fetching unread messages",
    search_email: "Searching emails",
    read_gmail: "Opening email",
    get_thread: "Loading thread",
    send_email: "Sending email",
    send_reply: "Sending reply",
    create_draft: "Saving draft",
    archive_email: "Archiving",
    trash_email: "Moving to trash",
    mark_read: "Marking as read",
    mark_unread: "Marking as unread",
    add_label: "Adding label",
    remove_label: "Removing label",
    list_labels: "Loading labels",
  };

  return (
    <div className="w-full rounded-xl border border-border bg-accent/10 px-4 py-3 flex items-center gap-3">
      {getStatusBadge(state as "input-available")}
      <span className="text-xs text-muted-foreground">
        {friendlyNames[toolName] || toolName.replace(/_/g, " ")}
      </span>
    </div>
  );
}

// ── Error State ──────────────────────────────────────────────────────────

function GmailErrorState({ errorText }: { errorText?: string }) {
  return (
    <div className="w-full rounded-xl border border-red-500/20 bg-red-500/[0.05] px-4 py-3 flex items-start gap-2.5">
      <IconAlertCircle className="size-4 text-red-400/70 shrink-0 mt-0.5" />
      <div className="text-xs text-red-300/80">
        {errorText || "Gmail operation failed"}
      </div>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function GmailTool({ part }: ToolRendererProps) {
  const toolName = getToolName(part.type);
  const isRunning =
    part.state === "input-streaming" || part.state === "input-available";

  if (isRunning) {
    return <GmailRunningState toolName={toolName} state={part.state} />;
  }

  if (part.state === "output-error") {
    return <GmailErrorState errorText={part.errorText} />;
  }

  if (part.state !== "output-available") return null;

  const data = parseOutput(part.output);
  const successMsg = extractSuccessMessage(part.output);

  if (!data) {
    return (
      <div className="w-full rounded-xl border border-border bg-card px-4 py-3 flex items-center gap-2.5">
        <IconCheck className="size-4 text-emerald-400/70 shrink-0" />
        <span className="text-sm text-foreground">
          {successMsg || "Done"}
        </span>
      </div>
    );
  }

  switch (toolName) {
    case "inbox_count":
      return <InboxCountCard data={data} />;

    case "list_unread":
      return <EmailListCard data={data} title="Unread messages" />;

    case "search_email":
      return <EmailListCard data={data} title="Search results" />;

    case "read_gmail":
      return <EmailPreviewCard data={data} />;

    case "send_email":
    case "send_reply":
    case "create_draft":
      return (
        <SentConfirmationCard
          data={data}
          message={successMsg}
          toolName={toolName}
        />
      );

    case "archive_email":
    case "trash_email":
    case "mark_read":
    case "mark_unread":
    case "add_label":
    case "remove_label":
      return <LabelActionCard data={data} toolName={toolName} />;

    default:
      return (
        <div className="w-full rounded-xl border border-border bg-card px-4 py-3 text-sm text-foreground">
          {successMsg || "Done"}
        </div>
      );
  }
}
