"use client";

import { CodeBlock } from "./code-block";
import { getStatusBadge } from "./tool";
import type { ToolRendererProps } from "./tool-renderer";

// ── Helpers ──────────────────────────────────────────────────────────────

function tryParseJSON(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null ? parsed : null;
  } catch {
    return null;
  }
}

function tryParseArray(value: unknown): unknown[] | null {
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function extractSuccessMessage(output: unknown): string {
  if (typeof output !== "string") return "";
  const jsonStart = output.indexOf("{");
  const arrStart = output.indexOf("[");
  const cut = jsonStart === -1 ? arrStart : arrStart === -1 ? jsonStart : Math.min(jsonStart, arrStart);
  if (cut <= 0) return "";
  return output.slice(0, cut).trim();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getToolName(type: string): string {
  return type.replace("tool-", "");
}

function getMimeTypeIcon(mimeType: string): string {
  if (mimeType.includes("folder")) return "📁";
  if (mimeType.includes("document")) return "📄";
  if (mimeType.includes("spreadsheet")) return "📊";
  if (mimeType.includes("presentation")) return "📽️";
  if (mimeType.includes("pdf")) return "📕";
  if (mimeType.includes("image")) return "🖼️";
  if (mimeType.includes("video")) return "🎬";
  if (mimeType.includes("audio")) return "🎵";
  return "📎";
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// ── Renderers ────────────────────────────────────────────────────────────

function FileRow({ file }: { file: Record<string, unknown> }) {
  const name = typeof file.name === "string" ? file.name : "Unknown";
  const mimeType = typeof file.mimeType === "string" ? file.mimeType : "";
  const size = typeof file.size === "string" ? parseInt(file.size, 10) : typeof file.size === "number" ? file.size : 0;
  const modified = typeof file.modifiedTime === "string" ? file.modifiedTime : "";
  const webViewLink = typeof file.webViewLink === "string" ? file.webViewLink : "";
  const owners = Array.isArray(file.owners)
    ? (file.owners as Record<string, unknown>[]).map((o) => typeof o.displayName === "string" ? o.displayName : typeof o.emailAddress === "string" ? o.emailAddress : "").filter(Boolean)
    : [];

  return (
    <div className="rounded border border-border bg-accent/10 px-3 py-2 space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-sm">{getMimeTypeIcon(mimeType)}</span>
        {webViewLink ? (
          <a href={webViewLink} target="_blank" rel="noreferrer" className="text-sm font-medium text-foreground hover:text-sky-300/80 truncate flex-1">
            {name}
          </a>
        ) : (
          <span className="text-sm font-medium text-foreground truncate flex-1">{name}</span>
        )}
      </div>
      <div className="flex items-center gap-2 text-[10px] text-muted-foreground flex-wrap">
        {mimeType && <span className="truncate max-w-[200px]">{mimeType}</span>}
        {size > 0 && <span>{formatBytes(size)}</span>}
        {modified && <span>{formatTime(modified)}</span>}
        {owners.length > 0 && <span>{owners.join(", ")}</span>}
      </div>
    </div>
  );
}

function FileListCard({ files, title }: { files: unknown[]; title: string }) {
  const validFiles = files.filter((f): f is Record<string, unknown> => typeof f === "object" && f !== null);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{title}</span>
        <span>{validFiles.length} file{validFiles.length !== 1 ? "s" : ""}</span>
      </div>
      {validFiles.length === 0 ? (
        <div className="text-sm text-muted-foreground/60 py-2">No files found.</div>
      ) : (
        <div className="space-y-1 max-h-80 overflow-y-auto">
          {validFiles.map((file, i) => (
            <FileRow key={`${String(file.id)}-${i}`} file={file} />
          ))}
        </div>
      )}
    </div>
  );
}

function FileInfoCard({ data }: { data: Record<string, unknown> }) {
  const name = typeof data.name === "string" ? data.name : "Unknown";
  const mimeType = typeof data.mimeType === "string" ? data.mimeType : "";
  const size = typeof data.size === "string" ? parseInt(data.size, 10) : typeof data.size === "number" ? data.size : 0;
  const modified = typeof data.modifiedTime === "string" ? data.modifiedTime : "";
  const description = typeof data.description === "string" ? data.description : "";
  const shared = data.shared === true;
  const trashed = data.trashed === true;
  const webViewLink = typeof data.webViewLink === "string" ? data.webViewLink : "";
  const owners = Array.isArray(data.owners)
    ? (data.owners as Record<string, unknown>[]).map((o) => typeof o.displayName === "string" ? o.displayName : typeof o.emailAddress === "string" ? o.emailAddress : "").filter(Boolean)
    : [];

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-lg">{getMimeTypeIcon(mimeType)}</span>
        <div className="text-sm font-semibold text-foreground">{name}</div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {mimeType && <span className="rounded-full border border-border bg-background px-2 py-0.5 text-[10px] text-muted-foreground">{mimeType}</span>}
        {shared && <span className="rounded-full border border-border bg-background px-2 py-0.5 text-[10px] text-sky-300/80">Shared</span>}
        {trashed && <span className="rounded-full border border-red-500/20 bg-red-500/[0.05] px-2 py-0.5 text-[10px] text-red-300/80">In Trash</span>}
        {size > 0 && <span className="text-[10px] text-muted-foreground">{formatBytes(size)}</span>}
      </div>
      {modified && <div className="text-xs text-muted-foreground">Modified {formatTime(modified)}</div>}
      {owners.length > 0 && <div className="text-xs text-muted-foreground">Owner{owners.length > 1 ? "s" : ""}: {owners.join(", ")}</div>}
      {description && <div className="text-xs text-muted-foreground/80">{description}</div>}
      {webViewLink && (
        <a href={webViewLink} target="_blank" rel="noreferrer" className="text-[10px] text-sky-300/80 hover:underline">
          Open in Drive
        </a>
      )}
    </div>
  );
}

function ExportedDocCard({ content }: { content: string }) {
  const truncated = content.length > 500 ? content.slice(0, 500) + "..." : content;

  return (
    <div className="space-y-2">
      <div className="text-xs text-muted-foreground">Exported content</div>
      <div className="rounded border border-border bg-accent/20 max-h-80 overflow-y-auto">
        <CodeBlock code={truncated} language="json" />
      </div>
    </div>
  );
}

function SharedDriveListCard({ drives }: { drives: unknown[] }) {
  const validDrives = drives.filter((d): d is Record<string, unknown> => typeof d === "object" && d !== null);

  return (
    <div className="space-y-1.5 max-h-60 overflow-y-auto">
      {validDrives.map((drive, i) => {
        const name = typeof drive.name === "string" ? drive.name : "Unknown";
        const id = typeof drive.id === "string" ? drive.id : "";

        return (
          <div key={`${id}-${i}`} className="flex items-center gap-2 rounded border border-border bg-accent/10 px-3 py-1.5">
            <span className="text-sm">📁</span>
            <span className="text-sm text-foreground flex-1 truncate">{name}</span>
            {id && <span className="text-[10px] text-muted-foreground font-mono">{id}</span>}
          </div>
        );
      })}
    </div>
  );
}

function ConfirmationCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-border bg-accent/20 px-3 py-2 text-sm text-foreground">
      {message}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function DriveTool({ part }: ToolRendererProps) {
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
        {part.errorText || "Drive operation failed"}
      </div>
    );
  }

  if (part.state !== "output-available") return null;

  switch (toolName) {
    case "search_drive": {
      const files = tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <FileListCard files={files} title="Search results" />
        </div>
      );
    }

    case "list_drive_files": {
      const files = tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <FileListCard files={files} title="Files" />
        </div>
      );
    }

    case "get_file_info": {
      const data = tryParseJSON(part.output);
      if (!data) break;
      return (
        <div className="w-full">
          <FileInfoCard data={data} />
        </div>
      );
    }

    case "export_google_doc":
    case "download_file": {
      const content = typeof part.output === "string" ? part.output : "";
      return (
        <div className="w-full">
          <ExportedDocCard content={content} />
        </div>
      );
    }

    case "create_folder": {
      const successMsg = extractSuccessMessage(part.output) || "Folder created.";
      return (
        <div className="w-full">
          <ConfirmationCard message={successMsg} />
        </div>
      );
    }

    case "list_shared_drives": {
      const data = tryParseJSON(part.output);
      const drives = data && Array.isArray(data.drives) ? data.drives : tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <SharedDriveListCard drives={drives} />
        </div>
      );
    }
  }

  // Fallback
  const data = tryParseJSON(part.output);
  if (data) {
    return (
      <div className="w-full overflow-hidden rounded border border-border bg-accent/20">
        <CodeBlock code={JSON.stringify(data, null, 2)} language="json" />
      </div>
    );
  }

  return (
    <div className="w-full rounded-lg border border-border bg-accent/20 px-3 py-2 text-sm text-foreground">
      {typeof part.output === "string" ? part.output : "Done"}
    </div>
  );
}
