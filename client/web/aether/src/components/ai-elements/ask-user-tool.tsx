import type { QuestionReplyPayload } from "#/lib/api";
import type { SessionState } from "#/lib/chat-runtime";

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

// ── Helpers (moved from tool-projection.tsx) ─────────────────────────────

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function formatKey(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase());
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

function renderAskUserOutput(part: ToolPartRecord, metadata: Record<string, unknown>) {
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

export type AskUserToolProps = {
  part: ToolPartRecord;
  questionRequest: SessionState["questionRequest"];
  onQuestionSubmit: (payload: QuestionReplyPayload) => void | Promise<void>;
  onQuestionDismiss: () => void | Promise<void>;
};

export function AskUserTool({
  part,
  questionRequest,
  onQuestionSubmit,
  onQuestionDismiss,
}: AskUserToolProps) {
  if (part.state === "input-available") {
    const request =
      questionRequest && questionRequest.toolCallId === part.toolCallId
        ? questionRequest
        : null;

    if (request) {
      return (
        <div className="w-full">
          <QuestionDock
            request={request}
            onSubmit={onQuestionSubmit}
            onDismiss={onQuestionDismiss}
          />
        </div>
      );
    }

    return (
      <div className="w-full rounded-lg border border-border bg-accent/10 px-3 py-2 text-xs text-muted-foreground">
        Waiting for response...
      </div>
    );
  }

  const metadata =
    typeof part.metadata === "object" && part.metadata !== null
      ? (part.metadata as Record<string, unknown>)
      : {};

  const output = renderAskUserOutput(part, metadata);
  if (output) return <div className="w-full">{output}</div>;

  if (part.state === "input-streaming") {
    return (
      <div className="w-full rounded-lg border border-border bg-accent/10 px-3 py-2 text-xs text-muted-foreground">
        Preparing prompt...
      </div>
    );
  }

  return null;
}
