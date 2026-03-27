import type { QuestionReplyPayload } from "#/lib/api";
import type { SessionState } from "#/lib/chat-runtime";
import type { ComponentType } from "react";

import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
} from "./tool";
import { renderToolProjection, hasInteractiveProjection } from "./tool-projection";
import { AskUserTool } from "./ask-user-tool";
import {
  ToolSandbox,
  type ToolSandboxSource,
} from "./tool-sandbox";

export type ToolPartRecord = {
  type: string;
  state: string;
  toolCallId: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  metadata?: unknown;
};

export type ToolRendererProps = {
  part: ToolPartRecord;
  questionRequest: SessionState["questionRequest"];
  onQuestionSubmit: (payload: QuestionReplyPayload) => void | Promise<void>;
  onQuestionDismiss: () => void | Promise<void>;
};

// ── Helpers ──────────────────────────────────────────────────────────────

function getSubtitle(input: unknown): string {
  if (!input || typeof input === "object") return "";
  const obj = input as Record<string, unknown>;
  return (
    (obj.query || obj.message_id || obj.name || obj.summary || "") as string
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isQuestionRequestForPart(
  questionRequest: SessionState["questionRequest"],
  part: ToolPartRecord,
) {
  return questionRequest && questionRequest.toolCallId === part.toolCallId;
}

function normalizeQuestionReplyPayload(payload: unknown): QuestionReplyPayload | null {
  if (typeof payload === "string") {
    return { answers: [payload] };
  }

  if (Array.isArray(payload) && payload.every((value) => typeof value === "string")) {
    return { answers: payload as string[] };
  }

  if (!isRecord(payload)) return null;

  if (Array.isArray(payload.answers)) {
    return {
      answers: payload.answers.filter((value): value is string => typeof value === "string"),
      data: isRecord(payload.data) ? payload.data : undefined,
    };
  }

  if (typeof payload.answer === "string") {
    return {
      answers: [payload.answer],
      data: isRecord(payload.data) ? payload.data : undefined,
    };
  }

  if (isRecord(payload.data)) {
    return { data: payload.data };
  }

  return null;
}

function handleSandboxHostAction(
  payload: unknown,
  part: ToolPartRecord,
  questionRequest: SessionState["questionRequest"],
  onQuestionSubmit: (payload: QuestionReplyPayload) => void | Promise<void>,
  onQuestionDismiss: () => void | Promise<void>,
) {
  if (!isRecord(payload)) {
    if (isQuestionRequestForPart(questionRequest, part)) {
      const reply = normalizeQuestionReplyPayload(payload);
      if (reply) return onQuestionSubmit(reply);
    }
    return;
  }

  const action = typeof payload.type === "string"
    ? payload.type
    : typeof payload.action === "string"
      ? payload.action
      : "";

  if (action === "dismiss" || action === "question.dismiss" || action === "question.reject") {
    if (isQuestionRequestForPart(questionRequest, part)) {
      return onQuestionDismiss();
    }
    return;
  }

  if (
    action === "submit" ||
    action === "question.submit" ||
    action === "question.reply" ||
    action === "reply"
  ) {
    if (isQuestionRequestForPart(questionRequest, part)) {
      const reply = normalizeQuestionReplyPayload(payload.payload ?? payload);
      if (reply) return onQuestionSubmit(reply);
    }
    return;
  }

  if (action === "open-url" && typeof payload.url === "string" && typeof window !== "undefined") {
    window.open(payload.url, "_blank", "noopener,noreferrer");
    return;
  }

  if (action === "copy" && typeof payload.text === "string" && typeof navigator !== "undefined") {
    void navigator.clipboard?.writeText(payload.text);
    return;
  }

  if (isQuestionRequestForPart(questionRequest, part)) {
    const reply = normalizeQuestionReplyPayload(payload);
    if (reply) return onQuestionSubmit(reply);
  }
}

function extractSandboxSource(metadata: unknown): ToolSandboxSource | null {
  if (!isRecord(metadata)) return null;
  const sandbox = metadata.sandbox;
  if (!isRecord(sandbox)) return null;
  const source = sandbox.source;
  if (!isRecord(source)) return null;

  // Validate source has at least one entry file
  const hasEntry = "main.ts" in source || "main.js" in source;
  if (!hasEntry) return null;

  // Convert to Record<string, string>
  const sourceFiles: Record<string, string> = {};
  for (const [key, value] of Object.entries(source)) {
    if (typeof value === "string") {
      sourceFiles[key] = value;
    }
  }

  return {
    source: sourceFiles,
    css: typeof sandbox.css === "string" ? sandbox.css : undefined,
    shadowDOM: typeof sandbox.shadowDOM === "boolean" ? sandbox.shadowDOM : undefined,
  };
}

// ── Default renderer — used when no sandbox is available ─────────────────

export function DefaultToolRenderer({
  part,
  questionRequest,
  onQuestionSubmit,
  onQuestionDismiss,
}: ToolRendererProps) {
  const toolName = part.type.replace("tool-", "");
  const subtitle = getSubtitle(part.input);

  const isRunning =
    part.state === "input-streaming" || part.state === "input-available";
  const isInteractive = hasInteractiveProjection(part);

  return (
    <Tool defaultOpen={isRunning || isInteractive}>
      <ToolHeader
        type={part.type as "tool-invocation"}
        state={part.state as "input-available"}
        title={subtitle ? `${toolName} ${subtitle}` : toolName}
      />
      <ToolContent>
        {(() => {
          const projection = renderToolProjection({
            part,
            questionRequest,
            onQuestionSubmit,
            onQuestionDismiss,
          });

          if (projection) {
            return projection;
          }

          return (
            <>
              <ToolInput input={part.input} />
              {(part.state === "output-available" ||
                part.state === "output-error") && (
                <ToolOutput
                  output={part.output}
                  errorText={part.errorText}
                />
              )}
            </>
          );
        })()}
      </ToolContent>
    </Tool>
  );
}

// ── Sandbox renderer — renders tool output via Arrow sandbox ─────────────

function SandboxToolRenderer({
  part,
  questionRequest,
  onQuestionSubmit,
  onQuestionDismiss,
}: ToolRendererProps) {
  const sandbox = extractSandboxSource(part.metadata);

  if (!sandbox) {
    return <DefaultToolRenderer
      part={part}
      questionRequest={null}
      onQuestionSubmit={onQuestionSubmit}
      onQuestionDismiss={onQuestionDismiss}
    />;
  }

  return (
    <ToolSandbox
      sandbox={sandbox}
      state={part.state}
      errorText={part.errorText}
      onOutput={(payload) =>
        handleSandboxHostAction(
          payload,
          part,
          questionRequest,
          onQuestionSubmit,
          onQuestionDismiss,
        )
      }
    />
  );
}

// ── Special-case renderers ───────────────────────────────────────────────
// These need custom interactivity that the sandbox can't provide.

const specialRenderers: Record<string, ComponentType<ToolRendererProps>> = {
  ask_user: AskUserTool,
};

// ── Public API ───────────────────────────────────────────────────────────

export function getToolRenderer(toolName: string, metadata?: unknown): ComponentType<ToolRendererProps> {
  // 1. Special cases that need host-side interactivity
  if (specialRenderers[toolName]) return specialRenderers[toolName];

  // 2. If tool metadata includes sandbox source, use it
  if (extractSandboxSource(metadata)) return SandboxToolRenderer;

  // 3. Fall back to default (JSON input/output viewer)
  return DefaultToolRenderer;
}
