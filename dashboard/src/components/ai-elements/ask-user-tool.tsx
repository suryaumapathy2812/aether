"use client";

import type { QuestionReplyPayload } from "@/lib/api";
import type { SessionState } from "@/lib/chat-runtime";

import { QuestionDock } from "./question-dock";
import { renderAskUserOutput } from "./tool-projection";

type ToolPartRecord = {
  type: string;
  state: string;
  toolCallId: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  metadata?: unknown;
};

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
