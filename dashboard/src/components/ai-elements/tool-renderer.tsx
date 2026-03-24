"use client";

import type { QuestionReplyPayload } from "@/lib/api";
import type { SessionState } from "@/lib/chat-runtime";
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

function getSubtitle(input: unknown): string {
  if (!input || typeof input !== "object") return "";
  const obj = input as Record<string, unknown>;
  return (
    (obj.query || obj.message_id || obj.name || obj.summary || "") as string
  );
}

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

const toolRenderers: Record<string, ComponentType<ToolRendererProps>> = {
  ask_user: AskUserTool,
};

export function getToolRenderer(toolName: string): ComponentType<ToolRendererProps> {
  return toolRenderers[toolName] ?? DefaultToolRenderer;
}
