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
import { GmailTool } from "./gmail-tool";
import { CalendarTool } from "./calendar-tool";
import { DriveTool } from "./drive-tool";
import { ContactsTool } from "./contacts-tool";

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

const prefixRenderers: Array<{
  prefix: string;
  renderer: ComponentType<ToolRendererProps>;
}> = [
  // Gmail
  { prefix: "inbox_count", renderer: GmailTool },
  { prefix: "list_unread", renderer: GmailTool },
  { prefix: "read_gmail", renderer: GmailTool },
  { prefix: "search_email", renderer: GmailTool },
  { prefix: "get_thread", renderer: GmailTool },
  { prefix: "send_email", renderer: GmailTool },
  { prefix: "send_reply", renderer: GmailTool },
  { prefix: "create_draft", renderer: GmailTool },
  { prefix: "archive_email", renderer: GmailTool },
  { prefix: "trash_email", renderer: GmailTool },
  { prefix: "mark_read", renderer: GmailTool },
  { prefix: "mark_unread", renderer: GmailTool },
  { prefix: "list_labels", renderer: GmailTool },
  { prefix: "add_label", renderer: GmailTool },
  { prefix: "remove_label", renderer: GmailTool },
  // Google Calendar
  { prefix: "upcoming_events", renderer: CalendarTool },
  { prefix: "search_events", renderer: CalendarTool },
  { prefix: "get_event", renderer: CalendarTool },
  { prefix: "create_event", renderer: CalendarTool },
  { prefix: "update_event", renderer: CalendarTool },
  { prefix: "delete_event", renderer: CalendarTool },
  { prefix: "list_calendars", renderer: CalendarTool },
  // Google Drive
  { prefix: "search_drive", renderer: DriveTool },
  { prefix: "list_drive_files", renderer: DriveTool },
  { prefix: "get_file_info", renderer: DriveTool },
  { prefix: "export_google_doc", renderer: DriveTool },
  { prefix: "download_file", renderer: DriveTool },
  { prefix: "create_folder", renderer: DriveTool },
  { prefix: "list_shared_drives", renderer: DriveTool },
  // Google Contacts
  { prefix: "search_contacts", renderer: ContactsTool },
  { prefix: "get_contact", renderer: ContactsTool },
];

export function getToolRenderer(toolName: string): ComponentType<ToolRendererProps> {
  if (toolRenderers[toolName]) return toolRenderers[toolName];
  for (const entry of prefixRenderers) {
    if (toolName === entry.prefix) return entry.renderer;
  }
  return DefaultToolRenderer;
}
