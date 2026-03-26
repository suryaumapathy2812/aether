import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "#/components/ui/collapsible";
import { cn } from "#/lib/utils";
import type { DynamicToolUIPart, ToolUIPart } from "ai";
import {
  IconCircleCheck,
  IconChevronDown,
  IconCircle,
  IconLoader2,
  IconCircleX,
} from "@tabler/icons-react";
import type { ComponentProps, ReactNode } from "react";
import { isValidElement } from "react";

import { CodeBlock } from "./code-block";

export type ToolProps = ComponentProps<typeof Collapsible>;

export const Tool = ({ className, ...props }: ToolProps) => (
  <Collapsible
    className={cn("group not-prose w-full", className)}
    {...props}
  />
);

export type ToolPart = ToolUIPart | DynamicToolUIPart;

export type ToolHeaderProps = {
  title?: string;
  className?: string;
} & (
  | { type: ToolUIPart["type"]; state: ToolUIPart["state"]; toolName?: never }
  | {
      type: DynamicToolUIPart["type"];
      state: DynamicToolUIPart["state"];
      toolName: string;
    }
);

const statusIcons: Record<ToolPart["state"], ReactNode> = {
  "approval-requested": <IconLoader2 className="size-3 text-muted-foreground animate-spin" />,
  "approval-responded": <IconCircleCheck className="size-3 text-muted-foreground" />,
  "input-available": <IconLoader2 className="size-3 text-muted-foreground animate-spin" />,
  "input-streaming": <IconCircle className="size-3 text-muted-foreground/50" />,
  "output-available": <IconCircleCheck className="size-3 text-emerald-500/70" />,
  "output-denied": <IconCircleX className="size-3 text-amber-500/70" />,
  "output-error": <IconCircleX className="size-3 text-red-400/70" />,
};

const statusLabels: Record<ToolPart["state"], string> = {
  "approval-requested": "Awaiting approval",
  "approval-responded": "Responded",
  "input-available": "Running",
  "input-streaming": "Pending",
  "output-available": "Done",
  "output-denied": "Denied",
  "output-error": "Error",
};

export const getStatusBadge = (status: ToolPart["state"]) => (
  <span className="inline-flex items-center gap-1 text-sm text-muted-foreground">
    {statusIcons[status]}
    {statusLabels[status]}
  </span>
);

export const ToolHeader = ({
  className,
  title,
  type,
  state,
  toolName,
}: ToolHeaderProps) => {
  const derivedName =
    type === "dynamic-tool" ? toolName : type.split("-").slice(1).join("-");

  return (
    <CollapsibleTrigger
      className={cn(
        "flex w-full items-center gap-2 py-1.5 text-sm text-muted-foreground hover:text-foreground/80 transition-colors",
        className
      )}
    >
      <span className="font-mono text-foreground/70">{title ?? derivedName}</span>
      {getStatusBadge(state)}
      <IconChevronDown className="ml-auto size-3 text-muted-foreground/50 transition-transform group-data-[state=open]:rotate-180" />
    </CollapsibleTrigger>
  );
};

export type ToolContentProps = ComponentProps<typeof CollapsibleContent>;

export const ToolContent = ({ className, ...props }: ToolContentProps) => (
  <CollapsibleContent
    className={cn(
      "space-y-2 py-2 text-xs",
      className
    )}
    {...props}
  />
);

export type ToolInputProps = ComponentProps<"div"> & {
  input: ToolPart["input"];
};

export const ToolInput = ({ className, input, ...props }: ToolInputProps) => {
  if (!input || (typeof input === "object" && Object.keys(input as object).length === 0)) {
    return null;
  }
  return (
    <div className={cn("overflow-hidden", className)} {...props}>
      <div className="rounded bg-accent/30 border border-border">
        <CodeBlock code={JSON.stringify(input, null, 2)} language="json" />
      </div>
    </div>
  );
};

export type ToolOutputProps = ComponentProps<"div"> & {
  output: ToolPart["output"];
  errorText: ToolPart["errorText"];
};

export const ToolOutput = ({
  className,
  output,
  errorText,
  ...props
}: ToolOutputProps) => {
  if (!(output || errorText)) {
    return null;
  }

  let Output = <div>{output as ReactNode}</div>;

  if (typeof output === "object" && !isValidElement(output)) {
    Output = (
      <CodeBlock code={JSON.stringify(output, null, 2)} language="json" />
    );
  } else if (typeof output === "string") {
    const truncated = output.length > 500 ? output.slice(0, 500) + "..." : output;
    Output = <CodeBlock code={truncated} language="json" />;
  }

  return (
    <div className={cn("overflow-hidden", className)} {...props}>
      <div
        className={cn(
          "rounded border text-xs",
          errorText
            ? "bg-red-500/[0.05] border-red-500/10 text-red-300/80"
            : "bg-accent/30 border-border"
        )}
      >
        {errorText && <div className="px-3 py-2 text-sm">{errorText}</div>}
        {!errorText && Output}
      </div>
    </div>
  );
};
