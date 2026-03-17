"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import type { SessionState } from "@/lib/chat-runtime";

type QuestionRequest = NonNullable<SessionState["questionRequest"]>;

export function QuestionDock({
  request,
  onSubmit,
  onDismiss,
}: {
  request: QuestionRequest;
  onSubmit: (answers: string[]) => void;
  onDismiss: () => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [customText, setCustomText] = useState("");
  const [showCustom, setShowCustom] = useState(false);

  function toggleOption(label: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  }

  function handleSubmit() {
    const answers = [...selected];
    if (showCustom && customText.trim()) {
      answers.push(customText.trim());
    }
    onSubmit(answers.length > 0 ? answers : ["(no selection)"]);
  }

  const canSubmit = selected.size > 0 || (showCustom && customText.trim().length > 0);

  return (
    <div className="border border-border rounded-xl bg-background p-4 space-y-3">
      {/* Header */}
      <div className="text-xs text-muted-foreground font-medium">
        {request.header}
      </div>

      {/* Question text */}
      <p className="text-sm">{request.question}</p>

      {/* Options */}
      {request.options.length > 0 && (
        <div className="space-y-1.5">
          {request.options.map((opt, i) => (
            <button
              key={i}
              onClick={() => toggleOption(opt.label)}
              className={cn(
                "w-full text-left px-3 py-2 rounded-lg text-sm border transition-colors",
                selected.has(opt.label)
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border hover:bg-accent"
              )}
            >
              <div className="font-medium">{opt.label}</div>
              {opt.description && (
                <div className="text-xs text-muted-foreground">
                  {opt.description}
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Custom answer */}
      {request.allowCustom &&
        (showCustom ? (
          <textarea
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            placeholder="Type your answer..."
            className="w-full px-3 py-2 text-sm border border-border rounded-lg bg-background resize-none min-h-[60px] focus:outline-none focus:ring-1 focus:ring-ring"
            autoFocus
          />
        ) : (
          <button
            onClick={() => setShowCustom(true)}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Type a custom answer
          </button>
        ))}

      {/* Footer buttons */}
      <div className="flex items-center justify-between pt-1">
        <button
          onClick={onDismiss}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          Dismiss
        </button>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="px-4 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
        >
          Submit
        </button>
      </div>
    </div>
  );
}
