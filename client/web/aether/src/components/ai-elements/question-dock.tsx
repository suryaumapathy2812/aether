import { useMemo, useState } from "react";

import type { QuestionReplyPayload } from "#/lib/api";
import type { SessionState } from "#/lib/chat-runtime";
import { cn } from "#/lib/utils";

type QuestionRequest = NonNullable<SessionState["questionRequest"]>;

function normalizeValue(value: string) {
  return value.trim();
}

export function QuestionDock({
  request,
  onSubmit,
  onDismiss,
}: {
  request: QuestionRequest;
  onSubmit: (payload: QuestionReplyPayload) => void | Promise<void>;
  onDismiss: () => void | Promise<void>;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [customText, setCustomText] = useState("");
  const [showCustom, setShowCustom] = useState(false);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  const kind = request.kind || "choice";

  const formValidity = useMemo(() => {
    if (kind !== "form") return true;
    return request.fields.every((field) => {
      if (!field.required) return true;
      return normalizeValue(formData[field.name] || "") !== "";
    });
  }, [formData, kind, request.fields]);

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

  async function submit(payload: QuestionReplyPayload) {
    if (submitting) return;
    setSubmitting(true);
    try {
      await onSubmit(payload);
    } finally {
      setSubmitting(false);
    }
  }

  async function dismiss() {
    if (submitting) return;
    setSubmitting(true);
    try {
      await onDismiss();
    } finally {
      setSubmitting(false);
    }
  }

  function handleChoiceSubmit() {
    const answers = [...selected];
    if (showCustom && customText.trim()) {
      answers.push(customText.trim());
    }
    void submit({ answers });
  }

  function handleFormSubmit() {
    const data = Object.fromEntries(
      request.fields
        .map((field) => [field.name, normalizeValue(formData[field.name] || "")])
        .filter(([, value]) => value !== ""),
    );
    void submit({ data });
  }

  const canSubmitChoice = selected.size > 0 || (showCustom && customText.trim().length > 0);

  return (
    <div className="rounded-xl border border-border bg-background p-4 space-y-3">
      <div className="text-xs text-muted-foreground font-medium">{request.header}</div>
      {request.question ? <p className="text-sm text-foreground">{request.question}</p> : null}

      {kind === "confirm" ? (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {(request.options.length > 0
              ? request.options
              : [
                  { label: "Yes", description: "Confirm" },
                  { label: "No", description: "Decline" },
                ]
            ).map((opt) => (
              <button
                key={opt.label}
                onClick={() => void submit({ answers: [opt.label] })}
                disabled={submitting}
                className="rounded-lg border border-border bg-accent px-3 py-2 text-sm transition-colors hover:bg-accent/80 disabled:opacity-50"
              >
                <div className="font-medium">{opt.label}</div>
                {opt.description ? (
                  <div className="text-xs text-muted-foreground">{opt.description}</div>
                ) : null}
              </button>
            ))}
          </div>
          <button
            onClick={() => void dismiss()}
            disabled={submitting}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
          >
            Dismiss
          </button>
        </div>
      ) : null}

      {kind === "form" ? (
        <div className="space-y-3">
          {request.fields.map((field) => {
            const value = formData[field.name] || "";
            const commonClassName =
              "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring";

            return (
              <label key={field.name} className="block space-y-1.5">
                <div className="text-sm font-medium text-foreground">
                  {field.label}
                  {field.required ? <span className="text-red-400"> *</span> : null}
                </div>
                {field.type === "textarea" ? (
                  <textarea
                    value={value}
                    onChange={(event) =>
                      setFormData((current) => ({ ...current, [field.name]: event.target.value }))
                    }
                    placeholder={field.placeholder || ""}
                    className={cn(commonClassName, "min-h-[90px] resize-y")}
                    disabled={submitting}
                  />
                ) : field.type === "select" ? (
                  <select
                    value={value}
                    onChange={(event) =>
                      setFormData((current) => ({ ...current, [field.name]: event.target.value }))
                    }
                    className={commonClassName}
                    disabled={submitting}
                  >
                    <option value="">Select...</option>
                    {(field.options || []).map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type={field.type === "email" || field.type === "number" ? field.type : "text"}
                    value={value}
                    onChange={(event) =>
                      setFormData((current) => ({ ...current, [field.name]: event.target.value }))
                    }
                    placeholder={field.placeholder || ""}
                    className={commonClassName}
                    disabled={submitting}
                  />
                )}
              </label>
            );
          })}

          <div className="flex items-center justify-between pt-1">
            <button
              onClick={() => void dismiss()}
              disabled={submitting}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
            >
              Dismiss
            </button>
            <button
              onClick={handleFormSubmit}
              disabled={!formValidity || submitting}
              className="px-4 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
            >
              {request.submitLabel || "Submit"}
            </button>
          </div>
        </div>
      ) : null}

      {kind === "choice" ? (
        <>
          {request.options.length > 0 && (
            <div className="space-y-1.5">
              {request.options.map((opt, i) => (
                <button
                  key={`${opt.label}-${i}`}
                  onClick={() => toggleOption(opt.label)}
                  disabled={submitting}
                  className={cn(
                    "w-full text-left px-3 py-2 rounded-lg text-sm border transition-colors disabled:opacity-50",
                    selected.has(opt.label)
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border hover:bg-accent",
                  )}
                >
                  <div className="font-medium">{opt.label}</div>
                  {opt.description ? <div className="text-xs text-muted-foreground">{opt.description}</div> : null}
                </button>
              ))}
            </div>
          )}

          {request.allowCustom &&
            (showCustom ? (
              <textarea
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                placeholder="Type your answer..."
                className="w-full px-3 py-2 text-sm border border-border rounded-lg bg-background resize-none min-h-[60px] focus:outline-none focus:ring-1 focus:ring-ring"
                autoFocus
                disabled={submitting}
              />
            ) : (
              <button
                onClick={() => setShowCustom(true)}
                disabled={submitting}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
              >
                Type a custom answer
              </button>
            ))}

          <div className="flex items-center justify-between pt-1">
            <button
              onClick={() => void dismiss()}
              disabled={submitting}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
            >
              Dismiss
            </button>
            <button
              onClick={handleChoiceSubmit}
              disabled={!canSubmitChoice || submitting}
              className="px-4 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
            >
              {request.submitLabel || "Submit"}
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
