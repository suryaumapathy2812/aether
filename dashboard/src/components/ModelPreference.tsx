"use client";

import { useEffect, useState } from "react";
import { useSession } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import {
  getUserPreference,
  setUserPreference,
  deleteUserPreference,
} from "@/lib/preferences";

const DEFAULT_MODEL = "minimax/minimax-m2.5";

export default function ModelPreference() {
  const { data: session } = useSession();
  const userId = session?.user?.id || "";

  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  useEffect(() => {
    if (!userId) return;
    getUserPreference(userId, "model").then((value) => {
      setModel(value || "");
      setLoading(false);
    });
  }, [userId]);

  async function handleSave() {
    if (!userId) return;
    setSaving(true);
    setMessage(null);
    const trimmed = model.trim();
    const success = await setUserPreference(userId, "model", trimmed);
    if (success) {
      setMessage({ type: "success", text: "Model saved" });
    } else {
      setMessage({ type: "error", text: "Failed to save" });
    }
    setSaving(false);
  }

  async function handleReset() {
    if (!userId) return;
    setSaving(true);
    setMessage(null);
    const success = await deleteUserPreference(userId, "model");
    if (success) {
      setModel("");
      setMessage({ type: "success", text: "Reset to default" });
    } else {
      setMessage({ type: "error", text: "Failed to reset" });
    }
    setSaving(false);
  }

  if (loading) return null;

  return (
    <div>
      <div className="flex gap-2">
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder={DEFAULT_MODEL}
          className="flex-1 h-8 px-3 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring text-foreground placeholder:text-muted-foreground/50"
        />
        <Button
          variant="ghost"
          size="sm"
          onClick={handleSave}
          disabled={saving}
          className="h-8 px-3 text-[12px] text-foreground/70 hover:text-foreground"
        >
          {saving ? "..." : "Save"}
        </Button>
        {model && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            disabled={saving}
            className="h-8 px-2 text-[12px] text-muted-foreground hover:text-foreground"
          >
            Reset
          </Button>
        )}
      </div>
      {message && (
        <p className={`text-[11px] mt-2 ${message.type === "success" ? "text-emerald-400/80" : "text-red-400/80"}`}>
          {message.text}
        </p>
      )}
    </div>
  );
}
