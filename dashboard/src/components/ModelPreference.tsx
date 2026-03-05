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
    <div className="py-3">
      <p className="text-[13px] text-foreground">Model</p>
      <p className="text-[11px] text-muted-foreground mt-0.5 mb-2">
        Override the default model for AI tasks
      </p>
      <div className="flex gap-2">
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder={DEFAULT_MODEL}
          className="flex-1 h-8 px-2 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>
      <div className="flex gap-2 mt-2">
        <Button
          variant="aether"
          size="aether"
          onClick={handleSave}
          disabled={saving}
          className="text-[12px] h-7"
        >
          {saving ? "Saving..." : "Save"}
        </Button>
        <Button
          variant="aether-link"
          size="aether-link"
          onClick={handleReset}
          disabled={saving || model === ""}
          className="text-[12px] h-7"
        >
          Reset
        </Button>
      </div>
      {message && (
        <p
          className={`text-[11px] mt-2 ${
            message.type === "success" ? "text-green-500" : "text-red-500"
          }`}
        >
          {message.text}
        </p>
      )}
    </div>
  );
}
