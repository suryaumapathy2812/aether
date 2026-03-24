"use client";

import { useEffect, useState } from "react";
import { useSession } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  getUserPreference,
  setUserPreference,
  deleteUserPreference,
} from "@/lib/preferences";
import { toast } from "sonner";

const DEFAULT_MODEL = "minimax/minimax-m2.5";

export default function ModelPreference({
  prefKey = "model",
  placeholder = DEFAULT_MODEL,
}: {
  prefKey?: string;
  placeholder?: string;
} = {}) {
  const { data: session } = useSession();
  const userId = session?.user?.id || "";

  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!userId) return;
    getUserPreference(userId, prefKey).then((value) => {
      setModel(value || "");
      setLoading(false);
    });
  }, [userId, prefKey]);

  async function handleSave() {
    if (!userId) return;
    setSaving(true);
    const trimmed = model.trim();
    const success = await setUserPreference(userId, prefKey, trimmed);
    if (success) {
      toast.success("Model saved");
    } else {
      toast.error("Failed to save");
    }
    setSaving(false);
  }

  async function handleReset() {
    if (!userId) return;
    setSaving(true);
    const success = await deleteUserPreference(userId, prefKey);
    if (success) {
      setModel("");
      toast.success("Reset to default");
    } else {
      toast.error("Failed to reset");
    }
    setSaving(false);
  }

  if (loading) return null;

  return (
    <div className="flex gap-2">
      <Input
        value={model}
        onChange={(e) => setModel(e.target.value)}
        placeholder={placeholder}
        className="flex-1"
      />
      <Button
        variant="secondary"
        size="sm"
        onClick={handleSave}
        disabled={saving}
      >
        {saving ? "..." : "Save"}
      </Button>
      {model && (
        <Button
          variant="ghost"
          size="sm"
          onClick={handleReset}
          disabled={saving}
          className="text-muted-foreground hover:text-foreground"
        >
          Reset
        </Button>
      )}
    </div>
  );
}
