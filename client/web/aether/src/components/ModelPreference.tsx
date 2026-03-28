import { useEffect, useMemo, useState } from "react";
import { useSession } from "#/lib/auth-client";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { Badge } from "#/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "#/components/ui/select";
import {
  getUserPreference,
  setUserPreference,
  deleteUserPreference,
} from "#/lib/preferences";
import {
  fetchLLMCatalog,
  findCatalogModel,
  isChatCapableModel,
  modelOutputs,
  modelSupports,
  PROVIDER_LABELS,
  type LLMCatalog,
  type ProviderName,
} from "#/lib/llm-catalog";
import { toast } from "sonner";

const DEFAULT_MODEL = "minimax/minimax-m2.5";
const PROVIDER_ORDER: ProviderName[] = ["anthropic", "openai", "google"];

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
  const [catalog, setCatalog] = useState<LLMCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  useEffect(() => {
    if (!userId) return;
    getUserPreference(userId, prefKey).then((value) => {
      setModel(value || "");
      setLoading(false);
    });
  }, [userId, prefKey]);

  useEffect(() => {
    let cancelled = false;

    void fetchLLMCatalog()
      .then((data) => {
        if (cancelled) return;
        setCatalog(data);
        setCatalogError(null);
      })
      .catch(() => {
        if (cancelled) return;
        setCatalogError("Model catalog unavailable");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const selectedModel = useMemo(() => {
    return findCatalogModel(catalog, model);
  }, [catalog, model]);

  const groupedModels = useMemo(() => {
    if (!catalog) return null;

    return PROVIDER_ORDER.map((provider) => ({
      provider,
      models: catalog.models
        .filter((entry) => entry.provider === provider && isChatCapableModel(entry))
        .sort((left, right) => left.name.localeCompare(right.name)),
    })).filter((group) => group.models.length > 0);
  }, [catalog]);

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
    <div className="space-y-4">
      {catalog && groupedModels && (
        <div className="space-y-2">
          <Select
            value={selectedModel?.id}
            onValueChange={(value) => setModel(value)}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Choose a model from the catalog" />
            </SelectTrigger>
            <SelectContent>
              {groupedModels.map((group) => (
                <SelectGroup key={group.provider}>
                  <SelectLabel>{PROVIDER_LABELS[group.provider]}</SelectLabel>
                  {group.models.map((entry) => (
                    <SelectItem key={entry.id} value={entry.id}>
                      {entry.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </div>
  );
}
