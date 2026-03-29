import { useEffect, useMemo, useState } from "react";
import { useSession } from "#/lib/auth-client";
import { Button } from "#/components/ui/button";
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
    let cancelled = false;
    if (!userId) {
      setLoading(false);
      return;
    }
    void getUserPreference(userId, prefKey).then((value) => {
      if (cancelled) return;
      setModel(value || "");
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
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

  async function persistModel(nextModel: string) {
    if (!userId) return;
    setSaving(true);
    const trimmed = nextModel.trim();
    const success = trimmed
      ? await setUserPreference(userId, prefKey, trimmed)
      : await deleteUserPreference(userId, prefKey);
    if (success) {
      toast.success(trimmed ? "Model saved" : "Reset to default");
      setModel(trimmed);
    } else {
      toast.error(trimmed ? "Failed to save" : "Failed to reset");
    }
    setSaving(false);
  }

  if (loading) return null;

  const hasOverride = model.trim().length > 0;
  const selectedValue = selectedModel?.id || model;

  return (
    <div className="space-y-4">
      {catalogError && (
        <p className="text-sm text-destructive">{catalogError}</p>
      )}
      {catalog && groupedModels && (
        <div className="space-y-2">
          <Select
            disabled={saving}
            value={selectedValue}
            onValueChange={(value) => {
              void persistModel(value);
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder={placeholder} />
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

      <div className="flex flex-wrap items-center gap-2">
        {selectedModel ? (
          <Badge variant="secondary">
            {PROVIDER_LABELS[selectedModel.provider]}
          </Badge>
        ) : hasOverride ? (
          <Badge variant="secondary">Custom override</Badge>
        ) : (
          <Badge variant="outline">Default model</Badge>
        )}

        <span className="text-sm text-muted-foreground">
          {saving
            ? "Saving model preference..."
            : hasOverride
              ? model
              : `Using default: ${placeholder}`}
        </span>
      </div>

      <div>
        <Button
          disabled={!hasOverride || saving}
          onClick={() => {
            void persistModel("");
          }}
          size="sm"
          type="button"
          variant="outline"
        >
          Reset to default
        </Button>
      </div>
    </div>
  );
}
