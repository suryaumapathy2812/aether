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
    if (!catalog) return null;
    const value = model.trim();
    if (!value) return null;
    return catalog.models.find((entry) => entry.id === value) || null;
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
      {catalog && (
        <div className="grid gap-3 md:grid-cols-3">
          {PROVIDER_ORDER.map((provider) => {
            const summary = catalog.provider_summary[provider];
            return (
              <div key={provider} className="rounded-lg border border-border/60 p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium">{PROVIDER_LABELS[provider]}</p>
                  <Badge variant="outline">{summary.total_models} models</Badge>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">{summary.notes}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Badge variant="secondary">
                    Image {summary.image_upload_models}/{summary.total_models}
                  </Badge>
                  <Badge variant="secondary">
                    Audio {summary.audio_upload_models}/{summary.total_models}
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {catalog && groupedModels && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Catalog models</p>
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
          <p className="text-xs text-muted-foreground">
            Catalog snapshot: {catalog.generated_at}. Source: OpenRouter.
          </p>
        </div>
      )}

      <div className="space-y-2">
        <p className="text-sm font-medium">Selected model ID</p>
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
      </div>

      {selectedModel && (
        <div className="rounded-lg border border-border/60 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium">{selectedModel.name}</p>
            <Badge variant="outline">{selectedModel.id}</Badge>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge variant={modelSupports(selectedModel, "image") ? "default" : "outline"}>
              Image input
            </Badge>
            <Badge variant={modelSupports(selectedModel, "audio") ? "default" : "outline"}>
              Audio input
            </Badge>
            <Badge variant={modelSupports(selectedModel, "file") ? "default" : "outline"}>
              File input
            </Badge>
            <Badge variant={modelSupports(selectedModel, "video") ? "default" : "outline"}>
              Video input
            </Badge>
            <Badge variant={modelOutputs(selectedModel, "image") ? "secondary" : "outline"}>
              Image output
            </Badge>
            <Badge variant={modelOutputs(selectedModel, "audio") ? "secondary" : "outline"}>
              Audio output
            </Badge>
          </div>
        </div>
      )}

      {!selectedModel && model.trim() && (
        <p className="text-xs text-muted-foreground">
          The current value is not in the bundled catalog. You can still save a custom model ID.
        </p>
      )}

      {catalogError && (
        <p className="text-xs text-muted-foreground">{catalogError}</p>
      )}
    </div>
  );
}
