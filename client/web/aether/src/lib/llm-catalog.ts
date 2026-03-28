export type ProviderName = "anthropic" | "openai" | "google";

export type ProviderSummary = {
  total_models: number;
  image_upload_models: number;
  audio_upload_models: number;
  notes: string;
};

export type CatalogModel = {
  provider: ProviderName;
  id: string;
  name: string;
  input_modalities: string[];
  output_modalities: string[];
};

export type LLMCatalog = {
  generated_at: string;
  source: string;
  provider_summary: Record<ProviderName, ProviderSummary>;
  models: CatalogModel[];
};

export const PROVIDER_LABELS: Record<ProviderName, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  google: "Google",
};

export function modelSupports(model: CatalogModel, modality: string): boolean {
  return model.input_modalities.includes(modality);
}

export function modelOutputs(model: CatalogModel, modality: string): boolean {
  return model.output_modalities.includes(modality);
}

export function isChatCapableModel(model: CatalogModel): boolean {
  return modelOutputs(model, "text");
}

export function findCatalogModel(
  catalog: LLMCatalog | null,
  modelId: string | null | undefined,
): CatalogModel | null {
  if (!catalog || !modelId) return null;
  const value = modelId.trim();
  if (!value) return null;
  const models = Array.isArray(catalog.models) ? catalog.models : [];
  return models.find((entry) => entry.id === value) || null;
}

function isProviderName(value: string): value is ProviderName {
  return value === "anthropic" || value === "openai" || value === "google";
}

function normalizeModalities(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function normalizeModel(input: unknown, providerHint?: ProviderName): CatalogModel | null {
  if (!input || typeof input !== "object") return null;
  const record = input as Record<string, unknown>;
  const id = typeof record.id === "string" ? record.id : "";
  const name = typeof record.name === "string" ? record.name : id;
  const provider =
    typeof record.provider === "string" && isProviderName(record.provider)
      ? record.provider
      : providerHint;

  if (!id || !provider) return null;

  return {
    provider,
    id,
    name,
    input_modalities: normalizeModalities(record.input_modalities),
    output_modalities: normalizeModalities(record.output_modalities),
  };
}

function normalizeProviderSummary(
  value: unknown,
  provider: ProviderName,
  models: CatalogModel[],
): ProviderSummary {
  const record = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const imageUploadModels = models.filter((model) => modelSupports(model, "image")).length;
  const audioUploadModels = models.filter((model) => modelSupports(model, "audio")).length;

  return {
    total_models:
      typeof record.total_models === "number" ? record.total_models : models.length,
    image_upload_models:
      typeof record.image_upload_models === "number"
        ? record.image_upload_models
        : imageUploadModels,
    audio_upload_models:
      typeof record.audio_upload_models === "number"
        ? record.audio_upload_models
        : audioUploadModels,
    notes:
      typeof record.notes === "string"
        ? record.notes
        : `${PROVIDER_LABELS[provider]} model support summary`,
  };
}

export function normalizeLLMCatalog(input: unknown): LLMCatalog {
  const record = input && typeof input === "object" ? (input as Record<string, unknown>) : {};
  const generatedAt =
    typeof record.generated_at === "string" ? record.generated_at : "";
  const source = typeof record.source === "string" ? record.source : "";

  let models: CatalogModel[] = [];
  if (Array.isArray(record.models)) {
    models = record.models
      .map((entry) => normalizeModel(entry))
      .filter((entry): entry is CatalogModel => entry !== null);
  } else if (record.providers && typeof record.providers === "object") {
    const providersRecord = record.providers as Record<string, unknown>;
    models = Object.entries(providersRecord).flatMap(([providerName, entries]) => {
      if (!isProviderName(providerName) || !Array.isArray(entries)) return [];
      return entries
        .map((entry) => normalizeModel(entry, providerName))
        .filter((entry): entry is CatalogModel => entry !== null);
    });
  }

  const grouped = {
    anthropic: models.filter((model) => model.provider === "anthropic"),
    openai: models.filter((model) => model.provider === "openai"),
    google: models.filter((model) => model.provider === "google"),
  };

  const providerSummaryRecord =
    record.provider_summary && typeof record.provider_summary === "object"
      ? (record.provider_summary as Record<string, unknown>)
      : {};

  return {
    generated_at: generatedAt,
    source,
    models,
    provider_summary: {
      anthropic: normalizeProviderSummary(
        providerSummaryRecord.anthropic,
        "anthropic",
        grouped.anthropic,
      ),
      openai: normalizeProviderSummary(
        providerSummaryRecord.openai,
        "openai",
        grouped.openai,
      ),
      google: normalizeProviderSummary(
        providerSummaryRecord.google,
        "google",
        grouped.google,
      ),
    },
  };
}

export async function fetchLLMCatalog(): Promise<LLMCatalog> {
  const response = await fetch(`/llm.json?v=${Date.UTC(2026, 2, 28)}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load model catalog");
  }

  return normalizeLLMCatalog(await response.json());
}
