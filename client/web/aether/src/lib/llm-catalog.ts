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
  return catalog.models.find((entry) => entry.id === value) || null;
}

export async function fetchLLMCatalog(): Promise<LLMCatalog> {
  const response = await fetch("/llm.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load model catalog");
  }

  return (await response.json()) as LLMCatalog;
}
