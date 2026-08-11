import type {
  ExamplesResponse,
  FeatureRecord,
  HealthResponse,
  ModelsResponse,
  PredictionResult,
  PredictAllResponse,
  SchemaResponse
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function responseErrorMessage(body: unknown): string {
  if (!isObject(body)) return "Request failed.";
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    const first = body.detail[0];
    if (isObject(first) && typeof first.msg === "string") return `Invalid request: ${first.msg}`;
    return "Invalid request. Check the feature values and try again.";
  }
  return "Request failed.";
}

export function isPredictionResult(value: unknown): value is PredictionResult {
  return (
    isObject(value) &&
    typeof value.model === "string" &&
    typeof value.attack_probability === "number" &&
    typeof value.threshold === "number" &&
    (value.prediction === 0 || value.prediction === 1) &&
    (value.label === "normal" || value.label === "attack")
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) }
    });
  } catch {
    throw new ApiError("Model service is unavailable. Live inference cannot be reached right now.");
  }

  const body: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(responseErrorMessage(body), response.status);
  }
  return body as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  models: () => request<ModelsResponse>("/models"),
  schema: () => request<SchemaResponse>("/schema"),
  examples: () => request<ExamplesResponse>("/examples"),
  predict: async (modelId: string, record: FeatureRecord) => {
    const result = await request<PredictionResult>(`/predict/${modelId}`, {
      method: "POST",
      body: JSON.stringify(record)
    });
    if (!isPredictionResult(result)) throw new ApiError("Model service returned an invalid prediction response.");
    return result;
  },
  predictAll: async (record: FeatureRecord) => {
    const response = await request<PredictAllResponse>("/predict-all", {
      method: "POST",
      body: JSON.stringify(record)
    });
    if (!Array.isArray(response.results) || !response.results.every(isPredictionResult)) {
      throw new ApiError("Model service returned an invalid comparison response.");
    }
    return response;
  }
};
