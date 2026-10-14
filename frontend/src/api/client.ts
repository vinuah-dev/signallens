import type {
  AlgorithmSpec,
  Dataset,
  FrameReport,
  FrameSelection,
  Limits,
  ModelSummary,
  Prediction,
  RowsPage,
  Sample,
  TrainSettings,
  WhatIf,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

/** Error shape returned by the API: { error: { code, message, details } }. */
export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
  }

  /** The server lost this dataset/model (restart or eviction). */
  get isGone() {
    return this.code === "dataset_not_found" || this.code === "model_not_found";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api${path}`, init);
  } catch {
    throw new ApiError("network", "Can't reach the SignalLens server. Check your connection and try again.", 0);
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // e.g. an HTML error page from a proxy
  }
  if (!res.ok) {
    const err = (body as { error?: { code: string; message: string; details?: Record<string, unknown> } } | null)?.error;
    if (err) throw new ApiError(err.code, err.message, res.status, err.details ?? {});
    const message =
      res.status === 413
        ? "The file is too large for the server to accept."
        : res.status >= 500
          ? "The server hit an unexpected problem. Please try again."
          : `Request failed (${res.status}).`;
    throw new ApiError(`http_${res.status}`, message, res.status);
  }
  return body as T;
}

const json = (data: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

export const api = {
  limits: () => request<Limits>("/limits"),
  algorithms: () => request<AlgorithmSpec[]>("/algorithms"),
  samples: () => request<Sample[]>("/samples"),
  loadSample: (key: string) => request<Dataset>(`/samples/${encodeURIComponent(key)}`, { method: "POST" }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Dataset>("/datasets", { method: "POST", body: form });
  },
  rows: (id: string, q: { offset: number; limit: number; missingOnly: boolean; sortBy?: string; descending?: boolean }) => {
    const params = new URLSearchParams({
      offset: String(q.offset),
      limit: String(q.limit),
      missing_only: String(q.missingOnly),
      descending: String(Boolean(q.descending)),
    });
    if (q.sortBy) params.set("sort_by", q.sortBy);
    return request<RowsPage>(`/datasets/${id}/rows?${params}`);
  },
  checkFrame: (id: string, sel: FrameSelection, signal?: AbortSignal) =>
    request<FrameReport>(`/datasets/${id}/frame`, { ...json(sel), signal }),
  train: (datasetId: string, sel: FrameSelection, settings: TrainSettings) =>
    request<ModelSummary>("/models", json({ dataset_id: datasetId, ...sel, ...settings })),
  predict: (modelId: string, rows: Record<string, unknown>[], signal?: AbortSignal) =>
    request<{ model_id: string; results: Prediction[] }>(`/models/${modelId}/predict`, { ...json({ rows, explain: true }), signal }),
  whatIf: (modelId: string, row: Record<string, unknown>, feature: string, signal?: AbortSignal) =>
    request<WhatIf>(`/models/${modelId}/what-if`, { ...json({ row, feature }), signal }),
};
