// Client-side checks give instant feedback; the server repeats every check
// and is the source of truth (these rules mirror backend/core/prediction.py).

import type { Limits, NumericFeature } from "../api/types";

export function checkFile(file: File, limits: Pick<Limits, "max_upload_bytes" | "extensions">): string | null {
  const name = file.name.toLowerCase();
  if (!limits.extensions.some((ext) => name.endsWith(ext))) {
    const ext = name.includes(".") ? name.split(".").pop() : "none";
    const excel = ext === "xlsx" || ext === "xls" ? " Export it from Excel with “Save as → CSV” first." : "";
    return `“.${ext}” files aren’t supported — choose a CSV file.${excel}`;
  }
  if (file.size === 0) return "This file is empty (0 bytes).";
  if (file.size > limits.max_upload_bytes) {
    return `This file is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is ${limits.max_upload_bytes / 1024 / 1024} MB.`;
  }
  return null;
}

export type FieldCheck =
  | { state: "empty" }
  | { state: "error"; message: string }
  | { state: "ok"; value: number; warning?: string };

export function checkNumber(raw: string, spec: Pick<NumericFeature, "min" | "max" | "integer">): FieldCheck {
  const text = raw.trim().replace(/,/g, "");
  if (text === "") return { state: "empty" };
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(text)) return { state: "error", message: "Not a number" };
  const value = Number(text);
  if (!Number.isFinite(value)) return { state: "error", message: "Must be a finite number" };
  if (value < spec.min || value > spec.max) {
    const span = spec.max - spec.min || Math.abs(spec.max) || 1;
    const far = value < spec.min - span || value > spec.max + span;
    return {
      state: "ok",
      value,
      warning: far ? "Far outside training range — unreliable" : "Outside training range — extrapolating",
    };
  }
  if (spec.integer && !Number.isInteger(value)) return { state: "ok", value, warning: "Training only had whole numbers" };
  return { state: "ok", value };
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
