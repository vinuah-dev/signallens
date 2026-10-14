import type { Cell } from "../api/types";

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

/** Human-scale number: integers get separators, others ~4 significant digits. */
export function num(v: number | null | undefined, maxSig = 4): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const abs = Math.abs(v);
  if (abs >= 1e6 || (abs > 0 && abs < 1e-3)) return v.toExponential(2);
  return Number(v.toPrecision(maxSig)).toLocaleString("en-US", { maximumFractionDigits: 6 });
}

export const score = (v: number | null | undefined, digits = 3) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function cell(v: Cell): string {
  if (v == null) return "";
  if (typeof v === "number") return num(v, 6);
  return String(v);
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

export const delimiterName = (d: string) =>
  ({ ",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe" })[d] ?? JSON.stringify(d);
