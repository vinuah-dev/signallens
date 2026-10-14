import { useState } from "react";
import { score } from "../lib/format";
import { ChartTip, scaleLinear, useWidth } from "./core";

export interface ImportanceItem {
  feature: string;
  mean: number;
  std: number;
}

/** Permutation importance: bar = average drop in balanced accuracy when the
 * feature is shuffled, whisker = ±1 standard deviation across repeats. */
export function ImportanceChart({ items, compare, limit = 15 }: { items: ImportanceItem[]; compare?: { label: string; items: ImportanceItem[] }; limit?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const shown = items.slice(0, limit);
  const other = new Map(compare?.items.map((i) => [i.feature, i]));
  const all = [...shown, ...(compare?.items ?? [])];
  const hi = Math.max(0.01, ...all.map((i) => i.mean + i.std));
  const lo = Math.min(0, ...all.map((i) => i.mean - i.std));
  const labelW = Math.min(190, Math.max(90, width * 0.3));
  const pad = { l: labelW, r: 56, t: 6, b: 24 };
  const rowH = compare ? 30 : 24;
  const height = pad.t + shown.length * rowH + pad.b;
  const x = scaleLinear([lo, hi], [pad.l, width - pad.r]);

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Permutation importance by feature">
        <line className="zero" x1={x(0)} x2={x(0)} y1={pad.t} y2={height - pad.b} />
        {shown.map((it, i) => {
          const y0 = pad.t + i * rowH;
          const barH = compare ? 10 : 12;
          const cmp = other.get(it.feature);
          return (
            <g key={it.feature} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect className="hit" x={0} y={y0} width={width} height={rowH} />
              <text className={`row-label ${hover === i ? "is-hover" : ""}`} x={pad.l - 10} y={y0 + rowH / 2 + 4} textAnchor="end">
                {it.feature.length > 26 ? `${it.feature.slice(0, 25)}…` : it.feature}
              </text>
              <rect
                className={`imp-bar ${it.mean < 0 ? "is-negative" : ""}`}
                x={Math.min(x(0), x(it.mean))}
                y={y0 + (compare ? 3 : (rowH - barH) / 2)}
                width={Math.abs(x(it.mean) - x(0))}
                height={barH}
              />
              <line className="whisker" x1={x(it.mean - it.std)} x2={x(it.mean + it.std)} y1={y0 + (compare ? 8 : rowH / 2)} y2={y0 + (compare ? 8 : rowH / 2)} />
              {cmp && (
                <rect
                  className="imp-bar is-compare"
                  x={Math.min(x(0), x(cmp.mean))}
                  y={y0 + 15}
                  width={Math.abs(x(cmp.mean) - x(0))}
                  height={barH}
                />
              )}
              <text className="value-label" x={width - pad.r + 8} y={y0 + rowH / 2 + 4}>
                {score(it.mean)}
              </text>
            </g>
          );
        })}
        <text className="tick-label" x={x(0)} y={height - 6} textAnchor="middle">
          0
        </text>
        <text className="tick-label" x={x(hi)} y={height - 6} textAnchor="end">
          {score(hi, 2)} drop in balanced acc.
        </text>
      </svg>
      {hover != null && (
        <ChartTip x={x(shown[hover].mean)} y={pad.t + hover * rowH} width={width}>
          <strong>{shown[hover].feature}</strong>
          <span>
            Shuffling it lowers balanced accuracy by {score(shown[hover].mean)} (± {score(shown[hover].std)}).
          </span>
          {other.get(shown[hover].feature) && (
            <span className="muted">
              {compare!.label}: {score(other.get(shown[hover].feature)!.mean)}
            </span>
          )}
          {shown[hover].mean <= 0.001 && <span className="muted">≈ 0: the model doesn’t rely on it (or another feature carries the same information).</span>}
        </ChartTip>
      )}
    </div>
  );
}
