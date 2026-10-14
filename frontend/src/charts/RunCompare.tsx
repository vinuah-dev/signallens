import { useState } from "react";
import type { ModelSummary } from "../api/types";
import { score } from "../lib/format";
import { runColour } from "../lib/palette";
import { ChartTip, scaleLinear, useWidth } from "./core";

const METRICS: { key: "accuracy" | "balanced_accuracy" | "f1_macro" | "roc_auc"; label: string; baseline?: (r: ModelSummary) => number }[] = [
  { key: "accuracy", label: "Accuracy", baseline: (r) => r.evaluation.baseline.accuracy },
  { key: "balanced_accuracy", label: "Balanced acc.", baseline: (r) => r.evaluation.baseline.balanced_accuracy },
  { key: "f1_macro", label: "Macro F1", baseline: (r) => r.evaluation.baseline.f1_macro },
  { key: "roc_auc", label: "ROC AUC", baseline: () => 0.5 },
];

/** One row per metric, one dot per run: shows at a glance which run wins
 * on which metric, and by how much relative to the naive baseline. */
export function RunCompare({ runs, activeId, onPick }: { runs: ModelSummary[]; activeId: string | null; onPick: (id: string) => void }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ run: number; metric: number } | null>(null);
  const metrics = METRICS.filter((m) => runs.some((r) => r.evaluation.metrics[m.key] != null));
  const values = runs.flatMap((r) => metrics.map((m) => r.evaluation.metrics[m.key] ?? 1));
  const baselines = runs.flatMap((r) => metrics.map((m) => m.baseline?.(r) ?? 1));
  const lo = Math.max(0, Math.floor(Math.min(...values, ...baselines) * 10) / 10 - 0.05);
  const pad = { l: 112, r: 18, t: 8, b: 26 };
  const rowH = 38;
  const height = pad.t + metrics.length * rowH + pad.b;
  const x = scaleLinear([lo, 1], [pad.l, width - pad.r]);
  const ticks = [lo, lo + (1 - lo) / 2, 1];

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Metric comparison across runs">
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={pad.t} y2={height - pad.b} />
            <text className="tick-label" x={x(t)} y={height - 8} textAnchor="middle">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {metrics.map((m, mi) => {
          const cy = pad.t + mi * rowH + rowH / 2;
          const base = runs.length ? m.baseline?.(runs[0]) : undefined;
          return (
            <g key={m.key}>
              <text className="row-label" x={pad.l - 14} y={cy + 4} textAnchor="end">
                {m.label}
              </text>
              <line className="row-rule" x1={pad.l} x2={width - pad.r} y1={cy} y2={cy} />
              {base != null && base >= lo && (
                <g className="baseline-mark">
                  <line x1={x(base)} x2={x(base)} y1={cy - 9} y2={cy + 9} />
                </g>
              )}
              {runs.map((r, ri) => {
                const v = r.evaluation.metrics[m.key];
                if (v == null) return null;
                const active = r.id === activeId;
                return (
                  <circle
                    key={r.id}
                    className={`run-dot ${active ? "is-active" : ""}`}
                    cx={x(v)}
                    cy={cy}
                    r={active ? 7 : 5}
                    fill={runColour(ri)}
                    onMouseEnter={() => setHover({ run: ri, metric: mi })}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => onPick(r.id)}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
      {hover && (
        <ChartTip
          x={x(runs[hover.run].evaluation.metrics[metrics[hover.metric].key] ?? 0)}
          y={pad.t + hover.metric * rowH}
          width={width}
        >
          <strong>
            #{hover.run + 1} {runs[hover.run].algorithm_label}
          </strong>
          <span>
            {metrics[hover.metric].label}: {score(runs[hover.run].evaluation.metrics[metrics[hover.metric].key])}
          </span>
          <span className="muted">baseline {score(metrics[hover.metric].baseline?.(runs[hover.run]))} · click to select</span>
        </ChartTip>
      )}
    </div>
  );
}
