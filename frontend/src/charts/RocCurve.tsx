import { useState } from "react";
import { pct } from "../lib/format";
import { scaleLinear, useWidth } from "./core";

export function RocCurve({ points, auc, positive }: { points: { fpr: number; tpr: number }[]; auc: number; positive: string }) {
  const [ref, width] = useWidth<HTMLDivElement>(360);
  const [hover, setHover] = useState<number | null>(null);
  const size = Math.min(width, 340);
  const pad = 44;
  const x = scaleLinear([0, 1], [pad, size - 10]);
  const y = scaleLinear([0, 1], [size - pad, 10]);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.fpr)},${y(p.tpr)}`).join("");
  const area = `${path}L${x(1)},${y(0)}L${x(0)},${y(0)}Z`;

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fpr = (e.clientX - rect.left) / rect.width;
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(p.fpr - fpr) < Math.abs(points[best].fpr - fpr)) best = i;
    });
    setHover(best);
  };
  const h = hover != null ? points[hover] : null;

  return (
    <div className="chart roc" ref={ref}>
      <svg width={size} height={size} role="img" aria-label={`ROC curve, area under curve ${auc.toFixed(2)}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} />
            <line className="grid" x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} />
            <text className="tick-label" x={x(t)} y={size - pad + 14} textAnchor="middle">
              {t}
            </text>
            <text className="tick-label" x={pad - 6} y={y(t) + 3} textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        <line className="chance" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
        <path className="roc-area" d={area} />
        <path className="roc-line" d={path} />
        {h && (
          <g>
            <line className="crosshair" x1={x(h.fpr)} x2={x(h.fpr)} y1={y(0)} y2={y(1)} />
            <circle className="roc-dot" cx={x(h.fpr)} cy={y(h.tpr)} r={4.5} />
          </g>
        )}
        <rect
          className="hit"
          x={x(0)}
          y={y(1)}
          width={x(1) - x(0)}
          height={y(0) - y(1)}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
        <text className="axis-title" x={x(0.5)} y={size - 2} textAnchor="middle">
          false alarm rate →
        </text>
        <text className="axis-title" transform={`translate(10, ${y(0.5)}) rotate(-90)`} textAnchor="middle">
          hit rate →
        </text>
      </svg>
      <p className="readout">
        {h
          ? `At this cut-off the model catches ${pct(h.tpr)} of real “${positive}” rows while falsely flagging ${pct(h.fpr)} of the others.`
          : `Each point is a different confidence cut-off for calling a row “${positive}”. The dashed diagonal is random guessing.`}
      </p>
    </div>
  );
}
