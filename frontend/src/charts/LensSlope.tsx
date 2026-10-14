import { useState } from "react";
import { score } from "../lib/format";
import { useWidth } from "./core";

export interface Lens {
  key: string;
  title: string;
  subtitle: string;
  values: Record<string, number>;
  format: (v: number) => string;
}

/** Slope chart of feature *ranks* under different lenses. Ranks are compared
 * rather than raw values because the lenses use incompatible units. Lines that
 * cross steeply mark features the data and the model disagree about. */
export function LensSlope({ lenses, limit = 14 }: { lenses: Lens[]; limit?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);

  const ranked = lenses.map((l) => Object.keys(l.values).sort((a, b) => l.values[b] - l.values[a]));
  // Keep features that rank highly under *any* lens.
  const keep = new Set<string>();
  for (let r = 0; keep.size < limit && r < ranked[0].length; r++) ranked.forEach((order) => order[r] && keep.size < limit && keep.add(order[r]));
  const orders = ranked.map((order) => order.filter((f) => keep.has(f)));
  const n = orders[0].length;

  const rowH = 26;
  const top = 54;
  const height = top + n * rowH + 8;
  const colW = width / lenses.length;
  const colX = (i: number) => colW * i + colW / 2;
  const y = (rank: number) => top + rank * rowH + rowH / 2;
  const label = (f: string) => (f.length > 18 ? `${f.slice(0, 17)}…` : f);
  const dotGap = Math.min(70, colW * 0.42);

  return (
    <div className="chart lens" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Feature rank under each lens">
        {lenses.map((l, i) => (
          <g key={l.key}>
            <text className="lens-title" x={colX(i)} y={18} textAnchor="middle">
              {l.title}
            </text>
            <text className="lens-sub" x={colX(i)} y={34} textAnchor="middle">
              {l.subtitle}
            </text>
          </g>
        ))}
        {[...keep].map((f) => {
          const pts = orders.map((o, i) => ({ x: colX(i), y: y(o.indexOf(f)) }));
          const shift = Math.abs(orders[0].indexOf(f) - orders[orders.length - 1].indexOf(f));
          const cls = hover ? (hover === f ? "is-hover" : "is-dim") : shift >= 3 ? "is-mover" : "";
          return (
            <g key={f} className={`lens-line ${cls}`} onMouseEnter={() => setHover(f)} onMouseLeave={() => setHover(null)}>
              {pts.slice(1).map((p, i) => (
                <line key={i} x1={pts[i].x + dotGap} y1={pts[i].y} x2={p.x - dotGap} y2={p.y} />
              ))}
              {pts.map((p, i) => (
                <g key={i}>
                  <rect className="hit" x={p.x - colW / 2} y={p.y - rowH / 2} width={colW} height={rowH} />
                  <text className="lens-label" x={p.x} y={p.y + 4} textAnchor="middle">
                    {label(f)}
                  </text>
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      <p className="readout" aria-live="polite">
        {hover
          ? lenses.map((l) => `${l.title}: ${l.format(l.values[hover] ?? 0)} (rank ${orders[lenses.indexOf(l)].indexOf(hover) + 1})`).join("  ·  ")
          : "Hover a feature to compare its scores. Steep lines (highlighted) mark features whose rank changes most between lenses."}
      </p>
    </div>
  );
}

export const fmt = { score: (v: number) => score(v), share: (v: number) => `${(v * 100).toFixed(1)}%` };
