import { useState } from "react";
import type { HistogramBars, HistogramBins } from "../api/types";
import { num, pct } from "../lib/format";
import { ChartTip, scaleLinear, useWidth } from "./core";

type Hist = HistogramBins | HistogramBars;

function normalise(h: Hist) {
  if (h.type === "bins") return h.bins.map((b) => ({ key: `${num(b.x0)} – ${num(b.x1)}`, count: b.count }));
  const bars = h.bars.map((b) => ({ key: b.label, count: b.count }));
  if (h.other) bars.push({ key: "other", count: h.other });
  return bars;
}

/** Tiny inline distribution for list rows. */
export function Sparkbars({ histogram, width = 72, height = 22 }: { histogram: Hist; width?: number; height?: number }) {
  const items = normalise(histogram).slice(0, 24);
  const max = Math.max(1, ...items.map((i) => i.count));
  const bw = width / Math.max(items.length, 1);
  return (
    <svg className="sparkbars" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {items.map((it, i) => {
        const h = Math.max(1, (it.count / max) * (height - 1));
        return <rect key={i} x={i * bw + 0.5} y={height - h} width={Math.max(1, bw - 1)} height={h} />;
      })}
    </svg>
  );
}

/** Full-size distribution chart with hover read-outs. */
export function Histogram({ histogram, total, height = 180 }: { histogram: Hist; total: number; height?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const items = normalise(histogram);
  const categorical = histogram.type === "bars";
  const pad = { l: 8, r: 8, t: 10, b: categorical ? 54 : 26 };
  const max = Math.max(1, ...items.map((i) => i.count));
  const y = scaleLinear([0, max], [height - pad.b, pad.t]);
  const bw = (width - pad.l - pad.r) / Math.max(items.length, 1);

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Distribution of values">
        <line className="axis" x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} />
        {items.map((it, i) => {
          const x = pad.l + i * bw;
          const top = y(it.count);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect className="hit" x={x} y={pad.t} width={bw} height={height - pad.b - pad.t} />
              <rect
                className={`bar ${hover === i ? "is-hover" : ""} ${it.key === "other" ? "is-other" : ""}`}
                x={x + Math.min(2, bw * 0.12)}
                y={top}
                width={Math.max(1, bw - Math.min(4, bw * 0.24))}
                height={Math.max(0, height - pad.b - top)}
              />
              {categorical && bw > 14 && (
                <text className="tick-label" transform={`translate(${x + bw / 2}, ${height - pad.b + 10}) rotate(35)`} textAnchor="start">
                  {it.key.length > 12 ? `${it.key.slice(0, 11)}…` : it.key}
                </text>
              )}
            </g>
          );
        })}
        {!categorical && histogram.type === "bins" && histogram.bins.length > 0 && (
          <>
            <text className="tick-label" x={pad.l} y={height - 8}>
              {num(histogram.bins[0].x0)}
            </text>
            <text className="tick-label" x={width - pad.r} y={height - 8} textAnchor="end">
              {num(histogram.bins[histogram.bins.length - 1].x1)}
            </text>
          </>
        )}
      </svg>
      {hover != null && (
        <ChartTip x={pad.l + hover * bw + bw / 2} y={y(items[hover].count)} width={width}>
          <strong>{items[hover].key}</strong>
          <span>
            {items[hover].count.toLocaleString()} rows · {pct(items[hover].count / Math.max(total, 1), 1)}
          </span>
        </ChartTip>
      )}
    </div>
  );
}
