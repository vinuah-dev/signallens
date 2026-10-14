import { useState } from "react";
import type { Sensitivity, WhatIf } from "../api/types";
import { cell, num, pct } from "../lib/format";
import { classColour } from "../lib/palette";
import { ChartTip, scaleLinear, useWidth } from "./core";

export function ProbabilityBars({ items, labels }: { items: { label: string; probability: number }[]; labels: string[] }) {
  return (
    <ol className="prob-bars">
      {items.map((p, i) => (
        <li key={p.label} className={i === 0 ? "is-top" : ""}>
          <span className="prob-label" title={p.label}>
            <i style={{ background: classColour(labels, p.label) }} />
            {p.label}
          </span>
          <span className="prob-track">
            <span style={{ width: `${p.probability * 100}%`, background: classColour(labels, p.label) }} />
          </span>
          <span className="prob-value">{pct(p.probability, 1)}</span>
        </li>
      ))}
    </ol>
  );
}

/** Diverging bars: how much each feature's actual value moved the predicted
 * class's probability, compared with resetting it to a typical value. */
export function SensitivityChart({ items, predicted, colour, limit = 10 }: { items: Sensitivity[]; predicted: string; colour: string; limit?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const shown = items.slice(0, limit);
  const m = Math.max(0.02, ...shown.map((s) => Math.abs(s.delta)));
  const labelW = Math.min(200, Math.max(110, width * 0.36));
  const pad = { l: labelW, r: 50 };
  const x = scaleLinear([-m, m], [pad.l, width - pad.r]);
  const rowH = 26;
  const height = shown.length * rowH + 26;

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Feature sensitivity for this prediction">
        <line className="zero" x1={x(0)} x2={x(0)} y1={0} y2={height - 20} />
        <text className="tick-label" x={x(-m)} y={height - 5}>
          ← lowers P({predicted})
        </text>
        <text className="tick-label" x={x(m)} y={height - 5} textAnchor="end">
          raises P({predicted}) →
        </text>
        {shown.map((s, i) => {
          const y0 = i * rowH;
          const v = s.value == null ? "empty" : cell(s.value);
          return (
            <g key={s.feature} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect className="hit" x={0} y={y0} width={width} height={rowH} />
              <text className="row-label" x={pad.l - 10} y={y0 + rowH / 2 + 4} textAnchor="end">
                <tspan>{s.feature.length > 18 ? `${s.feature.slice(0, 17)}…` : s.feature}</tspan>
                <tspan className="muted"> = {v.length > 10 ? `${v.slice(0, 9)}…` : v}</tspan>
              </text>
              <rect
                className={`sens-bar ${s.delta < 0 ? "is-negative" : ""}`}
                x={Math.min(x(0), x(s.delta))}
                y={y0 + 6}
                width={Math.max(1, Math.abs(x(s.delta) - x(0)))}
                height={rowH - 12}
                style={s.delta >= 0 ? { fill: colour } : undefined}
              />
              <text className="value-label" x={width - pad.r + 8} y={y0 + rowH / 2 + 4}>
                {s.delta >= 0 ? "+" : "−"}
                {Math.abs(s.delta * 100).toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <ChartTip x={x(shown[hover].delta)} y={hover * rowH} width={width}>
          <strong>
            {shown[hover].feature} = {shown[hover].value == null ? "empty" : cell(shown[hover].value)}
          </strong>
          <span>
            If it were typical ({cell(shown[hover].typical)}), P({predicted}) would be {pct(shown[hover].probability_if_typical, 1)} (
            {shown[hover].delta >= 0 ? "−" : "+"}
            {Math.abs(shown[hover].delta * 100).toFixed(1)} pts).
          </span>
        </ChartTip>
      )}
    </div>
  );
}

/** Probability of each class as one feature is swept across its training
 * range, everything else held at the current inputs. */
export function WhatIfChart({ data, labels }: { data: WhatIf; labels: string[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  if (data.kind === "categorical") {
    return (
      <div className="whatif-cats">
        {data.x.map((cat, i) => (
          <div key={String(cat)} className={`whatif-cat ${String(cat) === String(data.current) ? "is-current" : ""}`}>
            <span className="whatif-cat-label" title={String(cat)}>
              {String(cat)}
              {String(cat) === String(data.current) && <em> current</em>}
            </span>
            <span className="stack">
              {data.series.map((s) => (
                <span
                  key={s.label}
                  style={{ width: `${s.values[i] * 100}%`, background: classColour(labels, s.label) }}
                  title={`${s.label}: ${pct(s.values[i], 1)}`}
                />
              ))}
            </span>
          </div>
        ))}
      </div>
    );
  }

  const xs = data.x as number[];
  const pad = { l: 40, r: 14, t: 10, b: 30 };
  const height = 220;
  const x = scaleLinear([xs[0], xs[xs.length - 1]], [pad.l, width - pad.r]);
  const y = scaleLinear([0, 1], [height - pad.b, pad.t]);
  const current = typeof data.current === "number" ? data.current : null;

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left + pad.l;
    let best = 0;
    xs.forEach((v, i) => {
      if (Math.abs(x(v) - px) < Math.abs(x(xs[best]) - px)) best = i;
    });
    setHover(best);
  };

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={`Class probabilities as ${data.feature} changes`}>
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line className="grid" x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} />
            <text className="tick-label" x={pad.l - 6} y={y(t) + 3} textAnchor="end">
              {pct(t)}
            </text>
          </g>
        ))}
        <text className="tick-label" x={pad.l} y={height - 8}>
          {num(xs[0])}
        </text>
        <text className="tick-label" x={width - pad.r} y={height - 8} textAnchor="end">
          {num(xs[xs.length - 1])}
        </text>
        <text className="axis-title" x={(pad.l + width - pad.r) / 2} y={height - 8} textAnchor="middle">
          {data.feature} →
        </text>
        {data.series.map((s) => (
          <path
            key={s.label}
            className="whatif-line"
            stroke={classColour(labels, s.label)}
            d={s.values.map((v, i) => `${i ? "L" : "M"}${x(xs[i])},${y(v)}`).join("")}
          />
        ))}
        {current != null && current >= xs[0] && current <= xs[xs.length - 1] && (
          <g className="current-mark">
            <line x1={x(current)} x2={x(current)} y1={pad.t} y2={height - pad.b} />
            <text x={x(current) + 4} y={pad.t + 10}>
              now
            </text>
          </g>
        )}
        {hover != null && <line className="crosshair" x1={x(xs[hover])} x2={x(xs[hover])} y1={pad.t} y2={height - pad.b} />}
        <rect
          className="hit"
          x={pad.l}
          y={pad.t}
          width={width - pad.l - pad.r}
          height={height - pad.t - pad.b}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      {hover != null && (
        <ChartTip x={x(xs[hover])} y={pad.t} width={width}>
          <strong>
            {data.feature} = {num(xs[hover])}
          </strong>
          {[...data.series]
            .sort((a, b) => b.values[hover] - a.values[hover])
            .map((s) => (
              <span key={s.label}>
                <i className="swatch" style={{ background: classColour(labels, s.label) }} /> {s.label}: {pct(s.values[hover], 1)}
              </span>
            ))}
        </ChartTip>
      )}
    </div>
  );
}
