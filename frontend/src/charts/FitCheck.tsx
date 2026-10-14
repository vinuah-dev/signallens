import { score } from "../lib/format";
import { scaleLinear, useWidth } from "./core";

/** Macro-F1 measured three ways: on rows the model trained on, on held-out
 * cross-validation folds, and on the untouched test split. A big gap between
 * the first and the others is the visual signature of overfitting. */
export function FitCheck({ train, cv, test }: { train: number; cv: number[]; test: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(420);
  const all = [train, test, ...cv];
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.05) * 10) / 10);
  const pad = { l: 132, r: 46 };
  const x = scaleLinear([lo, 1], [pad.l, width - pad.r]);
  const rows: { label: string; values: number[]; cls: string }[] = [
    { label: "training rows", values: [train], cls: "fit-train" },
    { label: "CV folds", values: cv, cls: "fit-cv" },
    { label: "test rows", values: [test], cls: "fit-test" },
  ];
  const rowH = 30;
  const height = rows.length * rowH + 26;

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={`Macro F1: training ${score(train)}, test ${score(test)}`}>
        {[lo, (lo + 1) / 2, 1].map((t) => (
          <g key={t}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={6} y2={height - 20} />
            <text className="tick-label" x={x(t)} y={height - 6} textAnchor="middle">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = 16 + i * rowH;
          return (
            <g key={r.label} className={r.cls}>
              <text className="row-label" x={pad.l - 12} y={cy + 4} textAnchor="end">
                {r.label}
              </text>
              <line className="row-rule" x1={pad.l} x2={width - pad.r} y1={cy} y2={cy} />
              {r.values.length > 1 && (
                <line className="fit-range" x1={x(Math.min(...r.values))} x2={x(Math.max(...r.values))} y1={cy} y2={cy} />
              )}
              {r.values.map((v, j) => (
                <circle key={j} cx={x(v)} cy={cy} r={r.values.length > 1 ? 4 : 6} />
              ))}
              {r.values.length === 1 && (
                <text className="value-label" x={x(r.values[0]) + 10} y={cy + 4}>
                  {score(r.values[0], 2)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
