import { useState } from "react";
import { pct } from "../lib/format";

export type CellRef = { actual: number; predicted: number };

interface Props {
  labels: string[];
  matrix: number[][];
  normalise: boolean;
  selected: CellRef | null;
  onSelect: (cell: CellRef | null) => void;
}

/** Actual (rows) × predicted (columns). Diagonal = correct (teal), off-diagonal
 * = a specific mistake (vermilion); shade ∝ share of that actual class in
 * both display modes. */
export function ConfusionMatrix({ labels, matrix, normalise, selected, onSelect }: Props) {
  const [hover, setHover] = useState<CellRef | null>(null);
  const rowTotals = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const focus = hover ?? selected;

  const describe = ({ actual, predicted }: CellRef) => {
    const n = matrix[actual][predicted];
    const share = rowTotals[actual] ? n / rowTotals[actual] : 0;
    const a = labels[actual];
    const p = labels[predicted];
    if (actual === predicted)
      return `${n} of ${rowTotals[actual]} real “${a}” rows were correctly predicted as “${a}” (${pct(share)} recall).`;
    return `${n} row${n === 1 ? "" : "s"} that were really “${a}” were predicted as “${p}” — ${pct(share)} of real “${a}”.`;
  };

  return (
    <div className="cm">
      <div className="cm-scroll">
        <div className="cm-grid" style={{ gridTemplateColumns: `minmax(84px, max-content) repeat(${labels.length}, minmax(52px, 1fr))` }}>
          <div className="cm-corner">
            <span>actual ↓</span>
            <span>predicted →</span>
          </div>
          {labels.map((l, j) => (
            <div key={`h${j}`} className={`cm-head ${focus?.predicted === j ? "is-focus" : ""}`} title={l}>
              {l}
            </div>
          ))}
          {matrix.map((row, i) => (
            <div key={`r${i}`} className="cm-row" style={{ display: "contents" }}>
              <div className={`cm-side ${focus?.actual === i ? "is-focus" : ""}`} title={labels[i]}>
                {labels[i]}
              </div>
              {row.map((n, j) => {
                const share = rowTotals[i] ? n / rowTotals[i] : 0;
                const diag = i === j;
                // Shade by share of the actual class, so small classes stay readable.
                const alpha = n === 0 ? 0 : 0.1 + 0.85 * share;
                const isSel = selected?.actual === i && selected?.predicted === j;
                return (
                  <button
                    key={j}
                    type="button"
                    className={`cm-cell ${diag ? "is-diag" : "is-off"} ${isSel ? "is-selected" : ""} ${alpha > 0.55 ? "is-dark" : ""}`}
                    style={{ background: diag ? `rgba(29,110,115,${alpha})` : `rgba(224,83,31,${alpha})` }}
                    onMouseEnter={() => setHover({ actual: i, predicted: j })}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover({ actual: i, predicted: j })}
                    onBlur={() => setHover(null)}
                    onClick={() => onSelect(isSel ? null : { actual: i, predicted: j })}
                    aria-pressed={isSel}
                    aria-label={describe({ actual: i, predicted: j })}
                  >
                    {normalise ? pct(share) : n.toLocaleString()}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="readout" aria-live="polite">
        {focus ? describe(focus) : "Hover a cell to read it; click one to list those test rows below."}
      </p>
    </div>
  );
}
