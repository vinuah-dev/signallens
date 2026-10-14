import { runColour } from "../lib/palette";
import { score } from "../lib/format";
import { useDispatch, useStore } from "../state/store";

/** Compact run picker shown above the Inspect / Explain / Probe stages. */
export function RunSwitcher() {
  const { runs, activeRunId } = useStore();
  const dispatch = useDispatch();
  if (runs.length < 2) return null;
  return (
    <nav className="run-switcher" aria-label="Choose run">
      <span className="eyebrow">Viewing run</span>
      <div className="run-switcher-list">
        {runs.map((r, i) => (
          <button
            key={r.id}
            type="button"
            className={r.id === activeRunId ? "is-on" : ""}
            aria-pressed={r.id === activeRunId}
            onClick={() => dispatch({ type: "run/activate", id: r.id })}
          >
            <span className="run-chip" style={{ background: runColour(i) }}>
              #{i + 1}
            </span>
            {r.algorithm_label}
            <span className="mono-meta">{score(r.evaluation.metrics.f1_macro, 2)} F1</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
