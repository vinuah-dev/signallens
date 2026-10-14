import { useMemo, useState } from "react";
import type { ModelSummary } from "../api/types";
import { ConfusionMatrix, type CellRef } from "../charts/ConfusionMatrix";
import { FitCheck } from "../charts/FitCheck";
import { RocCurve } from "../charts/RocCurve";
import { Button, Glyph, IssueList, Meter, SectionHead, Segmented, Term } from "../components/ui";
import { cell, pct, score } from "../lib/format";
import { classColour } from "../lib/palette";
import { useActiveRun, useDispatch } from "../state/store";
import { RunSwitcher } from "./RunSwitcher";

export function InspectStage() {
  const run = useActiveRun()!;
  return (
    <div className="stage stage-inspect">
      <RunSwitcher />
      <Inspect key={run.id} run={run} />
    </div>
  );
}

function Inspect({ run }: { run: ModelSummary }) {
  const ev = run.evaluation;
  const m = ev.metrics;
  const [normalise, setNormalise] = useState(false);
  const [cellSel, setCellSel] = useState<CellRef | null>(null);

  const stats: { key: string; label: React.ReactNode; value: number | undefined; base?: number; baseLabel?: string }[] = [
    { key: "acc", label: <Term k="accuracy">Accuracy</Term>, value: m.accuracy, base: ev.baseline.accuracy, baseLabel: "always-majority guess" },
    { key: "bal", label: <Term k="balanced_accuracy">Balanced accuracy</Term>, value: m.balanced_accuracy, base: ev.baseline.balanced_accuracy, baseLabel: "always-majority guess" },
    { key: "f1", label: <><Term k="macro">Macro</Term> <Term k="f1">F1</Term></>, value: m.f1_macro, base: ev.baseline.f1_macro, baseLabel: "always-majority guess" },
    { key: "auc", label: <Term k="roc_auc">ROC AUC</Term>, value: m.roc_auc, base: 0.5, baseLabel: "random ranking" },
  ];

  return (
    <>
      {run.frame.imbalanced ? (
        // A dominant class makes plain accuracy look good for free, so the
        // headline switches to balanced accuracy (naive guess = 1 / #classes).
        <header className="verdict">
          <div className="verdict-number">
            <span className="eyebrow">{run.algorithm_label} · balanced accuracy</span>
            <span className="big">{pct(m.balanced_accuracy, 1)}</span>
          </div>
          <p className="verdict-text">
            “{ev.baseline.label}” makes up most rows, so always guessing it would already be {pct(ev.baseline.accuracy)} accurate while never
            finding anything else. Weighing each class equally, the model scores {pct(m.balanced_accuracy)} on{" "}
            <strong>{ev.n_test.toLocaleString()}</strong> unseen rows, versus {pct(ev.baseline.balanced_accuracy)} for that{" "}
            <Term k="baseline">naive guess</Term>.
          </p>
        </header>
      ) : (
        <header className="verdict">
          <div className="verdict-number">
            <span className="eyebrow">{run.algorithm_label} · test accuracy</span>
            <span className="big">{pct(m.accuracy, 1)}</span>
          </div>
          <p className="verdict-text">
            On <strong>{ev.n_test.toLocaleString()}</strong> rows it never saw during training, the model predicted <strong>{run.target}</strong> correctly{" "}
            {pct(m.accuracy)} of the time — versus {pct(ev.baseline.accuracy)} for a <Term k="baseline">naive baseline</Term>.
          </p>
        </header>
      )}

      <dl className="ledger ledger-metrics">
        {stats
          .filter((s) => s.value != null)
          .map((s) => (
            <div key={s.key}>
              <dt>{s.label}</dt>
              <dd>{score(s.value, 3)}</dd>
              {s.base != null && (
                <span className="ledger-sub">
                  <span className="vs-base">
                    <span className="vs-track">
                      <span className="vs-base-mark" style={{ left: `${s.base * 100}%` }} />
                      <span className="vs-value" style={{ width: `${(s.value ?? 0) * 100}%` }} />
                    </span>
                  </span>
                  {score(s.base, 2)} for {s.baseLabel}
                </span>
              )}
            </div>
          ))}
      </dl>

      <section className="block">
        <SectionHead index="04.1" title="Reading the result" kicker="Generated from this run’s numbers, in plain words." />
        <ul className="findings">
          {run.interpretation.map((line, i) => (
            <li key={i} className={`finding finding-${line.tone}`}>
              <Glyph level={line.tone === "good" ? "good" : line.tone} />
              <span>{line.text}</span>
            </li>
          ))}
        </ul>
        {run.warnings.length > 0 && (
          <details className="quiet-details">
            <summary>{run.warnings.length} data note(s) carried over from framing and training</summary>
            <IssueList issues={run.warnings} />
          </details>
        )}
      </section>

      <div className="two-col">
        <section className="block">
          <SectionHead
            index="04.2"
            title={<Term k="confusion_matrix">Confusion matrix</Term>}
            kicker="Where the right and wrong answers land."
            aside={
              <Segmented<"counts" | "rows">
                label="Matrix values"
                value={normalise ? "rows" : "counts"}
                onChange={(v) => setNormalise(v === "rows")}
                options={[
                  { value: "counts", label: "Counts" },
                  { value: "rows", label: "% of actual" },
                ]}
              />
            }
          />
          <ConfusionMatrix labels={ev.confusion.labels} matrix={ev.confusion.matrix} normalise={normalise} selected={cellSel} onSelect={setCellSel} />
        </section>

        <section className="block">
          <SectionHead index="04.3" title="Per class" kicker="Each class scored on its own." />
          <div className="table-wrap">
            <table className="class-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>
                    <Term k="precision">Precision</Term>
                  </th>
                  <th>
                    <Term k="recall">Recall</Term>
                  </th>
                  <th>
                    <Term k="f1">F1</Term>
                  </th>
                  <th className="num">Test rows</th>
                </tr>
              </thead>
              <tbody>
                {ev.per_class.map((c) => (
                  <tr key={c.label}>
                    <td>
                      <i className="swatch" style={{ background: classColour(run.labels, c.label) }} /> {c.label}
                    </td>
                    {(["precision", "recall", "f1"] as const).map((k) => (
                      <td key={k}>
                        <span className="metric-cell">
                          <Meter value={c[k]} tone={c[k] < 0.6 ? "signal" : "ink"} />
                          {score(c[k], 2)}
                        </span>
                      </td>
                    ))}
                    <td className="num">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <div className="two-col">
        <section className="block">
          <SectionHead
            index="04.4"
            title={<>Fit check · <Term k="overfitting">overfitting</Term></>}
            kicker="Macro F1 on rows the model trained on, on cross-validation folds, and on the test split."
          />
          <FitCheck train={ev.train_metrics.f1_macro} cv={run.cross_validation?.scores ?? []} test={m.f1_macro} />
          <p className="caption">
            {ev.train_metrics.f1_macro - m.f1_macro > 0.1
              ? "Training score is far above test score: the model fits training rows much better than new ones."
              : "Training, cross-validation and test scores sit close together: the model generalises about as well as it fits."}
          </p>
        </section>
        {ev.roc ? (
          <section className="block">
            <SectionHead index="04.5" title={<Term k="roc_auc">ROC curve</Term>} kicker={`Treating “${ev.roc.positive_label}” as the positive class. AUC ${ev.roc.auc.toFixed(3)}.`} />
            <RocCurve points={ev.roc.points} auc={ev.roc.auc} positive={ev.roc.positive_label} />
          </section>
        ) : (
          <section className="block">
            <SectionHead index="04.5" title="Training rows vs test rows" />
            <p className="muted">
              {ev.n_train.toLocaleString()} rows trained the model; {ev.n_test.toLocaleString()} were held back. ROC curves are shown for two-class
              problems; with {run.labels.length} classes the macro one-vs-rest AUC is {score(m.roc_auc)}.
            </p>
          </section>
        )}
      </div>

      <TestRows run={run} cellSel={cellSel} onClearCell={() => setCellSel(null)} />
    </>
  );
}

function TestRows({ run, cellSel, onClearCell }: { run: ModelSummary; cellSel: CellRef | null; onClearCell: () => void }) {
  const dispatch = useDispatch();
  const [filter, setFilter] = useState<"all" | "wrong" | "unsure">("all");
  const labels = run.evaluation.confusion.labels;
  const preview = run.importance.permutation.items.slice(0, 3).map((i) => i.feature);

  const rows = useMemo(() => {
    let rs = run.test_examples.rows;
    if (cellSel) rs = rs.filter((r) => r.actual === labels[cellSel.actual] && r.predicted === labels[cellSel.predicted]);
    if (filter === "wrong") rs = rs.filter((r) => r.actual !== r.predicted);
    if (filter === "unsure") rs = [...rs].sort((a, b) => a.confidence - b.confidence).slice(0, 40);
    return rs.slice(0, 120);
  }, [run, cellSel, filter, labels]);

  return (
    <section className="block">
      <SectionHead
        index="04.6"
        title="Test rows"
        kicker={`Individual held-out rows with the model’s answer${run.test_examples.truncated ? ` (a sample of ${run.test_examples.rows.length} of ${run.test_examples.total})` : ""}.`}
        aside={
          <Segmented<"all" | "wrong" | "unsure">
            label="Filter test rows"
            value={filter}
            onChange={setFilter}
            options={[
              { value: "all", label: "All" },
              { value: "wrong", label: "Mistakes" },
              { value: "unsure", label: "Least confident" },
            ]}
          />
        }
      />
      {cellSel && (
        <p className="filter-chip-row">
          <span className="filter-chip">
            actual “{labels[cellSel.actual]}” → predicted “{labels[cellSel.predicted]}”
            <button type="button" onClick={onClearCell} aria-label="Clear matrix filter">
              ×
            </button>
          </span>
        </p>
      )}
      <div className="table-wrap">
        <table className="test-table">
          <thead>
            <tr>
              <th className="idx">row</th>
              <th>Actual</th>
              <th>Predicted</th>
              <th>Confidence</th>
              {preview.map((f) => (
                <th key={f}>{f}</th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const wrong = r.actual !== r.predicted;
              return (
                <tr key={r.row} className={wrong ? "is-wrong" : ""}>
                  <td className="idx">{r.row + 1}</td>
                  <td>{r.actual}</td>
                  <td>
                    {wrong && <Glyph level="error" />} {r.predicted}
                  </td>
                  <td>
                    <span className="metric-cell">
                      <Meter value={r.confidence} tone={r.confidence < 0.6 ? "amber" : "ink"} />
                      {pct(r.confidence)}
                    </span>
                  </td>
                  {preview.map((f) => (
                    <td key={f} className={typeof r.values[f] === "number" ? "num" : ""}>
                      {r.values[f] == null ? <span className="muted">∅</span> : cell(r.values[f])}
                    </td>
                  ))}
                  <td>
                    <Button size="sm" variant="quiet" onClick={() => dispatch({ type: "probe/seed", values: r.values, actual: r.actual })}>
                      Probe →
                    </Button>
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td colSpan={5 + preview.length} className="empty-row">
                  {filter === "wrong" || cellSel ? "No test rows here — nothing was misclassified this way." : "No test rows."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
