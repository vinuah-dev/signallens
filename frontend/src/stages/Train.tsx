import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import type { AlgorithmSpec, ModelSummary, ParamSpec } from "../api/types";
import { RunCompare } from "../charts/RunCompare";
import { Button, Notice, Pulse, SectionHead, Term, Tip, Toggle } from "../components/ui";
import { pct, score } from "../lib/format";
import { runColour } from "../lib/palette";
import { useDispatch, useMeta, useStore } from "../state/store";

type ParamValues = Record<string, number | null>;
const defaults = (a: AlgorithmSpec): ParamValues => Object.fromEntries(a.params.map((p) => [p.name, p.default]));

export function TrainStage() {
  const { dataset, selection, frameReport, runs, activeRunId } = useStore();
  const { algorithms, limits } = useMeta();
  const dispatch = useDispatch();

  const [algo, setAlgo] = useState(algorithms[0].key);
  const [params, setParams] = useState<Record<string, ParamValues>>(() => Object.fromEntries(algorithms.map((a) => [a.key, defaults(a)])));
  const [testSize, setTestSize] = useState(limits.test_size.default);
  const [balance, setBalance] = useState(Boolean(frameReport?.imbalanced));
  const [seed, setSeed] = useState(42);
  const [job, setJob] = useState<{ label: string; step?: string; started: number } | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<ApiError | null>(null);
  const [justTrained, setJustTrained] = useState<ModelSummary | null>(null);

  useEffect(() => {
    if (!job) return;
    const t = setInterval(() => setElapsed((Date.now() - job.started) / 1000), 100);
    return () => clearInterval(t);
  }, [job]);

  const spec = algorithms.find((a) => a.key === algo)!;

  const runOne = async (a: AlgorithmSpec, p: ParamValues) => {
    const run = await api.train(dataset!.id, selection!, { algorithm: a.key, params: p, test_size: testSize, balance, random_state: seed });
    dispatch({ type: "run/added", run });
    return run;
  };

  const handle = async (label: string, work: (setStep: (s: string) => void) => Promise<ModelSummary | undefined>) => {
    setError(null);
    setJustTrained(null);
    setElapsed(0);
    const started = Date.now();
    setJob({ label, started });
    try {
      const run = await work((step) => setJob({ label, step, started }));
      if (run) setJustTrained(run);
    } catch (e) {
      const err = e instanceof ApiError ? e : new ApiError("unknown", String(e), 0);
      if (err.isGone) dispatch({ type: "lost" });
      setError(err);
    } finally {
      setJob(null);
    }
  };

  const trainSelected = () => handle(spec.label, () => runOne(spec, params[algo]));
  const trainAll = () =>
    handle("All algorithms", async (setStep) => {
      let last: ModelSummary | undefined;
      for (const [i, a] of algorithms.entries()) {
        setStep(`${i + 1} of ${algorithms.length}: ${a.label}`);
        last = await runOne(a, defaults(a));
      }
      return last;
    });

  const splitKeys = new Set(runs.map((r) => r.split_key));

  return (
    <div className="stage stage-train">
      <div className="intro">
        <h1 className="display display-sm">Fit a model — then another.</h1>
        <p className="lede">
          Every run uses a <Term k="stratified">stratified</Term> train/<Term k="test_split">test split</Term>. Preprocessing (filling gaps, scaling,
          encoding categories) is learned from the training rows only, then applied unchanged to the test rows.
        </p>
      </div>

      <div className="train-layout">
        <section className="block train-config">
          <SectionHead index="03.1" title="Algorithm" />
          <div className="algo-list" role="radiogroup" aria-label="Algorithm">
            {algorithms.map((a) => (
              <label key={a.key} className={`algo ${algo === a.key ? "is-on" : ""}`}>
                <input type="radio" name="algo" checked={algo === a.key} onChange={() => setAlgo(a.key)} />
                <span className="algo-name">{a.label}</span>
                <span className="algo-summary">{a.summary}</span>
                {algo === a.key && <span className="algo-strengths">{a.strengths}</span>}
              </label>
            ))}
          </div>

          {spec.params.length > 0 && (
            <>
              <h3 className="mini-head">Settings</h3>
              <div className="params">
                {spec.params.map((p) => (
                  <ParamField
                    key={p.name}
                    spec={p}
                    value={params[algo][p.name]}
                    onChange={(v) => setParams({ ...params, [algo]: { ...params[algo], [p.name]: v } })}
                  />
                ))}
                <Button size="sm" variant="quiet" onClick={() => setParams({ ...params, [algo]: defaults(spec) })}>
                  Reset to defaults
                </Button>
              </div>
            </>
          )}

          <h3 className="mini-head">Evaluation</h3>
          <div className="params">
            <div className="param">
              <div className="param-head">
                <label htmlFor="test-size">Held-out test share</label>
                <span className="mono-meta">{pct(testSize)}</span>
              </div>
              <input
                id="test-size"
                type="range"
                min={limits.test_size.min}
                max={limits.test_size.max}
                step={0.05}
                value={testSize}
                onChange={(e) => setTestSize(Number(e.target.value))}
              />
              <p className="param-help">
                {frameReport && `${Math.ceil(frameReport.rows_used * testSize).toLocaleString()} rows are hidden from training and used only for scoring.`}
              </p>
            </div>
            <Toggle checked={balance} onChange={setBalance}>
              <Term k="balancing">Balance classes</Term>
              {frameReport?.imbalanced && <span className="flag flag-warn">recommended</span>}
            </Toggle>
            <div className="param param-inline">
              <label htmlFor="seed">Random seed</label>
              <input id="seed" className="num-input" type="number" value={seed} onChange={(e) => setSeed(Math.trunc(Number(e.target.value) || 0))} />
              <Tip content="Fixes the split and any randomness in the model. Runs with the same seed and test share are scored on the exact same test rows.">
                <span className="muted small">why?</span>
              </Tip>
            </div>
          </div>

          <div className="btn-row train-actions">
            <Button variant="accent" size="lg" onClick={trainSelected} busy={job?.label === spec.label} disabled={job != null}>
              Train {spec.label.toLowerCase()}
            </Button>
            <Button onClick={trainAll} disabled={job != null}>
              Train all {algorithms.length} with defaults
            </Button>
          </div>

          {job && (
            <div className="training" role="status">
              <Pulse label="Training" />
              <div>
                <strong>Training {job.step ?? job.label}…</strong>
                <p className="muted small">
                  split → fit → score test rows → cross-validate → shuffle-test each feature · {elapsed.toFixed(1)} s
                </p>
              </div>
            </div>
          )}
          {error && (
            <Notice level="error" title={error.code === "training_failed" ? "Training failed" : "Can’t train with these settings"}>
              {error.message}
            </Notice>
          )}
          {justTrained && !job && (
            <Notice
              level="good"
              title={`${justTrained.algorithm_label} trained in ${justTrained.timing.total_seconds.toFixed(1)} s`}
              action={
                <Button size="sm" variant="primary" onClick={() => dispatch({ type: "stage", stage: "inspect" })}>
                  Inspect →
                </Button>
              }
            >
              Accuracy {pct(justTrained.evaluation.metrics.accuracy, 1)} on {justTrained.evaluation.n_test} unseen rows (baseline{" "}
              {pct(justTrained.evaluation.baseline.accuracy, 1)}).
            </Notice>
          )}
        </section>

        <section className="block train-runs">
          <SectionHead index="03.2" title="Runs" kicker="Select a run to inspect, explain and probe it." />
          {runs.length === 0 ? (
            <div className="runs-empty">
              <p>
                No runs yet. Train one model, or press <strong>Train all</strong> to compare every algorithm on the same split.
              </p>
            </div>
          ) : (
            <>
              <RunsTable runs={runs} activeId={activeRunId} currentTarget={selection?.target} />
              {runs.length > 1 && (
                <>
                  <h3 className="mini-head">Side by side</h3>
                  <RunCompare runs={runs} activeId={activeRunId} onPick={(id) => dispatch({ type: "run/activate", id })} />
                  <p className="caption">
                    Dots are runs (colours match the table). The grey tick on each line is the naive baseline. Further right is better.
                  </p>
                  {splitKeys.size > 1 && (
                    <Notice level="warn">
                      Some runs used a different target, features, test share or seed, so they were scored on different test rows. Compare
                      runs that share all four.
                    </Notice>
                  )}
                </>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

function paramSummary(r: ModelSummary): string {
  return Object.entries(r.params)
    .map(([k, v]) => `${k}=${v ?? "∞"}`)
    .join(" ");
}

function RunsTable({ runs, activeId, currentTarget }: { runs: ModelSummary[]; activeId: string | null; currentTarget?: string }) {
  const dispatch = useDispatch();
  const best = (key: "accuracy" | "balanced_accuracy" | "f1_macro" | "roc_auc") =>
    Math.max(...runs.map((r) => r.evaluation.metrics[key] ?? -1));
  const cell = (r: ModelSummary, key: "accuracy" | "balanced_accuracy" | "f1_macro" | "roc_auc") => {
    const v = r.evaluation.metrics[key];
    return <td className={`num ${runs.length > 1 && v === best(key) ? "is-best" : ""}`}>{score(v)}</td>;
  };
  return (
    <div className="table-wrap">
      <table className="runs-table">
        <thead>
          <tr>
            <th />
            <th>Model</th>
            <th className="num">
              <Term k="accuracy">Acc.</Term>
            </th>
            <th className="num">
              <Term k="balanced_accuracy">Bal. acc.</Term>
            </th>
            <th className="num">
              <Term k="f1">Macro F1</Term>
            </th>
            <th className="num">
              <Term k="roc_auc">AUC</Term>
            </th>
            <th className="num">
              <Term k="cross_validation">CV F1</Term>
            </th>
            <th />
          </tr>
        </thead>
        <tbody>
          {runs.map((r, i) => (
            <tr key={r.id} className={r.id === activeId ? "is-active" : ""} onClick={() => dispatch({ type: "run/activate", id: r.id })}>
              <td>
                <span className="run-chip" style={{ background: runColour(i) }}>
                  #{i + 1}
                </span>
              </td>
              <td>
                <strong>{r.algorithm_label}</strong>
                <span className="run-meta">
                  {paramSummary(r)} · test {pct(r.test_size)} · seed {r.random_state}
                  {r.balance && " · balanced"}
                  {currentTarget && r.target !== currentTarget && <span className="flag flag-warn">target: {r.target}</span>}
                </span>
              </td>
              {cell(r, "accuracy")}
              {cell(r, "balanced_accuracy")}
              {cell(r, "f1_macro")}
              {cell(r, "roc_auc")}
              <td className="num">
                {r.cross_validation?.mean != null ? (
                  <>
                    {score(r.cross_validation.mean)}
                    <span className="muted"> ±{score(r.cross_validation.std, 2)}</span>
                  </>
                ) : (
                  <Tip content={r.cross_validation?.skipped ?? "Not run"}>
                    <span className="muted">—</span>
                  </Tip>
                )}
              </td>
              <td>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Remove run ${i + 1}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    dispatch({ type: "run/removed", id: r.id });
                  }}
                >
                  ×
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* Slider + number box. Log-scaled params (e.g. C) slide in log space. */
function ParamField({ spec, value, onChange }: { spec: ParamSpec; value: number | null; onChange: (v: number | null) => void }) {
  const [text, setText] = useState(value == null ? "" : String(value));
  useEffect(() => setText(value == null ? "" : String(value)), [value]);
  const toSlider = (v: number) => (spec.log ? Math.log10(v) : v);
  const fromSlider = (s: number) => {
    const v = spec.log ? 10 ** s : s;
    return spec.type === "int" ? Math.round(v) : Number(v.toPrecision(3));
  };
  const lo = toSlider(spec.min);
  const hi = toSlider(spec.max);
  const invalid = text !== "" && (Number.isNaN(Number(text)) || Number(text) < spec.min || Number(text) > spec.max);

  return (
    <div className="param">
      <div className="param-head">
        <label htmlFor={`p-${spec.name}`}>{spec.label}</label>
        {spec.nullable && (
          <label className="nolimit">
            <input
              type="checkbox"
              checked={value == null}
              onChange={(e) => onChange(e.target.checked ? null : (spec.default ?? Math.round((spec.min + spec.max) / 4)))}
            />
            no limit
          </label>
        )}
      </div>
      <div className="param-controls">
        <input
          type="range"
          min={lo}
          max={hi}
          step={spec.log ? (hi - lo) / 200 : (spec.step ?? (spec.max - spec.min) / 100)}
          value={value == null ? hi : toSlider(value)}
          disabled={value == null}
          onChange={(e) => onChange(fromSlider(Number(e.target.value)))}
          aria-label={spec.label}
        />
        <input
          id={`p-${spec.name}`}
          className={`num-input ${invalid ? "is-invalid" : ""}`}
          type="text"
          inputMode="decimal"
          value={value == null ? "∞" : text}
          disabled={value == null}
          onChange={(e) => {
            setText(e.target.value);
            const v = Number(e.target.value);
            if (e.target.value !== "" && !Number.isNaN(v) && v >= spec.min && v <= spec.max) onChange(spec.type === "int" ? Math.round(v) : v);
          }}
          aria-invalid={invalid}
        />
      </div>
      <p className={`param-help ${invalid ? "is-error" : ""}`}>{invalid ? `Enter a value between ${spec.min} and ${spec.max}.` : spec.help}</p>
    </div>
  );
}
