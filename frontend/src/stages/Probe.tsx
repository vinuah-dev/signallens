import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type { Cell, CategoricalFeature, FeatureSpec, ModelSummary, NumericFeature, Prediction, WhatIf } from "../api/types";
import { ProbabilityBars, SensitivityChart, WhatIfChart } from "../charts/Probe";
import { Button, Glyph, Notice, Pulse, SectionHead, Term } from "../components/ui";
import { cell, num, pct } from "../lib/format";
import { classColour } from "../lib/palette";
import { checkNumber, clamp, type FieldCheck } from "../lib/validate";
import { useActiveRun, useDispatch, useStore, type State } from "../state/store";
import { RunSwitcher } from "./RunSwitcher";

type Raw = Record<string, string>;
interface Scenario {
  id: number;
  name: string;
  inputs: Record<string, Cell>;
  result: Prediction;
}

const typicalRaw = (schema: FeatureSpec[]): Raw =>
  Object.fromEntries(schema.map((s) => [s.name, s.kind === "numeric" ? String(Number(s.median.toPrecision(6))) : (s.mode ?? "")]));

const toRaw = (schema: FeatureSpec[], values: Record<string, Cell>): Raw =>
  Object.fromEntries(schema.map((s) => [s.name, values[s.name] == null ? "" : String(values[s.name])]));

export function ProbeStage() {
  const run = useActiveRun()!;
  const { probeSeed } = useStore();
  return (
    <div className="stage stage-probe">
      <RunSwitcher />
      <Probe key={run.id} run={run} seed={probeSeed} />
    </div>
  );
}

function Probe({ run, seed }: { run: ModelSummary; seed: State["probeSeed"] }) {
  const dispatch = useDispatch();
  const schema = useMemo(() => {
    const rank = new Map(run.importance.permutation.items.map((i, idx) => [i.feature, idx]));
    return [...run.schema].sort((a, b) => (rank.get(a.name) ?? 99) - (rank.get(b.name) ?? 99));
  }, [run]);
  const [raw, setRaw] = useState<Raw>(() => (seed ? toRaw(schema, seed.values) : typicalRaw(schema)));
  const [actual, setActual] = useState<string | null>(seed?.actual ?? null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [whatIfFeature, setWhatIfFeature] = useState<string>(schema[0]?.name ?? "");
  const [whatIf, setWhatIf] = useState<WhatIf | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);

  // A row sent from the Inspect stage replaces the current inputs.
  useEffect(() => {
    if (seed) {
      setRaw(toRaw(schema, seed.values));
      setActual(seed.actual ?? null);
    }
  }, [seed?.nonce]);

  const checks = useMemo(() => {
    const out: Record<string, FieldCheck> = {};
    for (const s of schema) {
      const v = raw[s.name] ?? "";
      out[s.name] = s.kind === "numeric" ? checkNumber(v, s) : v === "" ? { state: "empty" } : { state: "ok", value: 0 };
    }
    return out;
  }, [raw, schema]);
  const invalid = Object.entries(checks).filter(([, c]) => c.state === "error").map(([k]) => k);

  const payload = useMemo(() => {
    const row: Record<string, Cell> = {};
    for (const s of schema) {
      const c = checks[s.name];
      row[s.name] = c.state === "empty" ? null : s.kind === "numeric" ? (c.state === "ok" ? c.value : null) : raw[s.name];
    }
    return row;
  }, [checks, raw, schema]);
  const payloadKey = JSON.stringify(payload);

  // Live prediction (debounced) whenever the inputs are valid.
  useEffect(() => {
    if (invalid.length) return;
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await api.predict(run.id, [payload], ctrl.signal);
        setResult(res.results[0]);
        setError(null);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        const err = e instanceof ApiError ? e : new ApiError("unknown", String(e), 0);
        if (err.isGone) dispatch({ type: "lost" });
        setError(err);
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 220);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [payloadKey, run.id]);

  useEffect(() => {
    if (invalid.length || !whatIfFeature) return;
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api
        .whatIf(run.id, payload, whatIfFeature, ctrl.signal)
        .then(setWhatIf)
        .catch(() => !ctrl.signal.aborted && setWhatIf(null));
    }, 300);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [payloadKey, whatIfFeature, run.id]);

  const set = (name: string, value: string) => {
    setRaw((r) => ({ ...r, [name]: value }));
    setActual(null);
  };
  const randomTestRow = () => {
    const rows = run.test_examples.rows;
    const r = rows[Math.floor(Math.random() * rows.length)];
    setRaw(toRaw(schema, r.values));
    setActual(r.actual);
  };
  const pin = () => {
    if (!result) return;
    setScenarios((s) => [...s, { id: Date.now(), name: String.fromCharCode(65 + (s.length % 26)), inputs: payload, result }].slice(-6));
  };

  const predictedColour = result ? classColour(run.labels, result.predicted) : undefined;
  const warnings = result?.notes.filter((n) => n.level === "warn") ?? [];
  const infos = result?.notes.filter((n) => n.level === "info") ?? [];

  return (
    <>
      <div className="intro">
        <h1 className="display display-sm">Ask the model.</h1>
        <p className="lede">
          Change any input and the prediction updates live. Fields are ordered by how much the model relies on them. Leave a field empty to see how the model
          handles a missing value.
        </p>
      </div>

      <div className="probe-layout">
        <section className="block probe-form" aria-label="Feature inputs">
          <div className="btn-row probe-tools">
            <Button size="sm" onClick={() => (setRaw(typicalRaw(schema)), setActual(null))}>
              Typical values
            </Button>
            <Button size="sm" onClick={randomTestRow}>
              Random test row
            </Button>
            <Button size="sm" variant="quiet" onClick={() => (setRaw(Object.fromEntries(schema.map((s) => [s.name, ""]))), setActual(null))}>
              Clear all
            </Button>
          </div>
          {schema.map((s) =>
            s.kind === "numeric" ? (
              <NumberField key={s.name} spec={s} raw={raw[s.name] ?? ""} check={checks[s.name]} onChange={(v) => set(s.name, v)} />
            ) : (
              <CategoryField key={s.name} spec={s} raw={raw[s.name] ?? ""} onChange={(v) => set(s.name, v)} />
            ),
          )}
        </section>

        <section className="probe-result" aria-live="polite">
          {invalid.length > 0 && (
            <Notice level="error" title="Fix the highlighted inputs">
              {invalid.join(", ")} {invalid.length === 1 ? "isn’t" : "aren’t"} valid numbers, so there’s nothing to predict yet.
            </Notice>
          )}
          {error && <Notice level="error" title="Prediction failed">{error.message}</Notice>}
          {!result && !error && invalid.length === 0 && (
            <div className="probe-waiting">
              <Pulse label="Predicting" /> asking the model…
            </div>
          )}
          {result && (
            <div className={`answer ${loading ? "is-stale" : ""}`}>
              <span className="eyebrow">The model says</span>
              <div className="answer-main">
                <span className="answer-label" style={{ color: predictedColour }}>
                  {result.predicted}
                </span>
                <span className="answer-conf">{pct(result.confidence, 1)} probability</span>
                {loading && <Pulse small />}
              </div>
              {actual != null && (
                <p className={`answer-actual ${actual === result.predicted ? "is-right" : "is-wrong"}`}>
                  <Glyph level={actual === result.predicted ? "good" : "error"} />
                  <span>
                    This test row’s real label is <strong>{actual}</strong>.
                  </span>
                </p>
              )}
              <ProbabilityBars items={result.probabilities} labels={run.labels} />
              {warnings.length > 0 && (
                <ul className="issue-list">
                  {warnings.map((n) => (
                    <li key={n.feature + n.code} className="issue issue-warn">
                      <Glyph level="warn" />
                      <span className="issue-text">{n.message}</span>
                    </li>
                  ))}
                </ul>
              )}
              {infos.length > 0 && (
                <details className="quiet-details">
                  <summary>{infos.length} input note(s)</summary>
                  <ul className="issue-list">
                    {infos.map((n) => (
                      <li key={n.feature + n.code} className="issue issue-info">
                        <Glyph level="info" />
                        <span className="issue-text">{n.message}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              <div className="btn-row">
                <Button variant="primary" size="sm" onClick={pin} disabled={loading}>
                  Pin as scenario {String.fromCharCode(65 + (scenarios.length % 26))}
                </Button>
              </div>
            </div>
          )}
        </section>
      </div>

      {result?.sensitivity && (
        <div className="two-col">
          <section className="block">
            <SectionHead
              index="06.1"
              title={<Term k="sensitivity">What moved this prediction</Term>}
              kicker={`Change in P(${result.predicted}) if each feature were reset to a typical value (features already typical show 0). Model behaviour, not causes.`}
            />
            <SensitivityChart items={result.sensitivity} predicted={result.predicted} colour={predictedColour!} />
          </section>
          <section className="block">
            <SectionHead
              index="06.2"
              title="What if…"
              kicker="Sweep one feature across its training range; everything else stays as entered."
              aside={
                <select className="compact-select" value={whatIfFeature} onChange={(e) => setWhatIfFeature(e.target.value)} aria-label="Feature to sweep">
                  {schema.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name}
                    </option>
                  ))}
                </select>
              }
            />
            {whatIf && whatIf.feature === whatIfFeature ? (
              <>
                <WhatIfChart data={whatIf} labels={run.labels} />
                <ul className="legend legend-inline">
                  {run.labels.map((l) => (
                    <li key={l}>
                      <i style={{ background: classColour(run.labels, l) }} />
                      <span className="legend-label">{l}</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <div className="probe-waiting">
                <Pulse small /> sweeping {whatIfFeature}…
              </div>
            )}
          </section>
        </div>
      )}

      {result && (
        // Narrow screens stack the form above the answer; this dock keeps the
        // live prediction in view while editing inputs.
        <div className="probe-dock" aria-hidden="true">
          <span className="eyebrow">model says</span>
          <strong style={{ color: predictedColour }}>{result.predicted}</strong>
          <span className="mono-meta">{pct(result.confidence, 1)}</span>
          {warnings.length > 0 && <Glyph level="warn" />}
          {loading && <Pulse small />}
        </div>
      )}

      <Scenarios scenarios={scenarios} schema={schema} labels={run.labels} onRemove={(id) => setScenarios((s) => s.filter((x) => x.id !== id))} />
    </>
  );
}

function NumberField({ spec, raw, check, onChange }: { spec: NumericFeature; raw: string; check: FieldCheck; onChange: (v: string) => void }) {
  const id = `f-${spec.name}`;
  const value = check.state === "ok" ? check.value : spec.median;
  const step = spec.integer ? 1 : Number(((spec.max - spec.min) / 200).toPrecision(2)) || 0.01;
  const out = check.state === "ok" && (check.value < spec.min || check.value > spec.max);
  return (
    <div className={`field ${check.state === "error" ? "is-error" : ""} ${check.state === "ok" && check.warning ? "is-warn" : ""}`}>
      <div className="field-head">
        <label htmlFor={id}>{spec.name}</label>
        <span className="field-hint">
          {num(spec.min)} – {num(spec.max)} · typical {num(spec.median)}
        </span>
      </div>
      <div className="field-controls">
        <input
          type="range"
          min={spec.min}
          max={spec.max}
          step={step}
          value={clamp(value, spec.min, spec.max)}
          onChange={(e) => onChange(e.target.value)}
          aria-label={`${spec.name} slider`}
          tabIndex={-1}
        />
        <input
          id={id}
          className="num-input"
          type="text"
          inputMode="decimal"
          placeholder="empty"
          value={raw}
          onChange={(e) => onChange(e.target.value)}
          aria-invalid={check.state === "error"}
        />
      </div>
      {check.state === "error" && <p className="field-msg is-error">{check.message}</p>}
      {check.state === "ok" && check.warning && (
        <p className="field-msg is-warn">
          {check.warning}
          {out && (
            <button type="button" className="link-btn" onClick={() => onChange(String(clamp(check.value, spec.min, spec.max)))}>
              clip to range
            </button>
          )}
        </p>
      )}
      {check.state === "empty" && <p className="field-msg">empty — will be treated as missing</p>}
    </div>
  );
}

function CategoryField({ spec, raw, onChange }: { spec: CategoricalFeature; raw: string; onChange: (v: string) => void }) {
  const id = `f-${spec.name}`;
  const known = spec.categories.some((c) => c.value === raw);
  return (
    <div className={`field ${raw && !known ? "is-warn" : ""}`}>
      <div className="field-head">
        <label htmlFor={id}>{spec.name}</label>
        <span className="field-hint">
          {spec.n_categories} categories · most common {spec.mode}
        </span>
      </div>
      <select id={id} value={known || raw === "" ? raw : "__other__"} onChange={(e) => onChange(e.target.value === "__other__" ? raw : e.target.value)}>
        <option value="">(empty — missing)</option>
        {spec.categories.map((c) => (
          <option key={c.value} value={c.value}>
            {c.value} ({c.count})
          </option>
        ))}
        {raw && !known && <option value="__other__">{raw} (unseen)</option>}
      </select>
      {raw && !known && <p className="field-msg is-warn">Never seen in training — treated as unknown.</p>}
    </div>
  );
}

function Scenarios({ scenarios, schema, labels, onRemove }: { scenarios: Scenario[]; schema: FeatureSpec[]; labels: string[]; onRemove: (id: number) => void }) {
  if (scenarios.length === 0) {
    return (
      <section className="block">
        <SectionHead index="06.3" title="Scenarios" kicker="Pin predictions to compare different input combinations side by side." />
        <p className="muted">Nothing pinned yet. Adjust the inputs, press “Pin as scenario”, change something, pin again.</p>
      </section>
    );
  }
  const base = scenarios[0];
  return (
    <section className="block">
      <SectionHead index="06.3" title="Scenarios" kicker={`Differences from scenario ${base.name} are highlighted.`} />
      <div className="table-wrap">
        <table className="scenario-table">
          <thead>
            <tr>
              <th>Feature</th>
              {scenarios.map((s) => (
                <th key={s.id}>
                  <span className="scenario-name">{s.name}</span>
                  <button type="button" className="icon-btn" onClick={() => onRemove(s.id)} aria-label={`Remove scenario ${s.name}`}>
                    ×
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="scenario-outcome">
              <th>prediction</th>
              {scenarios.map((s) => (
                <td key={s.id} style={{ color: classColour(labels, s.result.predicted) }} className={s.result.predicted !== base.result.predicted ? "is-diff" : ""}>
                  <strong>{s.result.predicted}</strong> <span className="muted">{pct(s.result.confidence)}</span>
                </td>
              ))}
            </tr>
            {labels.length <= 6 &&
              labels.map((l) => (
                <tr key={l} className="scenario-prob">
                  <th>P({l})</th>
                  {scenarios.map((s) => {
                    const p = s.result.probabilities.find((x) => x.label === l)?.probability ?? 0;
                    return (
                      <td key={s.id}>
                        <span className="metric-cell">
                          <span className="meter">
                            <span style={{ width: `${p * 100}%`, background: classColour(labels, l) }} />
                          </span>
                          {pct(p)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            {schema.map((f) => (
              <tr key={f.name}>
                <th>{f.name}</th>
                {scenarios.map((s) => {
                  const v = s.inputs[f.name];
                  const diff = String(v) !== String(base.inputs[f.name]);
                  return (
                    <td key={s.id} className={`${diff ? "is-diff" : ""} ${typeof v === "number" ? "num" : ""}`}>
                      {v == null ? <span className="muted">∅</span> : cell(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
