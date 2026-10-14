import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type { ColumnProfile, FrameSelection, Issue, Profile } from "../api/types";
import { Sparkbars } from "../charts/Histogram";
import { Button, IssueList, KindBadge, Meter, Notice, Pulse, SectionHead, Term, Tip, Toggle } from "../components/ui";
import { pct } from "../lib/format";
import { classColour } from "../lib/palette";
import { useDispatch, useMeta, useStore } from "../state/store";

const USABLE = new Set(["numeric", "categorical"]);
const DROPPABLE = new Set(["leakage_suspect", "identifier_feature", "text_feature", "constant_feature", "mostly_missing"]);

function targetProblem(c: ColumnProfile, maxClasses: number): string | null {
  if (c.target_candidate) return null;
  if (c.is_constant) return "only one value";
  if (c.kind === "identifier") return "unique per row — an ID";
  if (c.kind === "text") return "free text";
  if (c.kind === "datetime") return "a date";
  if (c.unique > maxClasses) return `${c.unique.toLocaleString()} values — too many classes`;
  if (c.kind === "numeric") return `continuous number (${c.unique.toLocaleString()} values)`;
  return "not a set of classes";
}

export function defaultSelection(profile: Profile, target?: string | null): FrameSelection {
  const t = target ?? profile.suggested_target ?? profile.column_profiles.find((c) => c.target_candidate)?.name ?? "";
  return {
    target: t,
    features: profile.column_profiles.filter((c) => c.name !== t && USABLE.has(c.kind) && !c.is_constant).map((c) => c.name),
    categorical_overrides: [],
  };
}

export function FrameStage() {
  const { dataset, selection, frameReport } = useStore();
  const { limits } = useMeta();
  const dispatch = useDispatch();
  const profile = dataset!.profile;
  const sel = selection ?? defaultSelection(profile);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    if (!selection) dispatch({ type: "selection", selection: sel });
  }, []);

  // Live validation: every change to the selection is re-checked by the server
  // (debounced), which returns class balance, warnings and association scores.
  useEffect(() => {
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setChecking(true);
      try {
        const report = await api.checkFrame(dataset!.id, sel, ctrl.signal);
        dispatch({ type: "frame/report", report });
        setError(null);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        const err = e instanceof ApiError ? e : new ApiError("unknown", String(e), 0);
        if (err.isGone) dispatch({ type: "lost" });
        dispatch({ type: "frame/report", report: null });
        setError(err);
      } finally {
        if (!ctrl.signal.aborted) setChecking(false);
      }
    }, 280);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [dataset, sel.target, sel.features.join("\u0000"), sel.categorical_overrides.join("\u0000"), dispatch]);

  const update = (next: Partial<FrameSelection>) => {
    const merged = { ...sel, ...next };
    merged.categorical_overrides = merged.categorical_overrides.filter((o) => merged.features.includes(o));
    dispatch({ type: "selection", selection: merged });
  };
  const setTarget = (t: string) => update({ target: t, features: sel.features.filter((f) => f !== t) });
  const toggleFeature = (f: string, on: boolean) =>
    update({ features: on ? profile.column_profiles.map((c) => c.name).filter((n) => n === f || sel.features.includes(n)) : sel.features.filter((x) => x !== f) });
  const toggleOverride = (f: string, on: boolean) =>
    update({ categorical_overrides: on ? [...sel.categorical_overrides, f] : sel.categorical_overrides.filter((x) => x !== f) });

  const report = !error && frameReport?.target === sel.target ? frameReport : null;
  const issuesByColumn = useMemo(() => {
    const m = new Map<string, Issue[]>();
    report?.issues.forEach((i) => i.column && m.set(i.column, [...(m.get(i.column) ?? []), i]));
    return m;
  }, [report]);
  const association = new Map(report?.association.map((a) => [a.feature, a.score]));
  const flagged = [...new Set(report?.issues.filter((i) => i.column && DROPPABLE.has(i.code) && i.level === "warn").map((i) => i.column!))].filter((f) =>
    sel.features.includes(f),
  );

  const targets = profile.column_profiles.filter((c) => c.target_candidate);
  const nonTargets = profile.column_profiles.filter((c) => !c.target_candidate);
  const featureCols = profile.column_profiles.filter((c) => c.name !== sel.target);

  return (
    <div className="stage stage-frame">
      <div className="intro">
        <h1 className="display display-sm">What should the model predict?</h1>
        <p className="lede">
          Choose one <strong>target</strong> column (the answer) and the <strong>features</strong> the model may look at. Everything is
          re-checked as you change it.
        </p>
      </div>

      <div className="frame-layout">
        <div className="frame-main">
          <section className="block">
            <SectionHead index="02.1" title="Target" kicker="A column with a handful of repeated categories." />
            {targets.length === 0 && (
              <Notice level="error" title="No column can be a target">
                Every column is either unique per row, continuous, or constant. Add a column with categories (e.g. yes/no) to classify.
              </Notice>
            )}
            <ul className="target-list" role="radiogroup" aria-label="Target column">
              {targets.map((c) => (
                <li key={c.name}>
                  <label className={`target-option ${sel.target === c.name ? "is-on" : ""}`}>
                    <input type="radio" name="target" checked={sel.target === c.name} onChange={() => setTarget(c.name)} />
                    <span className="target-name">{c.name}</span>
                    <KindBadge kind={c.kind} />
                    <span className="mono-meta">{c.unique} classes</span>
                    <Sparkbars histogram={c.histogram} width={60} height={18} />
                  </label>
                </li>
              ))}
            </ul>
            {nonTargets.length > 0 && (
              <details className="quiet-details">
                <summary>{nonTargets.length} column(s) can’t be a target</summary>
                <ul className="reason-list">
                  {nonTargets.map((c) => (
                    <li key={c.name}>
                      <span>{c.name}</span> <span className="muted">— {targetProblem(c, limits.max_classes)}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </section>

          <section className="block">
            <SectionHead
              index="02.2"
              title="Features"
              kicker={`${sel.features.length} of ${featureCols.length} columns selected as inputs.`}
              aside={
                <div className="btn-row">
                  <Button size="sm" onClick={() => update({ features: defaultSelection(profile, sel.target).features })}>
                    Recommended
                  </Button>
                  <Button size="sm" onClick={() => update({ features: [] })}>
                    None
                  </Button>
                  {flagged.length > 0 && (
                    <Button size="sm" variant="accent" onClick={() => update({ features: sel.features.filter((f) => !flagged.includes(f)) })}>
                      Drop {flagged.length} flagged
                    </Button>
                  )}
                </div>
              }
            />
            <div className="table-wrap">
              <table className="feature-table">
                <thead>
                  <tr>
                    <th aria-label="Use as feature" />
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>
                      <Term k="association">Association</Term>
                    </th>
                    <th aria-label="Warnings" />
                  </tr>
                </thead>
                <tbody>
                  {featureCols.map((c) => {
                    const on = sel.features.includes(c.name);
                    const disabled = c.kind === "datetime";
                    const issues = issuesByColumn.get(c.name) ?? [];
                    const asCat = sel.categorical_overrides.includes(c.name);
                    const score = association.get(c.name);
                    return (
                      <tr key={c.name} className={`${on ? "is-on" : "is-off"} ${disabled ? "is-disabled" : ""}`}>
                        <td>
                          <input
                            type="checkbox"
                            checked={on}
                            disabled={disabled}
                            onChange={(e) => toggleFeature(c.name, e.target.checked)}
                            aria-label={`Use ${c.name} as a feature`}
                          />
                        </td>
                        <td className="feature-name">
                          {disabled ? (
                            <Tip content="Dates can’t be used directly. Derive month/weekday columns in your CSV.">{c.name}</Tip>
                          ) : (
                            c.name
                          )}
                        </td>
                        <td>
                          <span className="type-cell">
                            <KindBadge kind={c.kind} as={asCat ? "categorical" : undefined} />
                            {c.low_cardinality_numeric && on && (
                              <Toggle checked={asCat} onChange={(v) => toggleOverride(c.name, v)}>
                                <span className="small">as category</span>
                              </Toggle>
                            )}
                          </span>
                        </td>
                        <td className="num muted">{c.missing ? `${c.missing_pct}%` : "—"}</td>
                        <td>
                          {on && score != null ? (
                            <span className="assoc-cell">
                              <Meter value={score} tone={score >= 0.9 ? "signal" : "teal"} label={`association ${pct(score)}`} />
                              <span className="mono-meta">{score.toFixed(2)}</span>
                            </span>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                        <td>
                          {issues.length > 0 && (
                            <Tip content={issues.map((i) => i.message).join(" ")}>
                              <span className={`flag flag-${issues.some((i) => i.level === "warn") ? "warn" : "info"}`}>
                                {issues.some((i) => i.code === "leakage_suspect") ? "leak?" : issues.length === 1 ? "note" : `${issues.length} notes`}
                              </span>
                            </Tip>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        <aside className="frame-side" aria-live="polite">
          <div className="readiness">
            <div className="readiness-head">
              <span className="eyebrow">Readiness</span>
              {checking ? (
                <span className="status status-busy">
                  <Pulse small /> checking
                </span>
              ) : error ? (
                <span className="status status-bad">can’t train yet</span>
              ) : report ? (
                <span className="status status-good">ready to train</span>
              ) : null}
            </div>

            {error && (
              <Notice level="error" title={error.code === "network" ? "Connection problem" : "This selection won’t work"}>
                {error.message}
              </Notice>
            )}

            {report && (
              <>
                <p className="readiness-line">
                  Predict <strong>{report.target}</strong> from {report.features.length} feature{report.features.length === 1 ? "" : "s"} using{" "}
                  {report.rows_used.toLocaleString()} rows
                  {report.rows_dropped > 0 && <span className="muted"> ({report.rows_dropped} dropped)</span>}.
                </p>

                <h3 className="mini-head">
                  Class balance {report.imbalanced && <span className="flag flag-warn">imbalanced</span>}
                </h3>
                <ClassBalance classes={report.classes} />

                <IssueList
                  issues={report.issues}
                  actions={(i) =>
                    i.column && DROPPABLE.has(i.code) && sel.features.includes(i.column) ? (
                      <Button size="sm" variant="quiet" onClick={() => toggleFeature(i.column!, false)}>
                        Drop
                      </Button>
                    ) : null
                  }
                />
              </>
            )}

            <Button
              variant="accent"
              size="lg"
              className="wide"
              disabled={!report || checking}
              onClick={() => dispatch({ type: "stage", stage: "train" })}
            >
              Continue to training →
            </Button>
          </div>
        </aside>
      </div>
    </div>
  );
}

function ClassBalance({ classes }: { classes: { label: string; count: number }[] }) {
  const total = classes.reduce((a, c) => a + c.count, 0);
  const labels = classes.map((c) => c.label);
  const shown = classes.slice(0, 12);
  return (
    <div className="class-balance">
      <div className="stack stack-lg" role="img" aria-label="Share of each class">
        {classes.map((c) => (
          <span key={c.label} style={{ width: `${(c.count / total) * 100}%`, background: classColour(labels, c.label) }} title={`${c.label}: ${c.count}`} />
        ))}
      </div>
      <ul className="legend">
        {shown.map((c) => (
          <li key={c.label}>
            <i style={{ background: classColour(labels, c.label) }} />
            <span className="legend-label">{c.label}</span>
            <span className="mono-meta">
              {c.count.toLocaleString()} · {pct(c.count / total)}
            </span>
          </li>
        ))}
        {classes.length > shown.length && <li className="muted">+ {classes.length - shown.length} more</li>}
      </ul>
    </div>
  );
}
