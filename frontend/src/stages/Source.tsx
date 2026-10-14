import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { ApiError, api } from "../api/client";
import type { ColumnProfile, Dataset, Kind, RowsPage } from "../api/types";
import { Histogram, Sparkbars } from "../charts/Histogram";
import { HeroLens } from "../components/HeroLens";
import { Button, IssueList, KindBadge, Meter, Notice, Pulse, SectionHead, Segmented, Tip, Toggle } from "../components/ui";
import { bytes, cell, delimiterName, num, pct, plural } from "../lib/format";
import { checkFile } from "../lib/validate";
import { useDispatch, useMeta, useStore } from "../state/store";

export function SourceStage() {
  const { dataset } = useStore();
  return dataset ? <DatasetExplorer dataset={dataset} /> : <Intake />;
}

/* ------------------------------------------------------------------ intake */

const ERROR_HINTS: Record<string, string> = {
  unsupported_format: "SignalLens reads plain-text CSV files.",
  too_few_columns: "Open the file in a text editor and check what separates the values.",
  malformed_csv: "A row has more fields than the header — often an unquoted comma inside a value.",
  too_few_rows: "Add more rows; a model needs examples of every class to learn from and to be tested on.",
};

function Intake() {
  const { samples, limits } = useMeta();
  const dispatch = useDispatch();
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const load = async (label: string, run: () => Promise<Dataset>) => {
    setBusy(label);
    setError(null);
    try {
      dispatch({ type: "dataset/loaded", dataset: await run() });
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError("unknown", String(e), 0));
    } finally {
      setBusy(null);
    }
  };

  const takeFile = (file: File | undefined) => {
    if (!file) return;
    const problem = checkFile(file, limits);
    if (problem) {
      setError(new ApiError("invalid_file", problem, 0));
      return;
    }
    load(file.name, () => api.upload(file));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (busy) return;
    if (e.dataTransfer.files.length > 1) {
      setError(new ApiError("multiple_files", "Drop one file at a time.", 0));
      return;
    }
    takeFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="stage stage-source">
      <div className="hero">
        <div>
          <h1 className="display">
            Point the lens at <em>your</em> data.
          </h1>
          <p className="lede">
            Upload a table where one column is the thing you want to predict. SignalLens profiles every column, trains real
            scikit-learn classifiers on a held-out split, and shows what drives each prediction.
          </p>
          <ol className="promise">
            <li>
              <span className="promise-n">01</span>
              <strong>Profile</strong>
              <span>types, gaps and distributions of every column</span>
            </li>
            <li>
              <span className="promise-n">02</span>
              <strong>Train &amp; compare</strong>
              <span>four models, scored on rows they never saw</span>
            </li>
            <li>
              <span className="promise-n">03</span>
              <strong>Explain &amp; probe</strong>
              <span>what the model relies on, and live what-ifs</span>
            </li>
          </ol>
        </div>
        <div className="hero-art">
          <HeroLens />
        </div>
      </div>

      <div
        className={`dropzone ${drag ? "is-drag" : ""} ${busy ? "is-busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <input
          ref={input}
          type="file"
          accept={limits.extensions.join(",")}
          hidden
          onChange={(e) => {
            takeFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        {busy ? (
          <div className="dropzone-busy">
            <Pulse label="Reading file" />
            <p>
              Reading <strong>{busy}</strong> — detecting the separator, typing columns, counting gaps…
            </p>
          </div>
        ) : (
          <>
            <svg className="dropzone-mark" viewBox="0 0 64 40" aria-hidden="true">
              <rect x="1" y="1" width="62" height="38" rx="3" />
              <path d="M1 13h62M1 26h62M22 1v38M43 1v38" />
            </svg>
            <p className="dropzone-title">Drop a CSV here</p>
            <Button variant="primary" onClick={() => input.current?.click()}>
              Choose file…
            </Button>
            <p className="dropzone-rules">
              .csv / .tsv · up to {bytes(limits.max_upload_bytes)} · ≥ {limits.min_rows} rows · ≥ 2 columns · header row required
            </p>
          </>
        )}
      </div>

      {error && (
        <Notice level="error" title="That file can’t be used" action={<Button size="sm" onClick={() => setError(null)}>Dismiss</Button>}>
          {error.message}
          {ERROR_HINTS[error.code] && <span className="hint"> {ERROR_HINTS[error.code]}</span>}
        </Notice>
      )}

      <section className="samples">
        <SectionHead index="01.0" title="Or start from a sample" kicker="Bundled datasets, each chosen to exercise a different part of the pipeline." />
        <ul className="sample-list">
          {samples.map((s) => (
            <li key={s.key}>
              <div className="sample-shape" aria-hidden="true">
                <span className="sample-rows">{s.rows.toLocaleString()}</span>
                <span className="sample-x">rows × {s.columns} cols</span>
              </div>
              <div className="sample-text">
                <strong>{s.title}</strong>
                <p>{s.blurb}</p>
                <p className="sample-target">
                  predicts <code>{s.target}</code>
                </p>
                <div className="tags">
                  {s.tags.map((t) => (
                    <span key={t} className="tag">
                      {t}
                    </span>
                  ))}
                  <span className="tag tag-quiet">{bytes(s.bytes)}</span>
                </div>
              </div>
              <Button busy={busy === s.title} disabled={busy != null} onClick={() => load(s.title, () => api.loadSample(s.key))}>
                Load
              </Button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

/* ---------------------------------------------------------------- explorer */

type KindFilter = "all" | "numeric" | "categorical" | "other";

function DatasetExplorer({ dataset }: { dataset: Dataset }) {
  const dispatch = useDispatch();
  const { profile } = dataset;
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<KindFilter>("all");
  const [gapsOnly, setGapsOnly] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  const columns = useMemo(
    () =>
      profile.column_profiles.filter((c) => {
        if (query && !c.name.toLowerCase().includes(query.toLowerCase())) return false;
        if (gapsOnly && c.missing === 0) return false;
        if (kind === "other") return !["numeric", "categorical"].includes(c.kind);
        return kind === "all" || c.kind === kind;
      }),
    [profile, query, kind, gapsOnly],
  );

  return (
    <div className="stage stage-source">
      <header className="dataset-head">
        <div>
          <span className="eyebrow">{dataset.source === "sample" ? "Sample dataset" : "Uploaded file"}</span>
          <h1 className="display display-sm">{dataset.filename}</h1>
          <p className="mono-meta">
            {delimiterName(dataset.meta.delimiter)}-separated · {dataset.meta.encoding} · {bytes(dataset.meta.bytes)}
          </p>
        </div>
        <Button onClick={() => dispatch({ type: "dataset/cleared" })}>Use a different file</Button>
      </header>

      <dl className="ledger">
        <div>
          <dt>Rows</dt>
          <dd>{profile.rows.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Columns</dt>
          <dd>{profile.columns}</dd>
          <span className="ledger-sub">
            {(Object.entries(profile.kind_counts) as [Kind, number][])
              .filter(([, n]) => n)
              .map(([k, n]) => `${n} ${k}`)
              .join(" · ")}
          </span>
        </div>
        <div>
          <dt>Missing cells</dt>
          <dd>{pct(profile.missing_pct / 100, 1)}</dd>
          <span className="ledger-sub">{plural(profile.missing_cells, "cell")}</span>
        </div>
        <div>
          <dt>Rows with gaps</dt>
          <dd>{profile.rows_with_missing.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Duplicate rows</dt>
          <dd>{profile.duplicate_rows.toLocaleString()}</dd>
        </div>
      </dl>

      {dataset.warnings.length > 0 && (
        <div className="block">
          <h3 className="mini-head">What we fixed or noticed while reading the file</h3>
          <IssueList issues={dataset.warnings} />
        </div>
      )}

      <section className="block">
        <SectionHead
          index="01.1"
          title="Columns"
          kicker="Each column’s inferred type, gaps and distribution. Click a row for its full profile."
          aside={
            <div className="filters">
              <input className="search" type="search" placeholder="Find column…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Filter columns by name" />
              <Segmented<KindFilter>
                label="Filter by type"
                value={kind}
                onChange={setKind}
                options={[
                  { value: "all", label: "All" },
                  { value: "numeric", label: "Numeric" },
                  { value: "categorical", label: "Categorical" },
                  { value: "other", label: "Other" },
                ]}
              />
              <Toggle checked={gapsOnly} onChange={setGapsOnly}>
                Has gaps
              </Toggle>
            </div>
          }
        />
        <div className="table-wrap">
          <table className="columns-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Missing</th>
                <th className="num">Distinct</th>
                <th>Distribution</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((c) => (
                <ColumnRow key={c.name} c={c} rows={profile.rows} open={open === c.name} onToggle={() => setOpen(open === c.name ? null : c.name)} />
              ))}
              {columns.length === 0 && (
                <tr>
                  <td colSpan={6} className="empty-row">
                    No columns match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block">
        <RowsBrowser dataset={dataset} />
      </section>

      <footer className="stage-foot">
        <p>
          Next, pick the column to predict{profile.suggested_target ? <> — <strong>{profile.suggested_target}</strong> looks like a label</> : null}.
        </p>
        <Button variant="accent" size="lg" onClick={() => dispatch({ type: "stage", stage: "frame" })}>
          Frame the question →
        </Button>
      </footer>
    </div>
  );
}

function numericSummary(c: ColumnProfile): string {
  const s = c.stats;
  return s.median == null ? "" : `median ${num(s.median as number)} · ${num(s.min as number)} – ${num(s.max as number)}`;
}

function ColumnRow({ c, rows, open, onToggle }: { c: ColumnProfile; rows: number; open: boolean; onToggle: () => void }) {
  const present = rows - c.missing;
  const topShare = c.stats.top_count != null && present ? (c.stats.top_count as number) / present : null;
  return (
    <>
      <tr className={`column-row ${open ? "is-open" : ""}`} onClick={onToggle}>
        <td>
          <button type="button" className="row-toggle" aria-expanded={open} onClick={(e) => (e.stopPropagation(), onToggle())}>
            <span className="caret" aria-hidden="true">
              ▸
            </span>
            {c.name}
          </button>
          {c.is_constant && <span className="flag">constant</span>}
        </td>
        <td>
          <KindBadge kind={c.kind} />
        </td>
        <td>
          <span className="missing-cell">
            <Meter value={c.missing_pct / 100} tone={c.missing_pct > 40 ? "signal" : "amber"} label={`${c.missing_pct}% missing`} />
            <span className={c.missing ? "" : "muted"}>{c.missing ? `${c.missing_pct}%` : "none"}</span>
          </span>
        </td>
        <td className="num">{c.unique.toLocaleString()}</td>
        <td>
          <Sparkbars histogram={c.histogram} />
        </td>
        <td className="summary-cell">
          {c.kind === "numeric"
            ? numericSummary(c)
            : c.kind === "identifier" && c.unique_pct > 90
              ? "unique on every row"
              : topShare != null
              ? `top “${String(c.stats.top).slice(0, 16)}” · ${pct(topShare)}`
              : ""}
        </td>
      </tr>
      {open && (
        <tr className="column-detail">
          <td colSpan={6}>
            <div className="detail-grid">
              <div>
                <Histogram histogram={c.histogram} total={present} />
              </div>
              <dl className="stat-list">
                {c.kind === "numeric" ? (
                  (["mean", "std", "min", "q1", "median", "q3", "max", "zeros"] as const).map((k) => (
                    <div key={k}>
                      <dt>{{ mean: "mean", std: "std. dev.", min: "min", q1: "25th pct", median: "median", q3: "75th pct", max: "max", zeros: "zeros" }[k]}</dt>
                      <dd>{num(c.stats[k] as number)}</dd>
                    </div>
                  ))
                ) : (
                  <>
                    <div>
                      <dt>distinct values</dt>
                      <dd>{c.unique.toLocaleString()}</dd>
                    </div>
                    <div>
                      <dt>most common</dt>
                      <dd>{String(c.stats.top ?? "—")}</dd>
                    </div>
                    <div>
                      <dt>avg. length</dt>
                      <dd>{num(c.stats.avg_length as number)} chars</dd>
                    </div>
                  </>
                )}
                <div>
                  <dt>missing</dt>
                  <dd>{plural(c.missing, "row")}</dd>
                </div>
                <div>
                  <dt>stored as</dt>
                  <dd>{c.dtype}</dd>
                </div>
              </dl>
              <div className="samples-inline">
                <span className="eyebrow">first values</span>
                <code>{c.sample_values.map((v) => cell(v)).join("  ·  ")}</code>
                {c.low_cardinality_numeric && (
                  <p className="muted small">
                    Whole numbers with only {c.unique} values — these may be codes for categories. You can treat it as categorical in the next step.
                  </p>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function RowsBrowser({ dataset }: { dataset: Dataset }) {
  const dispatch = useDispatch();
  const [page, setPage] = useState<RowsPage>(dataset.preview);
  const [offset, setOffset] = useState(0);
  const [missingOnly, setMissingOnly] = useState(false);
  const [sort, setSort] = useState<{ by: string; desc: boolean } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const limit = 50;
  const first = useRef(true);

  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .rows(dataset.id, { offset, limit, missingOnly, sortBy: sort?.by, descending: sort?.desc })
      .then((p) => !cancelled && (setPage(p), setError(null)))
      .catch((e: ApiError) => {
        if (cancelled) return;
        if (e.isGone) dispatch({ type: "lost" });
        setError(e.message);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset.id, offset, missingOnly, sort, dispatch]);

  const toggleSort = (col: string) => {
    setOffset(0);
    setSort((s) => (s?.by !== col ? { by: col, desc: false } : s.desc ? null : { by: col, desc: true }));
  };

  return (
    <>
      <SectionHead
        index="01.2"
        title="Rows"
        kicker="The raw table after cleaning. Click a header to sort; empty cells are shown as ∅."
        aside={
          <Toggle
            checked={missingOnly}
            onChange={(v) => {
              setOffset(0);
              setMissingOnly(v);
            }}
          >
            Only rows with gaps
          </Toggle>
        }
      />
      {error && <Notice level="error">{error}</Notice>}
      <div className={`table-wrap data-table-wrap ${loading ? "is-loading" : ""}`}>
        <table className="data-table">
          <thead>
            <tr>
              <th className="idx">#</th>
              {page.columns.map((c) => (
                <th key={c} aria-sort={sort?.by === c ? (sort.desc ? "descending" : "ascending") : "none"}>
                  <button type="button" onClick={() => toggleSort(c)}>
                    {c}
                    <span className="sort-mark">{sort?.by === c ? (sort.desc ? "↓" : "↑") : ""}</span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.rows.map((r) => (
              <tr key={r.index}>
                <td className="idx">{r.index + 1}</td>
                {r.values.map((v, i) => (
                  <td key={i} className={v == null ? "is-null" : typeof v === "number" ? "num" : ""}>
                    {v == null ? "∅" : cell(v)}
                  </td>
                ))}
              </tr>
            ))}
            {page.rows.length === 0 && (
              <tr>
                <td colSpan={page.columns.length + 1} className="empty-row">
                  {missingOnly ? "No row has a missing value. 🎉" : "No rows."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <span className="mono-meta">
          {page.total === 0 ? "0 rows" : `${(offset + 1).toLocaleString()}–${Math.min(offset + limit, page.total).toLocaleString()} of ${page.total.toLocaleString()}`}
        </span>
        <Button size="sm" disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - limit))}>
          ← Prev
        </Button>
        <Button size="sm" disabled={offset + limit >= page.total || loading} onClick={() => setOffset(offset + limit)}>
          Next →
        </Button>
        {loading && <Pulse small label="Loading rows" />}
        <Tip content="Sorting and filtering run on the server, across the whole dataset, not just this page.">
          <span className="muted small">whole-dataset</span>
        </Tip>
      </div>
    </>
  );
}
