import { useEffect, useState } from "react";
import { ApiError, api } from "./api/client";
import { Button, Notice, Pulse } from "./components/ui";
import { pct } from "./lib/format";
import { ExplainStage } from "./stages/Explain";
import { FrameStage } from "./stages/Frame";
import { InspectStage } from "./stages/Inspect";
import { ProbeStage } from "./stages/Probe";
import { SourceStage } from "./stages/Source";
import { TrainStage } from "./stages/Train";
import { STAGES, StoreProvider, stageAvailability, useActiveRun, useDispatch, useStore, type Meta, type Stage } from "./state/store";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setError(null);
    Promise.all([api.limits(), api.algorithms(), api.samples()])
      .then(([limits, algorithms, samples]) => setMeta({ limits, algorithms, samples }))
      .catch((e) => setError(e instanceof ApiError ? e : new ApiError("unknown", String(e), 0)));
  }, [attempt]);

  if (error) {
    return (
      <div className="boot">
        <Wordmark />
        <Notice level="error" title="SignalLens can’t start" action={<Button onClick={() => setAttempt((a) => a + 1)}>Retry</Button>}>
          {error.message} If the app was idle, the server may be waking up — this can take ~30 seconds on free hosting.
        </Notice>
      </div>
    );
  }
  if (!meta) {
    return (
      <div className="boot">
        <Wordmark />
        <p className="boot-line">
          <Pulse label="Connecting" /> warming up the instrument…
        </p>
      </div>
    );
  }
  return (
    <StoreProvider meta={meta}>
      <Workbench />
    </StoreProvider>
  );
}

function Wordmark() {
  return (
    <span className="wordmark">
      <svg viewBox="0 0 32 32" aria-hidden="true">
        <circle cx="16" cy="16" r="10" />
        <path d="M2 16h7l2.5-5 4 10 3-7 2 2H30" />
      </svg>
      <span>
        Signal<em>Lens</em>
      </span>
    </span>
  );
}

const STAGE_INFO: Record<Stage, { n: string; title: string; idle: string }> = {
  source: { n: "01", title: "Source", idle: "Upload or pick a sample" },
  frame: { n: "02", title: "Frame", idle: "Choose target & inputs" },
  train: { n: "03", title: "Train", idle: "Fit & compare models" },
  inspect: { n: "04", title: "Inspect", idle: "Metrics & mistakes" },
  explain: { n: "05", title: "Explain", idle: "What drives the model" },
  probe: { n: "06", title: "Probe", idle: "Live what-if predictions" },
};

function Workbench() {
  const state = useStore();
  const dispatch = useDispatch();
  const run = useActiveRun();
  const available = stageAvailability(state);
  const { stage, dataset, frameReport, runs } = state;

  useEffect(() => {
    window.scrollTo({ top: 0 });
    // On narrow screens the rail is a horizontal strip; keep the current step visible.
    document.querySelector(".rail li.is-current")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [stage]);

  const status: Record<Stage, string | null> = {
    source: dataset ? `${dataset.filename} · ${dataset.profile.rows.toLocaleString()} × ${dataset.profile.columns}` : null,
    frame: frameReport ? `${frameReport.target} ← ${frameReport.features.length} features` : null,
    train: runs.length ? `${runs.length} run${runs.length > 1 ? "s" : ""}` : null,
    inspect: run ? `${run.algorithm_label} · ${pct(run.evaluation.metrics.accuracy, 1)}` : null,
    explain: run ? `top: ${run.importance.permutation.items[0]?.feature ?? "—"}` : null,
    probe: run ? `${run.features.length} inputs` : null,
  };
  const currentIndex = STAGES.indexOf(stage);

  return (
    <div className="app">
      <header className="masthead">
        <Wordmark />
        <span className="masthead-tag">an instrument for looking inside classifiers</span>
        <div className="masthead-actions">
          {dataset && (
            <Button size="sm" variant="quiet" onClick={() => dispatch({ type: "dataset/cleared" })}>
              New dataset
            </Button>
          )}
          <a className="masthead-link" href="https://github.com/vinuah-dev/signallens" target="_blank" rel="noreferrer">
            <span className="masthead-link-long">Source on </span>GitHub ↗
          </a>
        </div>
      </header>

      <div className="shell">
        <nav className="rail" aria-label="Workflow">
          <ol>
            {STAGES.map((s, i) => {
              const info = STAGE_INFO[s];
              const done = status[s] != null;
              return (
                <li key={s} className={`${s === stage ? "is-current" : ""} ${done ? "is-done" : ""} ${i < currentIndex ? "is-past" : ""}`}>
                  <button
                    type="button"
                    disabled={!available[s]}
                    aria-current={s === stage ? "step" : undefined}
                    onClick={() => dispatch({ type: "stage", stage: s })}
                  >
                    <span className="rail-node" aria-hidden="true" />
                    <span className="rail-n">{info.n}</span>
                    <span className="rail-title">{info.title}</span>
                    <span className="rail-status">{status[s] ?? info.idle}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>

        <main className="main" id="main">
          {state.lost && (
            <Notice
              level="error"
              title="The server no longer has this session’s data"
              action={
                <Button variant="primary" onClick={() => dispatch({ type: "dataset/cleared" })}>
                  Start over
                </Button>
              }
            >
              It may have restarted (free hosting sleeps when idle). Load the dataset again to continue — it only takes a moment.
            </Notice>
          )}
          {stage === "source" && <SourceStage />}
          {stage === "frame" && dataset && <FrameStage />}
          {stage === "train" && dataset && <TrainStage />}
          {stage === "inspect" && run && <InspectStage />}
          {stage === "explain" && run && <ExplainStage />}
          {stage === "probe" && run && <ProbeStage />}
        </main>
      </div>
    </div>
  );
}
