import { useState } from "react";
import type { ModelSummary } from "../api/types";
import { ImportanceChart } from "../charts/Importance";
import { LensSlope, fmt, type Lens } from "../charts/LensSlope";
import { SectionHead, Term } from "../components/ui";
import { useActiveRun, useStore } from "../state/store";
import { RunSwitcher } from "./RunSwitcher";

export function ExplainStage() {
  const run = useActiveRun()!;
  return (
    <div className="stage stage-explain">
      <RunSwitcher />
      <Explain key={run.id} run={run} />
    </div>
  );
}

function Explain({ run }: { run: ModelSummary }) {
  const { runs } = useStore();
  const peers = runs.filter((r) => r.id !== run.id && r.split_key === run.split_key);
  const [compareId, setCompareId] = useState<string>("");
  const compare = peers.find((r) => r.id === compareId);
  const perm = run.importance.permutation;
  const native = run.importance.native;
  const top = perm.items[0];
  const unused = perm.items.filter((i) => i.mean <= 0.001).map((i) => i.feature);

  const lenses: Lens[] = [
    {
      key: "assoc",
      title: "In the data",
      subtitle: "association with target",
      values: Object.fromEntries(run.association.map((a) => [a.feature, a.score])),
      format: fmt.score,
    },
  ];
  if (native) {
    lenses.push({
      key: "native",
      title: "Inside the model",
      subtitle: native.kind === "impurity" ? "tree split importance" : "|coefficient| share",
      values: Object.fromEntries(native.items.map((i) => [i.feature, i.share])),
      format: fmt.share,
    });
  }
  lenses.push({
    key: "perm",
    title: "On unseen rows",
    subtitle: "permutation importance",
    values: Object.fromEntries(perm.items.map((i) => [i.feature, i.mean])),
    format: fmt.score,
  });

  return (
    <>
      <div className="intro">
        <h1 className="display display-sm">What is the model paying attention to?</h1>
        <p className="lede">
          Three ways of asking “which features matter?” — and the one question none of them can answer.
        </p>
      </div>

      <section className="block">
        <SectionHead
          index="05.1"
          title={<Term k="permutation">Reliance on each feature</Term>}
          kicker={`Shuffle one feature across the ${perm.rows.toLocaleString()} test rows (${perm.repeats}× each) and measure how much balanced accuracy drops.`}
          aside={
            peers.length > 0 && (
              <label className="compare-select">
                <span className="small muted">compare with</span>
                <select value={compareId} onChange={(e) => setCompareId(e.target.value)}>
                  <option value="">— none —</option>
                  {peers.map((p) => (
                    <option key={p.id} value={p.id}>
                      #{runs.indexOf(p) + 1} {p.algorithm_label}
                    </option>
                  ))}
                </select>
              </label>
            )
          }
        />
        <ImportanceChart
          items={perm.items}
          compare={compare ? { label: `#${runs.indexOf(compare) + 1} ${compare.algorithm_label}`, items: compare.importance.permutation.items } : undefined}
        />
        <p className="caption">
          {compare ? (
            <>
              <span className="key key-ink" /> this run · <span className="key key-compare" /> #{runs.indexOf(compare) + 1} {compare.algorithm_label}. Different
              algorithms can reach similar scores by leaning on different features.{" "}
            </>
          ) : null}
          Whiskers show ±1 standard deviation across repeats.{" "}
          {top && top.mean > 0.001 && (
            <>
              Shuffling <strong>{top.feature}</strong> hurts most.{" "}
            </>
          )}
          {unused.length > 0 && <>{unused.length} feature(s) — e.g. {unused.slice(0, 3).join(", ")} — barely matter to this model; another feature may carry the same information.</>}
        </p>
      </section>

      <section className="block">
        <SectionHead
          index="05.2"
          title="Three lenses on the same features"
          kicker="Each column ranks features by a different question. Disagreements are the interesting part."
        />
        <LensSlope lenses={lenses} />
        <dl className="lens-notes">
          <div>
            <dt>
              <Term k="association">In the data</Term>
            </dt>
            <dd>How much the feature on its own tells you about {run.target}. Says nothing about whether the model uses it.</dd>
          </div>
          {native && (
            <div>
              <dt>
                <Term k={native.kind === "impurity" ? "impurity" : "coefficients"}>Inside the model</Term>
              </dt>
              <dd>The model’s own bookkeeping, measured on training data. Specific to {run.algorithm_label.toLowerCase()} — can’t be compared across algorithms.</dd>
            </div>
          )}
          <div>
            <dt>
              <Term k="permutation">On unseen rows</Term>
            </dt>
            <dd>How much the model’s test performance depends on the feature. Two correlated features can share — and hide — each other’s importance.</dd>
          </div>
        </dl>
      </section>

      <aside className="causation">
        <span className="section-index">§ 05.3</span>
        <h2>None of this is cause and effect.</h2>
        <p>
          These charts describe <em>patterns the model found</em> in this dataset. A feature can be important because it truly drives{" "}
          {run.target}, because it travels with something that does, or because of how the data was collected.
          {top && (
            <>
              {" "}High importance for <strong>{top.feature}</strong> does not mean changing {top.feature} would change {run.target}.
            </>
          )}
        </p>
        <p>
          Establishing cause needs controlled experiments or causal analysis — not a better classifier. Use these views to sanity-check the model
          (is it relying on something sensible? on a <Term k="leakage">leak</Term>?), not to make decisions about the world.
        </p>
      </aside>
    </>
  );
}
