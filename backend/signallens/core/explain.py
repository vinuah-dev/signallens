"""Explainability: which features the trained model relies on, globally and
for a single prediction.

Every number here describes the *model's behaviour*. None of it says that
changing a feature in the real world would change the outcome (causation)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .pipeline import output_feature_names, source_feature


def permutation_importances(pipeline, X_test: pd.DataFrame, y_test: pd.Series, random_state: int) -> dict:
    """How much balanced accuracy drops on the *test* split when one feature's
    values are shuffled (breaking its link to the target). Computed on the raw
    columns, so a one-hot encoded category counts as one feature."""
    Xs, ys = X_test, y_test
    if len(Xs) > config.PERMUTATION_MAX_ROWS:
        idx = Xs.sample(config.PERMUTATION_MAX_ROWS, random_state=random_state).index
        Xs, ys = Xs.loc[idx], ys.loc[idx]
    result = permutation_importance(
        pipeline,
        Xs,
        ys,
        scoring="balanced_accuracy",
        n_repeats=config.PERMUTATION_REPEATS,
        random_state=random_state,
    )
    items = [
        {"feature": f, "mean": float(m), "std": float(s)}
        for f, m, s in zip(Xs.columns, result.importances_mean, result.importances_std)
    ]
    items.sort(key=lambda d: d["mean"], reverse=True)
    return {"metric": "balanced_accuracy", "rows": int(len(Xs)), "repeats": config.PERMUTATION_REPEATS, "items": items}


def native_importances(pipeline) -> dict | None:
    """The model's own internal weighting, summed back to original columns.
    Impurity importance (trees) and |coefficients| (linear, on scaled inputs)
    are model-specific and can't be compared across algorithms."""
    model = pipeline.named_steps["model"]
    names = output_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values, kind = np.asarray(model.feature_importances_, dtype=float), "impurity"
    elif hasattr(model, "coef_"):
        values, kind = np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0), "coefficients"
    else:
        return None
    totals: dict[str, float] = defaultdict(float)
    for name, v in zip(names, values):
        totals[source_feature(name)] += float(v)
    grand = sum(totals.values()) or 1.0
    items = [{"feature": f, "share": v / grand} for f, v in totals.items()]
    items.sort(key=lambda d: d["share"], reverse=True)
    return {"kind": kind, "items": items}


def typical_value(spec: dict):
    return spec["median"] if spec["kind"] == "numeric" else spec["mode"]


def local_sensitivity(pipeline, row: pd.DataFrame, schema: list[dict], target_label: str) -> list[dict]:
    """For one input row: if each feature in turn were reset to a typical
    training value, how would the probability of ``target_label`` change?
    Positive delta = this feature's actual value pushes *toward* the label."""
    classes = [str(c) for c in pipeline.classes_]
    k = classes.index(target_label)
    variants = [row]
    for spec in schema:
        v = row.copy()
        v[spec["name"]] = typical_value(spec)
        variants.append(v)
    probs = pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, k]
    p0 = float(probs[0])
    out = []
    for spec, p in zip(schema, probs[1:]):
        current = row.iloc[0][spec["name"]]
        out.append(
            {
                "feature": spec["name"],
                "value": None if pd.isna(current) else (float(current) if spec["kind"] == "numeric" else str(current)),
                "typical": typical_value(spec),
                "probability_if_typical": float(p),
                "delta": p0 - float(p),
            }
        )
    out.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return out


def what_if_curve(pipeline, row: pd.DataFrame, spec: dict, points: int = 40) -> dict:
    """Sweep one feature across its training range, holding the rest fixed,
    and record every class probability (an individual conditional expectation curve)."""
    classes = [str(c) for c in pipeline.classes_]
    if spec["kind"] == "numeric":
        lo, hi = spec["p01"], spec["p99"]
        if lo == hi:
            lo, hi = spec["min"], spec["max"]
        grid = np.linspace(lo, hi, points)
        if spec["integer"]:
            grid = np.unique(np.round(grid))
        xs = [float(x) for x in grid]
    else:
        xs = [c["value"] for c in spec["categories"][:30]]
    variants = pd.concat([row] * len(xs), ignore_index=True)
    variants[spec["name"]] = xs
    proba = pipeline.predict_proba(variants)
    return {
        "feature": spec["name"],
        "kind": spec["kind"],
        "x": xs,
        "series": [{"label": c, "values": [float(v) for v in proba[:, i]]} for i, c in enumerate(classes)],
    }
