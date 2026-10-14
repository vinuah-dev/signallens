"""Orchestrates one training run: validate frame → split → fit → evaluate →
cross-validate → explain. Returns a ``TrainedModel`` that owns the fitted
Pipeline and everything the UI needs to show about it."""

from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config
from .algorithms import ALGORITHMS, build_estimator, resolve_params
from .errors import FrameValidationError, SignalLensError, TrainingError
from .evaluation import evaluate, interpret
from .explain import native_importances, permutation_importances
from .framing import FrameSpec, prepare_frame
from .pipeline import build_pipeline, to_category
from .profiling import jsonable

TEST_EXAMPLES_LIMIT = 500


@dataclass
class TrainConfig:
    algorithm: str
    params: dict = field(default_factory=dict)
    test_size: float = config.DEFAULT_TEST_SIZE
    balance: bool = False
    random_state: int = config.DEFAULT_RANDOM_STATE
    cross_validate: bool = True


@dataclass
class TrainedModel:
    id: str
    dataset_id: str
    pipeline: object
    spec: FrameSpec
    config: TrainConfig
    params: dict
    labels: list[str]
    schema: list[dict]
    summary: dict  # JSON-ready description of the run (no estimator objects)


def feature_schema(X_train: pd.DataFrame, kinds: dict[str, str]) -> list[dict]:
    """Per-feature facts learned from the *training split only*: the input
    form uses them for ranges/defaults and the validator for range checks."""
    schema = []
    for f, kind in kinds.items():
        s = X_train[f]
        missing_rate = float(s.isna().mean())
        if kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
            integer = bool(len(v) and np.all(np.mod(v, 1) == 0))
            p01, p99 = (np.percentile(v, [1, 99]) if len(v) else (0.0, 0.0))
            schema.append(
                {
                    "name": f,
                    "kind": "numeric",
                    "min": float(v.min()) if len(v) else 0.0,
                    "max": float(v.max()) if len(v) else 0.0,
                    "p01": float(p01),
                    "p99": float(p99),
                    "median": float(np.median(v)) if len(v) else 0.0,
                    "mean": float(v.mean()) if len(v) else 0.0,
                    "std": float(v.std()) if len(v) else 0.0,
                    "integer": integer,
                    "missing_rate": missing_rate,
                }
            )
        else:
            counts = s.map(to_category).dropna().value_counts()
            schema.append(
                {
                    "name": f,
                    "kind": "categorical",
                    "categories": [{"value": str(k), "count": int(c)} for k, c in counts.head(200).items()],
                    "n_categories": int(len(counts)),
                    "mode": str(counts.index[0]) if len(counts) else None,
                    "missing_rate": missing_rate,
                }
            )
    return schema


def _split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    n_classes = y.nunique()
    n_test = int(np.ceil(len(y) * test_size))
    if n_test < n_classes or len(y) - n_test < n_classes:
        raise FrameValidationError(
            f"With {len(y)} rows and {n_classes} classes, a {test_size:.0%} test split can't include every class on"
            " both sides. Use a larger test share or more data.",
            code="split_too_small",
        )
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _cross_validate(pipeline, X_train, y_train, random_state) -> dict | None:
    if len(X_train) > config.CV_MAX_TRAIN_ROWS:
        return {"skipped": f"Skipped: more than {config.CV_MAX_TRAIN_ROWS:,} training rows.", "scores": []}
    folds = int(min(5, y_train.value_counts().min()))
    if folds < 2:
        return {"skipped": "Skipped: a class has only one training row.", "scores": []}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clone(pipeline), X_train, y_train, scoring="f1_macro", cv=cv)
    return {"metric": "f1_macro", "folds": folds, "scores": [round(float(s), 4) for s in scores],
            "mean": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)}


def _test_examples(X_test, y_test, pipeline, labels) -> dict:
    Xs, ys = X_test, y_test
    truncated = len(Xs) > TEST_EXAMPLES_LIMIT
    if truncated:
        idx = Xs.sample(TEST_EXAMPLES_LIMIT, random_state=0).index
        Xs, ys = Xs.loc[idx], ys.loc[idx]
    pred = pipeline.predict(Xs)
    proba = pipeline.predict_proba(Xs)
    classes = [str(c) for c in pipeline.classes_]
    rows = []
    for i, (row_idx, row) in enumerate(Xs.iterrows()):
        rows.append(
            {
                "row": int(row_idx),
                "values": {k: jsonable(v) for k, v in row.items()},
                "actual": str(ys.loc[row_idx]),
                "predicted": str(pred[i]),
                "confidence": float(proba[i].max()),
                "probabilities": {c: float(p) for c, p in zip(classes, proba[i])},
            }
        )
    return {"truncated": truncated, "total": int(len(X_test)), "rows": rows}


def train_model(df: pd.DataFrame, profile: dict, dataset_id: str, spec: FrameSpec, cfg: TrainConfig) -> TrainedModel:
    started = time.perf_counter()
    params = resolve_params(cfg.algorithm, cfg.params)
    if not (config.MIN_TEST_SIZE <= cfg.test_size <= config.MAX_TEST_SIZE):
        raise FrameValidationError(
            f"Test share must be between {config.MIN_TEST_SIZE:.0%} and {config.MAX_TEST_SIZE:.0%}.", code="invalid_test_size"
        )

    frame = prepare_frame(df, profile, spec)
    X_train, X_test, y_train, y_test = _split(frame.X, frame.y, cfg.test_size, cfg.random_state)

    estimator = build_estimator(cfg.algorithm, params, cfg.balance, cfg.random_state)
    pipeline = build_pipeline(frame.kinds, estimator)
    run_warnings: list[dict] = []

    fit_started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            run_warnings.append(
                {"level": "warn", "code": "not_converged",
                 "message": "The optimiser stopped before fully converging; results may be slightly off. Try a smaller C."}
            )
    except SignalLensError:
        raise
    except Exception as exc:  # estimator-level failure: surface it, don't crash the API
        raise TrainingError(f"{ALGORITHMS[cfg.algorithm].label} failed to train: {exc}") from exc
    fit_seconds = time.perf_counter() - fit_started

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = evaluate(pipeline, X_train, y_train, X_test, y_test, frame.labels)
            importance = permutation_importances(pipeline, X_test, y_test, cfg.random_state)
            native = native_importances(pipeline)
            examples = _test_examples(X_test, y_test, pipeline, frame.labels)
    except Exception as exc:
        raise TrainingError(f"The model trained but evaluating it failed: {exc}") from exc

    cv = _cross_validate(pipeline, X_train, y_train, cfg.random_state) if cfg.cross_validate else None

    model_id = uuid.uuid4().hex[:12]
    schema = feature_schema(X_train, frame.kinds)
    summary = {
        "id": model_id,
        "dataset_id": dataset_id,
        "created_at": time.time(),
        "algorithm": cfg.algorithm,
        "algorithm_label": ALGORITHMS[cfg.algorithm].label,
        "params": params,
        "balance": cfg.balance,
        "test_size": cfg.test_size,
        "random_state": cfg.random_state,
        # Runs with the same split key were evaluated on the exact same test rows.
        "split_key": f"{spec.target}|{','.join(sorted(frame.kinds))}|{cfg.test_size}|{cfg.random_state}",
        "target": spec.target,
        "features": list(frame.kinds),
        "kinds": frame.kinds,
        "labels": frame.labels,
        "frame": {k: frame.report[k] for k in ("rows_used", "rows_dropped", "classes", "imbalanced", "imbalance_ratio")},
        "evaluation": ev,
        "cross_validation": cv,
        "interpretation": interpret(ev, cv, frame.report["imbalanced"]),
        "importance": {"permutation": importance, "native": native},
        "association": frame.report["association"],
        "schema": schema,
        "test_examples": examples,
        "warnings": frame.report["issues"] + run_warnings,
        "timing": {"fit_seconds": round(fit_seconds, 3), "total_seconds": round(time.perf_counter() - started, 3)},
    }
    return TrainedModel(
        id=model_id,
        dataset_id=dataset_id,
        pipeline=pipeline,
        spec=spec,
        config=cfg,
        params=params,
        labels=frame.labels,
        schema=schema,
        summary=summary,
    )
