"""Test-set metrics, a naive baseline for context, and plain-language
interpretation of what the numbers mean."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _r(v: float) -> float:
    return float(round(float(v), 4))


def score_block(y_true, y_pred) -> dict:
    return {
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": _r(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": _r(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": _r(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": _r(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": _r(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": _r(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _downsample_curve(fpr: np.ndarray, tpr: np.ndarray, max_points: int = 120) -> list[dict]:
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return [{"fpr": _r(a), "tpr": _r(b)} for a, b in zip(fpr, tpr)]


def evaluate(pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, labels: list[str]) -> dict:
    y_pred = pipeline.predict(X_test)
    metrics = score_block(y_test, y_pred)

    train_pred = pipeline.predict(X_train)
    train_metrics = {k: v for k, v in score_block(y_train, train_pred).items() if k in ("accuracy", "balanced_accuracy", "f1_macro")}

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    baseline = {
        "strategy": "most_frequent",
        "label": str(dummy.classes_[np.argmax(dummy.class_prior_)]),
        "accuracy": _r(accuracy_score(y_test, dummy_pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y_test, dummy_pred)),
        "f1_macro": _r(f1_score(y_test, dummy_pred, average="macro", zero_division=0)),
    }

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    per_class = [
        {"label": lab, "precision": _r(p), "recall": _r(r), "f1": _r(f), "support": int(s)}
        for lab, p, r, f, s in zip(labels, precision, recall, f1, support)
    ]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    roc = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        classes = [str(c) for c in pipeline.classes_]
        try:
            if len(classes) == 2:
                # Treat the rarer training class as "positive": it's usually the one people care about.
                counts = y_train.value_counts()
                positive = str(counts.idxmin()) if counts.min() != counts.max() else classes[1]
                pos_idx = classes.index(positive)
                fpr, tpr, _ = roc_curve(y_test, proba[:, pos_idx], pos_label=positive)
                metrics["roc_auc"] = _r(roc_auc_score((y_test == positive).astype(int), proba[:, pos_idx]))
                roc = {"positive_label": positive, "auc": metrics["roc_auc"], "points": _downsample_curve(fpr, tpr)}
            elif len(set(y_test)) == len(classes):
                metrics["roc_auc"] = _r(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes))
        except ValueError:
            # e.g. a class absent from the test split; AUC is undefined, so we omit it.
            pass

    return {
        "metrics": metrics,
        "train_metrics": train_metrics,
        "baseline": baseline,
        "per_class": per_class,
        "confusion": {"labels": labels, "matrix": matrix.astype(int).tolist()},
        "roc": roc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def interpret(ev: dict, cv: dict | None, imbalanced: bool) -> list[dict]:
    """Short, ordered sentences explaining the evaluation to a non-specialist.
    Each has a tone: good | warn | info."""
    out: list[dict] = []
    m, base, n_test = ev["metrics"], ev["baseline"], ev["n_test"]

    if imbalanced:
        # With a dominant class, plain accuracy rewards ignoring the small
        # classes, so the comparison with the naive guess uses balanced accuracy.
        lift = m["balanced_accuracy"] - base["balanced_accuracy"]
        tone = "good" if lift > 0.1 else "warn" if lift <= 0.02 else "info"
        text = (
            f"Correct on {_pct(m['accuracy'])} of the {n_test:,} held-out test rows. Always guessing '{base['label']}' would"
            f" score {_pct(base['accuracy'])} accuracy while never finding any other class. Balanced accuracy, which weighs"
            f" every class equally, is the fairer yardstick: {_pct(m['balanced_accuracy'])} for the model vs"
            f" {_pct(base['balanced_accuracy'])} for that guess"
        )
        text += "." if lift > 0.02 else " — the model barely separates the classes."
    else:
        lift = m["accuracy"] - base["accuracy"]
        tone = "good" if lift > 0.1 else "warn" if lift <= 0.02 else "info"
        text = (
            f"Correct on {_pct(m['accuracy'])} of the {n_test:,} held-out test rows. Always guessing the most common class"
            f" ('{base['label']}') would score {_pct(base['accuracy'])}"
        )
        text += f", so the model adds {lift * 100:+.0f} points." if lift > 0.02 else " — the model barely beats that naive guess."
    out.append({"tone": tone, "text": text})

    per_class = ev["per_class"]
    labels = ev["confusion"]["labels"]
    matrix = np.array(ev["confusion"]["matrix"])
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        out.append({"tone": "good", "text": "No test row was misclassified."})
    else:
        i, j = np.unravel_index(np.argmax(off), off.shape)
        out.append(
            {
                "tone": "info",
                "text": f"The most common mistake: {int(off[i, j])} row(s) that were really '{labels[i]}' were predicted as '{labels[j]}'.",
            }
        )
        weakest = min((c for c in per_class if c["support"] > 0), key=lambda c: c["recall"], default=None)
        if weakest and weakest["recall"] < 0.999:
            out.append(
                {
                    "tone": "warn" if weakest["recall"] < 0.6 else "info",
                    "text": f"It catches {_pct(weakest['recall'])} of real '{weakest['label']}' rows (recall) — its weakest class."
                    f" When it does predict '{weakest['label']}', it's right {_pct(weakest['precision'])} of the time (precision).",
                }
            )

    gap = ev["train_metrics"]["accuracy"] - m["accuracy"]
    if gap > 0.1:
        out.append(
            {
                "tone": "warn",
                "text": f"Training accuracy is {_pct(ev['train_metrics']['accuracy'])} but test accuracy is {_pct(m['accuracy'])}."
                " A gap this big suggests the model memorised training rows (overfitting) — try a simpler setting.",
            }
        )

    if cv and cv.get("scores"):
        lo, hi = min(cv["scores"]), max(cv["scores"])
        unstable = cv["std"] > 0.05
        out.append(
            {
                "tone": "warn" if unstable else "info",
                "text": f"Across {len(cv['scores'])} cross-validation folds of the training data, macro F1 ranged"
                f" {lo:.2f}–{hi:.2f}" + (" — results swing a lot depending on which rows are used." if unstable else ", so the score is fairly stable."),
            }
        )

    if n_test < 50:
        out.append(
            {
                "tone": "info",
                "text": f"With only {n_test} test rows, each mistake moves accuracy by {100 / n_test:.1f} points; treat the scores as rough.",
            }
        )

    roc = ev.get("roc")
    if roc:
        out.append(
            {
                "tone": "info",
                "text": f"ROC AUC is {roc['auc']:.2f}: given one random '{roc['positive_label']}' row and one random other row,"
                f" the model gives the '{roc['positive_label']}' row the higher score {_pct(roc['auc'])} of the time (50% = coin flip).",
            }
        )
    return out
