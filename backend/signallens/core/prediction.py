"""Validate user-entered feature values and run them through a trained model.

Policy:
  * non-numeric text in a numeric field         -> error (can't guess intent)
  * empty field                                 -> allowed, imputed with the training median/mode (noted)
  * number outside the training range            -> allowed, flagged as extrapolation
  * category never seen in training              -> allowed, flagged (encoded as 'unknown')
  * keys that aren't model features              -> ignored, flagged"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .errors import PredictionInputError
from .explain import local_sensitivity, typical_value
from .pipeline import to_category


def _fmt(v: float) -> str:
    return f"{v:,.4g}"


def validate_row(schema: list[dict], raw: dict) -> tuple[dict, list[dict]]:
    if not isinstance(raw, dict):
        raise PredictionInputError("Each input row must be an object of feature → value.")
    clean: dict = {}
    notes: list[dict] = []
    errors: dict[str, str] = {}
    names = {s["name"] for s in schema}

    for extra in sorted(set(raw) - names):
        notes.append({"level": "info", "feature": extra, "code": "ignored", "message": f"'{extra}' isn't a model feature and was ignored."})

    for spec in schema:
        f = spec["name"]
        value = raw.get(f)
        if value is None or (isinstance(value, str) and not value.strip()):
            clean[f] = np.nan
            if spec["kind"] == "numeric":
                how = f"fills it with the training median ({_fmt(spec['median'])}) and notes that it was missing"
            else:
                how = "treats it as the '(missing)' category, as it did for gaps in training"
            notes.append({"level": "info", "feature": f, "code": "imputed", "message": f"'{f}' is empty, so the model {how}."})
            continue

        if spec["kind"] == "numeric":
            if isinstance(value, bool):
                errors[f] = "Expected a number, got true/false."
                continue
            try:
                number = float(str(value).strip().replace(",", "")) if isinstance(value, str) else float(value)
            except (TypeError, ValueError):
                errors[f] = f"'{value}' isn't a number."
                continue
            if not math.isfinite(number):
                errors[f] = "Must be a finite number."
                continue
            clean[f] = number
            lo, hi = spec["min"], spec["max"]
            if number < lo or number > hi:
                span = (hi - lo) or abs(hi) or 1.0
                far = number < lo - span or number > hi + span
                notes.append(
                    {"level": "warn", "feature": f, "code": "out_of_range_far" if far else "out_of_range",
                     "message": f"{_fmt(number)} is outside the training range for '{f}' ({_fmt(lo)} – {_fmt(hi)}). The model never saw"
                                f" values like this, so the prediction is an extrapolation{' and may be unreliable' if far else ''}."}
                )
            elif spec["integer"] and not float(number).is_integer():
                notes.append({"level": "info", "feature": f, "code": "non_integer",
                              "message": f"'{f}' only had whole numbers in training; {_fmt(number)} is between them."})
        else:
            category = to_category(value)
            clean[f] = category
            known = {c["value"] for c in spec["categories"]}
            if category not in known and spec.get("n_categories", 0) <= len(spec["categories"]):
                notes.append(
                    {"level": "warn", "feature": f, "code": "unseen_category",
                     "message": f"'{category}' never appeared for '{f}' in training; the model treats it as an unknown/rare category."}
                )

    if errors:
        raise PredictionInputError(
            "Some inputs are invalid: " + "; ".join(f"{k}: {v}" for k, v in errors.items()),
            details={"fields": errors},
        )
    return clean, notes


def predict_rows(model, rows: list[dict], explain: bool = True) -> list[dict]:
    if not rows:
        raise PredictionInputError("Provide at least one input row.")
    if len(rows) > 100:
        raise PredictionInputError("At most 100 rows can be predicted per request.")
    validated = [validate_row(model.schema, r) for r in rows]
    frame = pd.DataFrame([v for v, _ in validated], columns=[s["name"] for s in model.schema])
    pipeline = model.pipeline
    classes = [str(c) for c in pipeline.classes_]
    proba = pipeline.predict_proba(frame)
    out = []
    for i, (clean, notes) in enumerate(validated):
        order = np.argsort(proba[i])[::-1]
        predicted = classes[int(order[0])]
        result = {
            "predicted": predicted,
            "confidence": float(proba[i][order[0]]),
            "probabilities": [{"label": classes[j], "probability": float(proba[i][j])} for j in order],
            "inputs": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in clean.items()},
            "notes": notes,
        }
        if explain:
            result["sensitivity"] = local_sensitivity(pipeline, frame.iloc[[i]].reset_index(drop=True), model.schema, predicted)
        out.append(result)
    return out
