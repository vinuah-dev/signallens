"""Turn "this column is the target, these are the features" into a validated
(X, y) pair, plus warnings a non-expert should see before training.

A *frame* is the modelling question: predict ``target`` from ``features``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config
from .errors import FrameValidationError

ASSOCIATION_SAMPLE_ROWS = 5_000


@dataclass
class FrameSpec:
    target: str
    features: list[str]
    categorical_overrides: list[str] = field(default_factory=list)


@dataclass
class PreparedFrame:
    X: pd.DataFrame
    y: pd.Series
    labels: list[str]
    kinds: dict[str, str]  # feature -> "numeric" | "categorical"
    report: dict


def _issue(level: str, code: str, message: str, column: str | None = None) -> dict:
    return {"level": level, "code": code, "message": message, "column": column}


def sort_labels(labels) -> list[str]:
    labels = [str(v) for v in labels]
    try:
        return sorted(labels, key=float)
    except ValueError:
        return sorted(labels)


def to_labels(y: pd.Series) -> pd.Series:
    """Class labels are always strings so '1', 'yes' and 'cat' behave the same
    way everywhere. Integer-valued floats ('1.0' due to NaN upcasting) become '1'."""
    if pd.api.types.is_numeric_dtype(y):
        arr = y.to_numpy(dtype=float)
        if np.all(np.mod(arr, 1) == 0):
            return pd.Series(arr.astype(np.int64).astype(str), index=y.index)
    return y.astype(str)


def resolve_kinds(profile_by_name: dict[str, dict], features: list[str], overrides: list[str]) -> dict[str, str]:
    kinds = {}
    for f in features:
        base = profile_by_name[f]["kind"]
        if base == "numeric":
            kinds[f] = "categorical" if f in overrides else "numeric"
        else:
            kinds[f] = "categorical"
    return kinds


def association_scores(X: pd.DataFrame, y: pd.Series, kinds: dict[str, str], random_state: int = 0) -> dict[str, float]:
    """Normalised mutual information between each feature and the target:
    0 = knowing the feature tells you nothing about the class, 1 = it fully
    determines it. This describes the *data*, not the model, and not causes."""
    if len(X) > ASSOCIATION_SAMPLE_ROWS:
        idx = X.sample(ASSOCIATION_SAMPLE_ROWS, random_state=random_state).index
        X, y = X.loc[idx], y.loc[idx]
    encoded = pd.DataFrame(index=X.index)
    discrete = []
    for col in X.columns:
        if kinds[col] == "numeric":
            s = pd.to_numeric(X[col], errors="coerce")
            encoded[col] = s.fillna(s.median() if s.notna().any() else 0.0)
            discrete.append(False)
        else:
            codes, _ = pd.factorize(X[col].astype(str).where(X[col].notna(), "__missing__"))
            encoded[col] = codes
            discrete.append(True)
    counts = y.value_counts(normalize=True).to_numpy()
    entropy = float(-(counts * np.log(counts)).sum())
    if entropy <= 0:
        return {c: 0.0 for c in X.columns}
    mi = mutual_info_classif(encoded.to_numpy(dtype=float), y.to_numpy(), discrete_features=np.array(discrete), random_state=random_state)
    return {c: float(np.clip(v / entropy, 0.0, 1.0)) for c, v in zip(X.columns, mi)}


def prepare_frame(df: pd.DataFrame, profile: dict, spec: FrameSpec, with_association: bool = True) -> PreparedFrame:
    by_name = {c["name"]: c for c in profile["column_profiles"]}
    target, features = spec.target, list(dict.fromkeys(spec.features))
    issues: list[dict] = []

    # --- structural checks -------------------------------------------------
    if not target:
        raise FrameValidationError("Choose a target column — the thing you want to predict.", code="no_target")
    if target not in df.columns:
        raise FrameValidationError(f"Target column '{target}' doesn't exist in this dataset.", code="target_not_found")
    if not features:
        raise FrameValidationError("Select at least one input feature.", code="no_features")
    unknown = [f for f in features if f not in df.columns]
    if unknown:
        raise FrameValidationError(f"Unknown feature column(s): {', '.join(unknown)}.", code="feature_not_found")
    if target in features:
        raise FrameValidationError(
            f"'{target}' is the target, so it can't also be an input — the model would just copy the answer.",
            code="target_in_features",
        )
    bad_overrides = [o for o in spec.categorical_overrides if o not in features]
    if bad_overrides:
        raise FrameValidationError(f"Type override for unselected column(s): {', '.join(bad_overrides)}.", code="bad_override")

    tprof = by_name[target]
    if tprof["kind"] in ("identifier", "text", "datetime"):
        raise FrameValidationError(
            f"'{target}' looks like {'an identifier' if tprof['kind'] == 'identifier' else 'free text' if tprof['kind'] == 'text' else 'a date'}"
            f" ({tprof['unique']:,} distinct values), not a set of classes. Pick a column with a few repeated categories.",
            code="target_not_categorical",
        )
    dt_features = [f for f in features if by_name[f]["kind"] == "datetime"]
    if dt_features:
        raise FrameValidationError(
            f"Date columns aren't supported as raw features ({', '.join(dt_features)}). Derive numeric parts such as"
            " month or weekday in your CSV, or deselect them.",
            code="datetime_feature",
        )

    # --- target ------------------------------------------------------------
    raw_y = df[target]
    missing_target = int(raw_y.isna().sum())
    if missing_target == len(df):
        raise FrameValidationError(f"Every value of '{target}' is missing.", code="target_empty")
    if missing_target:
        issues.append(_issue("warn", "target_missing_rows", f"{missing_target} row(s) have no '{target}' value and are excluded."))
    keep = raw_y.notna()

    present_y = raw_y[keep]
    if pd.api.types.is_numeric_dtype(present_y):
        integer_valued = bool(np.all(np.mod(present_y.to_numpy(dtype=float), 1) == 0))
        if not integer_valued and present_y.nunique() > 20:
            raise FrameValidationError(
                f"'{target}' is a continuous number ({present_y.nunique():,} distinct values). SignalLens builds"
                " classifiers, so the target must be a category. Bin it into groups (e.g. low/medium/high) first.",
                code="target_continuous",
            )
    y = to_labels(present_y)
    counts = y.value_counts()

    if len(counts) < 2:
        raise FrameValidationError(
            f"'{target}' only has one class ('{counts.index[0]}'). A classifier needs at least two classes to tell apart.",
            code="single_class",
        )
    if len(counts) > config.MAX_CLASSES:
        raise FrameValidationError(
            f"'{target}' has {len(counts):,} classes; the limit is {config.MAX_CLASSES}. It may be an ID or a continuous value.",
            code="too_many_classes",
        )

    rare = counts[counts < config.MIN_ROWS_PER_CLASS]
    if len(rare):
        if len(counts) - len(rare) < 2:
            raise FrameValidationError(
                f"After removing classes with fewer than {config.MIN_ROWS_PER_CLASS} rows, fewer than two classes remain.",
                code="too_few_per_class",
            )
        issues.append(
            _issue(
                "warn",
                "rare_classes_dropped",
                f"Class(es) {', '.join(repr(c) for c in rare.index[:5])} have a single row — too few to both train and test on,"
                " so those rows are excluded.",
            )
        )
        y = y[~y.isin(rare.index)]
        counts = y.value_counts()

    X = df.loc[y.index, features].copy()
    if len(X) < config.MIN_ROWS:
        raise FrameValidationError(
            f"Only {len(X)} usable rows remain; at least {config.MIN_ROWS} are needed.", code="too_few_rows"
        )

    # --- features ----------------------------------------------------------
    all_missing = [f for f in features if X[f].isna().all()]
    if all_missing:
        issues.append(_issue("warn", "empty_feature_dropped", f"Dropped feature(s) with no values: {', '.join(all_missing)}."))
        features = [f for f in features if f not in all_missing]
        X = X[features]
        if not features:
            raise FrameValidationError("None of the selected features has any values.", code="no_features")

    kinds = resolve_kinds(by_name, features, spec.categorical_overrides)
    for f in features:
        p = by_name[f]
        miss = float(X[f].isna().mean())
        if X[f].nunique(dropna=True) <= 1:
            issues.append(_issue("info", "constant_feature", f"'{f}' has the same value on every row, so it can't help the model.", f))
        if miss >= 0.4:
            issues.append(
                _issue("warn", "mostly_missing", f"'{f}' is {miss:.0%} missing. Gaps are filled from training data, which may blur its signal.", f)
            )
        if p["kind"] == "identifier":
            issues.append(
                _issue("warn", "identifier_feature", f"'{f}' looks like a row identifier. IDs rarely generalise and can let the model memorise rows.", f)
            )
        elif p["kind"] == "text":
            issues.append(
                _issue("warn", "text_feature", f"'{f}' is free text; it's treated as a category (only the {config.ONE_HOT_MAX_CATEGORIES} most common values are kept).", f)
            )
        elif kinds[f] == "categorical" and X[f].nunique() > config.ONE_HOT_MAX_CATEGORIES:
            issues.append(
                _issue("info", "high_cardinality", f"'{f}' has {X[f].nunique():,} categories; rarer ones are grouped as 'infrequent'.", f)
            )

    # --- class balance -----------------------------------------------------
    ratio = float(counts.min() / counts.max())
    imbalanced = ratio < config.IMBALANCE_WARNING_RATIO
    if imbalanced:
        issues.append(
            _issue(
                "warn",
                "class_imbalance",
                f"Classes are imbalanced: '{counts.idxmin()}' has {counts.min():,} rows vs {counts.max():,} for '{counts.idxmax()}'."
                " Accuracy can look high just by favouring the big class — watch balanced accuracy and per-class recall,"
                " and consider class balancing.",
            )
        )
    small = counts[counts < config.SMALL_CLASS_WARNING]
    if len(small):
        issues.append(
            _issue("info", "small_classes", f"{len(small)} class(es) have fewer than {config.SMALL_CLASS_WARNING} rows; their scores will be noisy.")
        )

    # --- association & leakage --------------------------------------------
    association: dict[str, float] = {}
    if with_association:
        association = association_scores(X, y, kinds)
        for f, score in association.items():
            if score >= config.LEAKAGE_ASSOCIATION_THRESHOLD and by_name[f]["kind"] != "identifier":
                issues.append(
                    _issue(
                        "warn",
                        "leakage_suspect",
                        f"'{f}' almost perfectly predicts '{target}' on its own ({score:.0%} association). If it's only known"
                        " after the outcome (or is derived from it), it leaks the answer — consider removing it.",
                        f,
                    )
                )

    labels = sort_labels(counts.index)
    report = {
        "target": target,
        "features": features,
        "kinds": kinds,
        "rows_used": int(len(X)),
        "rows_dropped": int(len(df) - len(X)),
        "classes": [{"label": lab, "count": int(counts[lab])} for lab in labels],
        "imbalance_ratio": ratio,
        "imbalanced": imbalanced,
        "association": [{"feature": f, "score": association[f]} for f in sorted(association, key=association.get, reverse=True)],
        "issues": issues,
    }
    return PreparedFrame(X=X, y=y, labels=labels, kinds=kinds, report=report)
