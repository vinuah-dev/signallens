"""Describe a DataFrame column by column: inferred kind, missingness and a
small statistical summary that the frontend can chart without the raw data."""

from __future__ import annotations

import re
import warnings as pywarnings

import numpy as np
import pandas as pd

from . import config

_ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|^row_?(id|num|number)?$|^unnamed)", re.I)
_DATE_HINT = re.compile(r"\d")
_DATE_SEPARATORS = re.compile(r"[-/:]")

KINDS = ("numeric", "categorical", "identifier", "text", "datetime")
USABLE_KINDS = ("numeric", "categorical")


def jsonable(value):
    """numpy/pandas scalars -> plain Python, NaN/inf -> None."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _is_integer_valued(values: pd.Series) -> bool:
    if values.empty:
        return False
    arr = values.to_numpy(dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(np.mod(arr, 1) == 0))


def _looks_datetime(present: pd.Series) -> bool:
    sample = present.astype(str).head(200)
    if sample.empty:
        return False
    hinted = sample.str.contains(_DATE_HINT) & sample.str.contains(_DATE_SEPARATORS)
    if hinted.mean() < 0.9:
        return False
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


def infer_kind(name: str, s: pd.Series) -> str:
    present = s.dropna()
    n = len(present)
    nunique = int(present.nunique())
    if n == 0:
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        if n >= 20 and nunique == n and _is_integer_valued(present):
            ordered = np.sort(present.to_numpy(dtype=float))
            consecutive = bool(np.all(np.diff(ordered) == 1))
            if _ID_NAME.search(name) or consecutive:
                return "identifier"
        return "numeric"

    as_str = present.astype(str)
    avg_len = float(as_str.str.len().mean())
    unique_ratio = nunique / n
    if _looks_datetime(present):
        return "datetime"
    if avg_len >= config.TEXT_MIN_AVG_LENGTH and unique_ratio > 0.5:
        return "text"
    if n >= 20 and unique_ratio > 0.9:
        return "identifier"
    if _ID_NAME.search(name) and unique_ratio > 0.5:
        return "identifier"
    return "categorical"


def _numeric_summary(present: pd.Series) -> tuple[dict, dict]:
    arr = present.to_numpy(dtype=float)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    stats = {
        "mean": jsonable(arr.mean()),
        "std": jsonable(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": jsonable(arr.min()),
        "q1": jsonable(q1),
        "median": jsonable(median),
        "q3": jsonable(q3),
        "max": jsonable(arr.max()),
        "zeros": int((arr == 0).sum()),
        "integer": _is_integer_valued(present),
    }
    nunique = len(np.unique(arr))
    if stats["integer"] and nunique <= 20:
        counts = present.value_counts().sort_index()
        hist = {"type": "bars", "bars": [{"label": str(int(k)), "count": int(v)} for k, v in counts.items()]}
    else:
        counts, edges = np.histogram(arr, bins=16)
        hist = {
            "type": "bins",
            "bins": [
                {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(c)} for i, c in enumerate(counts)
            ],
        }
    return stats, hist


def _categorical_summary(present: pd.Series, top: int = 12) -> tuple[dict, dict]:
    counts = present.astype(str).value_counts()
    bars = [{"label": str(k), "count": int(v)} for k, v in counts.head(top).items()]
    other = int(counts.iloc[top:].sum()) if len(counts) > top else 0
    stats = {
        "top": bars[0]["label"] if bars else None,
        "top_count": bars[0]["count"] if bars else 0,
        "other_count": other,
        "avg_length": jsonable(present.astype(str).str.len().mean()),
    }
    return stats, {"type": "bars", "bars": bars, "other": other}


def profile_column(name: str, s: pd.Series) -> dict:
    n_rows = len(s)
    present = s.dropna()
    missing = int(n_rows - len(present))
    nunique = int(present.nunique())
    kind = infer_kind(name, s)

    if kind == "numeric":
        stats, hist = _numeric_summary(present) if len(present) else ({}, {"type": "bars", "bars": []})
    else:
        stats, hist = _categorical_summary(present)

    low_card_numeric = kind == "numeric" and stats.get("integer", False) and nunique <= config.LOW_CARDINALITY_NUMERIC
    target_candidate = (
        2 <= nunique <= config.MAX_CLASSES
        and kind in ("categorical", "numeric")
        and (kind == "categorical" or (stats.get("integer") and nunique <= 20))
    )

    return {
        "name": name,
        "kind": kind,
        "dtype": str(s.dtype),
        "missing": missing,
        "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
        "unique": nunique,
        "unique_pct": round(100 * nunique / len(present), 2) if len(present) else 0.0,
        "is_constant": nunique <= 1,
        "high_cardinality": kind == "categorical" and nunique > config.ONE_HOT_MAX_CATEGORIES,
        "low_cardinality_numeric": bool(low_card_numeric),
        "target_candidate": bool(target_candidate),
        "stats": stats,
        "histogram": hist,
        "sample_values": [jsonable(v) for v in present.head(5).tolist()],
    }


def suggest_target(columns: list[dict]) -> str | None:
    candidates = [c for c in columns if c["target_candidate"]]
    if not candidates:
        return None
    # Convention: the label is usually the last column. Otherwise prefer the
    # candidate with the fewest classes, which is most likely a label.
    if columns[-1]["target_candidate"]:
        return columns[-1]["name"]
    return min(candidates, key=lambda c: (c["unique"], -columns.index(c)))["name"]


def profile_dataset(df: pd.DataFrame) -> dict:
    columns = [profile_column(str(c), df[c]) for c in df.columns]
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2) if total_cells else 0.0,
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "kind_counts": {k: sum(1 for c in columns if c["kind"] == k) for k in KINDS},
        "column_profiles": columns,
        "suggested_target": suggest_target(columns),
    }


def preview_rows(
    df: pd.DataFrame,
    offset: int = 0,
    limit: int = 50,
    missing_only: bool = False,
    sort_by: str | None = None,
    descending: bool = False,
) -> dict:
    view = df
    if missing_only:
        view = view[view.isna().any(axis=1)]
    if sort_by and sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=not descending, na_position="last", kind="stable")
    total = len(view)
    page = view.iloc[offset : offset + limit]
    rows = [
        {"index": int(idx), "values": [jsonable(v) for v in row]}
        for idx, row in zip(page.index, page.itertuples(index=False, name=None))
    ]
    return {"total": total, "offset": offset, "limit": limit, "columns": [str(c) for c in df.columns], "rows": rows}
