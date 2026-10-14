"""Column typing and summary statistics."""

import numpy as np
import pandas as pd

from signallens.core.profiling import preview_rows, profile_dataset

from .conftest import load_frame


def _by_name(profile):
    return {c["name"]: c for c in profile["column_profiles"]}


def test_kinds_are_inferred():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "row_number": np.arange(1, n + 1),
            "joined": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "age": rng.integers(18, 80, n),
            "plan": rng.choice(["basic", "pro"], n),
            "review": [f"This is a fairly long free-text review number {i} about the product." for i in range(n)],
            "label": rng.choice(["yes", "no"], n),
        }
    )
    _, profile = load_frame(df)
    cols = _by_name(profile)
    assert cols["customer_id"]["kind"] == "identifier"
    assert cols["row_number"]["kind"] == "identifier"
    assert cols["joined"]["kind"] == "datetime"
    assert cols["age"]["kind"] == "numeric"
    assert cols["plan"]["kind"] == "categorical"
    assert cols["review"]["kind"] == "text"
    assert profile["suggested_target"] == "label"


def test_missing_counts_and_stats(separable_df):
    df = separable_df.copy()
    df.loc[:9, "signal"] = np.nan
    loaded, profile = load_frame(df)
    sig = _by_name(profile)["signal"]
    assert sig["missing"] == 10
    assert sig["missing_pct"] == 5.0
    assert sig["stats"]["min"] <= sig["stats"]["median"] <= sig["stats"]["max"]
    assert sum(b["count"] for b in sig["histogram"]["bins"]) == len(df) - 10
    assert profile["rows_with_missing"] == 10


def test_categorical_histogram_counts(separable_df):
    _, profile = load_frame(separable_df)
    colour = _by_name(profile)["colour"]
    assert sum(b["count"] for b in colour["histogram"]["bars"]) == len(separable_df)
    assert colour["target_candidate"]


def test_constant_column_flagged():
    df = pd.DataFrame({"k": [1] * 12, "x": range(12), "label": ["a", "b"] * 6})
    _, profile = load_frame(df)
    assert _by_name(profile)["k"]["is_constant"]


def test_preview_paging_sorting_and_missing_filter(separable_df):
    df = separable_df.copy()
    df.loc[[5, 7], "noise"] = np.nan
    loaded, _ = load_frame(df)
    page = preview_rows(loaded, offset=10, limit=5)
    assert page["total"] == len(df) and len(page["rows"]) == 5 and page["rows"][0]["index"] == 10
    only_missing = preview_rows(loaded, missing_only=True)
    assert only_missing["total"] == 2
    assert all(None in r["values"] for r in only_missing["rows"])
    ordered = preview_rows(loaded, sort_by="signal", descending=True, limit=3)
    first = [r["values"][0] for r in ordered["rows"]]
    assert first == sorted(first, reverse=True)
